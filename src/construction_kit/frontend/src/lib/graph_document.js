// graph_document.js — translate between the frontend's flat grouped graph
// model and the backend's nested GraphDocument format.

function isGroupRef(nodeId) {
  return typeof nodeId === 'string' && nodeId.startsWith('group:');
}

function groupPathFromRef(nodeId) {
  return nodeId.slice(6);
}

function parentPath(path) {
  const dot = path.lastIndexOf('.');
  return dot >= 0 ? path.slice(0, dot) : '';
}

function pathAncestors(path) {
  const result = [];
  let current = path;
  while (true) {
    result.push(current);
    if (!current) break;
    current = parentPath(current);
  }
  return result;
}

function hasBoundaryPorts(groupInfo) {
  return !!(groupInfo?.type && groupInfo?.ports && (groupInfo.ports.in?.length || groupInfo.ports.out?.length));
}

function ownerScopeForGroup(path, groupsValue) {
  return pathAncestors(parentPath(path)).find((candidate) => hasBoundaryPorts(groupsValue[candidate])) || '';
}

function ownerScopeForNode(node, groupsValue) {
  return pathAncestors(node.group || '').find((candidate) => hasBoundaryPorts(groupsValue[candidate])) || '';
}

function stableGroupPaths(groupsValue) {
  return Object.keys(groupsValue)
    .filter((path) => hasBoundaryPorts(groupsValue[path]))
    .sort((a, b) => {
      const depthDiff = a.split('.').length - b.split('.').length;
      return depthDiff !== 0 ? depthDiff : a.localeCompare(b);
    });
}

function buildGroupNodeIds(nodesValue, groupsValue) {
  let nextId = nodesValue.reduce((max, node) => Math.max(max, node.id), 0) + 1;
  const ids = new Map();
  for (const path of stableGroupPaths(groupsValue)) {
    ids.set(path, nextId++);
  }
  return ids;
}

function endpointForScope(nodeId, portId, scopePath, nodeById, groupsValue, groupNodeIds) {
  if (isGroupRef(nodeId)) {
    const groupPath = groupPathFromRef(nodeId);
    if (ownerScopeForGroup(groupPath, groupsValue) !== scopePath) return null;
    const groupNodeId = groupNodeIds.get(groupPath);
    return groupNodeId ? { nodeId: groupNodeId, portId } : null;
  }

  const node = nodeById.get(nodeId);
  if (!node) return null;
  return ownerScopeForNode(node, groupsValue) === scopePath ? { nodeId, portId } : null;
}

function endpointRefForScope(ref, scopePath, nodeById, groupsValue, groupNodeIds) {
  if (!ref) return null;
  return endpointForScope(ref.nodeId, ref.portId, scopePath, nodeById, groupsValue, groupNodeIds);
}

function sortedParamEntries(params = {}) {
  return Object.keys(params)
    .sort()
    .reduce((result, key) => {
      result[key] = params[key];
      return result;
    }, {});
}

function sortGraph(graph) {
  graph.nodes.sort((a, b) => a.id - b.id);
  graph.edges.sort((a, b) => {
    const ak = `${a.from.nodeId}:${a.from.portId}:${a.to.nodeId}:${a.to.portId}`;
    const bk = `${b.from.nodeId}:${b.from.portId}:${b.to.nodeId}:${b.to.portId}`;
    return ak.localeCompare(bk);
  });

  for (const node of graph.nodes) {
    if (node.children) sortGraph(node.children);
  }

  return graph;
}

function serializeScope(scopePath, state, groupNodeIds) {
  const { nodes, edges, groups } = state;
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const graph = { nodes: [], edges: [] };

  const scopedNodes = nodes
    .filter((node) => ownerScopeForNode(node, groups) === scopePath)
    .slice()
    .sort((a, b) => a.id - b.id);

  for (const node of scopedNodes) {
    graph.nodes.push({
      id: node.id,
      type: node.type,
      x: node.x || 0,
      y: node.y || 0,
      params: sortedParamEntries(node.params || {}),
    });
  }

  const childGroups = stableGroupPaths(groups).filter((path) => ownerScopeForGroup(path, groups) === scopePath);
  for (const path of childGroups) {
    const info = groups[path];
    graph.nodes.push({
      id: groupNodeIds.get(path),
      type: info.type,
      x: info._x || 0,
      y: info._y || 0,
      params: sortedParamEntries(info.params || {}),
      children: serializeScope(path, state, groupNodeIds),
    });
  }

  for (const edge of edges) {
    const from = endpointForScope(edge.from.nodeId, edge.from.portId, scopePath, nodeById, groups, groupNodeIds);
    const to = endpointForScope(edge.to.nodeId, edge.to.portId, scopePath, nodeById, groups, groupNodeIds);
    if (!from || !to) continue;
    graph.edges.push({ from, to });
  }

  if (scopePath) {
    const info = groups[scopePath];
    const portMap = info?.portMap || {};

    for (const port of info?.ports?.in || []) {
      for (const ref of portMap[port.id] || []) {
        const target = endpointRefForScope(ref, scopePath, nodeById, groups, groupNodeIds);
        if (!target) continue;
        graph.edges.push({
          from: { nodeId: -1, portId: port.id },
          to: target,
        });
      }
    }

    for (const port of info?.ports?.out || []) {
      for (const ref of portMap[port.id] || []) {
        const source = endpointRefForScope(ref, scopePath, nodeById, groups, groupNodeIds);
        if (!source) continue;
        graph.edges.push({
          from: source,
          to: { nodeId: -1, portId: port.id },
        });
      }
    }
  }

  return sortGraph(graph);
}

export function serializeGraphDocument(state) {
  const groupNodeIds = buildGroupNodeIds(state.nodes, state.groups);
  return {
    version: 2,
    graph: serializeScope('', state, groupNodeIds),
  };
}

function componentFor(type, registryValue) {
  return registryValue?.components?.find((component) => component.type === type) || null;
}

function defaultPortDef(portId, direction) {
  return {
    id: portId,
    label: portId,
    dataType: direction === 'out' ? 'matrix' : 'matrix',
  };
}

function inferPorts(type, portMap, registryValue) {
  const inIds = Object.keys(portMap.in);
  const outIds = Object.keys(portMap.out);
  const comp = componentFor(type, registryValue);

  const pickPorts = (defs = [], ids, direction) => {
    if (defs.length) {
      const used = ids.length ? defs.filter((port) => ids.includes(port.id)) : defs;
      if (used.length) return used.map((port) => ({ ...port }));
    }
    return ids.map((id) => defaultPortDef(id, direction));
  };

  return {
    in: pickPorts(comp?.ports?.in, inIds, 'in'),
    out: pickPorts(comp?.ports?.out, outIds, 'out'),
  };
}

function sanitizeSegment(value) {
  return String(value || 'group')
    .replace(/[^a-zA-Z0-9_]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .toLowerCase() || 'group';
}

function labelFor(type, registryValue) {
  return componentFor(type, registryValue)?.label || type.replace(/_/g, ' ');
}

export function deserializeGraphDocument(docOrGraph, registryValue = null) {
  const sourceGraph = docOrGraph?.graph || docOrGraph || { nodes: [], edges: [] };
  const flatNodes = [];
  const flatEdges = [];
  const flatGroups = {};
  let maxNodeId = 0;

  function walk(graph, currentGroup = '') {
    const memberRefs = new Map();
    const localGroupPaths = new Map();
    const usedSegments = new Set();

    for (const node of graph.nodes || []) {
      maxNodeId = Math.max(maxNodeId, node.id || 0);

      if (node.children) {
        let segment = sanitizeSegment(`${node.type}_${node.id}`);
        while (usedSegments.has(segment)) segment += '_g';
        usedSegments.add(segment);

        const groupPath = currentGroup ? `${currentGroup}.${segment}` : segment;
        localGroupPaths.set(node.id, groupPath);
        memberRefs.set(node.id, `group:${groupPath}`);

        flatGroups[groupPath] = {
          label: labelFor(node.type, registryValue),
          type: node.type,
          params: { ...(node.params || {}) },
          _x: node.x || 0,
          _y: node.y || 0,
        };

        walk(node.children, groupPath);
      } else {
        flatNodes.push({
          id: node.id,
          type: node.type,
          group: currentGroup,
          x: node.x || 0,
          y: node.y || 0,
          params: { ...(node.params || {}) },
        });
        memberRefs.set(node.id, node.id);
      }
    }

    const boundaryMap = { in: {}, out: {} };

    for (const edge of graph.edges || []) {
      if (edge.from.nodeId === -1) {
        const ref = memberRefs.get(edge.to.nodeId);
        if (ref === undefined) continue;
        (boundaryMap.in[edge.from.portId] ||= []).push({ nodeId: ref, portId: edge.to.portId });
        continue;
      }

      if (edge.to.nodeId === -1) {
        const ref = memberRefs.get(edge.from.nodeId);
        if (ref === undefined) continue;
        (boundaryMap.out[edge.to.portId] ||= []).push({ nodeId: ref, portId: edge.from.portId });
        continue;
      }

      const fromRef = memberRefs.get(edge.from.nodeId);
      const toRef = memberRefs.get(edge.to.nodeId);
      if (fromRef === undefined || toRef === undefined) continue;

      flatEdges.push({
        from: { nodeId: fromRef, portId: edge.from.portId },
        to: { nodeId: toRef, portId: edge.to.portId },
      });
    }

    if (currentGroup) {
      flatGroups[currentGroup].portMap = { ...boundaryMap.in, ...boundaryMap.out };
      flatGroups[currentGroup].ports = inferPorts(flatGroups[currentGroup].type, boundaryMap, registryValue);
    }
  }

  walk(sourceGraph, '');

  let nextId = maxNodeId + 1;
  for (const edge of flatEdges) {
    edge.id = nextId++;
  }

  return {
    nodes: flatNodes.sort((a, b) => a.id - b.id),
    edges: flatEdges,
    groups: flatGroups,
    nextId,
  };
}
