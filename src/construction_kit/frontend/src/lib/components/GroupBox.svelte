<script>
  import { createEventDispatcher } from 'svelte';
  import { registry } from '../stores/ui.js';
  import { getCompDef, getDataTypeColor } from '../stores/ui.js';
  import { findNode } from '../stores/graph.js';
  import { portRank, portShapeAttrs } from '../utils/portShapes.js';

  export let groupPath;
  export let groupInfo;   // { label, type, params }
  export let x = 0;
  export let y = 0;
  export let w = 160;
  export let h = 70;
  export let inputEdges = [];   // edges coming from outside into this group
  export let outputEdges = [];  // edges going from inside this group to outside

  const dispatch = createEventDispatcher();

  $: comp = getCompDef(groupInfo?.type, $registry);
  $: color = comp?.color || '#666';
  $: label = groupInfo?.label || groupPath.split('.').pop();

  // Derive input ports from edges
  $: inPorts = deriveInputPorts(inputEdges, $registry);
  $: outPorts = deriveOutputPorts(outputEdges, $registry);

  // Auto-size height based on port count
  $: autoH = Math.max(h, 24 + Math.max(inPorts.length, outPorts.length) * 16);

  function deriveInputPorts(edges, reg) {
    const seen = new Map();
    for (const e of edges) {
      const key = `${e.from.nodeId}:${e.from.portId}`;
      if (seen.has(key)) continue;
      const srcNode = findNode(e.from.nodeId);
      const srcComp = srcNode ? getCompDef(srcNode.type, reg) : null;
      const srcPort = srcComp?.ports?.out?.find(p => p.id === e.from.portId);
      seen.set(key, {
        portId: e.to.portId || e.from.portId,
        dataType: srcPort?.dataType || 'matrix',
        shape: srcPort?.shape || [null, 'd'],
        label: srcPort?.label || e.from.portId,
      });
    }
    return [...seen.values()];
  }

  function deriveOutputPorts(edges, reg) {
    const seen = new Map();
    for (const e of edges) {
      const key = `${e.to.nodeId}:${e.to.portId}`;
      if (seen.has(key)) continue;
      const dstNode = findNode(e.to.nodeId);
      const dstComp = dstNode ? getCompDef(dstNode.type, reg) : null;
      const dstPort = dstComp?.ports?.in?.find(p => p.id === e.to.portId);
      seen.set(key, {
        portId: e.from.portId || e.to.portId,
        dataType: dstPort?.dataType || 'logits',
        shape: dstPort?.shape || [null, 'd'],
        label: dstPort?.label || e.to.portId,
      });
    }
    return [...seen.values()];
  }

  function onDblClick(e) {
    e.stopPropagation();
    dispatch('drillIn', { groupPath });
  }

  function onMouseDown(e) {
    e.stopPropagation();
    dispatch('groupMouseDown', { groupPath, event: e });
  }
</script>

<g
  class="group-box"
  transform="translate({x}, {y})"
  style="cursor: pointer"
  on:dblclick={onDblClick}
  on:mousedown={onMouseDown}
>
  <rect
    width={w} height={autoH} rx="8"
    fill="{color}11"
    stroke={color}
    stroke-width="1.5"
    stroke-dasharray="6,3"
  />
  <text
    x={w / 2} y="16"
    text-anchor="middle"
    fill={color}
    font-size="11"
    font-weight="600"
    pointer-events="none"
  >{label}</text>

  <!-- Input ports (left side) -->
  {#each inPorts as p, i}
    {@const py = 30 + i * 16}
    {@const dtColor = getDataTypeColor(p.dataType, $registry)}
    {@const rank = portRank(p.shape)}
    {@const ps = portShapeAttrs(rank, 0, py, 5)}
    {#if ps.tag === 'rect'}
      <rect {...ps.attrs} fill="transparent" stroke={dtColor} stroke-width="2" />
    {:else if ps.tag === 'polygon'}
      <polygon {...ps.attrs} fill="transparent" stroke={dtColor} stroke-width="2" />
    {:else}
      <circle {...ps.attrs} fill="transparent" stroke={dtColor} stroke-width="2" />
    {/if}
    <text x="10" y={py + 3} fill={dtColor} font-size="7" opacity="0.7" pointer-events="none">
      {p.label}
    </text>
  {/each}

  <!-- Output ports (right side) -->
  {#each outPorts as p, i}
    {@const py = 30 + i * 16}
    {@const dtColor = getDataTypeColor(p.dataType, $registry)}
    {@const rank = portRank(p.shape)}
    {@const ps = portShapeAttrs(rank, w, py, 5)}
    {#if ps.tag === 'rect'}
      <rect {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" />
    {:else if ps.tag === 'polygon'}
      <polygon {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" />
    {:else}
      <circle {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" />
    {/if}
    <text x={w - 10} y={py + 3} text-anchor="end" fill={dtColor} font-size="7" opacity="0.7" pointer-events="none">
      {p.label}
    </text>
  {/each}
</g>
