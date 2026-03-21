# microGPT Construction Kit — Frontend Design Document

## Overview

The construction kit is a visual tool for building, training, and experimenting
with small GPT-style language models. Users construct models by wiring components
in a graph, then build and train them. An integrated AI assistant (Claude) can
inspect and modify models through the same graph.

This document describes the frontend architecture for a Svelte-based rebuild.

## Core Concept: Flat Graph with Named Groups

All model components live in a single flat graph. There are no nested children
or boundary edges. The hierarchy users see (cooperative → transformers → layers →
math primitives) is purely visual — implemented as collapsible **named groups**.

### Nodes

Every node is a primitive or compound component:

```
{ id: 42, type: "matmul", group: "coop.xfmr_a.ffn_0", params: {}, x: 300, y: 40 }
```

- `id` — unique integer (or hash-based ID in future)
- `type` — component type from `components.json`
- `group` — dotted path defining visual hierarchy
- `params` — only locally-owned params; inherited params resolved from group chain
- `x, y` — canvas position (relative to group viewport)

### Edges

All edges connect directly between primitives. No boundary edges (nodeId: -1).

```
{ from: { nodeId: 42, portId: "out" }, to: { nodeId: 43, portId: "x" } }
```

### Groups

Groups are metadata, not containers. They define:

```
{
  "coop.xfmr_a": {
    label: "Transformer A",
    type: "transformer",
    params: { d: 64, n_layers: 3 }
  }
}
```

- `label` — display name
- `type` — component type (determines color, icon, available params)
- `params` — owned params (children inherit via the group path)

### Param Inheritance

The group path IS the inheritance chain:

```
"coop.xfmr_a.ffn_0" resolves d:
  1. "coop.xfmr_a.ffn_0".params → no d
  2. "coop.xfmr_a".params → d: 64 ✓
```

Key inherited param: `d` (the universal dimension). Everything flows from it.
FFN uses `ff_mult` (local, default 4) to compute `d_ff = d * ff_mult`.

## UI Architecture

### Component Hierarchy

```
App
├── LeftPanel
│   ├── TreeView          — group hierarchy, expandable
│   └── ChatPanel         — AI assistant
├── GraphCanvas
│   ├── NodeRenderer      — SVG node boxes with ports
│   ├── EdgeRenderer      — SVG bezier curves between ports
│   ├── GroupBox          — collapsed group representation
│   ├── DraftEdge        — wire being drawn
│   └── SelectionOverlay  — multi-select, lasso
├── BottomRail
│   ├── Palette           — drag-and-drop component palette
│   ├── TrainPanel        — training controls and charts
│   └── TestPanel         — text generation console
└── RightRail
    ├── CardList          — engine cards (models)
    ├── PropsPanel        — selected node/group params
    └── CardDetail        — build, train, test buttons
```

### State Management

Central store (Svelte writable stores):

```
graph         — { nodes: [], edges: [], groups: {} }
currentGroup  — string: dotted path of the group being viewed
selectedNode  — number | null: ID of selected node
activeCard    — number: index of active engine card
cards         — array of engine card objects
registry      — component definitions from components.json
```

Derived stores:

```
visibleNodes  — nodes where node.group === currentGroup
childGroups   — direct child groups of currentGroup
visibleEdges  — edges between visible nodes
groupPorts    — auto-derived ports for collapsed group boxes
resolvedParams — inherited params for the current group context
```

### GraphCanvas Component

SVG-based with D3 zoom for pan/zoom (or a lighter alternative).

**Rendering layers** (bottom to top):
1. Grid background
2. Edges (bezier curves)
3. Group boxes (collapsed child groups, dashed border)
4. Nodes (component boxes with ports)
5. Draft edge (wire being drawn)
6. Selection overlay

**Port rendering:**
- Shape indicates tensor rank: circle (scalar/tensor), diamond (vector), square (matrix)
- Color indicates data type: green (Token IDs), blue (Stream), orange (Logits), etc.
- Hollow = input, filled = output
- Small text labels next to each port
- Tooltip on hover: port ID, data type, shape

**Interactions:**
- Click node → select, show props
- Drag node → move
- Click port → start connection (crosshair cursor)
- Drag to port → complete connection (show draft edge while dragging)
- Double-click group box → drill into group (set currentGroup)
- Escape → go up one group level
- Delete key → remove selected node
- Drag from palette → add new node

### Port Connection Flow

1. User mousedown on output port → `connecting = { fromNodeId, fromPortId }`
2. Mousemove → render draft edge (dashed bezier from port to cursor)
3. Mouseup on input port → create edge, clear draft
4. Mouseup elsewhere → cancel, clear draft
5. Validate data type compatibility before connecting

### Group Navigation

- `currentGroup = ""` → show pipeline-level nodes and top-level groups
- `currentGroup = "coop"` → show nodes in "coop" group and its child groups
- `currentGroup = "coop.xfmr_a"` → show transformer internals
- Breadcrumbs: `Root ▸ Cooperative ▸ Transformer A` (clickable)
- Each child group renders as a dashed box with auto-derived ports

### Auto-Derived Group Ports

When a group is collapsed (shown as a box), its ports come from edges that
cross the group boundary:

- Edge from outside → inside = input port on the group box
- Edge from inside → outside = output port on the group box

No manual port definitions needed. The ports update automatically when edges change.

### Palette

Bottom rail, shows available component types for the current group context.
Drag to canvas to add. Categories: data, layers, math, experts, routers, etc.

### Property Panel

Right rail, shows params for the selected node or group:
- Local params: editable
- Inherited params: shown as read-only with resolved value and source
- Group params: shown when a group box is selected

### Engine Cards

Right rail, one card per model:
- Name (editable)
- Build / Rebuild button
- Model summary (params, experts, d, etc.)
- Train / Test / Reset buttons
- Loss sparkline
- Total steps trained

### AI Chat

Left panel tab. Sends graph state as viewscreen context.
Shows tool calls. Applies graph updates from AI responses.

## Data Flow: Build → Train

1. User clicks Build
2. Frontend serializes flat graph: `{ nodes, edges, groups }`
3. POST to `/api/build` with `graph_mode: true`
4. Server compiler reads flat graph:
   - Resolves params via `ParamContext` (walks group path)
   - Creates `ExecutableNode` for each non-skip node
   - Wires edges directly (no boundary resolution)
   - Topo-sorts into `ExecutableGraph`
5. Returns summary (params, experts, d, etc.)
6. User clicks Train
7. WebSocket streams step updates (loss, elapsed, samples)
8. Sparkline updates in real-time

## File Structure

```
src/construction_kit/
  frontend/                   — Svelte project
    src/
      App.svelte              — root layout
      lib/
        stores/
          graph.js            — graph state (nodes, edges, groups)
          cards.js            — engine card state
          ui.js               — UI state (selection, currentGroup, zoom)
        components/
          GraphCanvas.svelte  — main SVG canvas
          Node.svelte         — node box with ports
          Edge.svelte         — bezier edge
          GroupBox.svelte     — collapsed group representation
          Port.svelte         — port shape + label + tooltip
          DraftEdge.svelte    — wire being drawn
          Breadcrumbs.svelte  — group navigation
          Palette.svelte      — component palette
          CardList.svelte     — engine cards rail
          CardDetail.svelte   — single card expanded view
          PropsPanel.svelte   — param editor
          TreeView.svelte     — group hierarchy tree
          ChatPanel.svelte    — AI chat
          TrainPanel.svelte   — training controls
          TestPanel.svelte    — generation console
          Sparkline.svelte    — loss chart
        api/
          build.js            — /api/build, /api/train, /api/generate
          chat.js             — /api/chat
          projects.js         — /api/project/save, /api/project/load
        utils/
          portShapes.js       — createPortShape (circle/diamond/square)
          paramInherit.js     — resolveGroupParam, resolvedParams
          edgePath.js         — bezier curve calculation
    public/
      components.json         — component registry (served by Crystal)
      svg/                    — icons
    vite.config.js
    package.json
  public/                     — build output (served by Crystal server)
    index.html
    app.js                    — compiled Svelte bundle
    style.css
  server.cr                   — Crystal HTTP + WebSocket server
  compiler.cr                 — flat graph compiler
  graph.cr                    — data model (Node, Edge, GroupInfo, ParamContext)
  builder.cr                  — dataset + training orchestration
  executable_graph.cr         — runtime graph executor
  executable_node.cr          — legacy node wrappers
  math_nodes.cr               — math primitive executors
```

## Migration Plan

1. **Set up Svelte project** in `frontend/`, configure Vite to output to `public/`
2. **Port graph state** — use `graph.js` and `defaults.js` from flat-graph branch
3. **Build GraphCanvas** — SVG rendering with D3 zoom, port connections
4. **Build Node/Edge/Port** — individual SVG components
5. **Build GroupBox** — collapsed group with auto-derived ports
6. **Build Palette** — drag to add components
7. **Build CardList** — engine management, build/train
8. **Build PropsPanel** — param editing with inheritance display
9. **Port ChatPanel** — AI integration
10. **Port TrainPanel/TestPanel** — training and generation
11. **Switch Crystal server** to serve from new `public/` output
12. **Remove old app.js** — clean break

Each step produces a working (if incomplete) UI that can be tested.

## Design Principles

- **What you see is what runs.** The visual graph IS the model.
- **Groups are facades.** They organize, they don't contain. The flat graph is the truth.
- **Params inherit.** Children don't store copies. Change `d` once, it flows everywhere.
- **Edges are direct.** No boundary ports, no resolution. Primitive to primitive.
- **Components are reusable.** Select nodes → "Save as Component" → reusable group template.
- **The AI can see and edit everything.** The viewscreen shows the full graph state.
