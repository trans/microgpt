<script>
  export let edge;        // { id, from: {nodeId, portId}, to: {nodeId, portId} }
  export let fromPos;     // { x, y } or null
  export let toPos;       // { x, y } or null
  export let color = '#555';

  $: visible = fromPos && toPos;
  $: d = visible ? bezierPath(fromPos, toPos) : '';

  function bezierPath(from, to) {
    const dx = Math.abs(to.x - from.x) * 0.5;
    return `M${from.x},${from.y} C${from.x + dx},${from.y} ${to.x - dx},${to.y} ${to.x},${to.y}`;
  }
</script>

{#if visible}
  <path
    class="edge"
    {d}
    fill="none"
    stroke={color}
    stroke-width="1.5"
    data-edge-id={edge.id}
  />
{/if}
