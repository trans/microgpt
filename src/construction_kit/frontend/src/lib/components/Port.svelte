<script>
  import { portRank, portShapeAttrs } from '../utils/portShapes.js';

  export let portDef;     // { id, label, dataType, shape }
  export let isOutput;    // boolean
  export let px;          // x position
  export let py;          // y position
  export let color;       // data type color
  export let r = 6;       // port radius

  let showTooltip = false;
  let tooltipX = 0, tooltipY = 0;

  $: rank = portRank(portDef.shape);
  $: shape = portShapeAttrs(rank, px, py, r);
  $: fill = isOutput ? color : 'transparent';
  $: strokeWidth = isOutput ? 2 : 2.5;

  function onEnter(e) {
    const rect = e.target.closest('svg')?.getBoundingClientRect();
    if (!rect) return;
    tooltipX = e.clientX - rect.left + 12;
    tooltipY = e.clientY - rect.top - 10;
    showTooltip = true;
  }
</script>

<g
  class="port"
  class:port-in={!isOutput}
  class:port-out={isOutput}
  style="cursor: crosshair"
  data-port-id={portDef.id}
  data-is-output={isOutput}
  on:mouseenter={onEnter}
  on:mouseleave={() => showTooltip = false}
  on:mousedown
>
  <!-- Invisible hit area -->
  <circle cx={px} cy={py} r={r + 5} fill="transparent" stroke="none" />

  <!-- Visible shape -->
  {#if shape.tag === 'rect'}
    <rect {...shape.attrs} {fill} stroke={color} stroke-width={strokeWidth} pointer-events="none" />
  {:else if shape.tag === 'polygon'}
    <polygon {...shape.attrs} {fill} stroke={color} stroke-width={strokeWidth} pointer-events="none" />
  {:else}
    <circle {...shape.attrs} {fill} stroke={color} stroke-width={strokeWidth} pointer-events="none" />
  {/if}

  <!-- Label -->
  {#if isOutput}
    <text x={px - r - 4} y={py + 3} text-anchor="end" fill={color} font-size="8" opacity="0.7" pointer-events="none">
      {portDef.label || portDef.id}
    </text>
  {:else}
    <text x={px + r + 4} y={py + 3} fill={color} font-size="8" opacity="0.7" pointer-events="none">
      {portDef.label || portDef.id}
    </text>
  {/if}
</g>

{#if showTooltip}
  <g class="port-tooltip-g" transform="translate({tooltipX}, {tooltipY})">
    <rect x="0" y="-12" width="140" height="30" rx="4" fill="#1a1a2e" stroke="#444" opacity="0.95" />
    <text x="4" y="0" fill="#ddd" font-size="10" font-weight="600">{portDef.id}</text>
    <text x="4" y="12" fill="#888" font-size="9">{isOutput ? 'output' : 'input'}: {portDef.dataType || ''}</text>
  </g>
{/if}
