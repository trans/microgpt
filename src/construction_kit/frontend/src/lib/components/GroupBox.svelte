<script>
  import { createEventDispatcher } from 'svelte';
  import { registry } from '../stores/ui.js';
  import { getCompDef } from '../stores/ui.js';

  export let groupPath;
  export let groupInfo;   // { label, type, params }
  export let x = 0;
  export let y = 0;
  export let w = 160;
  export let h = 70;

  const dispatch = createEventDispatcher();

  $: comp = getCompDef(groupInfo?.type, $registry);
  $: color = comp?.color || '#666';
  $: label = groupInfo?.label || groupPath.split('.').pop();

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
    width={w} height={h} rx="8"
    fill="{color}11"
    stroke={color}
    stroke-width="1.5"
    stroke-dasharray="6,3"
  />
  <text
    x="8" y="-8"
    fill={color}
    font-size="12"
    font-weight="600"
    pointer-events="none"
  >{label}</text>
</g>
