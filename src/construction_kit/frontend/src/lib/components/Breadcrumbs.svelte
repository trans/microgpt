<script>
  import { createEventDispatcher } from 'svelte';
  import { currentGroup } from '../stores/ui.js';
  import { groups } from '../stores/graph.js';
  import { registry, getCompDef } from '../stores/ui.js';

  const dispatch = createEventDispatcher();

  $: parts = $currentGroup ? $currentGroup.split('.') : [];
  $: crumbs = buildCrumbs(parts, $groups, $registry);

  function buildCrumbs(parts, groupsVal, reg) {
    const result = [{ label: 'Root', path: '', color: 'var(--text-dim, #888)' }];
    let path = '';
    for (const part of parts) {
      path += (path ? '.' : '') + part;
      const info = groupsVal[path];
      const comp = info ? getCompDef(info.type, reg) : null;
      result.push({
        label: info?.label || part,
        path,
        color: comp?.color || 'var(--text-dim, #888)',
      });
    }
    return result;
  }

  function navigate(path) {
    currentGroup.set(path);
  }
</script>

<nav class="breadcrumbs">
  {#each crumbs as crumb, i}
    {#if i > 0}
      <span class="sep">&#x25B8;</span>
    {/if}
    <button
      class="crumb"
      class:active={crumb.path === $currentGroup}
      style="color: {crumb.color}"
      on:click={() => navigate(crumb.path)}
    >{crumb.label}</button>
  {/each}
</nav>

<style>
  .breadcrumbs {
    display: flex;
    align-items: center;
    gap: 2px;
    padding: 8px 12px;
    font-size: 14px;
  }
  .sep { color: #666; font-size: 12px; margin: 0 2px; }
  .crumb {
    background: none;
    border: none;
    cursor: pointer;
    padding: 2px 6px;
    font-weight: 600;
    font-family: inherit;
    font-size: inherit;
  }
  .crumb:hover { text-decoration: underline; }
  .crumb.active { opacity: 0.6; }
</style>
