<script>
  import { onMount, onDestroy } from 'svelte';
  import { registry, currentGroup } from './lib/stores/ui.js';
  import { nodes, edges, groups, serializeGraph } from './lib/stores/graph.js';
  import { createDemoGraph } from './lib/defaults.js';
  import { createEditor } from './lib/rete/editor.js';
  import { syncScopeToRete, setupReteListeners } from './lib/rete/sync.js';
  import Breadcrumbs from './lib/components/Breadcrumbs.svelte';

  let loaded = false;
  let containerEl;
  let editorInstance = null;
  let dataFile = 'data/input.txt';
  let buildBusy = false;
  let buildError = '';
  let buildResult = null;
  const cardId = 'default';

  onMount(async () => {
    const resp = await fetch('/components.json');
    const data = await resp.json();
    registry.set(data);
    createDemoGraph();
    loaded = true;
  });

  let lastSyncedGroup = null;

  // When loaded + container ready, create Rete editor
  $: if (loaded && containerEl && !editorInstance) {
    initEditor();
  }

  // Re-sync when currentGroup changes (but not on initial load — initEditor handles that)
  $: if (editorInstance && loaded && lastSyncedGroup !== $currentGroup) {
    syncScope($currentGroup);
  }

  async function initEditor() {
    editorInstance = await createEditor(containerEl);
    setupReteListeners(editorInstance.editor, editorInstance.area, handleDrillIn);
    lastSyncedGroup = $currentGroup;
    await syncScopeToRete(editorInstance.editor, editorInstance.area);
  }

  async function syncScope(group) {
    if (!editorInstance) return;
    lastSyncedGroup = group;
    await syncScopeToRete(editorInstance.editor, editorInstance.area);
  }

  function handleDrillIn(groupPath) {
    currentGroup.set(groupPath);
  }

  async function buildModel() {
    buildBusy = true;
    buildError = '';

    try {
      const payload = {
        ...serializeGraph(),
        card_id: cardId,
        data_file: dataFile,
        graph_mode: true,
      };

      const resp = await fetch('/api/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data?.details?.join('\n') || data?.error || 'Build failed');
      }

      buildResult = data;
    } catch (err) {
      buildError = err.message || String(err);
      buildResult = null;
    } finally {
      buildBusy = false;
    }
  }

  function exportGraph() {
    const blob = new Blob([JSON.stringify(serializeGraph(), null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'microgpt-graph.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  onDestroy(() => {
    editorInstance?.destroy();
  });
</script>

<div id="app">
  {#if loaded}
    <div id="left-panel">
      <div class="panel-header">
        <h3>microGPT</h3>
      </div>
      <div class="panel-body">
        <p class="info">
          {$nodes.length} nodes &middot; {$edges.length} edges &middot; {Object.keys($groups).length} groups
        </p>
      </div>
    </div>

    <div id="canvas-wrap">
      <Breadcrumbs />
      <div class="rete-container" bind:this={containerEl}></div>
    </div>

    <div id="right-panel">
      <div class="panel-header">
        <h3>Build</h3>
      </div>
      <div class="panel-body">
        <label class="field">
          <span>Training Data</span>
          <input bind:value={dataFile} />
        </label>

        <div class="actions">
          <button on:click={buildModel} disabled={buildBusy}>
            {buildBusy ? 'Building…' : 'Build Model'}
          </button>
          <button class="secondary" on:click={exportGraph}>Export JSON</button>
        </div>

        {#if buildError}
          <p class="error">{buildError}</p>
        {/if}

        {#if buildResult?.summary}
          <div class="summary">
            <p><strong>Params</strong> {buildResult.summary.total_params}</p>
            <p><strong>Experts</strong> {buildResult.summary.n_experts}</p>
            <p><strong>Router</strong> {buildResult.summary.router}</p>
            <p><strong>Seq Len</strong> {buildResult.summary.seq_len}</p>
            <p><strong>Vocab</strong> {buildResult.summary.vocab_size}</p>
            {#if buildResult.model_hash}
              <p><strong>Hash</strong> <code>{buildResult.model_hash}</code></p>
            {/if}
            {#if buildResult.graph_warnings?.length}
              <div class="warnings">
                <strong>Warnings</strong>
                {#each buildResult.graph_warnings as warning}
                  <p>{warning}</p>
                {/each}
              </div>
            {/if}
          </div>
        {:else}
          <p class="info">Build the current graph or export it as backend JSON.</p>
        {/if}
      </div>
    </div>
  {:else}
    <div class="loading">Loading...</div>
  {/if}
</div>

<style>
  :global(html, body) {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    height: 100%;
    overflow: hidden;
  }
  :global(*) { box-sizing: border-box; }

  #app {
    display: flex;
    height: 100vh;
  }
  #left-panel, #right-panel {
    width: 260px;
    background: #16213e;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #left-panel { border-right: 1px solid #333; }
  #right-panel { border-left: 1px solid #333; }
  #canvas-wrap {
    flex: 1;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .rete-container {
    flex: 1;
    width: 100%;
    height: 100%;
  }
  .panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid #333;
  }
  .panel-header h3 {
    margin: 0;
    font-size: 14px;
    color: #4a90d9;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .panel-body {
    padding: 12px 16px;
    flex: 1;
    overflow-y: auto;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 12px;
  }
  .field span {
    font-size: 11px;
    color: #8ea5c0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .field input {
    border: 1px solid #31415d;
    background: #0f1a31;
    color: inherit;
    padding: 9px 10px;
    border-radius: 8px;
  }
  .actions {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
  }
  button {
    border: 0;
    border-radius: 8px;
    padding: 9px 12px;
    background: #4a90d9;
    color: white;
    cursor: pointer;
    font-weight: 700;
  }
  button.secondary {
    background: #31415d;
  }
  button:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .summary p, .warnings p {
    margin: 0 0 8px;
    font-size: 12px;
  }
  .summary strong, .warnings strong {
    color: #8ea5c0;
    margin-right: 8px;
  }
  .warnings {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #2a3951;
  }
  .error {
    color: #ff8a8a;
    font-size: 12px;
    white-space: pre-wrap;
  }
  .info {
    font-size: 11px;
    color: #666;
  }
  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #666;
  }
</style>
