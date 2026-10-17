<script lang="ts">
	import RecordedTransformer from '../RecordedTransformer.svelte';
	import HybridProjection from './HybridProjection.svelte';
	import HybridCore from './HybridCore.svelte';
	import HybridOutput from './HybridOutput.svelte';
	import HybridInspector from './HybridInspector.svelte';
	import { loadHybridRecording, selectHybridBlock } from '$lib/hybrid/adapter';
	import { hybridRecording, hybridLoading, hybridError, hybridInspector } from '$lib/hybrid/store';
	import { blockIdx } from '~/store';
</script>

<div class="hybrid-transformer">
	<div class="hybrid-toolbar">
		<span>All 38 blocks</span>
		<div class="block-selector" aria-label="Hybrid model blocks">
			{#each $hybridRecording?.document.layers ?? [] as layer}<button
					class:selected={$blockIdx === layer.index}
					class:hybrid={layer.type === 'hybrid'}
					aria-label={`Block ${layer.index + 1}: ${layer.type === 'hybrid' ? 'attention and Mamba' : 'Mamba'}`}
					aria-pressed={$blockIdx === layer.index}
					on:click={() => selectHybridBlock(layer.index)}>{layer.index + 1}</button
				>{/each}
		</div>
		<small>Purple: attention + gated FFN + Mamba · other blocks: Mamba</small>
	</div>
	<RecordedTransformer
		modelLabel="Zamba2 · Transformer–Mamba"
		customQKV={HybridProjection}
		customAttention={HybridCore}
		customFeedForward={HybridOutput}
		loadRecording={loadHybridRecording}
	/>
	{#if $hybridLoading}<div class="status" role="status">
			Loading block {$blockIdx + 1} · actual recorded tensors…
		</div>{/if}
	{#if $hybridError}<div class="status" role="alert">{$hybridError}</div>{/if}
	{#if $hybridInspector}<HybridInspector
			stage={$hybridInspector}
			on:close={() => hybridInspector.set(null)}
		/>{/if}
</div>

<style>
	.hybrid-transformer {
		width: 100%;
		height: 100%;
		display: flex;
		flex-direction: column;
	}
	.hybrid-toolbar {
		position: relative;
		z-index: 150;
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 5px 40px 10px;
		color: #90749f;
		font:
			11px Arial,
			sans-serif;
		flex-shrink: 0;
	}
	.block-selector {
		display: flex;
		gap: 2px;
		flex: 1;
	}
	.block-selector button {
		border: 1px solid #e6daee;
		border-radius: 3px;
		background: #fff;
		color: #ac97bb;
		padding: 5px 0;
		min-width: 20px;
		flex: 1;
		cursor: pointer;
	}
	.block-selector button.hybrid {
		background: #efe2f8;
		color: #9060af;
		border-color: #d7bde8;
	}
	.block-selector button.selected {
		background: #a575c6;
		color: white;
		border-color: #a575c6;
	}
	.hybrid-toolbar small {
		font-size: 9px;
		max-width: 170px;
	}
	.hybrid-transformer :global(.main-section) {
		flex: 1;
		min-height: 0;
	}
	.status {
		position: fixed;
		z-index: 1100;
		right: 30px;
		top: 180px;
		padding: 12px 20px;
		background: #fcf9fe;
		color: #8862a4;
		border: 1px solid #d6bce8;
		border-radius: 6px;
		box-shadow: 0 4px 18px #68458d22;
		font-size: 12px;
	}
</style>
