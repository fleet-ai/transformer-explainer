<script lang="ts">
	import RecordedTransformer from '../RecordedTransformer.svelte';
	import HybridProjection from './HybridProjection.svelte';
	import HybridCore from './HybridCore.svelte';
	import HybridOutput from './HybridOutput.svelte';
	import HybridPopups from './HybridPopups.svelte';
	import { loadHybridRecording, selectHybridBlock } from '$lib/hybrid/adapter';
	import { hybridRecording, hybridLoading, hybridError, hybridStage } from '$lib/hybrid/store';
	import { blockIdx } from '~/store';
</script>

<div class="hybrid-transformer">
	<div class="hybrid-toolbar">
		<span>All 38 blocks</span>
		<div class="block-selector" aria-label="Hybrid model blocks">
			{#each $hybridRecording?.document.layers ?? [] as layer}{#if layer.type === 'hybrid'}<span
						class="hybrid-pair"
						class:selected={$blockIdx === layer.index}
						><button
							class="half transformer"
							class:on={$blockIdx === layer.index && $hybridStage === 'transformer'}
							aria-label={`Block ${layer.index + 1}, half 1: shared transformer`}
							aria-pressed={$blockIdx === layer.index && $hybridStage === 'transformer'}
							on:click={() => selectHybridBlock(layer.index, 'transformer')}
							>{layer.index + 1}</button
						><button
							class="half mamba"
							class:on={$blockIdx === layer.index && $hybridStage === 'mamba'}
							aria-label={`Block ${layer.index + 1}, half 2: Mamba mixer`}
							aria-pressed={$blockIdx === layer.index && $hybridStage === 'mamba'}
							on:click={() => selectHybridBlock(layer.index, 'mamba')}>M</button
						></span
					>{:else}<button
						class:selected={$blockIdx === layer.index}
						aria-label={`Block ${layer.index + 1}: Mamba`}
						aria-pressed={$blockIdx === layer.index}
						on:click={() => selectHybridBlock(layer.index)}>{layer.index + 1}</button
					>{/if}{/each}
		</div>
		<small>Purple: shared transformer, then Mamba · other blocks: Mamba</small>
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
	<HybridPopups />
</div>

<style>
	@media (max-width: 1150px) {
		.hybrid-toolbar small {
			display: none;
		}
	}
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
		padding: 5px 12px 10px;
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
	.block-selector button.selected,
	.block-selector .half.on {
		background: #a575c6;
		color: white;
		border-color: #a575c6;
	}
	/* A hybrid block is two steps: its shared transformer, then its Mamba mixer. */
	.hybrid-pair {
		display: flex;
		flex: 2;
		gap: 1px;
	}
	.block-selector .half {
		min-width: 0;
	}
	.block-selector .half.transformer {
		background: #efe2f8;
		color: #9060af;
		border-color: #d7bde8;
		border-radius: 3px 0 0 3px;
	}
	.block-selector .half.mamba {
		border-radius: 0 3px 3px 0;
		font-size: 9px;
	}
	.block-selector .half.transformer.on,
	.block-selector .half.mamba.on {
		background: #a575c6;
		color: white;
		border-color: #a575c6;
	}
	.hybrid-toolbar small {
		font-size: 9px;
		max-width: 120px;
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
