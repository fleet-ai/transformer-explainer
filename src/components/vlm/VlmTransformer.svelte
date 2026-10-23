<script lang="ts">
	import { blockIdx } from '~/store';
	import RecordedTransformer from '../RecordedTransformer.svelte';
	import VlmEmbedding from './VlmEmbedding.svelte';
	import VlmFeedForward from './VlmFeedForward.svelte';
	import VlmConnectorOutput from './VlmConnectorOutput.svelte';
	import VlmPopups from './VlmPopups.svelte';
	import { loadVlmRecording, selectVlmPhase } from '$lib/vlm/adapter';
	import { vlmPhase, vlmInspectorStage, vlmLoading, vlmError } from '$lib/vlm/store';
</script>

<div class="vlm-transformer">
	<RecordedTransformer
		modelLabel="Vision-language Transformer"
		customEmbedding={VlmEmbedding}
		customFeedForward={VlmFeedForward}
		customOutput={$vlmPhase === 'vision' ? VlmConnectorOutput : null}
		loadRecording={loadVlmRecording}
	/>
	{#if $vlmLoading}<div class="loading" role="status">
			Loading {$vlmPhase} block {$blockIdx + 1} · full recorded tensors…
		</div>{/if}{#if $vlmError}<div class="loading" role="alert">
			{$vlmError}<button on:click={() => selectVlmPhase($vlmPhase)}>Retry</button>
		</div>{/if}
	{#if $vlmInspectorStage}<VlmPopups
			stage={$vlmInspectorStage}
			on:close={() => vlmInspectorStage.set(null)}
		/>{/if}
</div>

<style>
	.vlm-transformer :global(.main-section) {
		flex: 1;
		min-height: 0;
	}
	.vlm-transformer {
		display: flex;
		flex-direction: column;
		width: 100%;
		height: 100%;
		min-height: 0;
	}
	.loading {
		position: fixed;
		right: 30px;
		top: 140px;
		z-index: 1000;
		padding: 10px 16px;
		border: 1px solid #e5e7eb;
		box-shadow:
			0 4px 6px -1px #0000001a,
			0 2px 4px -2px #0000001a;
		background: white;
		color: #6b7280;
		border-radius: 6px;
		font-size: 12px;
	}
	.loading button {
		margin-left: 10px;
		color: #8b5cf6;
	}
</style>
