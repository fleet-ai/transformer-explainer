<script lang="ts">
	import { base } from '$app/paths';
	import { blockIdx } from '~/store';
	import RecordedTransformer from '../RecordedTransformer.svelte';
	import VlmEmbedding from './VlmEmbedding.svelte';
	import VlmFeedForward from './VlmFeedForward.svelte';
	import VlmConnectorOutput from './VlmConnectorOutput.svelte';
	import VlmInspector from './VlmInspector.svelte';
	import { loadVlmRecording, selectVlmPhase, selectVlmPage } from '$lib/vlm/adapter';
	import {
		vlmRecording,
		vlmPhase,
		vlmVisiblePositions,
		vlmInspectorStage,
		vlmLoading,
		vlmError
	} from '$lib/vlm/store';
	$: stream = $vlmRecording?.document.streams[$vlmPhase];
	$: page = Math.floor(($vlmVisiblePositions[0] ?? 0) / 8);
</script>

<div class="vlm-transformer">
	{#if $vlmRecording && stream}<div class="vlm-toolbar">
			<img
				src={`${base}/${$vlmRecording.document.image.src}`}
				alt={$vlmRecording.document.image.alt}
			/>
			<div class="details">
				<div class="phases">
					<button class:active={$vlmPhase === 'vision'} on:click={() => selectVlmPhase('vision')}
						>Vision · 12 blocks</button
					><button on:click={() => vlmInspectorStage.set('connector')}
						>Connector · 256 → 16 visual tokens</button
					><button
						class:active={$vlmPhase === 'language'}
						on:click={() => selectVlmPhase('language')}>Language · 30 blocks</button
					>
				</div>
				<div class="window">
					<button disabled={page === 0} on:click={() => selectVlmPage(page - 1)}>‹</button><span
						>Position group {page + 1} / {Math.ceil(stream.sequenceLength / 8)}</span
					><button
						disabled={page >= Math.ceil(stream.sequenceLength / 8) - 1}
						on:click={() => selectVlmPage(page + 1)}>›</button
					><small
						>Showing {$vlmVisiblePositions.length} of {stream.sequenceLength} positions; attention keeps
						its full denominator.</small
					>
				</div>
				<small>Prompt: {$vlmRecording.document.prompt}</small>
			</div>
			<a href={`${base}/traces/vlm/manifest.json`} target="_blank" rel="noreferrer"
				>Trace provenance ↗</a
			>
		</div>{/if}
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
	{#if $vlmInspectorStage}<VlmInspector
			stage={$vlmInspectorStage}
			on:close={() => vlmInspectorStage.set(null)}
		/>{/if}
</div>

<style>
	.vlm-transformer {
		display: flex;
		flex-direction: column;
		width: 100%;
		min-height: 100%;
		position: relative;
	}
	.vlm-toolbar {
		display: flex;
		align-items: center;
		gap: 16px;
		padding: 10px 30px;
		border-bottom: 1px solid #eee7f5;
		background: white;
		position: relative;
		z-index: 50;
		flex-shrink: 0;
		color: #8c72a1;
	}
	.vlm-toolbar img {
		width: 72px;
		height: 72px;
		border: 1px solid #e2d8ec;
	}
	.details {
		flex: 1;
	}
	.phases {
		display: flex;
		gap: 0;
	}
	.phases button {
		font-size: 11px;
		border: 1px solid #dbc9e9;
		background: #fcf9fe;
		padding: 6px 12px;
	}
	.phases button.active {
		background: #ab80cb;
		color: white;
	}
	.window {
		display: flex;
		align-items: center;
		gap: 9px;
		margin: 7px 0;
		font-size: 11px;
	}
	.window button {
		border: 1px solid #ddcdeb;
		border-radius: 3px;
		padding: 0 8px;
		background: #fbf8fd;
	}
	.window button:disabled {
		opacity: 0.3;
	}
	small,
	a {
		font-size: 10px;
	}
	a {
		white-space: nowrap;
	}
	.loading {
		position: fixed;
		right: 30px;
		top: 180px;
		z-index: 1000;
		padding: 12px 20px;
		border: 1px solid #d6bce8;
		box-shadow: 0 4px 18px #68458d22;
		background: #fdfaff;
		color: #8862a4;
		border-radius: 6px;
		font-size: 12px;
	}
</style>
