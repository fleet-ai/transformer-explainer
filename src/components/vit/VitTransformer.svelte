<script lang="ts">
	import { recordedAssetUrl } from '$lib/recorded-json.js';

	import { base } from '$app/paths';
	import RecordedTransformer from '../RecordedTransformer.svelte';
	import VitEmbedding from './VitEmbedding.svelte';
	import VitFeedForward from './VitFeedForward.svelte';
	import VitInspector from './VitInspector.svelte';
	import { loadVitRecording, showVitPatchPage, showVitPositions } from '$lib/vit/adapter';
	import { vitRecording, vitVisiblePositions, vitInspectorStage } from '$lib/vit/store';
	let page = 0;
	function choosePatch(patch: number) {
		page = Math.floor(patch / 8);
		showVitPatchPage(page);
	}
</script>

<div class="vit-transformer">
	{#if $vitRecording}<div class="vit-toolbar">
			<div
				class="image-grid"
				style={`background-image:url('${base}/${$vitRecording.document.image.src}')`}
				aria-label="Select image patches"
			>
				{#each Array(196) as _, patch}<button
						class:visible={$vitVisiblePositions.includes(patch + 1)}
						aria-label={`Show patch ${Math.floor(patch / 14)},${patch % 14}`}
						title={`Patch ${Math.floor(patch / 14)},${patch % 14}`}
						on:click={() => choosePatch(patch)}
					></button>{/each}
			</div>
			<div class="selection">
				<strong>Full DeiT-tiny · 12 blocks · 3 heads · 197 positions</strong>
				<div>
					<button
						disabled={page === 0}
						on:click={() => {
							page--;
							showVitPatchPage(page);
						}}>‹</button
					><span>Patch group {page + 1} / 25</span><button
						disabled={page === 24}
						on:click={() => {
							page++;
							showVitPatchPage(page);
						}}>›</button
					><button on:click={() => vitInspectorStage.set('embedding')}>Image → patches ↗</button
					><button on:click={() => vitInspectorStage.set('classifier')}
						>[CLS] → 1,000 classes ↗</button
					>
				</div>
				<small
					>Diagram: [CLS] + {$vitVisiblePositions.length - 1} patches. All 197 positions participate
					in every recorded attention row; displayed probabilities are not renormalized.</small
				>
			</div>
			<a href={recordedAssetUrl(`${base}/traces/vit/manifest.json`)} target="_blank" rel="noreferrer"
				>Trace provenance ↗</a
			>
		</div>{/if}
	<RecordedTransformer
		modelLabel="Vision Transformer"
		customEmbedding={VitEmbedding}
		customFeedForward={VitFeedForward}
		loadRecording={loadVitRecording}
	/>
	{#if $vitInspectorStage}<VitInspector
			stage={$vitInspectorStage}
			on:close={() => vitInspectorStage.set(null)}
		/>{/if}
</div>

<style>
	.vit-transformer {
		display: flex;
		flex-direction: column;
		width: 100%;
		min-height: 100%;
	}
	.vit-toolbar {
		display: flex;
		align-items: center;
		gap: 16px;
		padding: 10px 30px;
		border-bottom: 1px solid #eee7f5;
		background: white;
		position: relative;
		z-index: 50;
		color: #8c72a1;
		flex-shrink: 0;
	}
	.image-grid {
		display: grid;
		grid-template-columns: repeat(14, 1fr);
		width: 84px;
		height: 84px;
		background-size: cover;
		flex-shrink: 0;
	}
	.image-grid button {
		border: 0.3px solid #d1c5dd;
	}
	.image-grid button.visible {
		border: 1px solid #9b62c6;
		background: #c8a2e944;
	}
	.image-grid button:hover {
		background: #a465d488;
	}
	.selection {
		flex: 1;
		min-width: 0;
	}
	.selection strong {
		font-size: 12px;
	}
	.selection div {
		display: flex;
		gap: 8px;
		align-items: center;
		margin: 6px 0;
		font-size: 11px;
	}
	.selection button {
		border: 1px solid #ddcdeb;
		border-radius: 3px;
		padding: 3px 8px;
		background: #fbf8fd;
	}
	.selection button:disabled {
		opacity: 0.3;
	}
	small {
		font-size: 10px;
	}
	a {
		font-size: 10px;
		white-space: nowrap;
	}
</style>
