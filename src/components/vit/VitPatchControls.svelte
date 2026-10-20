<script lang="ts">
	import { base } from '$app/paths';
	import { recordedAssetUrl } from '$lib/recorded-json.js';
	import { showVitPatchPage } from '$lib/vit/adapter';
	import { vitRecording, vitVisiblePositions, vitInspectorStage } from '$lib/vit/store';
	$: page = Math.floor((($vitVisiblePositions[1] ?? 1) - 1) / 8);
	function choosePatch(patch: number) {
		showVitPatchPage(Math.floor(patch / 8));
	}
</script>

{#if $vitRecording}<div class="vit-patch-controls">
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
			<div class="patch-pager">
				<button
					aria-label="Previous patch group"
					disabled={page === 0}
					on:click={() => {
						page--;
						showVitPatchPage(page);
					}}>‹</button
				><span>Patch group {page + 1} / 25</span><button
					aria-label="Next patch group"
					disabled={page === 24}
					on:click={() => {
						page++;
						showVitPatchPage(page);
					}}>›</button
				>
			</div>
			<div class="patch-actions">
				<button on:click={() => vitInspectorStage.set('embedding')}>Image → patches ↗</button
				><button on:click={() => vitInspectorStage.set('classifier')}
					>[CLS] → 1,000 classes ↗</button
				>
			</div>
			<small
				>Diagram: [CLS] + {$vitVisiblePositions.length - 1} patches. All 197 positions participate in
				every recorded attention row; displayed probabilities are not renormalized.</small
			>
		</div>
		<a href={recordedAssetUrl(`${base}/traces/vit/manifest.json`)} target="_blank" rel="noreferrer"
			>Trace provenance ↗</a
		>
	</div>{/if}

<style>
	.vit-patch-controls {
		display: flex;
		flex-direction: column;
		align-items: stretch;
		gap: 10px;
		width: 180px;
		margin-top: 18px;
		padding: 12px 8px 0;
		border-top: 1px solid #eee7f5;
		color: #8c72a1;
		position: relative;
		z-index: 50;
	}
	.image-grid {
		display: grid;
		grid-template-columns: repeat(14, 1fr);
		width: 98px;
		height: 98px;
		background-size: cover;
		align-self: center;
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
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.selection strong {
		font-size: 11px;
		line-height: 1.5;
		text-align: center;
	}
	.patch-pager {
		display: flex;
		gap: 5px;
		align-items: center;
		justify-content: space-between;
		font-size: 10px;
		white-space: nowrap;
	}
	.patch-actions {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.selection button {
		border: 1px solid #ddcdeb;
		border-radius: 3px;
		padding: 4px 6px;
		background: #fbf8fd;
		font-size: 11px;
	}
	.selection button:disabled {
		opacity: 0.3;
	}
	small {
		font-size: 9px;
		line-height: 1.45;
	}
	a {
		font-size: 10px;
	}
</style>
