<script lang="ts">
	import { base } from '$app/paths';
	import { AngleLeftOutline, AngleRightOutline } from 'flowbite-svelte-icons';
	import { recordedAssetUrl } from '$lib/recorded-json.js';
	import { showVitPatchPage } from '$lib/vit/adapter';
	import { vitRecording, vitVisiblePositions } from '$lib/vit/store';
	/** The image is the input; its grid picks which 8 patches the diagram shows beside [CLS]. */
	const PAGE = 8;
	$: grid = $vitRecording?.document.image.gridSize ?? 14;
	$: patches = grid * grid;
	$: pages = Math.ceil(patches / PAGE);
	$: page = Math.floor((($vitVisiblePositions[1] ?? 1) - 1) / PAGE);
	$: first = page * PAGE;
	$: last = Math.min(patches, first + PAGE) - 1;
	function go(next: number) {
		showVitPatchPage(Math.max(0, Math.min(pages - 1, next)));
	}
</script>

{#if $vitRecording}<div class="vit-patch-controls" role="none" on:click|stopPropagation>
		<div
			class="image-grid"
			style={`background-image:url('${base}/${$vitRecording.document.image.src}');--grid:${grid}`}
			role="group"
			aria-label="Choose which image patches to show"
		>
			{#each Array(patches) as _, patch}<button
					class:visible={$vitVisiblePositions.includes(patch + 1)}
					aria-label={`Show patch ${Math.floor(patch / grid)},${patch % grid}`}
					title={`Patch ${Math.floor(patch / grid)},${patch % grid}`}
					on:click={() => go(Math.floor(patch / PAGE))}
				></button>{/each}
		</div>
		<div class="pager">
			<span class="count" title={`Patches ${first}–${last} of ${patches}`}
				>{page + 1} of {pages}</span
			><button aria-label="Previous patches" disabled={page === 0} on:click={() => go(page - 1)}
				><AngleLeftOutline size="sm" /></button
			><button aria-label="Next patches" disabled={page === pages - 1} on:click={() => go(page + 1)}
				><AngleRightOutline size="sm" /></button
			>
		</div>
		<small
			>[CLS] + {$vitVisiblePositions.length - 1} of {patches} patches shown. Attention uses all {$vitRecording
				.document.sequenceLength} positions.</small
		>
		<a href={recordedAssetUrl(`${base}/traces/vit/manifest.json`)} target="_blank" rel="noreferrer"
			>Trace provenance ↗</a
		>
	</div>{/if}

<style>
	.vit-patch-controls {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 6px;
		width: 108px;
		margin-top: 18px;
		padding-top: 10px;
		border-top: 1px solid #f3f4f6;
		color: #9ca3af;
		position: relative;
		z-index: 50;
		user-select: none;
	}
	.image-grid {
		display: grid;
		grid-template-columns: repeat(var(--grid), 1fr);
		width: 70px;
		height: 70px;
		background-size: cover;
		border: 1px solid #e5e7eb;
	}
	.image-grid button {
		border: 0.3px solid #e5e7eb88;
		cursor: pointer;
	}
	.image-grid button.visible {
		border: 1px solid #9b62c6;
		background: #c8a2e944;
	}
	.image-grid button:hover {
		background: #a465d488;
	}
	.pager {
		display: flex;
		align-items: center;
		gap: 3px;
		font-size: 11px;
		white-space: nowrap;
	}
	.count {
		font-weight: 500;
		margin-right: 2px;
	}
	.pager button {
		border-radius: 0.3rem;
		padding: 0.1rem;
		border: 1px solid #d1d5db;
		color: #9ca3af;
		background: white;
		box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
		cursor: pointer;
		line-height: 0;
	}
	.pager button:disabled {
		color: #e5e7eb;
		border-color: #e5e7eb;
		cursor: default;
	}
	small {
		font-size: 9px;
		line-height: 1.4;
		text-align: center;
	}
	a {
		font-size: 9.5px;
		color: #a78bca;
	}
	@media (max-height: 650px) {
		.image-grid {
			width: 56px;
			height: 56px;
		}
		small {
			display: none;
		}
		.vit-patch-controls {
			margin-top: 8px;
		}
	}
</style>
