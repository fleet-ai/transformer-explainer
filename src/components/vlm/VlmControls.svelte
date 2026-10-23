<script lang="ts">
	import { base } from '$app/paths';
	import { recordedAssetUrl } from '$lib/recorded-json.js';
	import { moeTokenIndex } from '~/store';
	import { selectVlmPhase, selectVlmPage } from '$lib/vlm/adapter';
	import {
		vlmRecording,
		vlmPhase,
		vlmVisiblePositions,
		vlmSelectedPosition,
		vlmVisualToken
	} from '$lib/vlm/store';

	/** The tab's own controls, stacked under the token list as on the ViT tab: the image, the
	 *  stream (vision or language) and which window of positions the diagram shows. */
	const WINDOW = 8;
	$: document = $vlmRecording?.document;
	$: stream = document?.streams[$vlmPhase];
	$: n = stream?.sequenceLength ?? 0;
	$: page = Math.floor(($vlmVisiblePositions[0] ?? 0) / WINDOW);
	$: pages = Math.ceil(n / WINDOW);
	$: grid = document?.image.gridSize ?? 16;
	$: blocks = grid / (document?.image.pixelShuffleFactor ?? 4);
	$: shownFrom = ($vlmVisiblePositions[0] ?? 0) + 1;
	$: shownTo = Math.min(n, ($vlmVisiblePositions[0] ?? 0) + WINDOW);
	$: imagePositions = document?.imageTokenPositions ?? [];
	$: cells =
		$vlmPhase === 'vision'
			? Array.from({ length: grid * grid }, (_, patch) => ({
					position: patch,
					label: `patch ${Math.floor(patch / grid)},${patch % grid}`
				}))
			: imagePositions.map((position, k) => ({
					position,
					label: `image ${k} · position ${position}`
				}));
	function show(position: number) {
		selectVlmPage(Math.floor(position / WINDOW));
		const index = $vlmVisiblePositions.indexOf(position);
		if (index >= 0) {
			moeTokenIndex.set(index);
			vlmSelectedPosition.set(position);
		}
		if ($vlmPhase === 'language' && imagePositions.includes(position))
			vlmVisualToken.set(imagePositions.indexOf(position));
	}
</script>

{#if document && stream}<div class="vlm-controls">
		<div
			class="image-grid"
			style={`background-image:url('${base}/${document.image.src}');--cells:${$vlmPhase === 'vision' ? grid : blocks}`}
			role="group"
			aria-label={$vlmPhase === 'vision' ? 'Select image patches' : 'Select image tokens'}
		>
			{#each cells as cell}<button
					type="button"
					class:visible={$vlmVisiblePositions.includes(cell.position)}
					aria-label={`Show ${cell.label}`}
					title={cell.label}
					on:click={() => show(cell.position)}
				></button>{/each}
		</div>
		<div class="phases" role="group" aria-label="Stream">
			<button
				type="button"
				class:active={$vlmPhase === 'vision'}
				aria-pressed={$vlmPhase === 'vision'}
				title="Vision encoder · {document.streams.vision.layers
					.length} blocks over the image patches"
				on:click={() => selectVlmPhase('vision')}
				>Vision · {document.streams.vision.layers.length}</button
			><button
				type="button"
				class:active={$vlmPhase === 'language'}
				aria-pressed={$vlmPhase === 'language'}
				title="Language decoder · {document.streams.language.layers
					.length} blocks over text and image tokens"
				on:click={() => selectVlmPhase('language')}
				>Language · {document.streams.language.layers.length}</button
			>
		</div>
		<div class="pager">
			<button
				type="button"
				aria-label="Previous positions"
				disabled={page === 0}
				on:click={() => selectVlmPage(page - 1)}>‹</button
			><span
				>{$vlmPhase === 'vision' ? 'Patches' : 'Positions'}
				{shownFrom}–{shownTo} / {n}</span
			><button
				type="button"
				aria-label="Next positions"
				disabled={page >= pages - 1}
				on:click={() => selectVlmPage(page + 1)}>›</button
			>
		</div>
		<small
			>{$vlmPhase === 'vision'
				? `All ${n} patches attend to each other; the diagram shows ${$vlmVisiblePositions.length}.`
				: `Image tokens sit at positions ${imagePositions[0]}–${imagePositions[imagePositions.length - 1]}. The last position is always shown.`}</small
		>
		<small class="prompt">Prompt: {document.prompt}</small>
		<a href={recordedAssetUrl(`${base}/traces/vlm/manifest.json`)} target="_blank" rel="noreferrer"
			>Trace provenance ↗</a
		>
	</div>{/if}

<style>
	.vlm-controls {
		display: flex;
		flex-direction: column;
		align-items: stretch;
		gap: 6px;
		width: 108px;
		margin-top: 18px;
		padding: 8px 4px 0;
		border-top: 1px solid #eee7f5;
		color: #8c72a1;
		position: relative;
		z-index: 50;
	}
	.image-grid {
		display: grid;
		grid-template-columns: repeat(var(--cells), 1fr);
		width: 64px;
		height: 64px;
		background-size: cover;
		align-self: center;
		border: 1px solid #e5e7eb;
	}
	.image-grid button {
		border: 0.3px solid #d1c5dd66;
		padding: 0;
		background: transparent;
		cursor: pointer;
	}
	.image-grid button.visible {
		border: 1px solid #9b62c6;
		background: #c8a2e944;
	}
	.image-grid button:hover {
		background: #a465d488;
	}
	.phases {
		display: flex;
		flex-direction: column;
	}
	.phases button,
	.pager button {
		border: 1px solid #e5e7eb;
		background: white;
		color: #6b7280;
		font-size: 10px;
		cursor: pointer;
	}
	.phases button {
		padding: 4px 5px;
		text-align: left;
		font-size: 9.5px;
		white-space: nowrap;
	}
	.phases button + button {
		border-top: 0;
	}
	.phases button.active {
		background: #8b5cf6;
		border-color: #8b5cf6;
		color: white;
	}
	.pager {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 4px;
		font-size: 9.5px;
		white-space: nowrap;
		color: #6b7280;
	}
	.pager button {
		border-radius: 0.3rem;
		padding: 0 5px;
		line-height: 16px;
		box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
	}
	.pager button:disabled {
		opacity: 0.35;
		cursor: default;
	}
	small {
		font-size: 9px;
		line-height: 1.45;
		color: #a898ba;
	}
	a {
		font-size: 9.5px;
		color: #9a61bf;
	}
	button:focus-visible {
		outline: 2px solid #b596cf;
		outline-offset: 1px;
	}
	@media (max-height: 760px) {
		.image-grid {
			width: 52px;
			height: 52px;
		}
		small {
			display: none;
		}
		.vlm-controls {
			margin-top: 8px;
		}
	}
</style>
