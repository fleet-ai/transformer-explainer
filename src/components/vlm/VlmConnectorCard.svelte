<script lang="ts">
	import { tick } from 'svelte';
	import { AngleLeftOutline, AngleRightOutline } from 'flowbite-svelte-icons';
	import { base } from '$app/paths';
	import { bandRedraw, headGap } from '~/store';
	import { selectVlmPhase } from '$lib/vlm/adapter';
	import {
		vlmRecording,
		vlmPhase,
		vlmVisiblePositions,
		vlmSelectedPosition,
		vlmVisualToken,
		vlmInspectorStage
	} from '$lib/vlm/store';

	/** The image path into the language model, one image token at a time: the 16 patch vectors of
	 *  its 4 × 4 block are laid end to end (pixel shuffle), projected to the language width, and
	 *  placed among the text tokens. The other image tokens are stacked behind. */
	export let className: string | undefined = undefined;
	type Hot = 'shuffle' | 'projector' | 'merge';
	const MAX_GHOSTS = 5;
	const MAX_WIDTH = 180;
	/** Band thickness follows the real width; the 12,288-number packed vector is BAND_PX. */
	const BAND_PX = 36;
	const SVG_HEIGHT = 278;
	let columnWidth = 200;
	let hovered: Hot | null = null;

	$: document = $vlmRecording?.document;
	$: image = document?.image;
	$: grid = image?.gridSize ?? 16;
	$: factor = image?.pixelShuffleFactor ?? 4;
	$: blocks = grid / factor;
	$: tokenCount = blocks * blocks;
	$: group = factor * factor;
	$: visionHidden = document?.streams.vision.model.hiddenSize ?? 768;
	$: languageHidden = document?.streams.language.model.hiddenSize ?? 576;
	$: packed = group * visionHidden;
	$: positions = document?.imageTokenPositions ?? [];
	$: sequence = document?.streams.language.sequenceLength ?? 38;
	$: k = Math.min(Math.max(0, $vlmVisualToken), tokenCount - 1);
	$: target = positions[k] ?? 0;
	$: ghosts = Math.min(MAX_GHOSTS, Math.max(0, tokenCount - 1));
	$: width = Math.max(150, Math.min(MAX_WIDTH, columnWidth - ghosts * headGap.x - 6));
	$: fw = width - 20;
	$: px = (size: number) => Math.max(2.5, (size / packed) * BAND_PX);
	$: strand = (visionHidden / packed) * BAND_PX;

	const blockOf = (patch: number, size: number, f: number) =>
		Math.floor(Math.floor(patch / size) / f) * (size / f) + Math.floor((patch % size) / f);
	// Selecting a patch anywhere on the page shows the image token it belongs to.
	$: if ($vlmPhase === 'vision') vlmVisualToken.set(blockOf($vlmSelectedPosition, grid, factor));
	/** Which shown patch rows feed this image token; the band layer reads it per band. */
	$: members = $vlmVisiblePositions.map((p) => (blockOf(p, grid, factor) === k ? 1 : 0)).join('');
	$: if (width || members || k >= 0) tick().then(() => bandRedraw.update((count) => count + 1));

	// Flow geometry.
	const inY = 30;
	const shuffleY = 108;
	const projY = 182;
	const stripY = 232;
	const BOX_W = 84;
	const BOX_H = 30;
	const xc = 50;
	$: strands = Array.from({ length: group }, (_, s) => {
		const y = inY - BAND_PX / 2 + (s + 0.5) * strand;
		const x = xc + BAND_PX / 2 - (s + 0.5) * strand;
		return `M8 ${y} C${x} ${y} ${x} ${y} ${x} ${shuffleY - BOX_H / 2}`;
	});
	$: cellW = fw / sequence;
	$: targetX = (target + 0.5) * cellW;
	$: gridSize = Math.min(52, fw - BOX_W - 14);
	$: gridX = fw - gridSize;
	$: cell = gridSize / grid;
	$: boxes = <{ id: Hot; y: number; label: string; sub: string }[]>[
		{
			id: 'shuffle',
			y: shuffleY,
			label: 'Pixel shuffle',
			sub: `${group} × ${visionHidden} → ${packed.toLocaleString('en-US')}`
		},
		{
			id: 'projector',
			y: projY,
			label: 'Wproj',
			sub: `${packed.toLocaleString('en-US')} → ${languageHidden}`
		}
	];

	function open(id: Hot) {
		vlmInspectorStage.set(id);
	}
	function keyOpen(event: KeyboardEvent, id: Hot) {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			open(id);
		}
	}
	function choose(next: number) {
		vlmVisualToken.set((next + tokenCount) % tokenCount);
	}
	const titles: Record<Hot, string> = {
		shuffle: 'pixel shuffle (4 × 4 patches → 1 packed vector)',
		projector: 'connector projection (Wproj)',
		merge: 'text and image tokens in the language input'
	};
</script>

<div class={`vlm-connector ${className ?? ''}`}>
	<div class="title">Vision → language</div>
	<div class="content" bind:clientWidth={columnWidth}>
		{#if document && image}
			<div class="stack" style={`width:${width}px;height:360px;`}>
				{#each Array(ghosts) as _, ghost}
					<div
						class="ghost"
						style={`width:${width}px;height:360px;transform:translate(${(ghost + 1) * headGap.x}px,${(ghost + 1) * headGap.y}px);z-index:${ghosts - ghost};opacity:${Math.max(0.2, 1 - (ghost + 1) * 0.16)};`}
					></div>
				{/each}
				<div class="card" style={`width:${width}px;height:360px;z-index:${ghosts + 1};`}>
					<header>
						<strong>Connector · image {k}</strong>
						<span>{grid * grid} × {visionHidden} → {tokenCount} × {languageHidden}</span>
						<button
							type="button"
							class="post-norm"
							title="The vision output already includes the encoder's final LayerNorm"
							on:click={() => vlmInspectorStage.set('post-norm')}
							>input · after post-LayerNorm ↗</button
						>
					</header>
					<svg
						width={fw}
						height={SVG_HEIGHT}
						viewBox={`0 0 ${fw} ${SVG_HEIGHT}`}
						role="group"
						aria-label={`${group} patch vectors of ${visionHidden} packed into ${packed}, projected to ${languageHidden}, placed at language position ${target}`}
					>
						{#each strands as d}<path
								{d}
								style={`stroke-width:${Math.max(1, strand - 0.5)}px`}
								class:hot={hovered === 'shuffle'}
								role="button"
								tabindex="-1"
								aria-label={`Open ${titles.shuffle}`}
								on:click={() => open('shuffle')}
								on:keydown={(event) => keyOpen(event, 'shuffle')}
								on:mouseenter={() => (hovered = 'shuffle')}
								on:mouseleave={() => (hovered = null)}
							/>{/each}
						<path
							d={`M${xc} ${shuffleY + BOX_H / 2} V${projY - BOX_H / 2}`}
							style={`stroke-width:${BAND_PX}px`}
							class:hot={hovered === 'shuffle'}
							role="button"
							tabindex="0"
							aria-label={`Open ${titles.shuffle}`}
							on:click={() => open('shuffle')}
							on:keydown={(event) => keyOpen(event, 'shuffle')}
							on:mouseenter={() => (hovered = 'shuffle')}
							on:mouseleave={() => (hovered = null)}
						/>
						<path
							d={`M${xc} ${projY + BOX_H / 2} C${xc} ${stripY - 12} ${targetX} ${stripY - 18} ${targetX} ${stripY}`}
							style={`stroke-width:${px(languageHidden)}px`}
							class="out-band"
							class:hot={hovered === 'projector'}
							role="button"
							tabindex="0"
							aria-label={`Open ${titles.projector}`}
							on:click={() => open('projector')}
							on:keydown={(event) => keyOpen(event, 'projector')}
							on:mouseenter={() => (hovered = 'projector')}
							on:mouseleave={() => (hovered = null)}
						/>
						<rect
							class="bar vlm-connector-in"
							data-members={members}
							x="0"
							y={inY - BAND_PX / 2}
							width="8"
							height={BAND_PX}
						/>
						<text class="dim" x={xc + BAND_PX / 2 + 4} y={(shuffleY + projY) / 2 + 3}
							>{packed.toLocaleString('en-US')}</text
						>
						<text class="dim" x={xc + 6} y={projY + BOX_H / 2 + 14}>{languageHidden}</text>
						<text class="dim" x="12" y={inY + BAND_PX / 2 + 10}>{group} × {visionHidden}</text>

						<!-- The real input image with the 4 × 4 block this token packs. -->
						<g class="patches" aria-hidden="true">
							<image
								href={`${base}/${image.src}`}
								x={gridX}
								y={inY - gridSize / 2 + 8}
								width={gridSize}
								height={gridSize}
								preserveAspectRatio="none"
							/>
							{#each Array(grid + 1) as _, line}<line
									x1={gridX + line * cell}
									x2={gridX + line * cell}
									y1={inY - gridSize / 2 + 8}
									y2={inY + gridSize / 2 + 8}
								/><line
									x1={gridX}
									x2={gridX + gridSize}
									y1={inY - gridSize / 2 + 8 + line * cell}
									y2={inY - gridSize / 2 + 8 + line * cell}
								/>{/each}
							<rect
								class="block"
								x={gridX + (k % blocks) * factor * cell}
								y={inY - gridSize / 2 + 8 + Math.floor(k / blocks) * factor * cell}
								width={factor * cell}
								height={factor * cell}
							/>
						</g>
						<text class="dim" x={fw} y={inY + gridSize / 2 + 18} text-anchor="end"
							>block {Math.floor(k / blocks)},{k % blocks}</text
						>

						{#each boxes as box}<foreignObject
								x={xc - BOX_W / 2}
								y={box.y - BOX_H / 2}
								width={BOX_W}
								height={BOX_H}
								><button
									type="button"
									class:hot={hovered === box.id}
									title={titles[box.id]}
									aria-label={`Open ${titles[box.id]}`}
									on:click={() => open(box.id)}
									on:mouseenter={() => (hovered = box.id)}
									on:mouseleave={() => (hovered = null)}
									on:focus={() => (hovered = box.id)}
									on:blur={() => (hovered = null)}>{box.label}<small>{box.sub}</small></button
								></foreignObject
							>{/each}

						<text class="dim" x="0" y={stripY - 6}>language input · {sequence} positions</text>
						<g
							class="strip"
							class:hot={hovered === 'merge'}
							role="button"
							tabindex="0"
							aria-label={`Open ${titles.merge}`}
							on:click={() => open('merge')}
							on:keydown={(event) => keyOpen(event, 'merge')}
							on:mouseenter={() => (hovered = 'merge')}
							on:mouseleave={() => (hovered = null)}
						>
							<rect class="hit" x="0" y={stripY - 2} width={fw} height="16" />
							{#each Array(sequence) as _, position}<rect
									x={position * cellW}
									y={stripY}
									width={Math.max(1, cellW - 1)}
									height="12"
									class:image-cell={positions.includes(position)}
									class:target={position === target}
								/>{/each}
						</g>
						<text class="target-label" x="0" y={stripY + 26}>image {k} → position {target}</text>
						<text class="dim" x="0" y={stripY + 38}
							>{positions.length} image + {sequence - positions.length} text tokens</text
						>
					</svg>
					<footer>
						<span class="behind">{tokenCount - 1} more behind</span>
						<span class="pager"
							><span class="count">{k + 1} of {tokenCount}</span><button
								type="button"
								aria-label="Previous image token"
								on:click={() => choose(k - 1)}><AngleLeftOutline size="sm" /></button
							><button type="button" aria-label="Next image token" on:click={() => choose(k + 1)}
								><AngleRightOutline size="sm" /></button
							></span
						>
					</footer>
				</div>
			</div>
			<button
				type="button"
				class="continue"
				style={`margin-top:${ghosts * headGap.y + 10}px`}
				on:click={() => selectVlmPhase('language')}
				>Continue in the {document.streams.language.layers.length} language blocks →</button
			>
		{/if}
	</div>
</div>

<style>
	.vlm-connector {
		min-width: 0;
	}
	.title {
		pointer-events: none;
	}
	.content {
		display: block;
		position: relative;
		min-width: 0;
	}
	.stack {
		position: relative;
	}
	.card,
	.ghost {
		position: absolute;
		left: 0;
		top: 0;
		box-sizing: border-box;
		background: white;
		border: 1px solid #f3f4f6;
		border-radius: 4px;
		box-shadow:
			0 4px 6px -1px rgba(3, 7, 18, 0.15),
			0 2px 4px -2px rgba(3, 7, 18, 0.08);
	}
	.card {
		display: flex;
		flex-direction: column;
		padding: 8px 10px 6px;
		user-select: none;
		-webkit-user-select: none;
		font:
			11px/1.3 Arial,
			sans-serif;
		color: #9ca3af;
	}
	header {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		margin-bottom: 6px;
	}
	header strong {
		color: #6b7280;
		font-weight: 500;
		font-size: 13px;
	}
	header span {
		color: #a898ba;
		font-size: 9.5px;
		white-space: nowrap;
	}
	.post-norm {
		margin-top: 2px;
		padding: 0;
		border: 0;
		background: none;
		color: #9a61bf;
		font-size: 9px;
		cursor: pointer;
	}
	svg {
		display: block;
		overflow: visible;
	}
	path {
		fill: none;
		stroke: #dfe3fb;
		pointer-events: stroke;
		cursor: pointer;
		transition: stroke 0.12s;
	}
	path.hot {
		stroke: #cfd5f7;
	}
	path.out-band {
		stroke: #d8c4f3;
	}
	path.out-band.hot {
		stroke: #c4a6ee;
	}
	path:focus-visible {
		outline: 2px solid #a779c8;
	}
	.bar {
		fill: #e9d5ff;
	}
	.dim {
		fill: #a898ba;
		font-size: 8.5px;
	}
	.target-label {
		fill: #8b5cf6;
		font-size: 9px;
		font-weight: 600;
	}
	.patches line {
		stroke: #ffffff;
		stroke-width: 0.4;
		opacity: 0.7;
	}
	.patches image {
		opacity: 0.85;
	}
	.patches .block {
		fill: #a78bfa55;
		stroke: #7c3aed;
		stroke-width: 1.2;
	}
	.strip {
		cursor: pointer;
		outline: none;
	}
	.strip rect {
		fill: #e5e7eb;
	}
	.strip rect.hit {
		fill: transparent;
	}
	.strip rect.image-cell {
		fill: #ddd0f3;
	}
	.strip rect.target {
		fill: #8b5cf6;
	}
	.strip.hot rect:not(.hit) {
		fill: #d1d5db;
	}
	.strip.hot rect.image-cell {
		fill: #c9b3ef;
	}
	.strip.hot rect.target,
	.strip:focus-visible rect.target {
		fill: #7c3aed;
	}
	foreignObject button {
		width: 100%;
		height: 100%;
		padding: 0;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			10.5px/1.05 Arial,
			sans-serif;
		cursor: pointer;
		display: grid;
		place-items: center;
		align-content: center;
	}
	foreignObject button small {
		display: block;
		font-size: 8px;
		color: #8a6aa3;
	}
	foreignObject button.hot {
		background: #f1e9f9;
		border-color: #b9a0d4;
	}
	foreignObject button:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: -2px;
	}
	footer {
		margin-top: auto;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 6px;
	}
	.behind {
		color: #b7a9c3;
		font-size: 8.5px;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.pager {
		display: flex;
		align-items: center;
		gap: 3px;
		color: #9ca3af;
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
	.pager button:hover {
		color: #6b7280;
		border-color: #9ca3af;
	}
	.continue {
		display: block;
		padding: 0;
		border: 0;
		background: none;
		color: #9861be;
		font-size: 11px;
		cursor: pointer;
		text-align: left;
	}
	.continue:hover {
		color: #7c3aed;
	}
</style>
