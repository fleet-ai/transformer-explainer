<script lang="ts">
	import { AngleLeftOutline, AngleRightOutline } from 'flowbite-svelte-icons';
	import { blockIdx, headGap, moeExample, moeTokenIndex } from '~/store';
	import { percent } from '$lib/moe/format';
	import type { ExpertEntry } from '$lib/moe/types';
	import ExpertProjectionPopup from './ExpertProjectionPopup.svelte';
	import ExpertElementwisePopup from './ExpertElementwisePopup.svelte';

	/** Every network that receives this token's vector; one is shown, the rest are stacked behind. */
	export let entries: ExpertEntry[] = [];
	export let index = 0;
	export let hiddenSize = 1024;
	export let token = 'x';
	export let width = 170;
	export let height = 330;
	export let maxGhosts = 5;

	const FLOW_HEIGHT = 266;
	const BOX_HEIGHT = 26;
	const BAND_PX = 30;
	let flowWidth = 150;
	type Step = 'up' | 'gate' | 'silu' | 'product' | 'down';
	const STEPS: Step[] = ['up', 'gate', 'silu', 'product', 'down'];
	let step: Step | null = null;
	/** Bands and the box of one operation light up together. */
	let hovered: Step | null = null;

	$: entry = entries[index] ?? entries[0];
	$: intermediate = entry?.size ?? 512;
	$: scale = BAND_PX / Math.max(hiddenSize, intermediate);
	$: hiddenPx = Math.max(10, hiddenSize * scale);
	$: interPx = Math.max(4, intermediate * scale);
	$: ghosts = Math.min(maxGhosts, Math.max(0, entries.length - 1));
	/** One anchor per expert output, offset like the stack; the Sankey layer routes each to the sum. */
	$: maxWeight = Math.max(...entries.map((item) => item.weight), 0.0001);
	$: taps = entries.map((_, order) => {
		const item = entries[(order + index) % entries.length];
		return { order, flow: (item.weight / maxWeight) * (order === 0 ? 1 : 0.75) };
	});
	$: projectionStep = step === 'up' || step === 'gate' || step === 'down' ? step : null;
	$: formulaStep = step === 'silu' || step === 'product' ? step : null;
	$: if (entry || $moeTokenIndex >= 0 || $blockIdx >= 0) step = null;
	// Flow geometry: the vector enters at the left edge, every projection sits on its own row,
	// and the result leaves at the right edge.
	$: xl = flowWidth * 0.26;
	$: xr = flowWidth * 0.74;
	$: xc = flowWidth / 2;
	$: boxWidth = Math.max(40, Math.min(56, flowWidth * 0.42));
	const yIn = 44;
	const rowY = 92;
	const siluY = 138;
	const prodY = 184;
	const downY = 226;
	const outY = 250;
	const half = BOX_HEIGHT / 2;
	$: mid1 = (rowY + half + prodY - 14) / 2;
	$: mid2 = (siluY + half + prodY - 14) / 2;
	$: bands = [
		{
			id: 'up',
			w: hiddenPx,
			d: `M8 ${yIn} C${xl} ${yIn} ${xl} ${rowY - 34} ${xl} ${rowY - half}`
		},
		{
			id: 'gate',
			w: hiddenPx,
			d: `M8 ${yIn} C${xr} ${yIn} ${xr} ${rowY - 34} ${xr} ${rowY - half}`
		},
		{
			id: 'up',
			w: interPx,
			d: `M${xl} ${rowY + half} C${xl} ${mid1} ${xc} ${mid1} ${xc} ${prodY - 14}`
		},
		{ id: 'gate', w: interPx, d: `M${xr} ${rowY + half} V${siluY - half}` },
		{
			id: 'silu',
			w: interPx,
			d: `M${xr} ${siluY + half} C${xr} ${mid2} ${xc} ${mid2} ${xc} ${prodY - 14}`
		},
		{ id: 'down', w: interPx, d: `M${xc} ${prodY + 14} V${downY - half}` },
		{
			id: 'down',
			w: hiddenPx,
			d: `M${xc} ${downY + half} C${xc} ${outY} ${xc + 24} ${outY} ${flowWidth - 8} ${outY}`
		}
	];
	$: boxes = [
		{ id: 'up', label: 'Wup', x: xl, y: rowY, w: boxWidth },
		{ id: 'gate', label: 'Wgate', x: xr, y: rowY, w: boxWidth },
		{ id: 'silu', label: 'SiLU', x: xr, y: siluY, w: boxWidth },
		{ id: 'product', label: '⊙', x: xc, y: prodY, w: 30, h: 30 },
		{ id: 'down', label: 'Wdown', x: xc, y: downY, w: boxWidth }
	];

	const stepName: Record<string, string> = {
		up: 'up projection (Wup)',
		gate: 'gate projection (Wgate)',
		silu: 'SiLU activation',
		product: 'element-wise product of up and SiLU(gate)',
		down: 'down projection (Wdown)'
	};

	function previous() {
		index = index > 0 ? index - 1 : entries.length - 1;
	}
	function next() {
		index = index < entries.length - 1 ? index + 1 : 0;
	}
	function select(id: string) {
		if (STEPS.includes(id as Step)) step = id as Step;
	}
</script>

<div class="expert-stack" style={`width:${width}px;height:${height}px;`}>
	{#each Array(ghosts) as _, ghost}
		<div
			class="ghost-card"
			style={`width:${width}px;height:${height}px;transform:translate(${(ghost + 1) * headGap.x}px,${(ghost + 1) * headGap.y}px);z-index:${ghosts - ghost};opacity:${Math.max(0.2, 1 - (ghost + 1) * 0.16)};`}
		></div>
	{/each}
	<div class="expert-card" style={`width:${width}px;height:${height}px;z-index:${ghosts + 1};`}>
		{#if entry}
			<header>
				<strong>{entry.label}</strong>
				<span
					>{entry.kind === 'routed'
						? `${percent(entry.weight, 1)} of the mix`
						: entry.kind === 'shared'
							? 'always on · weight 1'
							: 'every token'}</span
				>
			</header>
			<div class="flow" bind:clientWidth={flowWidth}>
				<svg
					width={flowWidth}
					height={FLOW_HEIGHT}
					viewBox={`0 0 ${flowWidth} ${FLOW_HEIGHT}`}
					role="group"
					aria-label={`${hiddenSize} input to up and gate projections, SiLU, product, down projection, ${hiddenSize} output`}
				>
					{#each bands as band}<path
							d={band.d}
							style={`stroke-width:${band.w}px`}
							class:hot={hovered === band.id}
							class:chosen={step === band.id}
							role="button"
							tabindex="0"
							aria-label={`Open ${stepName[band.id]}`}
							on:click={() => select(band.id)}
							on:mouseenter={() => (hovered = band.id as Step)}
							on:mouseleave={() => (hovered = null)}
							on:focus={() => (hovered = band.id as Step)}
							on:blur={() => (hovered = null)}
							on:keydown={(event) => {
								if (event.key === 'Enter' || event.key === ' ') {
									event.preventDefault();
									select(band.id);
								}
							}}
						/>{/each}
					<rect class="flow-in" x="0" y={yIn - hiddenPx / 2} width="8" height={hiddenPx} />
					<rect
						class="flow-out"
						x={flowWidth - 8}
						y={outY - hiddenPx / 2}
						width="8"
						height={hiddenPx}
					/>
					{#each taps as tap}<rect
							class="expert-tap"
							x={flowWidth - 8 + Math.min(tap.order, ghosts) * headGap.x}
							y={outY - hiddenPx / 2 + Math.min(tap.order, ghosts) * headGap.y}
							width="8"
							height={hiddenPx}
							data-flow={tap.flow.toFixed(2)}
						/>{/each}
					{#each boxes as box}<foreignObject
							x={box.x - box.w / 2}
							y={box.y - (box.h ?? BOX_HEIGHT) / 2}
							width={box.w}
							height={box.h ?? BOX_HEIGHT}
							><button
								type="button"
								class:hot={hovered === box.id}
								class:glyph={box.id === 'product'}
								title={stepName[box.id]}
								aria-label={`Open ${stepName[box.id]} for ${entry.label}`}
								aria-pressed={step === box.id}
								on:click={() => select(box.id)}
								on:mouseenter={() => (hovered = box.id as Step)}
								on:mouseleave={() => (hovered = null)}
								on:focus={() => (hovered = box.id as Step)}
								on:blur={() => (hovered = null)}>{box.label}</button
							></foreignObject
						>{/each}
				</svg>
			</div>
			<footer>
				<span class="dims">{hiddenSize} → {intermediate} → {hiddenSize}</span>
				{#if entries.length > 1}
					<span class="pager"
						><span class="count">{index + 1} of {entries.length}</span><button
							type="button"
							aria-label="Previous expert"
							on:click={previous}><AngleLeftOutline size="sm" /></button
						><button type="button" aria-label="Next expert" on:click={next}
							><AngleRightOutline size="sm" /></button
						></span
					>
				{/if}
			</footer>
		{/if}
	</div>
</div>

{#if entry && formulaStep}
	{#key `${$moeExample?.id}-${$blockIdx}-${$moeTokenIndex}-${entry.id}-${formulaStep}`}
		<ExpertElementwisePopup
			kind={formulaStep}
			intermediateSize={entry.size}
			{token}
			label={entry.label}
			on:close={() => (step = null)}
		/>
	{/key}
{:else if entry && projectionStep}
	{#key `${$moeExample?.id}-${$blockIdx}-${$moeTokenIndex}-${entry.id}-${projectionStep}`}
		<ExpertProjectionPopup
			step={projectionStep}
			{hiddenSize}
			intermediateSize={entry.size}
			{token}
			label={entry.label}
			on:close={() => (step = null)}
		/>
	{/key}
{/if}

<style>
	.expert-stack {
		position: relative;
	}
	.ghost-card,
	.expert-card {
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
	.expert-card {
		user-select: none;
		-webkit-user-select: none;
		display: flex;
		flex-direction: column;
		padding: 8px 10px 8px;
		font:
			11px/1.3 Arial,
			sans-serif;
		color: #9ca3af;
	}
	header {
		display: flex;
		flex-direction: column;
		margin-bottom: 4px;
	}
	header strong {
		color: #6b7280;
		font-weight: 500;
		font-size: 13px;
	}
	header span {
		color: #a898ba;
		font-size: 10px;
	}
	.flow {
		flex: 0 0 auto;
		min-width: 0;
	}
	svg {
		display: block;
		overflow: visible;
	}
	path {
		fill: none;
		stroke: #dfe3fb;
		/* Only the painted band reacts, never the empty space inside a curve's bounding box. */
		pointer-events: stroke;
		cursor: pointer;
		transition: stroke 0.12s;
	}
	path.hot,
	path.chosen {
		stroke: #cfd5f7;
	}
	path:focus-visible {
		outline: 2px solid #a779c8;
	}
	.expert-tap {
		fill: none;
		pointer-events: none;
	}
	.flow-in {
		fill: #e9d5ff;
	}
	.flow-out {
		fill: #e9d5ff;
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
			10.5px Arial,
			sans-serif;
		cursor: pointer;
	}
	foreignObject button.hot {
		background: #f1e9f9;
		border-color: #b9a0d4;
	}
	foreignObject button[aria-pressed='true'] {
		background: #e0c8ef;
		border-color: #b28bce;
	}
	foreignObject button.glyph {
		font-size: 16px;
		line-height: 1;
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
	.dims {
		color: #a898ba;
		font-size: 10px;
		white-space: nowrap;
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
</style>
