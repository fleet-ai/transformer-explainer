<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { scaleLinear } from 'd3';
	import { headGap } from '~/store';
	import { headCardGeometry } from './metrics';

	/** One attention head drawn like the Dense explainer: Key on top, Query beside the matrix, Value below, Out on the right. */
	export let id: string;
	export let title = 'Attention';
	export let queryTokens: string[] = [];
	export let keyTokens: string[] = [];
	export let matrix: number[][] = [];
	export let causal = false;
	export let pitch = 8;
	export let cell = 12;
	export let head = 0;
	export let heads = 8;
	export let selected = 0;
	export let keyNote = '';
	const dispatch = createEventDispatcher<{
		head: number;
		select: number;
		/** The clicked row: a key or value row for K and V, a query row otherwise. */
		open: { kind: 'q' | 'k' | 'v' | 'matrix' | 'out'; row: number };
	}>();
	let hotRow = -1;
	let hotCol = -1;

	$: rows = queryTokens.length;
	$: cols = keyTokens.length;
	$: g = headCardGeometry(rows, cols, pitch, cell);
	$: shade = scaleLinear<string>().domain([0, 1]).range(['#ede9fe', '#7425d5']);
	$: font = Math.max(8, Math.min(11, pitch + 1.5));
	$: dashH = Math.max(2, Math.min(3, pitch * 0.35));
	const masked = (row: number, col: number) => causal && col > row;
	$: keyY = (i: number) => g.kTop + i * pitch + pitch / 2;
	$: queryY = (i: number) => g.qTop + i * pitch + pitch / 2;
	$: valueY = (i: number) => g.vTop + i * pitch + pitch / 2;
	$: colX = (j: number) => g.mLeft + (j + 0.5) * cell;
	$: rowY = (i: number) => g.mTop + (i + 0.5) * cell;
	$: keyPath = (i: number) => {
		const x = colX(i);
		const r = Math.min(6, cell / 2);
		return `M${g.lineX},${keyY(i)} H${x - r} A${r},${r} 0 0 1 ${x},${keyY(i) + r} V${g.mTop + cell * 0.1}`;
	};
	$: queryPath = (i: number) => {
		const reach = (g.mLeft - g.lineX) * 0.6;
		return `M${g.lineX},${queryY(i)} C${g.lineX + reach},${queryY(i)} ${g.mLeft - reach},${rowY(i)} ${g.mLeft - 1},${rowY(i)}`;
	};
	// Out = A · V: a band from the attention rows and one from the values into the Out column.
	$: weightedBand = `M${g.mRight},${g.mTop} C${g.mRight + 18},${g.mTop} ${g.outX - 18},${g.mTop} ${g.outX},${g.mTop} V${g.mTop + rows * cell} C${g.outX - 18},${g.mTop + rows * cell} ${g.mRight + 18},${g.mTop + rows * cell} ${g.mRight},${g.mTop + rows * cell} Z`;
	$: valueBand = (() => {
		const mid = (g.lineX + g.outX) / 2;
		const bottom = g.vTop + cols * pitch;
		const outBottom = g.mTop + rows * cell;
		return `M${g.lineX},${g.vTop} C${mid},${g.vTop} ${mid},${g.mTop} ${g.outX},${g.mTop} V${outBottom} C${mid},${outBottom} ${mid},${bottom} ${g.lineX},${bottom} Z`;
	})();
	$: lit = (row: number, col: number) => hotRow === row || hotCol === col;
</script>

<div class="head-card-wrap" style={`width:${g.width}px;height:${g.height}px`}>
	{#each Array(Math.min(5, Math.max(0, heads - 1))) as _, ghost}
		<div
			class="head-ghost"
			style={`transform:translate(${(ghost + 1) * headGap.x}px,${(ghost + 1) * headGap.y}px);opacity:${Math.max(0.25, 1 - (ghost + 1) * 0.15)};`}
		></div>
	{/each}
	<div class="head-face" data-b={`${id}:card`}>
		<svg
			width={g.width}
			height={g.height}
			viewBox={`0 0 ${g.width} ${g.height}`}
			role="group"
			aria-label={`${title}, head ${head + 1} of ${heads}`}
			on:mouseleave={() => {
				hotRow = -1;
				hotCol = -1;
			}}
		>
			<defs>
				<linearGradient id={`${id}-vband`} x1="0" x2="1" y1="0" y2="0">
					<stop offset="0" stop-color="#bbf7d0" stop-opacity="0.55" />
					<stop offset="1" stop-color="#ddd6fe" stop-opacity="0.55" />
				</linearGradient>
			</defs>
			<path class="value-band" d={valueBand} fill={`url(#${id}-vband)`} />
			<path class="weighted-band" d={weightedBand} />
			<!-- Lines from every key to its column and every query to its row. -->
			{#each keyTokens as _, i}<path
					class="key-line"
					class:lit={hotCol === i}
					d={keyPath(i)}
				/>{/each}
			{#each queryTokens as _, i}<path
					class="query-line"
					class:lit={hotRow === i || selected === i}
					d={queryPath(i)}
				/>{/each}

			<text class="list-title key" x={g.dashX + 12} text-anchor="end" y={g.kTop - 6}
				>Key{keyNote}</text
			>
			<text class="list-title query" x={g.dashX + 12} text-anchor="end" y={g.qTop - 6}>Query</text>
			<text class="list-title value" x={g.dashX + 12} text-anchor="end" y={g.vTop - 6}
				>Value{keyNote}</text
			>
			<text class="list-title out" x={g.outX + 6} y={g.mTop - 8} text-anchor="middle">Out</text>

			{#each keyTokens as token, i}
				<text
					class="token-label"
					class:hot={hotCol === i}
					x={g.labelRight}
					y={keyY(i)}
					style={`font-size:${font}px`}>{token}</text
				>
				<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
				<rect
					class="dash key"
					data-b={`${id}:k`}
					data-i={i}
					x={g.dashX}
					y={keyY(i) - dashH / 2}
					width="12"
					height={dashH}
					on:click={() => dispatch('open', { kind: 'k', row: i })}
					><title>Key · {token}</title></rect
				>
			{/each}
			{#each queryTokens as token, i}
				<text
					class="token-label"
					class:hot={hotRow === i || selected === i}
					x={g.labelRight}
					y={queryY(i)}
					style={`font-size:${font}px`}>{token}</text
				>
				<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
				<rect
					class="dash query"
					data-b={`${id}:q`}
					data-i={i}
					x={g.dashX}
					y={queryY(i) - dashH / 2}
					width="12"
					height={dashH}
					on:click={() => {
						dispatch('select', i);
						dispatch('open', { kind: 'q', row: i });
					}}><title>Query · {token}</title></rect
				>
			{/each}
			{#each keyTokens as token, i}
				<text
					class="token-label"
					class:hot={hotCol === i}
					x={g.labelRight}
					y={valueY(i)}
					style={`font-size:${font}px`}>{token}</text
				>
				<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
				<rect
					class="dash value"
					data-b={`${id}:v`}
					data-i={i}
					x={g.dashX}
					y={valueY(i) - dashH / 2}
					width="12"
					height={dashH}
					on:click={() => dispatch('open', { kind: 'v', row: i })}
					><title>Value · {token}</title></rect
				>
			{/each}

			{#each matrix as line, row}{#each line as value, col}
					<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
					<circle
						class="mcell"
						cx={colX(col)}
						cy={rowY(row)}
						r={cell * 0.42}
						fill={masked(row, col) ? '#e5e7eb' : shade(value)}
						opacity={masked(row, col) ? 0.55 : lit(row, col) || hotRow < 0 ? 1 : 0.55}
						on:mouseenter={() => {
							hotRow = row;
							hotCol = col;
						}}
						on:click={() => dispatch('open', { kind: 'matrix', row })}
						><title
							>{queryTokens[row]} → {keyTokens[col]}: {masked(row, col)
								? 'masked'
								: value.toFixed(3)}</title
						></circle
					>
				{/each}{/each}

			{#each queryTokens as token, i}
				<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
				<rect
					class="dash out"
					data-b={`${id}:out`}
					data-i={i}
					x={g.outX}
					y={rowY(i) - dashH / 2}
					width="12"
					height={dashH}
					on:click={() => dispatch('open', { kind: 'out', row: i })}
					><title>Head output · {token}</title></rect
				>
			{/each}

			<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
			<text
				class="matrix-label"
				x={g.mLeft + (cols * cell) / 2}
				y={g.mTop + rows * cell + 20}
				text-anchor="middle"
				on:click={() => dispatch('open', { kind: 'matrix', row: selected })}>Attention ⊕</text
			>
		</svg>
		<div class="pager">
			<span>Head {head + 1} of {heads}</span>
			<button aria-label="Previous head" on:click={() => dispatch('head', -1)}>‹</button>
			<button aria-label="Next head" on:click={() => dispatch('head', 1)}>›</button>
		</div>
	</div>
</div>

<style>
	.head-card-wrap {
		position: relative;
		flex: none;
	}
	.head-ghost,
	.head-face {
		position: absolute;
		inset: 0;
		box-sizing: border-box;
		background: white;
		border: 1px solid #f3f4f6;
		border-radius: 2px;
		box-shadow:
			0 4px 6px -1px rgba(3, 7, 18, 0.12),
			0 2px 4px -2px rgba(3, 7, 18, 0.06);
	}
	.head-ghost {
		pointer-events: none;
	}
	.head-face svg {
		display: block;
		position: relative;
		z-index: 3;
		user-select: none;
	}
	.value-band {
		pointer-events: none;
	}
	.weighted-band {
		fill: #ddd6fe;
		opacity: 0.45;
		pointer-events: none;
	}
	.key-line,
	.query-line {
		fill: none;
		stroke-width: 1.2;
		pointer-events: none;
		transition: opacity 0.1s;
	}
	.key-line {
		stroke: #b49ae3;
		opacity: 0.55;
	}
	.query-line {
		stroke: #93b8f3;
		opacity: 0.65;
	}
	.key-line.lit,
	.query-line.lit {
		opacity: 1;
		stroke-width: 1.8;
	}
	.list-title {
		font:
			13px Arial,
			sans-serif;
	}
	.list-title.key {
		fill: #f87171;
	}
	.list-title.query {
		fill: #60a5fa;
	}
	.list-title.value {
		fill: #22c55e;
	}
	.list-title.out {
		fill: #a78bfa;
	}
	.token-label {
		fill: #4b5563;
		text-anchor: end;
		dominant-baseline: middle;
		font-family: Arial, sans-serif;
	}
	.token-label.hot {
		fill: #111827;
		font-weight: 700;
	}
	.dash {
		cursor: pointer;
	}
	.dash.key {
		fill: #f87171;
	}
	.dash.query {
		fill: #60a5fa;
	}
	.dash.value {
		fill: #34d399;
	}
	.dash.out {
		fill: #8b5cf6;
	}
	.mcell {
		cursor: pointer;
		transition: opacity 0.1s;
	}
	.matrix-label {
		font:
			15px Arial,
			sans-serif;
		fill: #6b7280;
		cursor: pointer;
	}
	.pager {
		position: absolute;
		right: 12px;
		bottom: 8px;
		z-index: 4;
		display: flex;
		align-items: center;
		gap: 4px;
		font:
			500 14px Arial,
			sans-serif;
		color: #6b7280;
	}
	.pager span {
		margin-right: 4px;
	}
	.pager button {
		width: 20px;
		height: 22px;
		padding: 0;
		border: 1px solid #d1d5db;
		border-radius: 4px;
		background: white;
		color: #6b7280;
		cursor: pointer;
		font-size: 14px;
		line-height: 1;
	}
	.pager button:hover {
		background: #f5f3ff;
		color: #7c3aed;
	}
</style>
