<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	export let hiddenSize = 1024;
	export let intermediateSize = 512;
	export let gated = true;
	export let activation = 'SiLU';
	export let selected = 'up';
	export let token = 'x';
	export let projectionsOnly = false;
	export let denseStyle = false;
	export let dimensional = false;
	const dispatch = createEventDispatcher<{ select: string }>();
	let width = 700;
	let paths: { id: string; d: string; size?: number }[] = [];
	$: height = gated ? 164 : 112;
	$: left = 60;
	$: boxWidth = Math.max(60, Math.min(100, width * 0.18));
	$: up = width * 0.31;
	$: act = width * 0.51;
	$: product = width * 0.7;
	$: right = width - 32;
	$: mid = gated ? 82 : 65;
	$: paths = gated
		? [
				{
					id: 'up',
					d: `M${left} ${mid} C${up - 50} ${mid},${up - 50} 53,${up} 53 H${act} C${product - 35} 53,${product - 35} ${mid},${product} ${mid}`
				},
				{ id: 'gate', d: `M${left} ${mid} C${up - 50} ${mid},${up - 50} 116,${up} 116 H${act}` },
				{
					id: 'product',
					d: `M${act} 116 C${product - 35} 116,${product - 35} ${mid},${product} ${mid}`
				},
				{ id: 'down', d: `M${product} ${mid} H${right}` }
			]
		: [
				{ id: 'up', d: `M${left} ${mid} H${act}` },
				{ id: 'activation', d: `M${act} ${mid} H${product}` },
				{ id: 'down', d: `M${product} ${mid} H${right}` }
			];
	$: bandScale = 42 / Math.max(hiddenSize, intermediateSize);
	$: if (dimensional) {
		const down = (product + right) / 2;
		paths = gated
			? [
					{
						id: 'up',
						d: `M${left} ${mid} C${up - 50} ${mid},${up - 50} 53,${up} 53`,
						size: hiddenSize
					},
					{
						id: 'up',
						d: `M${up} 53 H${act} C${product - 35} 53,${product - 35} ${mid},${product} ${mid}`,
						size: intermediateSize
					},
					{
						id: 'gate',
						d: `M${left} ${mid} C${up - 50} ${mid},${up - 50} 116,${up} 116`,
						size: hiddenSize
					},
					{ id: 'gate', d: `M${up} 116 H${act}`, size: intermediateSize },
					{
						id: 'product',
						d: `M${act} 116 C${product - 35} 116,${product - 35} ${mid},${product} ${mid}`,
						size: intermediateSize
					},
					{ id: 'down', d: `M${product} ${mid} H${down}`, size: intermediateSize },
					{ id: 'down', d: `M${down} ${mid} H${right}`, size: hiddenSize }
				]
			: [
					{ id: 'up', d: `M${left} ${mid} H${up}`, size: hiddenSize },
					{ id: 'up', d: `M${up} ${mid} H${act}`, size: intermediateSize },
					{ id: 'activation', d: `M${act} ${mid} H${product}`, size: intermediateSize },
					{ id: 'down', d: `M${product} ${mid} H${down}`, size: intermediateSize },
					{ id: 'down', d: `M${down} ${mid} H${right}`, size: hiddenSize }
				];
	}

	$: boxes = gated
		? [
				{ id: 'up', label: 'Wup', x: up, y: 53 },
				{ id: 'gate', label: 'Wgate', x: up, y: 116 },
				{ id: 'activation', label: activation, x: act, y: 116 },
				{ id: 'product', label: '⊙', x: product, y: mid },
				{ id: 'down', label: 'Wdown', x: (product + right) / 2, y: mid }
			]
		: [
				{ id: 'up', label: 'Wup', x: up, y: mid },
				{ id: 'activation', label: activation, x: act, y: mid },
				{ id: 'down', label: 'Wdown', x: (product + right) / 2, y: mid }
			];
	const interactive = (id: string) => !projectionsOnly || ['up', 'gate', 'down'].includes(id);
	const select = (id: string) => {
		if (interactive(id)) dispatch('select', id);
	};
</script>

<div class="feed-forward-flow" class:dense-style={denseStyle} bind:clientWidth={width}>
	<div class="hint">
		{projectionsOnly
			? gated
				? 'Click Wup, Wgate, or Wdown to see its multiplication'
				: 'Click Wup or Wdown to see its multiplication'
			: 'Click a band or box to inspect its operation'}
	</div>
	<svg
		viewBox={`0 0 ${width} ${height}`}
		role="group"
		aria-label={`${hiddenSize} input → ${gated ? 'up and gate projections' : 'up projection'} → ${activation} → ${intermediateSize} intermediate → down projection → ${hiddenSize} output`}
	>
		{#each paths as p}{#if interactive(p.id)}<path
					d={p.d}
					style={`stroke-width:${dimensional ? (p.size ?? hiddenSize) * bandScale : 28}px`}
					class:chosen={selected === p.id}
					role="button"
					tabindex="0"
					aria-label={`Inspect ${p.id} band`}
					on:click={() => select(p.id)}
					on:keydown={(e) => {
						if (e.key === 'Enter' || e.key === ' ') {
							e.preventDefault();
							select(p.id);
						}
					}}
				/>{:else}<path
					d={p.d}
					style={`stroke-width:${dimensional ? (p.size ?? hiddenSize) * bandScale : 28}px`}
					class="static-band"
				/>{/if}{/each}
		<rect x={left - 4} y={mid - 18} width="8" height="36" /><rect
			x={right - 4}
			y={mid - 18}
			width="8"
			height="36"
		/>
		<text x={left - 9} y={mid + 4} text-anchor="end"
			>{token.length > 7 ? token.slice(0, 6) + '…' : token}</text
		>
		<text x={left} y={mid + 40} text-anchor="middle">{hiddenSize}</text>
		<text x={up} y="21" text-anchor="middle">{intermediateSize}</text>
		<text x={right} y={mid + 40} text-anchor="end">{hiddenSize}</text>
		{#each boxes as box}<foreignObject
				x={box.x - (box.id === 'product' ? 22 : boxWidth / 2)}
				y={box.y - 22}
				width={box.id === 'product' ? 44 : boxWidth}
				height="46"
				>{#if interactive(box.id)}<button
						type="button"
						aria-label={`Inspect ${box.id === 'activation' ? activation : box.id} operation`}
						aria-pressed={selected === box.id}
						on:click={() => select(box.id)}>{box.label}</button
					>{:else}<div class="static-box">{box.label}</div>{/if}</foreignObject
			>{/each}
	</svg>
</div>

<style>
	.feed-forward-flow {
		width: 100%;
		min-width: 0;
		margin: 4px 0;
	}
	.hint {
		font:
			14px Arial,
			sans-serif;
		color: #7e668d;
		margin-bottom: 4px;
	}
	svg {
		width: 100%;
		display: block;
		overflow: visible;
	}
	path {
		fill: none;
		stroke: #ded4ee;
		stroke-width: 28;
		pointer-events: bounding-box;
		cursor: pointer;
	}
	path.chosen,
	path:hover {
		stroke: #bf99da;
	}
	path:focus-visible {
		outline: 2px solid #a779c8;
	}
	rect {
		fill: #bfd8fc;
	}
	text {
		font:
			14px Arial,
			sans-serif;
		fill: #725687;
	}
	button {
		width: 100%;
		height: 44px;
		padding: 0 2px;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			14px Arial,
			sans-serif;
		cursor: pointer;
	}
	button[aria-pressed='true'] {
		background: #e0c8ef;
		border-color: #b28bce;
	}
	button:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: -2px;
	}
	.static-box {
		height: 44px;
		display: grid;
		place-items: center;
		border: 1px solid #e5e7eb;
		border-radius: 4px;
		background: white;
		color: #4b5563;
		font:
			14px system-ui,
			sans-serif;
		box-sizing: border-box;
	}
	path.static-band {
		cursor: default;
		pointer-events: none;
	}
	.dense-style {
		font-family: system-ui, sans-serif;
	}
	.dense-style .hint {
		color: #9ca3af;
		font:
			13px system-ui,
			sans-serif;
		margin-bottom: 10px;
	}
	.dense-style path {
		stroke: #e9eafd;
		pointer-events: stroke;
	}
	.dense-style path.chosen,
	.dense-style path:not(.static-band):hover {
		stroke: #dcd7fe;
	}
	.dense-style button {
		background: white;
		color: #4b5563;
		border: 1px solid #e5e7eb;
		font-family: system-ui, sans-serif;
		box-shadow: 0 1px 2px #0000000d;
	}
	.dense-style button[aria-pressed='true'] {
		background: #f4f2ff;
		border-color: #c4b5fd;
	}
	.dense-style text {
		fill: #6b7280;
		font-family: system-ui, sans-serif;
	}
</style>
