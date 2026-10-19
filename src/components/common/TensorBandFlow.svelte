<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { operationGraph } from './operation-graphs.js';
	export let kind = 'projection';
	export let selected = '';
	export let hiddenSize = 512;
	export let outputSize = hiddenSize;
	export let headSize = 64;
	export let heads = 8;
	export let kvHeads = heads;
	export let sequenceLength = 6;
	export let normalization = 'RMSNorm';
	export let causal = false;
	export let relativeBias = false;
	export let scale = '';
	export let rotary = false;
	export let label = 'Recorded operation';
	export let token = 'x';
	let width = 900;
	const dispatch = createEventDispatcher<{ select: string }>();
	$: graph = operationGraph(kind, {
		hiddenSize,
		outputSize,
		headSize,
		heads,
		kvHeads,
		sequenceLength,
		normalization,
		causal,
		relativeBias,
		scale,
		rotary,
		label
	});
	$: boxWidth = Math.min(132, Math.max(72, width * graph.boxWidth));
	const choose = (id: string) => {
		if (id) dispatch('select', id);
	};
	const x = (n: any, w: number, bw: number) => {
		const center = 24 + n.x * (w - 48);
		return n.terminal ? center : Math.max(bw / 2 + 4, Math.min(w - bw / 2 - 4, center));
	};
	function path(edge: any, w: number, bw: number, layout: typeof graph) {
		const a = layout.nodes.find((n: any) => n.id === edge.from),
			b = layout.nodes.find((n: any) => n.id === edge.to);
		if (!a || !b) return '';
		const ax = x(a, w, bw),
			bx = x(b, w, bw),
			c = (ax + bx) / 2;
		if (edge.via != null)
			return `M${ax} ${a.y} C${ax + 30} ${a.y},${ax + 30} ${edge.via},${ax + 60} ${edge.via} H${bx - 60} C${bx - 30} ${edge.via},${bx - 30} ${b.y},${bx} ${b.y}`;
		return `M${ax} ${a.y} C${c} ${a.y},${c} ${b.y},${bx} ${b.y}`;
	}
</script>

<div class="tensor-band-flow" bind:clientWidth={width}>
	<div class="flow-heading">
		<span>Click a band or box to inspect its operation</span><small>{graph.caption}</small>
	</div>
	<svg viewBox={`0 0 ${width} ${graph.height}`} role="group" aria-label={label + ' bands'}>
		{#each graph.edges as edge}
			{#if edge.action}
				<path
					d={path(edge, width, boxWidth, graph)}
					class="band"
					class:chosen={selected === edge.action}
					style={`--band:${edge.color ?? '#ded4ee'}`}
					role="button"
					tabindex="0"
					aria-label={`Inspect ${edge.label ?? edge.action} band`}
					aria-pressed={selected === edge.action}
					on:click={() => choose(edge.action)}
					on:keydown={(e) => {
						if (e.key === 'Enter' || e.key === ' ') {
							e.preventDefault();
							choose(edge.action);
						}
					}}
				/>
			{:else}<path d={path(edge, width, boxWidth, graph)} class="bypass" />{/if}
			{#if edge.label && edge.via != null}<text
					x={(x(
						graph.nodes.find((n) => n.id === edge.from),
						width,
						boxWidth
					) +
						x(
							graph.nodes.find((n) => n.id === edge.to),
							width,
							boxWidth
						)) /
						2}
					y={edge.via - 16}
					text-anchor="middle">{edge.label}</text
				>{/if}
		{/each}
		{#each graph.nodes as node}
			{#if node.terminal}
				<rect
					class="terminal"
					x={x(node, width, boxWidth) - 4}
					y={node.y - 17}
					width="8"
					height="34"
				/>
				<text
					x={x(node, width, boxWidth)}
					y={node.y - 27}
					text-anchor={node.x < 0.1 ? 'start' : node.x > 0.9 ? 'end' : 'middle'}
					>{node.label === 'Token' ? token : node.label}</text
				>
				<text
					class="dimension"
					x={x(node, width, boxWidth)}
					y={node.y + 37}
					text-anchor={node.x < 0.1 ? 'start' : node.x > 0.9 ? 'end' : 'middle'}>{node.dim}</text
				>
			{:else}
				<foreignObject
					x={x(node, width, boxWidth) - boxWidth / 2}
					y={node.y - 23}
					width={boxWidth}
					height="48"
					><button
						type="button"
						aria-label={`Inspect ${node.label}`}
						aria-pressed={selected === node.action}
						on:click={() => choose(node.action)}>{node.label}</button
					></foreignObject
				>
				{#if node.dim}<text
						class="dimension"
						x={x(node, width, boxWidth)}
						y={node.y - 31}
						text-anchor="middle">{node.dim}</text
					>{/if}
			{/if}
		{/each}
	</svg>
</div>

<style>
	.tensor-band-flow {
		width: 100%;
		min-width: 0;
		margin: 4px 0 8px;
		color: #7e668d;
	}
	.flow-heading {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		font:
			14px Arial,
			sans-serif;
		margin-bottom: 4px;
	}
	.flow-heading small {
		font-size: 12px;
		text-align: right;
	}
	svg {
		display: block;
		width: 100%;
		overflow: visible;
	}
	.band {
		fill: none;
		stroke: var(--band);
		stroke-width: 25;
		cursor: pointer;
		pointer-events: stroke;
	}
	.band.chosen,
	.band:hover {
		stroke: #bf99da;
	}
	.band:focus-visible {
		stroke: #a779c8;
		outline: none;
	}
	.bypass {
		fill: none;
		stroke: #c5e7d9;
		stroke-width: 25;
	}
	.terminal {
		fill: #bfd8fc;
	}
	text {
		font:
			13px Arial,
			sans-serif;
		fill: #725687;
	}
	.dimension {
		font-size: 13px;
		font-weight: 600;
	}
	button {
		width: 100%;
		height: 46px;
		padding: 2px 5px;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			14px/1.15 Arial,
			sans-serif;
		cursor: pointer;
	}
	button[aria-pressed='true'] {
		background: #e0c8ef;
		border-color: #b28bce;
	}
	button:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: -3px;
	}
</style>
