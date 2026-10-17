<script lang="ts">
	import { onMount } from 'svelte';

	export let expanded = false;
	let overlay: SVGSVGElement;
	let paths: { kind: string; d: string; x: number; y: number }[] = [];
	let destination = { x: 0, y: 0 };
	let schedule = () => {};
	$: {
		expanded;
		schedule();
	}

	onMount(() => {
		const card = overlay.closest('.multi-head')!;
		let frame = 0;
		function measure() {
			frame = 0;
			const target = card.querySelector(
				expanded ? '.attention-qk .matrix' : '.attention-result .matrix'
			);
			const flow = card.querySelector('.head-flow');
			if (!target || !flow) return;
			const bounds = overlay.getBoundingClientRect();
			const scale = bounds.width / overlay.clientWidth;
			if (!scale) return;
			const matrix = target.getBoundingClientRect();
			if (!matrix.width) return;
			const x = (matrix.left + matrix.width / 2 - bounds.left) / scale;
			const y = (matrix.top - bounds.top) / scale;
			destination = { x, y: y - 20 };
			const right = (flow.getBoundingClientRect().right - bounds.left) / scale;
			paths = ['query', 'key'].flatMap((kind, index) => {
				const source = flow.querySelector(`[data-rope-output="${kind}"]`);
				if (!source) return [];
				const point = source.getBoundingClientRect();
				const sx = (point.x - bounds.left) / scale;
				const sy = (point.y - bounds.top) / scale;
				const lane = Math.max(right + 12 + index * 10, x + 65 + index * 10);
				const ey = y - 64 + index * 12;
				const ex = x + (index ? 12 : -12);
				return [
					{
						kind,
						x: ex,
						y: y - 34,
						d: `M ${sx} ${sy} H ${lane - 8} Q ${lane} ${sy} ${lane} ${sy + 8} V ${ey - 8} Q ${lane} ${ey} ${lane - 8} ${ey} H ${ex + 8} Q ${ex} ${ey} ${ex} ${ey + 8} V ${y - 34}`
					}
				];
			});
		}
		schedule = () => {
			if (!frame) frame = requestAnimationFrame(measure);
		};
		const resize = new ResizeObserver(schedule);
		const mutation = new MutationObserver(schedule);
		// Track the existing expand/collapse animation without a permanent animation loop.
		card
			.querySelectorAll('.head-flow, .head-block, .attention-matrix-container, .attention-matrix')
			.forEach((node) => {
				resize.observe(node);
				mutation.observe(node, { attributes: true, attributeFilter: ['style', 'class'] });
			});
		resize.observe(card);
		schedule();
		return () => {
			resize.disconnect();
			mutation.disconnect();
			cancelAnimationFrame(frame);
			schedule = () => {};
		};
	});
</script>

<svg
	bind:this={overlay}
	class="rope-attention-links"
	role="img"
	aria-label="Rotated query and key flow from RoPE into the attention dot product. Value bypasses RoPE."
>
	{#each paths as path}
		<g class={path.kind}>
			<path d={path.d} />
			<path
				d={`M ${path.x - 3} ${path.y - 4} L ${path.x} ${path.y} L ${path.x + 3} ${path.y - 4}`}
			/>
		</g>
	{/each}
	{#if paths.length}
		<text x={destination.x} y={destination.y} text-anchor="middle"
			>{expanded ? 'Q′K′ᵀ · Dot product' : 'Q′K′ᵀ → scale / mask → softmax'}</text
		>
		<path
			class="into-matrix"
			d={`M ${destination.x} ${destination.y + 5} V ${destination.y + 15} m -3 -4 l 3 4 l 3 -4`}
		/>
	{/if}
</svg>

<style>
	.rope-attention-links {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		overflow: visible;
		pointer-events: none;
		z-index: 31;
	}
	path {
		fill: none;
		stroke-width: 1.5;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.query {
		stroke: #80afe9;
	}
	.key {
		stroke: #e9a0a5;
	}
	.into-matrix {
		stroke: #b49ace;
	}
	text {
		font:
			10px Arial,
			Helvetica,
			sans-serif;
		fill: #87669e;
		paint-order: stroke;
		stroke: white;
		stroke-width: 5px;
		stroke-linejoin: round;
	}
</style>
