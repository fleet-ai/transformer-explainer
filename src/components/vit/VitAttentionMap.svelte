<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { tensor, type TensorData } from '$lib/vit/types';
	export let layer: TensorData;
	export let head = 0;
	export let query = 0;
	export let count = 197;
	export let causal = false;
	const dispatch = createEventDispatcher<{ query: number }>();
	let canvas: HTMLCanvasElement;
	let hover: { q: number; k: number; value: number } | null = null;
	$: if (canvas && layer) paint(layer, head, query, count);
	function paint(data: TensorData, h: number, q: number, n: number) {
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const values = tensor(data, 'attention');
		const image = ctx.createImageData(n, n);
		for (let i = 0; i < n * n; i++) {
			const t = Math.sqrt(values[h * n * n + i]);
			image.data[i * 4] = 255 - (255 - 113) * t;
			image.data[i * 4 + 1] = 255 - (255 - 54) * t;
			image.data[i * 4 + 2] = 255 - (255 - 176) * t;
			image.data[i * 4 + 3] = 255;
		}
		ctx.putImageData(image, 0, 0);
		ctx.strokeStyle = '#3169cf';
		ctx.lineWidth = 0.5;
		ctx.strokeRect(0, q, n, 1);
	}
	function position(e: MouseEvent) {
		const rect = canvas.getBoundingClientRect();
		return {
			q: Math.min(
				count - 1,
				Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * count))
			),
			k: Math.min(
				count - 1,
				Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * count))
			)
		};
	}
	function move(e: MouseEvent) {
		const { q, k } = position(e);
		hover = { q, k, value: tensor(layer, 'attention')[(head * count + q) * count + k] };
	}
</script>

<div class="full-map">
	<canvas
		bind:this={canvas}
		width={count}
		height={count}
		on:mousemove={move}
		on:mouseleave={() => (hover = null)}
		on:click={(e) => dispatch('query', position(e).q)}
		role="img"
		aria-label={`Complete ${count} by ${count} ${causal ? 'causal' : 'bidirectional'} attention matrix for head ${head + 1}; click a row to choose its query.`}
	></canvas>
	<div>
		{hover
			? `Query ${hover.q} → key ${hover.k}: ${(hover.value * 100).toFixed(5)}%`
			: `All ${count} queries × ${count} keys · click a query row`}
	</div>
</div>

<style>
	.full-map canvas {
		width: 250px;
		height: 250px;
		image-rendering: pixelated;
		border: 1px solid #d9c8e8;
		cursor: crosshair;
	}
	.full-map div {
		font-size: 10px;
		color: #9378a7;
		margin-top: 6px;
		width: 250px;
	}
</style>
