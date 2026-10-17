<script lang="ts">
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import tailwindConfig from '../../../tailwind.config';
	import resolveConfig from 'tailwindcss/resolveConfig';
	import { vectorHeight } from '~/store';

	const { theme } = resolveConfig(tailwindConfig);
	const palettes: Record<string, { 100: string; 400: string }> = {
		gray: theme.colors.gray, blue: theme.colors.blue, red: theme.colors.red,
		green: theme.colors.green, purple: theme.colors.purple, indigo: theme.colors.indigo
	};

	export let active: boolean = false;
	export let data: number[] | undefined = undefined;
	export let label: string | undefined = undefined;
	export let colorScale: string | ((t: number, i: number) => string) | undefined = undefined;

	let canvas: HTMLCanvasElement;

	const lineHeight = 1;
	$: hasCapturedValues = !!data?.length && data.every(Number.isFinite);
	$: description = label ?? (hasCapturedValues
		? `Recorded activation preview, first ${data!.length} coordinates. Colors normalized within this vector.`
		: 'Activation values not captured. This neutral vector is schematic.');

	$: colorKey = typeof colorScale === 'string' ? colorScale : 'gray';
	$: palette = palettes[colorKey] ?? palettes.gray;
	$: color =
		typeof colorScale === 'function'
			? colorScale
			: d3.interpolate(palette[100], palette[400]);

	function drawCanvas(values: number[] | undefined, palette: (value: number, index: number) => string) {
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const width = canvas.parentElement?.clientWidth ?? 0;
		const height = canvas.parentElement?.clientHeight ?? 0;

		ctx.clearRect(0, 0, width, height);

		let pixelRatio = 4;
		canvas.width = width * pixelRatio;
		canvas.height = height * pixelRatio;
		canvas.style.width = `${width}px`;
		canvas.style.height = `${height}px`;

		const captured = values?.length && values.every(Number.isFinite) ? values : null;
		const extent = captured ? Math.max(...captured.map(Math.abs), 1e-12) : 1;
		for (let i = 0; i < height / lineHeight; i++) {
			// Stretch only the recorded preview coordinates over the visual strip.
			// Missing activations remain uniform; no random values or invented tail.
			const index = captured ? Math.min(captured.length - 1, Math.floor(i * lineHeight / height * captured.length)) : 0;
			const normalized = captured ? 0.5 + captured[index] / (2 * extent) : 0.5;
			ctx.fillStyle = captured ? palette(normalized, index) : theme.colors.gray[200];
			ctx.fillRect(0, i * lineHeight * pixelRatio, width * pixelRatio, lineHeight * pixelRatio);
		}
	}

	onMount(() => {
		const redraw = () => drawCanvas(data, color);
		const unsubscribe = vectorHeight.subscribe(redraw);
		const observer = new ResizeObserver(redraw);
		if (canvas.parentElement) observer.observe(canvas.parentElement);
		return () => {
			unsubscribe();
			observer.disconnect();
		};
	});

	$: if (canvas && color) {
		drawCanvas(data, color);
	}
</script>

<canvas class:active bind:this={canvas} aria-label={description} title={description}></canvas>

<style lang="scss">
	canvas {
		position: absolute;
		top: 0;
		left: 0;
		opacity: 0;
		display: block;
		width: 100%;
		height: 100%;
		overflow: hidden;
		transition: opacity 0.2s;

		&.active {
			opacity: 1;
		}

		&:hover {
			opacity: 1;
		}
	}
</style>
