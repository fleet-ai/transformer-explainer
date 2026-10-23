<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { bandRedraw } from '~/store';
	import type { Box, Link } from './bands';

	/** Draws the bands between marked elements of one page; every pair of token strips gets its own band. */
	export let root: HTMLElement | undefined = undefined;
	export let links: Link[] = [];

	let drawn: { d: string; color: string; opacity: number; width: number }[] = [];
	let labels: { text: string; x: number; y: number }[] = [];
	let size = { width: 0, height: 0 };
	let ready = false;

	function boxes(name: string, origin: DOMRect): Box[] {
		return [...(root?.querySelectorAll<HTMLElement>(`[data-b="${name}"]`) ?? [])]
			.filter((element) => element.getClientRects().length)
			.sort((a, b) => Number(a.dataset.i ?? 0) - Number(b.dataset.i ?? 0))
			.map((element) => {
				const rect = element.getBoundingClientRect();
				const left = rect.left - origin.left;
				const top = rect.top - origin.top;
				return {
					left,
					right: left + rect.width,
					top,
					bottom: top + rect.height,
					cx: left + rect.width / 2,
					cy: top + rect.height / 2,
					width: rect.width,
					height: rect.height
				};
			});
	}
	const curve = (source: Box, target: Box) => {
		const reach = Math.max(10, (target.left - source.right) * 0.5);
		return `M${source.right},${source.cy} C${source.right + reach},${source.cy} ${target.left - reach},${target.cy} ${target.left},${target.cy}`;
	};

	function draw() {
		if (!root) return;
		const origin = root.getBoundingClientRect();
		size = { width: origin.width, height: origin.height };
		const next: typeof drawn = [];
		const nextLabels: typeof labels = [];
		for (const link of links) {
			const from = boxes(link.from, origin);
			const to = boxes(link.to, origin);
			const pairs = link.group
				? [{ source: link.last ? from[from.length - 1] : from[0], target: to[0], row: 0 }]
				: from.map((source, row) => ({ source, target: to[row], row }));
			for (const { source, target, row } of pairs) {
				if (!source || !target) continue;
				if (link.label)
					nextLabels.push({ text: link.label.text, ...link.label.at(source, target) });
				next.push({
					d: (link.path ?? curve)(source, target),
					color: link.color,
					opacity: typeof link.opacity === 'function' ? link.opacity(row) : (link.opacity ?? 0.5),
					width:
						link.width?.(source, target) ??
						Math.max(1.5, Math.min(source.height, target.height) * 0.92)
				});
			}
		}
		drawn = next;
		labels = nextLabels;
		ready = true;
	}
	$: if (root) (links, tick().then(draw));
	$: ($bandRedraw, root && tick().then(draw));
	const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => draw());
	let observed: HTMLElement | undefined;
	$: if (observer && root && root !== observed) {
		if (observed) observer.unobserve(observed);
		observer.observe(root);
		observed = root;
	}
	onMount(() => {
		window.addEventListener('resize', draw);
		if (document.fonts) document.fonts.ready.then(draw);
		return () => {
			observer?.disconnect();
			window.removeEventListener('resize', draw);
		};
	});
</script>

<svg
	class="bands"
	class:ready
	width={size.width}
	height={size.height}
	viewBox={`0 0 ${size.width} ${size.height}`}
	aria-hidden="true"
>
	{#each drawn as band}<path
			d={band.d}
			style={`stroke:${band.color};stroke-width:${band.width}px;opacity:${band.opacity}`}
		/>{/each}
	{#each labels as label}<text x={label.x} y={label.y}>{label.text}</text>{/each}
</svg>

<style>
	.bands {
		position: absolute;
		left: 0;
		top: 0;
		z-index: 2;
		pointer-events: none;
		overflow: visible;
		opacity: 0;
		transition: opacity 0.15s;
	}
	.bands.ready {
		opacity: 1;
	}
	path {
		fill: none;
		stroke-linejoin: round;
	}
	text {
		font:
			600 10.5px Arial,
			sans-serif;
		fill: #694684;
	}
</style>
