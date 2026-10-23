<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	/** The Dense explainer's MLP: each token's vector widens four times into the 2048D layer and narrows back. */
	export let id: string;
	export let labels: string[] = [];
	export let height = 300;
	export let selected = -1;
	const dispatch = createEventDispatcher<{ open: { kind: 'up' | 'down'; row: number } }>();
	const LABEL_W = 70;
	const IN_X = LABEL_W + 6;
	const HID_X = IN_X + 12 + 64;
	const OUT_X = HID_X + 16 + 64;
	const WIDTH = OUT_X + 12;
	$: rows = labels.length;
	$: pitch = height / Math.max(rows, 1);
	$: thin = Math.max(2.5, (pitch - 3) / 4);
	$: wide = Math.min(pitch - 3, thin * 4);
	$: y = (row: number) => row * pitch + pitch / 2;
	$: font = Math.max(8, Math.min(13, pitch * 0.45));
	const band = (x0: number, h0: number, x1: number, h1: number, cy: number) => {
		const mid = (x0 + x1) / 2;
		return `M${x0},${cy - h0 / 2} C${mid},${cy - h0 / 2} ${mid},${cy - h1 / 2} ${x1},${cy - h1 / 2} V${cy + h1 / 2} C${mid},${cy + h1 / 2} ${mid},${cy + h0 / 2} ${x0},${cy + h0 / 2} Z`;
	};
	let hot = -1;
</script>

<div class="mlp-stage" style={`width:${WIDTH}px;height:${height}px`}>
	<svg
		width={WIDTH}
		{height}
		viewBox={`0 0 ${WIDTH} ${height}`}
		role="group"
		aria-label="Feed-forward, 512 to 2048 to 512 with ReLU"
		on:mouseleave={() => (hot = -1)}
	>
		{#each labels as label, row}
			<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
			<path
				class="expand"
				class:lit={hot === row || selected === row}
				d={band(IN_X + 12, thin, HID_X, wide, y(row))}
				on:mouseenter={() => (hot = row)}
				on:click={() => dispatch('open', { kind: 'up', row })}
				><title>{label} · up projection 512 → 2048</title></path
			>
			<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
			<path
				class="narrow"
				class:lit={hot === row || selected === row}
				d={band(HID_X + 16, wide, OUT_X, thin, y(row))}
				on:mouseenter={() => (hot = row)}
				on:click={() => dispatch('open', { kind: 'down', row })}
				><title>{label} · ReLU, then down projection 2048 → 512</title></path
			>
			<text
				class="mlabel"
				class:hot={selected === row}
				x={LABEL_W}
				y={y(row)}
				style={`font-size:${font}px`}>{label}</text
			>
			<rect
				class="vec in"
				data-b={`${id}:in`}
				data-i={row}
				x={IN_X}
				y={y(row) - thin / 2}
				width="12"
				height={thin}
			/>
			<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
			<rect
				class="hcol"
				x={HID_X}
				y={y(row) - wide / 2}
				width="16"
				height={wide}
				on:click={() => dispatch('open', { kind: 'down', row })}
			/>
			<rect
				class="vec out"
				data-b={`${id}:out`}
				data-i={row}
				x={OUT_X}
				y={y(row) - thin / 2}
				width="12"
				height={thin}
			/>
		{/each}
	</svg>
</div>

<style>
	.mlp-stage {
		position: relative;
		flex: none;
	}
	svg {
		display: block;
		position: relative;
		z-index: 3;
		overflow: visible;
		user-select: none;
	}
	.expand,
	.narrow {
		fill: #c7d2fe;
		opacity: 0.35;
		cursor: pointer;
		transition: opacity 0.1s;
	}
	.narrow {
		fill: #bfdbfe;
	}
	.expand.lit,
	.narrow.lit {
		opacity: 0.7;
	}
	.mlabel {
		fill: #4b5563;
		text-anchor: end;
		dominant-baseline: middle;
		font-family: Arial, sans-serif;
	}
	.mlabel.hot {
		fill: #111827;
		font-weight: 700;
	}
	.vec.in {
		fill: #c4b5fd;
	}
	.vec.out {
		fill: #93c5fd;
	}
	.hcol {
		fill: #a5b4fc;
		cursor: pointer;
	}
</style>
