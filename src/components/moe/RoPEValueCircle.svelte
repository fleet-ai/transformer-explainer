<script lang="ts">
	import { rotaryCircle, rotaryValueVectors } from '$lib/moe/rotaryCircle';

	export let before: [number, number];
	export let after: [number, number];
	export let indices: [number, number];
	export let cosine: number;
	export let sine: number;
	export let selectedCoordinate: number;
	export let compact = false;
	$: vectors = rotaryValueVectors(before, after);
	$: rotation = rotaryCircle(cosine, sine);
	$: angle =
		rotation.degrees !== 0 && rotation.degrees < 0.01
			? rotation.degrees.toExponential(2)
			: rotation.degrees.toFixed(2);
	$: selectedAxis = indices.indexOf(selectedCoordinate);
	$: selected = selectedAxis !== -1;
	$: arc =
		rotation.showArc && vectors.beforeDirection && vectors.afterDirection
			? `M ${110 + 20 * vectors.beforeDirection.x} ${73 + 20 * vectors.beforeDirection.y} A 20 20 0 ${rotation.largeArc ? 1 : 0} 0 ${110 + 20 * vectors.afterDirection.x} ${73 + 20 * vectors.afterDirection.y}`
			: '';
	$: endpoints = [
		{
			name: 'Before',
			className: 'before',
			point: vectors.before,
			direction: vectors.beforeDirection,
			values: before
		},
		{
			name: 'After',
			className: 'after',
			point: vectors.after,
			direction: vectors.afterDirection,
			values: after
		}
	];
	function value(number: number, digits = 6): string {
		return number !== 0 && Math.abs(number) < 0.0001
			? number.toExponential(3)
			: number.toFixed(digits);
	}
	function arrow(point: { x: number; y: number }, direction: { x: number; y: number }): string {
		const x = 110 + 49 * point.x;
		const y = 73 + 49 * point.y;
		return `${x},${y} ${x - 7 * direction.x - 3 * direction.y},${y - 7 * direction.y + 3 * direction.x} ${x - 7 * direction.x + 3 * direction.y},${y - 7 * direction.y - 3 * direction.x}`;
	}
</script>

<figure class="value-circle" class:selected>
	<svg
		viewBox="0 0 220 146"
		role="img"
		aria-label={`Recorded value vectors for coordinates ${indices.join(' and ')}. Before (${before.join(', ')}), after (${after.join(', ')}). Both use radius ${vectors.scale}.`}
	>
		<title>Recorded pair {indices[0]} ↔ {indices[1]} before and after RoPE</title>
		<circle class="rim" cx="110" cy="73" r="49" />
		<line class="axis" x1="41" y1="73" x2="180" y2="73" />
		<line class="axis" x1="110" y1="13" x2="110" y2="134" />
		<text class="axis-label" class:active={selectedAxis === 0} x="179" y="68">x[{indices[0]}]</text>
		<text class="axis-label" class:active={selectedAxis === 1} x="116" y="16">x[{indices[1]}]</text>
		<text class="zero" x="114" y="85">0</text>
		{#if arc}<path class="angle" d={arc} />{/if}
		{#each endpoints as endpoint}
			<g class={endpoint.className}>
				{#if selected}
					<line
						class="projection"
						x1={110 + 49 * endpoint.point.x}
						y1={73 + 49 * endpoint.point.y}
						x2={selectedAxis === 0 ? 110 + 49 * endpoint.point.x : 110}
						y2={selectedAxis === 0 ? 73 : 73 + 49 * endpoint.point.y}
					/>
					<circle
						class="axis-point"
						cx={selectedAxis === 0 ? 110 + 49 * endpoint.point.x : 110}
						cy={selectedAxis === 0 ? 73 : 73 + 49 * endpoint.point.y}
						r="2.7"
					/>
				{/if}
				<line
					class="vector"
					x1="110"
					y1="73"
					x2={110 + 49 * endpoint.point.x}
					y2={73 + 49 * endpoint.point.y}
				>
					<title>{endpoint.name}: ({endpoint.values.join(', ')})</title>
				</line>
				{#if endpoint.direction}<polygon
						class="arrow"
						points={arrow(endpoint.point, endpoint.direction)}
					/>{/if}
			</g>
		{/each}
		<circle class="origin" cx="110" cy="73" r="2" />
	</svg>
	<figcaption class:compact>
		{#if compact}
			<div
				class="compact-legend"
				title={rotation.radians === 0 ? '0° rotation: before and after lines overlap.' : undefined}
			>
				<span class="before"><i></i>Before</span>
				<span class="after"><i></i>After</span>
			</div>
			<small>Shared radius {value(vectors.scale, 4)}</small>
		{:else}
			<span class="before"><i></i>Before ({value(before[0], 4)}, {value(before[1], 4)})</span>
			<span class="after"><i></i>After ({value(after[0], 4)}, {value(after[1], 4)})</span>
			<small>θ ≈ {angle}° (mod 360°) · radius {value(vectors.scale, 4)}</small>
			{#if selected}<span class="selected-value"
					>x[{selectedCoordinate}]: {value(before[selectedAxis])} → {value(
						after[selectedAxis]
					)}</span
				>{/if}
			{#if rotation.radians === 0}<small>0° rotation · lines overlap</small>{/if}
		{/if}
	</figcaption>
</figure>

<style>
	.value-circle {
		width: 220px;
		flex: 0 0 220px;
		margin: 0;
		color: #9b8ca8;
	}
	svg {
		display: block;
		width: 220px;
		height: 146px;
	}
	.rim {
		fill: none;
		stroke: #ded3e9;
		stroke-width: 1.2;
	}
	.axis {
		stroke: #e3dce9;
		stroke-width: 1;
	}
	.axis-label,
	.zero {
		font:
			10px Arial,
			sans-serif;
		fill: #a79ab2;
	}
	.axis-label.active {
		fill: #8053a4;
		font-weight: 600;
	}
	.zero {
		font-size: 8px;
	}
	.angle {
		fill: none;
		stroke: #bba0d2;
		stroke-width: 1.2;
		stroke-dasharray: 2 2;
	}
	.before {
		color: #5089c9;
		stroke: #5089c9;
	}
	.after {
		color: #9468b7;
		stroke: #9468b7;
	}
	.vector {
		stroke-width: 2.3;
	}
	.before .vector {
		stroke-dasharray: 5 2;
		stroke-width: 3;
	}
	.projection {
		stroke-width: 1.2;
		stroke-dasharray: 2 3;
		opacity: 0.65;
	}
	.arrow {
		fill: currentColor;
		stroke: none;
	}
	.axis-point {
		fill: white;
		stroke-width: 1.4;
	}
	.origin {
		fill: #a694b7;
	}
	figcaption {
		text-align: center;
		font:
			10px/1.5 'SFMono-Regular',
			Consolas,
			monospace;
	}
	figcaption span {
		display: block;
	}
	figcaption.compact {
		min-height: 30px;
	}
	.compact-legend {
		display: flex;
		justify-content: center;
		gap: 14px;
		white-space: nowrap;
	}
	i {
		display: inline-block;
		width: 13px;
		border-top: 2px solid currentColor;
		margin: 0 5px 3px 0;
	}
	.before i {
		border-top-style: dashed;
	}
	figcaption small {
		display: block;
		font:
			9px/1.5 Arial,
			sans-serif;
		color: #a694b7;
	}
	.selected-value {
		font-size: 9px;
		color: #8053a4;
		margin-top: 3px;
	}
</style>
