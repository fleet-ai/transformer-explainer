<script lang="ts">
	import { rotaryCircle } from '$lib/moe/rotaryCircle';

	export let cosine: number;
	export let sine: number;
	export let pairIndex: number;
	export let selected = false;
	export let compact = false;
	$: geometry = rotaryCircle(cosine, sine);
	$: pointX = 90 + 43 * geometry.x;
	$: pointY = 65 + 43 * geometry.y;
	$: arc = geometry.showArc
		? `M 107 65 A 17 17 0 ${geometry.largeArc ? 1 : 0} 0 ${90 + 17 * geometry.x} ${65 + 17 * geometry.y}`
		: '';
	function value(number: number): string {
		return number !== 0 && Math.abs(number) < 0.0001 ? number.toExponential(5) : number.toFixed(6);
	}
	$: angle =
		geometry.degrees !== 0 && geometry.degrees < 0.01
			? geometry.degrees.toExponential(2)
			: geometry.degrees.toFixed(2);
</script>

<figure class:selected class="unit-circle">
	<svg
		viewBox="0 0 180 132"
		role="img"
		aria-label={`Unit circle for pair ${pairIndex}: cosine ${value(cosine)}, sine ${value(sine)}, angle ${angle} degrees modulo 360`}
	>
		<title>Recorded (cos θ, sin θ) for pair {pairIndex}</title>
		<line class="axis" x1="26" y1="65" x2="154" y2="65" />
		<line class="axis" x1="90" y1="14" x2="90" y2="116" />
		<circle class="rim" cx="90" cy="65" r="43" />
		{#each [45, 135, 225, 315] as degrees}
			<circle
				class="tick"
				cx={90 + 43 * Math.cos((degrees * Math.PI) / 180)}
				cy={65 - 43 * Math.sin((degrees * Math.PI) / 180)}
				r="1.5"
			/>
		{/each}
		<text class="cardinal" x="157" y="68">0°</text>
		<text class="cardinal centered" x="90" y="10">90°</text>
		<text class="cardinal end" x="23" y="68">180°</text>
		<text class="cardinal centered" x="90" y="128">270°</text>
		<line class="projection cos" x1={pointX} y1={pointY} x2="90" y2={pointY} />
		<line class="projection sin" x1={pointX} y1={pointY} x2={pointX} y2="65" />
		<line class="radius" x1="90" y1="65" x2={pointX} y2={pointY} />
		<line class="component cos" x1="90" y1="65" x2={pointX} y2="65" />
		<line class="component sin" x1="90" y1="65" x2="90" y2={pointY} />
		{#if arc}<path class="angle" d={arc} />{/if}
		<circle class="origin" cx="90" cy="65" r="2" />
		<circle class="selected-point" cx={pointX} cy={pointY} r="4" />
	</svg>
	<figcaption class:compact>
		<span class="theta">θ<sub>{pairIndex}</sub> ≈ {angle}° <small>(mod 360°)</small></span>
		<span class="cos">x = cos θ = {value(cosine)}</span>
		<span class="sin">y = sin θ = {value(sine)}</span>
	</figcaption>
</figure>

<style>
	.unit-circle {
		width: 180px;
		flex: 0 0 180px;
		margin: 0;
		color: #a694b7;
	}
	svg {
		display: block;
		width: 180px;
		height: 132px;
		overflow: visible;
	}
	.axis {
		stroke: #ddd5e5;
		stroke-width: 1;
	}
	.rim {
		fill: none;
		stroke: #c9b5dc;
		stroke-width: 1.3;
	}
	.tick {
		fill: #c9b5dc;
	}
	.cardinal {
		font:
			9px Arial,
			sans-serif;
		fill: #9b8ca8;
	}
	.centered {
		text-anchor: middle;
	}
	.end {
		text-anchor: end;
	}
	.projection {
		stroke-width: 1.2;
		stroke-dasharray: 3 3;
		fill: none;
		opacity: 0.7;
	}
	.cos {
		color: #5089c9;
		stroke: #5089c9;
	}
	.sin {
		color: #ce7c90;
		stroke: #ce7c90;
	}
	.component {
		stroke-width: 2.5;
		stroke-linecap: round;
	}
	.radius {
		stroke: #9c79bd;
		stroke-width: 1.5;
	}
	.angle {
		fill: none;
		stroke: #a379ce;
		stroke-width: 1.3;
	}
	.origin {
		fill: #a694b7;
	}
	.selected-point {
		fill: #9468b7;
		stroke: white;
		stroke-width: 1.5;
	}
	.selected .selected-point {
		fill: #8350b6;
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
		min-height: 49px;
		white-space: nowrap;
	}
	.theta {
		color: #9468b7;
		margin: 1px 0 3px;
		font-family: Arial, sans-serif;
	}
	.theta small {
		color: #a694b7;
		font-size: 9px;
	}
</style>
