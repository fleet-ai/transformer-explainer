<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { CloseOutline, RefreshOutline } from 'flowbite-svelte-icons';
	import { projectionSweep } from '$lib/moe/projection-sweep';
	/** Activations, the gate product and the residual addition work on each coordinate alone,
	 *  unlike the matrix projections. */
	export let kind: 'silu' | 'product' | 'gelu' | 'residual';
	/** Length of the vector the operation runs over (the hidden size for a residual addition). */
	export let intermediateSize: number;
	export let label: string;
	export let token: string;
	const dispatch = createEventDispatcher<{ close: void }>();
	let panel: HTMLDivElement;
	let frame = 0;
	let reduced = false;
	let progress = projectionSweep(0, 512);

	const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));
	const silu = (value: number) => value * sigmoid(value);
	/** The tanh approximation of GELU, as the pretrained vision encoders use it. */
	const gelu = (value: number) =>
		0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
	$: plotted = kind === 'silu' || kind === 'gelu';
	$: title = {
		silu: 'SiLU activation',
		gelu: 'GELU activation',
		product: 'Gate product ⊙',
		residual: 'Residual addition'
	}[kind];
	$: names = {
		silu: { input: 'Gate projection g', middle: '', output: 'Activated gate a' },
		gelu: { input: 'Up projection u', middle: '', output: 'Activated hidden h' },
		product: { input: 'Up projection u', middle: 'SiLU(gate) a', output: 'Gated product h' },
		residual: {
			input: 'Block stream x',
			middle: 'Sublayer output f(x)',
			output: 'Updated stream r'
		}
	}[kind];

	// The activation curve; it is the formula drawn out, with no recorded values on it.
	const W = 280;
	const H = 170;
	const PAD = 18;
	const xMin = -6;
	const xMax = 6;
	$: activation = kind === 'gelu' ? gelu : silu;
	$: yMin = (kind === 'gelu' ? gelu(-0.752) : silu(-1.278)) - 0.05;
	$: yMax = activation(xMax);
	const sx = (x: number) => PAD + ((x - xMin) / (xMax - xMin)) * (W - PAD * 2);
	$: sy = (y: number) => H - PAD - ((y - yMin) / (yMax - yMin)) * (H - PAD * 2);
	$: curve = Array.from({ length: 121 }, (_, k) => {
		const x = xMin + ((xMax - xMin) * k) / 120;
		return `${k === 0 ? 'M' : 'L'}${sx(x).toFixed(1)} ${sy(activation(x)).toFixed(1)}`;
	}).join(' ');

	function portal(element: HTMLElement) {
		document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	function replay() {
		cancelAnimationFrame(frame);
		const start = performance.now();
		progress = projectionSweep(reduced ? 3500 : 0, intermediateSize);
		if (reduced) return;
		const tick = (now: number) => {
			progress = projectionSweep(now - start, intermediateSize);
			if (!progress.complete) frame = requestAnimationFrame(tick);
		};
		frame = requestAnimationFrame(tick);
	}
	function keydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			event.preventDefault();
			event.stopImmediatePropagation();
			dispatch('close');
		} else if (event.key === 'Tab') {
			const targets = Array.from(panel.querySelectorAll<HTMLElement>('button, summary, a[href]'));
			const first = targets[0],
				last = targets[targets.length - 1];
			if (
				event.shiftKey &&
				(document.activeElement === first || document.activeElement === panel)
			) {
				event.preventDefault();
				last?.focus();
			} else if (!event.shiftKey && document.activeElement === last) {
				event.preventDefault();
				first?.focus();
			}
		}
	}
	onMount(() => {
		const previous = document.activeElement;
		const media = matchMedia('(prefers-reduced-motion: reduce)');
		reduced = media.matches;
		const change = () => {
			reduced = media.matches;
			if (reduced) {
				cancelAnimationFrame(frame);
				progress = projectionSweep(3500, intermediateSize);
			}
		};
		media.addEventListener('change', change);
		panel.focus({ preventScroll: true });
		replay();
		return () => {
			cancelAnimationFrame(frame);
			media.removeEventListener('change', change);
			if (previous instanceof HTMLElement && previous.isConnected)
				previous.focus({ preventScroll: true });
		};
	});
</script>

<svelte:window on:keydown|capture={keydown} />
<div class="popup-layer" use:portal>
	<div class="shade"></div>
	<div
		class="popup"
		bind:this={panel}
		role="dialog"
		aria-modal="true"
		aria-label={`${label} · ${title}`}
		tabindex="-1"
	>
		<header>
			<h3>{label} · {title}</h3>
			<div class="controls">
				<button aria-label="Replay animation" on:click={replay}><RefreshOutline /></button>
				<button aria-label="Close" on:click={() => dispatch('close')}><CloseOutline /></button>
			</div>
		</header>
		<div class="body">
			<p class="notice">
				Element-wise · every coordinate is computed on its own, with no sum across coordinates ·
				schematic animation
			</p>
			<div class="equation">
				<div class="tensor">
					<div>
						{names.input}
						<span class="token">“{token}”</span>
					</div>
					<div class="cells input" class:working={!progress.complete} aria-hidden="true">
						{#each Array(32) as _}<i></i>{/each}
					</div>
					<strong>(1, {intermediateSize})</strong>
				</div>
				<span class="operator">{plotted ? '→' : kind === 'residual' ? '+' : '⊙'}</span>
				{#if plotted}
					<div class="plot">
						<div>
							{kind === 'gelu' ? 'GELU(u) ≈ ½u · (1 + tanh(…))' : 'SiLU(g) = g · σ(g)'}
						</div>
						<svg
							viewBox={`0 0 ${W} ${H}`}
							role="img"
							aria-label={kind === 'gelu'
								? 'The GELU curve, tanh approximation'
								: 'The SiLU curve, g times sigmoid of g'}
						>
							<line class="axis" x1={sx(xMin)} x2={sx(xMax)} y1={sy(0)} y2={sy(0)} />
							<line class="axis" x1={sx(0)} x2={sx(0)} y1={sy(yMin)} y2={sy(yMax)} />
							<path class="curve" d={curve} />
							<text class="axis-label" x={W - PAD} y={sy(0) + 14} text-anchor="end"
								>{kind === 'gelu' ? 'u' : 'g'}</text
							>
						</svg>
						<strong>applied to each of {intermediateSize} coordinates</strong>
					</div>
				{:else}
					<div class="tensor">
						<div>{names.middle}</div>
						<div class="cells activated" aria-hidden="true">
							{#each Array(32) as _}<i></i>{/each}
						</div>
						<strong>(1, {intermediateSize})</strong>
					</div>
				{/if}
				<span class="operator">=</span>
				<div class="tensor">
					<div>{names.output}</div>
					<div class="cells output" aria-hidden="true">
						{#each Array(32) as _, i}<i
								class:done={i < progress.completed}
								class:active={i === progress.active}
							></i>{/each}
					</div>
					<strong>(1, {intermediateSize})</strong>
				</div>
			</div>
			<div class="formula-lines" aria-hidden="true">
				{#if kind === 'silu'}
					<div>
						a<sub>{progress.coordinate}</sub> = g<sub>{progress.coordinate}</sub> · σ(g<sub
							>{progress.coordinate}</sub
						>) = g<sub>{progress.coordinate}</sub> / (1 + e<sup
							>−g<sub>{progress.coordinate}</sub></sup
						>)
					</div>
				{:else if kind === 'gelu'}
					<div>
						h<sub>{progress.coordinate}</sub> = ½ u<sub>{progress.coordinate}</sub> · (1 +
						tanh(√(2/π) · (u<sub>{progress.coordinate}</sub> + 0.044715 u<sub
							>{progress.coordinate}</sub
						><sup>3</sup>)))
					</div>
				{:else if kind === 'residual'}
					<div>
						r<sub>{progress.coordinate}</sub> = x<sub>{progress.coordinate}</sub> + f(x)<sub
							>{progress.coordinate}</sub
						>
					</div>
				{:else}
					<div>
						a<sub>{progress.coordinate}</sub> = SiLU(g<sub>{progress.coordinate}</sub>) = g<sub
							>{progress.coordinate}</sub
						>
						· σ(g<sub>{progress.coordinate}</sub>) = g<sub>{progress.coordinate}</sub> / (1 + e<sup
							>−g<sub>{progress.coordinate}</sub></sup
						>)
					</div>
					<div>
						h<sub>{progress.coordinate}</sub> = u<sub>{progress.coordinate}</sub> · a<sub
							>{progress.coordinate}</sub
						>
					</div>
				{/if}
			</div>
			<p class="status" role="status">
				{progress.complete
					? `Complete · output (1, ${intermediateSize})`
					: 'Applying across the vector'} · schematic sweep
			</p>
			<slot />
		</div>
	</div>
</div>

<style>
	.popup-layer {
		position: fixed;
		inset: 0;
		z-index: 1400;
		display: grid;
		place-items: center;
		padding: 16px;
		box-sizing: border-box;
		font:
			14px/1.5 ui-sans-serif,
			system-ui,
			sans-serif;
		color: #111827;
	}
	.shade {
		position: absolute;
		inset: 0;
		background: #ffffffc7;
	}
	.popup {
		position: relative;
		width: min(1000px, 100%);
		max-height: calc(100dvh - 32px);
		border: 1px solid #e5e7eb;
		background: #fff;
		border-radius: 8px;
		box-shadow:
			0 4px 6px -1px #0000001a,
			0 2px 4px -2px #0000001a;
		display: flex;
		flex-direction: column;
		outline: none;
	}
	header {
		background: #f3f4f6;
		border-bottom: 1px solid #e5e7eb;
		border-radius: 8px 8px 0 0;
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 10px 16px;
		gap: 8px;
	}
	h3 {
		font-size: 18px;
		font-weight: 600;
		margin: 0;
	}
	.controls {
		display: flex;
		gap: 8px;
	}
	button {
		display: grid;
		place-items: center;
		width: 30px;
		height: 30px;
		padding: 4px;
		border: 0;
		background: transparent;
		color: #6b7280;
		cursor: pointer;
	}
	button :global(svg) {
		width: 22px;
		height: 22px;
	}
	.body {
		padding: 16px 24px 20px;
		min-height: 0;
		overflow: auto;
	}
	.notice {
		color: #9ca3af;
		font-size: 12px;
		margin: 0;
	}
	.equation {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 22px minmax(0, 1.15fr) 22px minmax(0, 1fr);
		gap: 10px;
		align-items: center;
		margin-top: 24px;
	}
	.tensor,
	.plot {
		min-width: 0;
		display: grid;
		align-items: center;
		text-align: center;
	}
	.tensor {
		grid-template-rows: 34px 110px 34px;
	}
	.plot {
		grid-template-rows: 34px auto 34px;
	}
	.plot svg {
		width: 100%;
		height: auto;
		max-height: 170px;
	}
	.tensor strong,
	.plot strong {
		font-size: 18px;
		font-weight: 400;
		white-space: nowrap;
	}
	.plot strong {
		font-size: 13px;
		color: #6b7280;
		white-space: normal;
	}
	.token {
		font-size: 12px;
		color: #9ca3af;
	}
	.cells {
		display: grid;
		grid-template-columns: repeat(32, minmax(0, 1fr));
		gap: 1px;
		height: 26px;
	}
	i {
		background: #c8ccd3;
	}
	.input i {
		background: #cdc0fc;
	}
	.input.working i {
		background: #b7a3f1;
	}
	.activated i {
		background: #b9c8fd;
	}
	.output i {
		background: #f3f4f6;
	}
	.output i.done {
		background: #b9c8fd;
	}
	.output i.active {
		background: #8da6f5;
	}
	.operator {
		font-size: 26px;
		color: #374151;
		text-align: center;
	}
	.axis {
		stroke: #d1d5db;
		stroke-width: 1;
	}
	.curve {
		fill: none;
		stroke: #8da6f5;
		stroke-width: 2.5;
	}
	.axis-label {
		fill: #9ca3af;
		font:
			italic 12px Georgia,
			serif;
	}
	.formula-lines {
		display: grid;
		gap: 10px;
		font:
			italic 24px Georgia,
			serif;
		color: #6b7280;
		text-align: center;
		margin: 24px 0 18px;
	}
	sub,
	sup {
		font-size: 12px;
	}
	.status {
		color: #9ca3af;
		font-size: 12px;
		text-align: center;
		margin: 0 0 16px;
	}
	@media (max-width: 600px) {
		.body {
			padding: 12px;
		}
		h3 {
			font-size: 16px;
		}
		.equation {
			gap: 5px;
			grid-template-columns: minmax(0, 1fr) 12px minmax(0, 1.1fr) 12px minmax(0, 1fr);
		}
		.tensor {
			font-size: 11px;
			grid-template-rows: 42px 70px 30px;
		}
		.tensor strong {
			font-size: 12px;
		}
		.operator {
			font-size: 18px;
		}
		.formula-lines {
			font-size: 19px;
		}
	}
</style>
