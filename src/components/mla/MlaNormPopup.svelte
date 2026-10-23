<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { CloseOutline, RefreshOutline } from 'flowbite-svelte-icons';
	import { projectionSweep } from '$lib/moe/projection-sweep';
	/** RMSNorm on the KV latent: divide by the root mean square, then scale each coordinate.
	 *  Other architectures reuse it for their block norms, including LayerNorm. */
	export let size = 512;
	export let rotary = 64;
	export let epsilon = 1e-6;
	export let token = 'x';
	export let kind: 'rmsnorm' | 'layernorm' = 'rmsnorm';
	export let title = 'MLA · Latent RMSNorm';
	export let inputLabel = 'Raw KV latent';
	export let outputLabel = 'Normalized latent';
	$: layer = kind === 'layernorm';
	const dispatch = createEventDispatcher<{ close: void }>();
	let panel: HTMLDivElement;
	let frame = 0;
	let reduced = false;
	let progress = projectionSweep(0, 512);
	$: j = progress.coordinate;

	function portal(element: HTMLElement) {
		document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	function replay() {
		cancelAnimationFrame(frame);
		const start = performance.now();
		progress = projectionSweep(reduced ? 3500 : 0, size);
		if (reduced) return;
		const tick = (now: number) => {
			progress = projectionSweep(now - start, size);
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
				progress = projectionSweep(3500, size);
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
		aria-label={title}
		tabindex="-1"
	>
		<header>
			<h3>{title}</h3>
			<div class="controls">
				<button aria-label="Replay animation" on:click={replay}><RefreshOutline /></button>
				<button aria-label="Close" on:click={() => dispatch('close')}><CloseOutline /></button>
			</div>
		</header>
		<div class="body">
			<p class="notice">
				No weights to multiply · {layer ? 'a mean and a scale are' : 'one scale is'} computed from the
				whole vector, then applied to every coordinate · schematic animation
			</p>
			<div class="equation">
				<div class="tensor">
					<div>{inputLabel} <span class="token">“{token}”</span></div>
					<div class="cells input" class:working={!progress.complete} aria-hidden="true">
						{#each Array(32) as _}<i></i>{/each}
					</div>
					<strong>(1, {size})</strong>
				</div>
				<span class="operator">→</span>
				<div class="norm-box" aria-hidden="true">
					<strong>{layer ? 'LayerNorm' : 'RMSNorm'}</strong>
					<span>{layer ? '− μ · ÷ σ · × γ + β' : '÷ RMS · × γ'}</span>
				</div>
				<span class="operator">=</span>
				<div class="tensor">
					<div>{outputLabel}</div>
					<div class="cells output" aria-hidden="true">
						{#each Array(32) as _, i}<i
								class:done={i < progress.completed}
								class:active={i === progress.active}
							></i>{/each}
					</div>
					<strong>(1, {size})</strong>
				</div>
			</div>
			<div class="formula-lines" aria-hidden="true">
				{#if layer}
					<div>
						μ = (1/{size}) ∑<sub>d=1…{size}</sub> x<sub>d</sub>, σ² = (1/{size}) ∑<sub
							>d=1…{size}</sub
						>
						(x<sub>d</sub> − μ)², ε = {epsilon.toExponential(0)}
					</div>
					<div>
						y<sub>{j}</sub> = γ<sub>{j}</sub> · (x<sub>{j}</sub> − μ) / √(σ² + ε) + β<sub>{j}</sub>
					</div>
				{:else}
					<div>
						RMS(x) = √( (1/{size}) ∑<sub>d=1…{size}</sub> x<sub>d</sub>² + ε ), ε = {epsilon.toExponential(
							0
						)}
					</div>
					<div>y<sub>{j}</sub> = γ<sub>{j}</sub> · x<sub>{j}</sub> / RMS(x)</div>
				{/if}
			</div>
			<p class="status" role="status">
				{progress.complete ? `Complete · output (1, ${size})` : 'Normalizing across the vector'} · schematic
				sweep
			</p>
			<slot>
				<p class="notice">
					γ is a learned scale per coordinate. The {rotary}D rotary key skips this step. The cache
					stores the normalized latent together with the rotated key, {size + rotary} numbers per token
					in this layer.
				</p>
			</slot>
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
		grid-template-columns: minmax(0, 1fr) 26px 150px 26px minmax(0, 1fr);
		gap: 10px;
		align-items: center;
		margin-top: 24px;
	}
	.tensor {
		min-width: 0;
		display: grid;
		grid-template-rows: 34px 60px 34px;
		align-items: center;
		text-align: center;
	}
	.tensor strong {
		font-size: 18px;
		font-weight: 400;
		white-space: nowrap;
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
	.cells i {
		background: #cdc0fc;
	}
	.cells.input.working i {
		background: #b7a3f1;
	}
	.cells.output i {
		background: #f3f4f6;
	}
	.cells.output i.done {
		background: #b9c8fd;
	}
	.cells.output i.active {
		background: #8da6f5;
	}
	.operator {
		font-size: 26px;
		color: #374151;
		text-align: center;
	}
	.norm-box {
		display: grid;
		place-items: center;
		gap: 2px;
		height: 62px;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			600 15px Arial,
			sans-serif;
	}
	.norm-box span {
		font:
			400 11px Arial,
			sans-serif;
		color: #8a6aa3;
	}
	.formula-lines {
		display: grid;
		gap: 10px;
		font:
			italic 22px Georgia,
			serif;
		color: #6b7280;
		text-align: center;
		margin: 26px 0 18px;
	}
	sub {
		font-size: 12px;
	}
	.status {
		color: #9ca3af;
		font-size: 12px;
		text-align: center;
		margin: 0 0 12px;
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
			grid-template-columns: minmax(0, 1fr) 14px 90px 14px minmax(0, 1fr);
		}
		.formula-lines {
			font-size: 17px;
		}
	}
</style>
