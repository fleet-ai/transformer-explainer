<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { CloseOutline, RefreshOutline } from 'flowbite-svelte-icons';
	import { projectionSweep } from '$lib/moe/projection-sweep';
	import type { ExpertEntry } from '$lib/moe/types';
	/** The weighted sum of every selected expert's output, then the residual scaling. */
	export let entries: ExpertEntry[] = [];
	export let hiddenSize = 1024;
	export let token = 'x';
	export let residualMultiplier = 1;
	export let shared = false;
	const dispatch = createEventDispatcher<{ close: void }>();
	let panel: HTMLDivElement;
	let frame = 0;
	let reduced = false;
	let progress = projectionSweep(0, 1024);

	$: routed = entries.filter((entry) => entry.kind === 'routed');
	$: maxWeight = Math.max(...entries.map((entry) => entry.weight), 0.0001);
	$: scaled = residualMultiplier !== 1;
	$: j = progress.coordinate;

	function portal(element: HTMLElement) {
		document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	function replay() {
		cancelAnimationFrame(frame);
		const start = performance.now();
		progress = projectionSweep(reduced ? 3500 : 0, hiddenSize);
		if (reduced) return;
		const tick = (now: number) => {
			progress = projectionSweep(now - start, hiddenSize);
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
				progress = projectionSweep(3500, hiddenSize);
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
		aria-label="Weighted sum of the expert outputs"
		tabindex="-1"
	>
		<header>
			<h3>Σ · Weighted sum</h3>
			<div class="controls">
				<button aria-label="Replay animation" on:click={replay}><RefreshOutline /></button>
				<button aria-label="Close" on:click={() => dispatch('close')}><CloseOutline /></button>
			</div>
		</header>
		<div class="body">
			<p class="notice">
				Each expert returns a full (1, {hiddenSize}) vector. The router's weight scales it, then the
				scaled vectors are added coordinate by coordinate · schematic animation
			</p>
			<div class="equation">
				<div class="experts" aria-label="Expert outputs scaled by their router weights">
					<div class="heading">Expert outputs <span class="token">“{token}”</span></div>
					{#each entries as entry, i}
						<div class="expert-row">
							<span class="plus" aria-hidden="true">{i === 0 ? '' : '+'}</span>
							<span class="name">{entry.kind === 'routed' ? `E${entry.id}` : 'Shared'}</span>
							<span class="weight" title="Router weight, drawn to scale" aria-hidden="true"
								><i style={`width:${(entry.weight / maxWeight) * 100}%`}></i></span
							>
							<span class="cells" aria-hidden="true">
								{#each Array(24) as _}<i></i>{/each}
							</span>
						</div>
					{/each}
					<div class="legend">bar = router weight w<sub>e</sub> · strip = (1, {hiddenSize})</div>
				</div>
				<span class="operator">=</span>
				<div class="sum">
					<div>Weighted sum y</div>
					<div class="cells wide output" aria-hidden="true">
						{#each Array(32) as _, i}<i
								class:done={i < progress.completed}
								class:active={i === progress.active}
							></i>{/each}
					</div>
					<strong>(1, {hiddenSize})</strong>
				</div>
			</div>
			<div class="formula-lines" aria-hidden="true">
				<div>
					y<sub>{j}</sub> = ∑<sub>e ∈ top-{routed.length}</sub> w<sub>e</sub> · Expert<sub>e</sub
					>(x)<sub>{j}</sub>{#if shared}
						+ Shared(x)<sub>{j}</sub>{/if}
				</div>
				<div>
					out<sub>{j}</sub> = x<sub>{j}</sub> + {#if scaled}{residualMultiplier} ·
					{/if}y<sub>{j}</sub>
				</div>
			</div>
			<p class="status" role="status">
				{progress.complete ? `Complete · output (1, ${hiddenSize})` : 'Adding across the vector'} · schematic
				sweep
			</p>
			<p class="notice">
				{#if shared}The routed weights are the router's probabilities for the selected experts,
					without renormalising; the always-on shared experts are added with weight 1.
				{:else}The weights are a softmax over the {routed.length} selected router logits, so they sum
					to 1.{/if}
				{#if scaled}
					The sum is scaled by {residualMultiplier} before it joins the residual stream.{:else}
					The sum joins the residual stream unscaled.{/if}
			</p>
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
		grid-template-columns: minmax(0, 1.5fr) 26px minmax(0, 1fr);
		gap: 14px;
		align-items: center;
		margin-top: 22px;
	}
	.experts {
		display: grid;
		gap: 5px;
		min-width: 0;
	}
	.heading,
	.sum > div:first-child {
		text-align: center;
		color: #111827;
		margin-bottom: 4px;
	}
	.token {
		font-size: 12px;
		color: #9ca3af;
	}
	.expert-row {
		display: grid;
		grid-template-columns: 14px 44px 84px minmax(0, 1fr);
		gap: 8px;
		align-items: center;
	}
	.plus {
		color: #6b7280;
		font-size: 16px;
		text-align: center;
	}
	.name {
		font:
			12px ui-monospace,
			monospace;
		color: #6b7280;
	}
	.weight {
		display: block;
		height: 8px;
		background: #f3f4f6;
	}
	.weight i {
		display: block;
		height: 100%;
		background: #b49bdf;
	}
	.cells {
		display: grid;
		grid-template-columns: repeat(24, minmax(0, 1fr));
		gap: 1px;
		height: 16px;
	}
	.cells.wide {
		grid-template-columns: repeat(32, minmax(0, 1fr));
		height: 26px;
	}
	.cells i {
		background: #cdd6fb;
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
	.legend {
		color: #9ca3af;
		font-size: 12px;
		text-align: center;
		margin-top: 4px;
	}
	.operator {
		font-size: 26px;
		color: #374151;
		text-align: center;
	}
	.sum {
		display: grid;
		gap: 8px;
		text-align: center;
	}
	.sum strong {
		font-size: 18px;
		font-weight: 400;
		white-space: nowrap;
	}
	.formula-lines {
		display: grid;
		gap: 10px;
		font:
			italic 24px Georgia,
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
			grid-template-columns: minmax(0, 1fr);
		}
		.operator {
			font-size: 20px;
		}
		.expert-row {
			grid-template-columns: 10px 38px 48px minmax(0, 1fr);
			gap: 4px;
		}
		.formula-lines {
			font-size: 19px;
		}
	}
</style>
