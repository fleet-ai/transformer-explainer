<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { CloseOutline, RefreshOutline } from 'flowbite-svelte-icons';
	import { projectionSweep } from '$lib/moe/projection-sweep';
	export let projection: {
		label: string;
		input: string;
		weight: string;
		output: string;
		inputSize: number;
		outputSize: number;
	};
	export let token: string;
	const dispatch = createEventDispatcher<{ close: void }>();
	let panel: HTMLDivElement;
	let frame = 0;
	let reduced = false;
	let progress = projectionSweep(0, 512);
	$: ({ inputSize, outputSize } = projection);
	function portal(element: HTMLElement) {
		document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	function replay() {
		cancelAnimationFrame(frame);
		const start = performance.now();
		progress = projectionSweep(reduced ? 3500 : 0, outputSize);
		if (reduced) return;
		const tick = (now: number) => {
			progress = projectionSweep(now - start, outputSize);
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
				progress = projectionSweep(3500, outputSize);
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
<div class="projection-layer" use:portal>
	<div class="shade"></div>
	<div
		class="projection-popup"
		bind:this={panel}
		role="dialog"
		aria-modal="true"
		aria-label={`MLA · ${projection.label}`}
		tabindex="-1"
	>
		<header>
			<h3>MLA · {projection.label}</h3>
			<div class="controls">
				<button aria-label="Replay multiplication" on:click={replay}><RefreshOutline /></button>
				<button aria-label="Close multiplication" on:click={() => dispatch('close')}
					><CloseOutline /></button
				>
			</div>
		</header>
		<div class="body">
			<p class="notice">Schematic animation · full tensor dimensions · bias-free</p>
			<div
				class="equation"
				aria-label={`Input (1, ${inputSize}) times weights (${inputSize}, ${outputSize}) equals output (1, ${outputSize})`}
			>
				<div class="tensor">
					<div>
						{projection.input} <span class="token">“{token}”</span>
					</div>
					<div
						class="projection-vector input"
						class:working={!progress.complete}
						aria-hidden="true"
					>
						{#each Array(32) as _}<i></i>{/each}
					</div>
					<strong>(1, {inputSize})</strong>
				</div>
				<span class="operator">×</span>
				<div class="tensor">
					<div>{projection.weight}</div>
					<div class="matrix" aria-hidden="true">
						{#each Array(256) as _, i}<i class:active={i % 32 === progress.active}></i>{/each}
					</div>
					<strong>({inputSize}, {outputSize})</strong>
				</div>
				<span class="operator">=</span>
				<div class="tensor">
					<div>{projection.output}</div>
					<div class="projection-vector output" aria-hidden="true">
						{#each Array(32) as _, i}<i
								class:done={i < progress.completed}
								class:active={i === progress.active}
							></i>{/each}
					</div>
					<strong>(1, {outputSize})</strong>
				</div>
			</div>
			<div class="formula" aria-hidden="true">
				y<sub>{progress.coordinate}</sub> = ∑<sub>d=1…{inputSize}</sub> x<sub>d</sub> · W<sub
					>{progress.coordinate},d</sub
				>
			</div>
			<p class="status" role="status">
				{progress.complete
					? `Complete · output (1, ${outputSize})`
					: 'Multiplying across the output'} · schematic sweep
			</p>
			<slot />
		</div>
	</div>
</div>

<style>
	.projection-layer {
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
	.projection-popup {
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
	.tensor {
		min-width: 0;
		display: grid;
		grid-template-rows: 34px 110px 34px;
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
	.projection-vector,
	.matrix {
		display: grid;
		grid-template-columns: repeat(32, minmax(0, 1fr));
		gap: 1px;
	}
	.projection-vector {
		height: 26px;
	}
	.matrix {
		height: 88px;
		grid-template-rows: repeat(8, 1fr);
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
	.matrix i.active {
		background: #9ca3af;
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
	.formula {
		font:
			italic 24px Georgia,
			serif;
		color: #6b7280;
		text-align: center;
		margin: 24px 0 18px;
	}
	sub {
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
		.matrix {
			height: 60px;
		}
		.projection-vector {
			height: 22px;
		}
		.operator {
			font-size: 18px;
		}
		.formula {
			font-size: 19px;
		}
	}
</style>
