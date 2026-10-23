<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { CloseOutline, RefreshOutline } from 'flowbite-svelte-icons';
	import { projectionSweep } from '$lib/moe/projection-sweep';
	/** The shared formula-popup shell (portal, shade, focus trap, replayable schematic sweep) for
	 *  the vision-language operations that have no matrix multiplication of their own. */
	export let title: string;
	export let notice: string;
	/** Length of the swept output, so the formula can name its current coordinate. */
	export let size = 32;
	/** Number of schematic cells the sweep walks through. */
	export let cells = 32;
	const dispatch = createEventDispatcher<{ close: void }>();
	let panel: HTMLDivElement;
	let frame = 0;
	let reduced = false;
	let progress = projectionSweep(0, size, cells);

	function portal(element: HTMLElement) {
		document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	function replay() {
		cancelAnimationFrame(frame);
		const start = performance.now();
		progress = projectionSweep(reduced ? 3500 : 0, size, cells);
		if (reduced) return;
		const tick = (now: number) => {
			progress = projectionSweep(now - start, size, cells);
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
				progress = projectionSweep(3500, size, cells);
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
			<p class="notice">{notice}</p>
			<slot {progress} />
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
	.controls button {
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
	.controls button :global(svg) {
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
	/* Content parts shared by every popup body, matching the MoE and MLA formula popups. */
	.body :global(.equation) {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 22px minmax(0, 1.15fr) 22px minmax(0, 1fr);
		gap: 10px;
		align-items: center;
		margin-top: 24px;
	}
	.body :global(.tensor) {
		min-width: 0;
		display: grid;
		grid-template-rows: 34px 110px 34px;
		align-items: center;
		text-align: center;
	}
	.body :global(.tensor strong) {
		font-size: 18px;
		font-weight: 400;
		white-space: nowrap;
	}
	.body :global(.token) {
		font-size: 12px;
		color: #9ca3af;
	}
	.body :global(.cells) {
		display: grid;
		grid-template-columns: repeat(var(--cells, 32), minmax(0, 1fr));
		gap: 1px;
		height: 26px;
	}
	.body :global(.cells i) {
		background: #c8ccd3;
	}
	.body :global(.cells.input i) {
		background: #cdc0fc;
	}
	.body :global(.cells.activated i) {
		background: #b9c8fd;
	}
	.body :global(.cells.output i) {
		background: #f3f4f6;
	}
	.body :global(.cells.output i.done) {
		background: #b9c8fd;
	}
	.body :global(.cells i.active) {
		background: #8da6f5;
	}
	.body :global(.operator) {
		font-size: 26px;
		color: #374151;
		text-align: center;
	}
	.body :global(.formula-lines) {
		display: grid;
		gap: 10px;
		font:
			italic 24px Georgia,
			serif;
		color: #6b7280;
		text-align: center;
		margin: 24px 0 18px;
	}
	.body :global(.formula-lines sub),
	.body :global(.formula-lines sup) {
		font-size: 12px;
	}
	.body :global(.status) {
		color: #9ca3af;
		font-size: 12px;
		text-align: center;
		margin: 0 0 16px;
	}
	.body :global(.vlm-note) {
		color: #6b7280;
		font-size: 13px;
		line-height: 1.5;
		margin: 6px 0 0;
		text-align: center;
	}
	@media (max-width: 600px) {
		.body {
			padding: 12px;
		}
		h3 {
			font-size: 16px;
		}
		.body :global(.equation) {
			gap: 5px;
			grid-template-columns: minmax(0, 1fr) 12px minmax(0, 1.1fr) 12px minmax(0, 1fr);
		}
		.body :global(.tensor) {
			font-size: 11px;
			grid-template-rows: 42px 70px 30px;
		}
		.body :global(.tensor strong) {
			font-size: 12px;
		}
		.body :global(.operator) {
			font-size: 18px;
		}
		.body :global(.formula-lines) {
			font-size: 19px;
		}
	}
</style>
