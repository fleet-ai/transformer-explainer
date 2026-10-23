<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { CloseOutline, RefreshOutline } from 'flowbite-svelte-icons';
	import { attentionHeadIdx, moeExample, moeModel, moeTokenIndex } from '~/store';
	import { headMapping } from '$lib/moe/attention';
	import { tokenLabel } from '$lib/moe/format';
	import { projectionSweep } from '$lib/moe/projection-sweep';
	import TokenInspectorLayout from './TokenInspectorLayout.svelte';
	import TokenPicker from './TokenPicker.svelte';

	/** Rotary position encoding: Q and K rotate pair by pair, V passes through. Formulas only. */
	export let kind: 'query' | 'key' = 'query';
	const dispatch = createEventDispatcher<{ close: void; attention: void }>();
	let panel: HTMLDivElement;
	let frame = 0;
	let reduced = false;
	let progress = projectionSweep(0, 64);

	$: isMLA = $moeModel?.architecture === 'mla';
	/** Grouped-query models share each key head; Zamba2 gives every query head its own key. */
	$: sharedKey = ($moeModel?.keyValueHeads ?? 8) < ($moeModel?.attentionHeads ?? 16);
	$: headSize = isMLA ? ($moeModel?.rotarySize ?? 64) : ($moeModel?.attentionHeadSize ?? 64);
	$: half = headSize / 2;
	$: base = ($moeModel?.ropeTheta ?? 10000).toLocaleString('en-US');
	$: mapping = headMapping(
		$attentionHeadIdx,
		$moeModel?.attentionHeads ?? 16,
		(isMLA ? 1 : $moeModel?.keyValueHeads) ?? 8
	);
	$: position = $moeTokenIndex;
	$: tokenOptions = ($moeExample?.tokens ?? []).map((token) => ({
		value: token.index,
		label: `${token.index}: ${tokenLabel(token.text)}`
	}));

	// Schematic rotation: one frequency, so the angle simply grows with the position.
	const STEP = 28;
	const START = 24;
	const C = 100;
	const R = 78;
	$: turn = position * STEP;
	const point = (degrees: number, radius = R) => ({
		x: C + radius * Math.cos((degrees * Math.PI) / 180),
		y: C - radius * Math.sin((degrees * Math.PI) / 180)
	});
	$: before = point(START);
	$: after = point(START + turn);
	$: arc = (() => {
		const from = point(START, 34);
		const to = point(START + turn, 34);
		return `M${from.x} ${from.y} A34 34 0 ${turn > 180 ? 1 : 0} 0 ${to.x} ${to.y}`;
	})();
	$: label = point(START + turn / 2, 50);

	function portal(element: HTMLElement) {
		document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	function replay() {
		cancelAnimationFrame(frame);
		const start = performance.now();
		progress = projectionSweep(reduced ? 3500 : 0, headSize);
		if (reduced) return;
		const tick = (now: number) => {
			progress = projectionSweep(now - start, headSize);
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
				progress = projectionSweep(3500, headSize);
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
		aria-label="RoPE · rotate Q and K"
		tabindex="-1"
	>
		<header>
			<h3>RoPE · Rotate Q and K</h3>
			<div class="controls">
				<button aria-label="Replay animation" on:click={replay}><RefreshOutline /></button>
				<button aria-label="Close" on:click={() => dispatch('close')}><CloseOutline /></button>
			</div>
		</header>
		<div class="body">
			<p class="notice">
				No weights · each pair of coordinates is rotated by an angle set by the token's position ·
				the size stays {headSize}D · schematic animation
			</p>
			<TokenInspectorLayout gap="10px"
				><TokenPicker
					slot="tokens"
					label="RoPE token"
					options={tokenOptions}
					bind:value={$moeTokenIndex}
				/>
				<div class="rows">
					{#each [{ id: 'query', name: isMLA ? `Query ${$attentionHeadIdx + 1} · rotary part` : `Query ${$attentionHeadIdx + 1}`, out: 'Q′', tone: 'blue' }, { id: 'key', name: isMLA ? 'Shared rotary key' : `${sharedKey ? 'Shared key' : 'Key'} ${mapping.keyValueHead + 1}`, out: 'K′', tone: 'red' }] as row}
						<div class={`row ${row.tone}`} class:focus={kind === row.id}>
							<div class="tensor">
								<div>{row.name}</div>
								<div class="cells input" aria-hidden="true">
									{#each Array(32) as _}<i></i>{/each}
								</div>
								<strong>(1, {headSize})</strong>
							</div>
							<span class="band" aria-hidden="true"></span>
							<div class="rope-box" aria-hidden="true">
								<span class="glyph">↻</span>
								<span>RoPE</span>
								<small>θ = pos · ω</small>
							</div>
							<span class="band" aria-hidden="true"></span>
							<div class="tensor">
								<div>{row.out} rotated</div>
								<div class="cells output" aria-hidden="true">
									{#each Array(32) as _, i}<i
											class:done={i < progress.completed}
											class:active={i === progress.active}
										></i>{/each}
								</div>
								<strong>(1, {headSize})</strong>
							</div>
						</div>
					{/each}
					<div class="row green bypass">
						<div class="tensor">
							<div>Value</div>
							<div class="cells input" aria-hidden="true">
								{#each Array(32) as _}<i></i>{/each}
							</div>
							<strong>(1, {$moeModel?.valueHeadSize ?? headSize})</strong>
						</div>
						<span class="band long" aria-hidden="true"></span>
						<div class="bypass-note">V bypasses RoPE · unchanged</div>
						<span class="band long" aria-hidden="true"></span>
						<div class="tensor">
							<div>Value</div>
							<div class="cells input" aria-hidden="true">
								{#each Array(32) as _}<i></i>{/each}
							</div>
							<strong>(1, {$moeModel?.valueHeadSize ?? headSize})</strong>
						</div>
					</div>
				</div>
				<div class="pair">
					<div class="dial">
						<svg viewBox="0 0 200 200" role="img" aria-label="A coordinate pair rotated by RoPE">
							<circle class="ring" cx={C} cy={C} r={R} />
							<line class="axis" x1="12" x2="188" y1={C} y2={C} />
							<line class="axis" x1={C} x2={C} y1="12" y2="188" />
							<text class="axis-label" x="190" y={C + 14} text-anchor="end"
								>x<tspan dy="3">i</tspan></text
							>
							<text class="axis-label" x={C + 6} y="20">x<tspan dy="3">i+{half}</tspan></text>
							{#if position > 0}<path class="arc" d={arc} /><text
									class="theta"
									x={label.x}
									y={label.y}
									text-anchor="middle">θ</text
								>{/if}
							<line class="before" x1={C} y1={C} x2={before.x} y2={before.y} />
							<line class="after" x1={C} y1={C} x2={after.x} y2={after.y} />
						</svg>
						<div class="legend">
							<span class="before-key">before</span><span class="after-key">after RoPE</span>
						</div>
						<p class="notice">
							{position === 0
								? 'Position 0 is not rotated.'
								: `Position ${position}: the pair turns ${position} × ω.`} Schematic · every pair has
							its own frequency ω.
						</p>
					</div>
					<div class="formulas" aria-hidden="true">
						<div>
							x′<sub>i</sub> = x<sub>i</sub> cos θ<sub>i</sub> − x<sub>i+{half}</sub> sin θ<sub
								>i</sub
							>
						</div>
						<div>
							x′<sub>i+{half}</sub> = x<sub>i+{half}</sub> cos θ<sub>i</sub> + x<sub>i</sub> sin θ<sub
								>i</sub
							>
						</div>
						<div class="angle">
							θ<sub>i</sub> = pos · {base}<sup>−2i/{headSize}</sup>, i = 0 … {half - 1}
						</div>
					</div>
				</div>
				<p class="notice foot">
					Pairs are (i, i + {half}). Early pairs spin fast and late pairs barely move, so each pair
					encodes position at a different scale. After rotation, Q′·K′ depends only on how far apart
					the two tokens are. {#if isMLA}Only the {headSize}D rotary branch rotates; the content
						branch bypasses RoPE.{/if}
					<button class="link" on:click={() => dispatch('attention')}
						>See attention operations →</button
					>
				</p>
			</TokenInspectorLayout>
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
	.rows {
		display: grid;
		gap: 12px;
		margin-top: 12px;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 36px 128px 36px minmax(0, 1fr);
		align-items: center;
		padding: 6px 8px;
		border-radius: 6px;
		border: 1px solid transparent;
	}
	.row.focus {
		border-color: #e5e7eb;
		background: #fafafa;
	}
	.tensor {
		display: grid;
		grid-template-rows: 26px 26px 26px;
		text-align: center;
		min-width: 0;
		align-items: center;
	}
	.tensor strong {
		font-size: 15px;
		font-weight: 400;
		white-space: nowrap;
	}
	.cells {
		display: grid;
		grid-template-columns: repeat(32, minmax(0, 1fr));
		gap: 1px;
		height: 26px;
	}
	.band {
		display: block;
		height: 26px;
		background: #dfe3fb;
	}
	.band.long {
		grid-column: auto;
	}
	.blue .cells i {
		background: #c9dcfb;
	}
	.red .cells i {
		background: #f8cccc;
	}
	.green .cells i {
		background: #c9ecd3;
	}
	.blue .cells.output i {
		background: #f3f4f6;
	}
	.red .cells.output i {
		background: #f3f4f6;
	}
	.blue .cells.output i.done {
		background: #93b7f3;
	}
	.red .cells.output i.done {
		background: #ee9f9f;
	}
	.blue .cells.output i.active {
		background: #5f95ec;
	}
	.red .cells.output i.active {
		background: #e66f6f;
	}
	.rope-box {
		display: grid;
		place-items: center;
		align-content: center;
		gap: 0;
		height: 62px;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			600 15px Arial,
			sans-serif;
		line-height: 1.15;
	}
	.rope-box small {
		font:
			400 10px Arial,
			sans-serif;
		color: #8a6aa3;
	}
	.glyph {
		font-size: 20px;
		line-height: 1;
	}
	.bypass {
		grid-template-columns: minmax(0, 1fr) 36px 128px 36px minmax(0, 1fr);
	}
	.bypass-note {
		text-align: center;
		font-size: 12px;
		color: #4d8a5f;
		line-height: 1.3;
	}
	.pair {
		display: grid;
		grid-template-columns: minmax(220px, 280px) minmax(0, 1fr);
		gap: 24px;
		align-items: center;
		border-top: 1px solid #e5e7eb;
		margin-top: 16px;
		padding-top: 14px;
	}
	.dial svg {
		width: 100%;
		max-width: 240px;
		display: block;
		margin: 0 auto;
	}
	.ring {
		fill: none;
		stroke: #d9d2ec;
		stroke-width: 1.2;
	}
	.axis {
		stroke: #e5e7eb;
		stroke-width: 1;
	}
	.axis-label {
		fill: #9ca3af;
		font:
			italic 12px Georgia,
			serif;
	}
	.arc {
		fill: none;
		stroke: #9a7bc8;
		stroke-width: 1.5;
		stroke-dasharray: 3 2;
	}
	.theta {
		fill: #7c5aa8;
		font:
			italic 14px Georgia,
			serif;
	}
	.before {
		stroke: #5f95ec;
		stroke-width: 2.5;
		stroke-dasharray: 5 3;
	}
	.after {
		stroke: #8f5fc0;
		stroke-width: 3;
	}
	.legend {
		display: flex;
		justify-content: center;
		gap: 14px;
		font-size: 12px;
		margin-top: 2px;
	}
	.before-key {
		color: #5f95ec;
	}
	.after-key {
		color: #8f5fc0;
	}
	.dial .notice {
		text-align: center;
		margin-top: 4px;
	}
	.formulas {
		display: grid;
		gap: 12px;
		font:
			italic 21px Georgia,
			serif;
		color: #6b7280;
		text-align: center;
	}
	.formulas .angle {
		font-size: 19px;
		margin-top: 6px;
	}
	sub,
	sup {
		font-size: 12px;
	}
	.foot {
		margin-top: 14px;
		line-height: 1.6;
	}
	.link {
		border: 0;
		background: none;
		padding: 0;
		margin-left: 6px;
		color: #7c5aa8;
		text-decoration: underline;
		cursor: pointer;
		font: inherit;
	}
	@media (max-width: 700px) {
		.body {
			padding: 12px;
		}
		.row,
		.bypass {
			grid-template-columns: minmax(0, 1fr) 14px 76px 14px minmax(0, 1fr);
		}
		.rope-box {
			font-size: 12px;
		}
		.rope-box small {
			display: none;
		}
		.pair {
			grid-template-columns: minmax(0, 1fr);
		}
		.formulas {
			font-size: 17px;
		}
	}
</style>
