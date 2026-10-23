<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { CloseOutline, RefreshOutline } from 'flowbite-svelte-icons';
	/** One attention head as formulas plus a schematic sweep over its query rows. */
	export let variant: 'encoder' | 'decoder' | 'cross' = 'encoder';
	export let rows = 6;
	export let cols = 6;
	export let head = 0;
	export let heads = 8;
	export let headSize = 64;
	export let selected = 0;
	export let token = '';
	const dispatch = createEventDispatcher<{ close: void }>();
	const ROW_MS = 320;
	let panel: HTMLDivElement;
	let frame = 0;
	let reduced = false;
	let done = 0;

	$: title =
		variant === 'cross'
			? 'Decoder block 1 · Cross-attention'
			: variant === 'decoder'
				? 'Decoder block 1 · Causal self-attention'
				: 'Encoder block 1 · Self-attention';
	$: complete = done >= rows;
	$: active = Math.min(done, rows - 1);

	function portal(element: HTMLElement) {
		document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	function replay() {
		cancelAnimationFrame(frame);
		done = reduced ? rows : 0;
		if (reduced) return;
		const start = performance.now();
		const step = (now: number) => {
			done = Math.min(rows, Math.floor((now - start) / ROW_MS));
			if (done < rows) frame = requestAnimationFrame(step);
		};
		frame = requestAnimationFrame(step);
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
	const masked = (row: number, col: number) => variant === 'decoder' && col > row;
	onMount(() => {
		const previous = document.activeElement;
		const media = matchMedia('(prefers-reduced-motion: reduce)');
		reduced = media.matches;
		const change = () => {
			reduced = media.matches;
			if (reduced) {
				cancelAnimationFrame(frame);
				done = rows;
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
		aria-label={`T5 · ${title}`}
		tabindex="-1"
	>
		<header>
			<h3>T5 · {title}</h3>
			<div class="controls">
				<button aria-label="Replay animation" on:click={replay}><RefreshOutline /></button>
				<button aria-label="Close" on:click={() => dispatch('close')}><CloseOutline /></button>
			</div>
		</header>
		<div class="body">
			<p class="notice">
				Head {head + 1} of {heads} · rows are {variant === 'cross' ? 'decoder' : ''} query tokens, columns
				are {variant === 'cross' ? 'encoder' : ''} key tokens · schematic animation
			</p>
			<div class="equation" aria-hidden="true">
				<div class="tensor">
					<div>Q <span class="token">“{token}”</span></div>
					<div class="tens q" style={`--rows:${rows}`}>
						{#each Array(rows) as _, row}<i
								class:active={row === active && !complete}
								class:selected={row === selected}
							></i>{/each}
					</div>
					<strong>({rows}, {headSize})</strong>
				</div>
				<span class="operator">·</span>
				<div class="tensor">
					<div>Kᵀ</div>
					<div class="tens k" style={`--cols:${cols}`}>
						{#each Array(cols) as _}<i></i>{/each}
					</div>
					<strong>({headSize}, {cols})</strong>
				</div>
				<span class="operator">→</span>
				<div class="tensor">
					<div>softmax</div>
					<div class="amap" style={`--rows:${rows};--cols:${cols}`}>
						{#each Array(rows) as _, row}{#each Array(cols) as _, col}<i
									class:masked={masked(row, col)}
									class:done={row < done && !masked(row, col)}
									class:active={row === active && !complete && !masked(row, col)}
								></i>{/each}{/each}
					</div>
					<strong>A ({rows}, {cols})</strong>
				</div>
				<span class="operator">·</span>
				<div class="tensor">
					<div>V</div>
					<div class="tens v" style={`--rows:${cols}`}>
						{#each Array(cols) as _}<i></i>{/each}
					</div>
					<strong>({cols}, {headSize})</strong>
				</div>
				<span class="operator">=</span>
				<div class="tensor">
					<div>Out</div>
					<div class="tens o" style={`--rows:${rows}`}>
						{#each Array(rows) as _, row}<i class:done={row < done}></i>{/each}
					</div>
					<strong>({rows}, {headSize})</strong>
				</div>
			</div>
			<div class="formula-lines" aria-hidden="true">
				{#if variant === 'cross'}
					<div>s<sub>ij</sub> = q<sub>i</sub> · k<sub>j</sub></div>
				{:else}
					<div>s<sub>ij</sub> = q<sub>i</sub> · k<sub>j</sub> + b<sub>h</sub>(j − i)</div>
				{/if}
				{#if variant === 'decoder'}
					<div>s<sub>ij</sub> = −∞ for j &gt; i</div>
				{/if}
				<div>
					a<sub>ij</sub> = exp(s<sub>ij</sub>) / ∑<sub>j′</sub> exp(s<sub>ij′</sub>) · o<sub>i</sub>
					= ∑<sub>j</sub>
					a<sub>ij</sub> v<sub>j</sub>
				</div>
			</div>
			<p class="status" role="status">
				{complete
					? `Complete · out (${rows}, ${headSize}) for this head`
					: `Query row ${active + 1} of ${rows}`} · schematic sweep
			</p>
			<p class="notice">
				{#if variant === 'cross'}
					k<sub>j</sub> and v<sub>j</sub> are projections of the encoder's final hidden state, so
					every decoder token can look at all {cols} source tokens. Cross-attention has no position bias.
					T5 does not divide by √{headSize}; that scale is folded into its weights.
				{:else}
					b<sub>h</sub>(j − i) is a learned bias for head {head + 1} that depends only on the bucketed
					distance between the tokens. Block 1 holds it, and later blocks reuse it.
					{variant === 'decoder'
						? 'The mask stops a token from reading tokens that come after it.'
						: 'The encoder is bidirectional: every token reads every other token.'}
					T5 does not divide by √{headSize}; that scale is folded into its weights.
				{/if}
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
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 14px;
		margin-top: 24px;
	}
	.tensor {
		display: grid;
		grid-template-rows: 30px 132px 30px;
		align-items: center;
		justify-items: center;
		text-align: center;
	}
	.tensor strong {
		font-size: 15px;
		font-weight: 400;
		white-space: nowrap;
	}
	.token {
		font-size: 12px;
		color: #9ca3af;
	}
	.tens {
		display: grid;
		gap: 2px;
	}
	.tens.q,
	.tens.o,
	.tens.v {
		grid-template-rows: repeat(var(--rows), minmax(0, 1fr));
		width: 46px;
		height: 120px;
	}
	.tens.k {
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
		width: 120px;
		height: 46px;
	}
	.tens i,
	.amap i {
		display: block;
		border-radius: 1px;
	}
	.tens.q i {
		background: #bfdbfe;
	}
	.tens.q i.selected {
		outline: 1.5px solid #6fa0ee;
	}
	.tens.q i.active {
		background: #6fa0ee;
	}
	.tens.k i {
		background: #fecaca;
	}
	.tens.v i {
		background: #bbf7d0;
	}
	.tens.o i {
		background: #f3f4f6;
	}
	.tens.o i.done {
		background: #d8c4fb;
	}
	.amap {
		display: grid;
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
		grid-template-rows: repeat(var(--rows), minmax(0, 1fr));
		gap: 2px;
		width: min(220px, calc(var(--cols) * 20px));
		height: min(120px, calc(var(--rows) * 20px));
	}
	.amap i {
		background: #f3f4f6;
	}
	.amap i.done {
		background: #c9b2f6;
	}
	.amap i.active {
		background: #8b5fe0;
	}
	.amap i.masked {
		background: repeating-linear-gradient(45deg, #f3f4f6 0 3px, #e5e7eb 3px 5px);
	}
	.operator {
		font-size: 24px;
		color: #374151;
	}
	.formula-lines {
		display: grid;
		gap: 8px;
		font:
			italic 21px Georgia,
			serif;
		color: #6b7280;
		text-align: center;
		margin: 26px 0 16px;
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
	@media (max-width: 760px) {
		.equation {
			flex-wrap: wrap;
		}
		.formula-lines {
			font-size: 16px;
		}
	}
</style>
