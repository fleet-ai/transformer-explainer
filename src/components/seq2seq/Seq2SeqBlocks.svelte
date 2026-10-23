<script lang="ts">
	import { onMount, tick } from 'svelte';
	import {
		attentionHeadIdx,
		attentionHeadIdxTemp,
		bandRedraw,
		sampling,
		temperature
	} from '~/store';
	import { recordedProbabilities } from '$lib/moe/adapter';
	import { loadT5Recording } from '$lib/seq2seq/adapter';
	import { t5Document, t5Example } from '$lib/seq2seq/store';
	import type { T5Attention } from '$lib/seq2seq/types';
	import AnimatedMatrixPopup from '../common/AnimatedMatrixPopup.svelte';
	import PairAttentionPopup from './pair/PairAttentionPopup.svelte';
	import PairBands from './pair/PairBands.svelte';
	import PairBank from './pair/PairBank.svelte';
	import PairBlocks from './pair/PairBlocks.svelte';
	import PairHeadCard from './pair/PairHeadCard.svelte';
	import PairMlp from './pair/PairMlp.svelte';
	import PairOutput from './pair/PairOutput.svelte';
	import PairStrips from './pair/PairStrips.svelte';
	import type { Link } from './pair/bands';
	import { ROW_HEADING, STEP_TITLE, headCardHeight } from './pair/metrics';

	/** Encoder block 1 and decoder block 1 on one page, in the Dense explainer's visual language.
	 *  The encoder's final hidden state feeds decoder block 1's cross-attention keys and values. */
	const ROW_GAP = 30;
	let active = false;
	let error = '';
	let disposed = false;
	let cleanup: (() => void) | void;
	let root: HTMLElement | undefined;
	let wrapHeight = 0;
	let wrapWidth = 0;
	let encSelected = -1;
	let decSelected = -1;

	$: ex = $t5Example;
	$: heads = $t5Document?.model.attentionHeads ?? 8;
	$: head = $attentionHeadIdx;
	$: enc = ex?.encoder;
	$: dec = ex?.decoder;
	$: cross = ex?.crossAttention[0];
	// The T5 recording adds scores and probabilities to each attention layer.
	$: encOps = enc?.attentionOperations.layers[0] as T5Attention | undefined;
	$: decOps = dec?.attentionOperations.layers[0] as T5Attention | undefined;
	$: encTokens = enc?.example.tokens.map((token) => token.text) ?? [];
	$: decTokens = dec?.example.tokens.map((token) => token.text) ?? [];
	$: encSel = encSelected < 0 ? Math.max(0, encTokens.length - 1) : encSelected;
	$: decSel = decSelected < 0 ? Math.max(0, decTokens.length - 1) : decSelected;
	$: candidates =
		$t5Document && dec
			? recordedProbabilities(
					dec.example.nextTokenLogits,
					$t5Document.vocabulary,
					$temperature,
					$sampling
				)
					.slice(0, 8)
					.map(({ token, probability }) => ({ token, probability }))
			: [];

	// Pick the list pitch so both rows fit the viewport; below the minimum the page scrolls.
	$: m = Math.max(encTokens.length, 1);
	$: n = Math.max(decTokens.length, 1);
	const cardsAt = (p: number, m: number, n: number) =>
		headCardHeight(m, m, p, cellFor(p)) +
		Math.max(headCardHeight(n, n, p, cellFor(p)), headCardHeight(n, m, p, cellFor(p)));
	const cellFor = (p: number) => Math.min(22, Math.max(10, Math.round(p * 1.5)));
	// Two headings, two step-title rows, the gap, padding, and the ghost cards below the last card.
	$: fixed = 2 * (ROW_HEADING + STEP_TITLE) + ROW_GAP + 8 + 40;
	$: pitch = (() => {
		let p = 11;
		while (p > 7 && cardsAt(p, m, n) > wrapHeight - fixed) p -= 0.2;
		return Math.round(p * 10) / 10;
	})();
	$: cell = Math.min(cellFor(pitch), Math.max(10, Math.floor((wrapWidth - 1290) / (m + m))));
	$: encHeight = headCardHeight(m, m, pitch, cell);
	$: decHeight = Math.max(headCardHeight(n, n, pitch, cell), headCardHeight(n, m, pitch, cell));
	$: if (active && (pitch || cell || wrapWidth || head >= 0))
		tick().then(() => bandRedraw.update((count) => count + 1));

	function onHead(delta: number) {
		const next = (head + delta + heads) % heads;
		attentionHeadIdx.set(next);
		attentionHeadIdxTemp.set(next);
	}
	type Variant = 'encoder' | 'decoder' | 'cross';
	type Projection = {
		label: string;
		input: string;
		weight: string;
		output: string;
		inputSize: number;
		outputSize: number;
	};
	let popup:
		| { kind: 'projection'; block: string; projection: Projection; token: string; note: string }
		| { kind: 'attention'; variant: Variant; selected: number; token: string }
		| null = null;

	/** Q, K and V for one attention card, as one projection to all 8 heads. */
	function openStrip(variant: Variant, which: 'q' | 'k' | 'v', row: number) {
		const fromEncoder = variant === 'cross' && which !== 'q';
		const tokens = variant === 'encoder' || fromEncoder ? encTokens : decTokens;
		const name = { q: 'Query', k: 'Key', v: 'Value' }[which];
		const slice = `Head ${head + 1} reads coordinates ${head * 64 + 1}–${(head + 1) * 64}.`;
		popup = {
			kind: 'projection',
			block: `T5 · ${variant === 'encoder' ? 'Encoder' : 'Decoder'} block 1`,
			projection: {
				label: `${variant === 'cross' ? 'Cross-attention ' : ''}${name.toLowerCase()} projection`,
				input: fromEncoder ? 'Encoder hidden state' : 'RMSNorm(x)',
				weight: `W${which}ᵀ`,
				output: `${which.toUpperCase()} · ${heads} heads`,
				inputSize: 512,
				outputSize: 512
			},
			token: tokens[Math.min(row, tokens.length - 1)] ?? '',
			note: fromEncoder
				? `The input is the encoder's final hidden state, after all 6 encoder blocks. This page draws it into decoder block 1; blocks 2–6 project the same hidden state with their own weights. ${slice}`
				: `${heads} heads × 64 = 512, so the projection keeps the width. ${slice}`
		};
	}
	function openFeedForward(variant: 'encoder' | 'decoder', kind: 'up' | 'down', row: number) {
		const tokens = variant === 'encoder' ? encTokens : decTokens;
		popup = {
			kind: 'projection',
			block: `T5 · ${variant === 'encoder' ? 'Encoder' : 'Decoder'} block 1`,
			projection:
				kind === 'up'
					? {
							label: 'Up projection',
							input: 'RMSNorm(x)',
							weight: 'Wᵢᵀ',
							output: 'U',
							inputSize: 512,
							outputSize: 2048
						}
					: {
							label: 'Down projection',
							input: 'ReLU(U)',
							weight: 'Wₒᵀ',
							output: 'Y',
							inputSize: 2048,
							outputSize: 512
						},
			token: tokens[row] ?? '',
			note:
				kind === 'up'
					? 'ReLU then sets every negative coordinate of U to zero: h = max(0, U).'
					: 'Y is added back to the residual stream: x + Y.'
		};
	}
	function openOutput() {
		popup = {
			kind: 'projection',
			block: 'T5 · Decoder',
			projection: {
				label: 'LM head',
				input: 'RMSNorm(h) · 512^−½',
				weight: 'Eᵀ (tied)',
				output: 'logits',
				inputSize: 512,
				outputSize: $t5Document?.model.vocabSize ?? 32128
			},
			token: decTokens[decTokens.length - 1] ?? '',
			note: 'The input is the last decoder position after all 6 decoder blocks. T5 reuses the token embedding as this matrix and scales the hidden state by 1/√512 first. Softmax over the logits, with the temperature and sampling controls, gives the list.'
		};
	}
	function openCard(variant: Variant, kind: string, row: number) {
		if (kind === 'q' || kind === 'k' || kind === 'v') openStrip(variant, kind, row);
		else
			popup = {
				kind: 'attention',
				variant,
				selected: row,
				token: (variant === 'encoder' ? encTokens : decTokens)[row] ?? ''
			};
	}

	const QUERY = '#93b8f3';
	const KEY = '#f3a5a5';
	const VALUE = '#86dcad';
	const PURPLE = '#c9b8f3';
	const BLUE = '#b8cdf6';
	const lineOpacity = (selected: number) => (row: number) => (row === selected ? 0.95 : 0.4);
	const bandOpacity = (selected: number) => (row: number) => (row === selected ? 0.55 : 0.22);
	/** Embedding → bank → head card, as the Dense explainer draws its Q · K · V paths. */
	const attentionLinks = (prefix: string, card: string, selected: number): Link[] => [
		{
			from: `${prefix}-tok`,
			to: `${prefix}-bank:tok`,
			color: BLUE,
			opacity: bandOpacity(selected)
		},
		{ from: `${prefix}-bank:q`, to: `${card}:q`, color: QUERY, opacity: lineOpacity(selected) },
		{ from: `${prefix}-bank:k`, to: `${card}:k`, color: KEY, opacity: lineOpacity(selected) },
		{ from: `${prefix}-bank:v`, to: `${card}:v`, color: VALUE, opacity: lineOpacity(selected) }
	];
	/** The feed-forward and the remaining blocks after an attention card's Out column. */
	const tailLinks = (prefix: string, card: string, selected: number): Link[] => [
		{
			from: `${card}:out`,
			to: `${prefix}-mlp:in`,
			color: PURPLE,
			opacity: bandOpacity(selected),
			width: (_source, target) => target.height
		},
		{
			from: `${prefix}-mlp:out`,
			to: `${prefix}-blk:in`,
			color: BLUE,
			opacity: bandOpacity(selected)
		}
	];
	$: links = <Link[]>[
		...attentionLinks('enc', 'enc-self', encSel),
		...tailLinks('enc', 'enc-self', encSel),
		{ from: 'enc-blk:out', to: 'enc-hid', color: BLUE, opacity: bandOpacity(encSel) },
		{
			from: 'enc-hid:lane',
			to: 'dec-cross:card',
			group: true,
			color: '#b9c6f6',
			opacity: 0.85,
			width: () => 18,
			path: (source, target) => {
				// Run between the two rows, then drop into the card above its Key list.
				const lane = target.top - STEP_TITLE - ROW_HEADING - ROW_GAP / 2;
				const x = target.left + 40;
				return `M${source.cx},${source.bottom} V${lane} H${x} V${target.top}`;
			},
			label: {
				text: 'encoder hidden state → keys and values of decoder block 1 cross-attention',
				at: (_source, target) => ({
					x: target.left + 60,
					y: target.top - STEP_TITLE - ROW_HEADING - ROW_GAP / 2 - 13
				})
			}
		},
		...attentionLinks('dec', 'dec-self', decSel),
		{
			from: 'dec-self:out',
			to: 'dec-cross:q',
			color: QUERY,
			opacity: lineOpacity(decSel),
			width: () => 2
		},
		...tailLinks('dec', 'dec-cross', decSel),
		{
			from: 'dec-blk:out',
			to: 'dec-out:list',
			group: true,
			last: true,
			color: PURPLE,
			opacity: 0.45,
			width: (source) => source.height
		}
	];

	onMount(() => {
		(async () => {
			try {
				cleanup = await loadT5Recording();
				if (disposed) {
					cleanup?.();
					return;
				}
				active = true;
			} catch (reason) {
				error = reason instanceof Error ? reason.message : 'Could not load the T5 recording';
			}
		})();
		return () => {
			disposed = true;
			cleanup?.();
		};
	});
</script>

<svelte:head>
	<title>Transformer Explainer · T5 · Encoder–decoder</title>
</svelte:head>

<div class="pair-page">
	<div class="toolbar">
		<span>Full model translation <b>{$t5Example?.translation ?? 'Loading…'}</b></span>
		<span class="scope"
			>Block 1 of 6 in each stack · all {heads} heads · bidirectional encoder, causal decoder</span
		>
	</div>
	{#if error}<div class="status" role="alert">{error}</div>
	{:else if !active}<div class="status" role="status">Loading recorded inference…</div>{/if}
	<div class="wrap" bind:clientHeight={wrapHeight} bind:clientWidth={wrapWidth}>
		{#if active && ex && enc && dec && cross && encOps && decOps}
			<div class="board" bind:this={root} style={`--row-gap:${ROW_GAP}px`}>
				<PairBands {root} {links} />

				<h2 style="grid-column:1 / -1;grid-row:1">Encoder Block 1</h2>
				<div class="step-title" style="grid-column:1;grid-row:2">Embedding</div>
				<div class="step-title" style="grid-column:2 / span 3;grid-row:2;justify-self:center">
					Multi-head Self Attention <small>bidirectional</small>
				</div>
				<div class="step-title" style="grid-column:5;grid-row:2">
					MLP <small>512 → 2048 → 512 · ReLU</small>
				</div>
				<div class="step-title small" style="grid-column:6;grid-row:2">
					<b>5 more</b> identical Encoder Blocks
				</div>
				<div class="step-title" style="grid-column:7;grid-row:2">Encoder hidden state</div>
				<div style="grid-column:1;grid-row:3">
					<PairStrips
						id="enc-tok"
						labels={encTokens}
						vectors={enc.example.embeddingPreview}
						height={encHeight}
						selected={encSel}
						on:select={(e) => (encSelected = e.detail)}
					/>
				</div>
				<div style="grid-column:2;grid-row:3">
					<PairBank id="enc-bank" rows={m} height={encHeight} selected={encSel} />
				</div>
				<div style="grid-column:3 / span 2;grid-row:3;justify-self:center">
					<PairHeadCard
						id="enc-self"
						title="Encoder self-attention"
						queryTokens={encTokens}
						keyTokens={encTokens}
						matrix={encOps.probabilities[head]}
						{pitch}
						{cell}
						{head}
						{heads}
						selected={encSel}
						on:head={(e) => onHead(e.detail)}
						on:select={(e) => (encSelected = e.detail)}
						on:open={(e) => openCard('encoder', e.detail, encSel)}
					/>
				</div>
				<div style="grid-column:5;grid-row:3">
					<PairMlp
						id="enc-mlp"
						labels={encTokens}
						height={encHeight}
						selected={encSel}
						on:open={(e) => openFeedForward('encoder', e.detail.kind, e.detail.row)}
					/>
				</div>
				<div style="grid-column:6;grid-row:3;justify-self:center">
					<PairBlocks id="enc-blk" rows={m} height={encHeight} />
				</div>
				<div style="grid-column:7;grid-row:3">
					<PairStrips
						id="enc-hid"
						labels={encTokens}
						vectors={enc.example.finalHiddenPreview}
						color="indigo"
						height={encHeight}
						selected={encSel}
						labelSide="right"
						interactive={false}
					/>
				</div>

				<h2 class="second" style="grid-column:1 / -1;grid-row:4">Decoder Block 1</h2>
				<div class="step-title" style="grid-column:1;grid-row:5">Embedding</div>
				<div class="step-title" style="grid-column:2 / span 2;grid-row:5;justify-self:center">
					Multi-head Self Attention <small>causal</small>
				</div>
				<div class="step-title" style="grid-column:4;grid-row:5;justify-self:center">
					Cross Attention
				</div>
				<div class="step-title" style="grid-column:5;grid-row:5">
					MLP <small>512 → 2048 → 512 · ReLU</small>
				</div>
				<div class="step-title small" style="grid-column:6;grid-row:5">
					<b>5 more</b> identical Decoder Blocks
				</div>
				<div class="step-title" style="grid-column:7;grid-row:5">Probabilities</div>
				<div style="grid-column:1;grid-row:6">
					<PairStrips
						id="dec-tok"
						labels={decTokens}
						vectors={dec.example.embeddingPreview}
						height={decHeight}
						selected={decSel}
						on:select={(e) => (decSelected = e.detail)}
					/>
				</div>
				<div style="grid-column:2;grid-row:6">
					<PairBank id="dec-bank" rows={n} height={decHeight} selected={decSel} />
				</div>
				<div style="grid-column:3;grid-row:6">
					<PairHeadCard
						id="dec-self"
						title="Decoder self-attention"
						queryTokens={decTokens}
						keyTokens={decTokens}
						matrix={decOps.probabilities[head]}
						causal
						{pitch}
						{cell}
						{head}
						{heads}
						selected={decSel}
						on:head={(e) => onHead(e.detail)}
						on:select={(e) => (decSelected = e.detail)}
						on:open={(e) => openCard('decoder', e.detail, decSel)}
					/>
				</div>
				<div style="grid-column:4;grid-row:6">
					<PairHeadCard
						id="dec-cross"
						title="Cross-attention"
						queryTokens={decTokens}
						keyTokens={encTokens}
						matrix={cross.probabilities[head]}
						keyNote=" · encoder"
						{pitch}
						{cell}
						{head}
						{heads}
						selected={decSel}
						on:head={(e) => onHead(e.detail)}
						on:select={(e) => (decSelected = e.detail)}
						on:open={(e) => openCard('cross', e.detail, decSel)}
					/>
				</div>
				<div style="grid-column:5;grid-row:6">
					<PairMlp
						id="dec-mlp"
						labels={decTokens}
						height={decHeight}
						selected={decSel}
						on:open={(e) => openFeedForward('decoder', e.detail.kind, e.detail.row)}
					/>
				</div>
				<div style="grid-column:6;grid-row:6;justify-self:center">
					<PairBlocks id="dec-blk" rows={n} height={decHeight} />
				</div>
				<div style="grid-column:7;grid-row:6">
					<PairOutput id="dec-out" {candidates} height={decHeight} on:open={openOutput} />
				</div>
			</div>
		{/if}
	</div>
	{#if popup?.kind === 'projection'}
		<AnimatedMatrixPopup
			label={popup.block}
			projection={popup.projection}
			token={popup.token}
			on:close={() => (popup = null)}><p class="pair-note">{popup.note}</p></AnimatedMatrixPopup
		>
	{:else if popup?.kind === 'attention'}
		<PairAttentionPopup
			variant={popup.variant}
			rows={popup.variant === 'encoder' ? encTokens.length : decTokens.length}
			cols={popup.variant === 'decoder' ? decTokens.length : encTokens.length}
			{head}
			{heads}
			selected={popup.selected}
			token={popup.token}
			on:close={() => (popup = null)}
		/>
	{/if}
</div>

<style>
	.pair-page {
		width: 100%;
		height: 100%;
		display: flex;
		flex-direction: column;
		min-width: 0;
		overflow: hidden;
	}
	.toolbar {
		position: relative;
		z-index: 150;
		flex: 0 0 30px;
		padding: 0 24px 4px;
		display: flex;
		align-items: center;
		gap: 16px;
		color: #9986ab;
		font:
			12px Arial,
			sans-serif;
	}
	.toolbar b {
		color: #765889;
		margin-left: 8px;
		font-weight: 500;
	}
	.scope {
		margin-left: auto;
		color: #a798b5;
	}
	@media (max-width: 1400px) {
		.scope {
			display: none;
		}
	}
	.status {
		padding: 4rem 3rem;
		color: #9ca3af;
	}
	.wrap {
		flex: 1;
		min-height: 0;
		position: relative;
		overflow: auto;
	}
	.board {
		position: relative;
		isolation: isolate;
		box-sizing: border-box;
		min-width: 1280px;
		width: 100%;
		padding: 0 28px 8px;
		display: grid;
		grid-template-columns: repeat(7, auto);
		justify-content: space-between;
		align-items: start;
		column-gap: 20px;
	}
	h2 {
		margin: 0;
		height: 20px;
		font:
			600 17px/20px Arial,
			sans-serif;
		color: #374151;
		white-space: nowrap;
	}
	h2.second {
		margin-top: var(--row-gap);
	}
	.step-title {
		height: 30px;
		display: flex;
		align-items: center;
		gap: 6px;
		font:
			17px Arial,
			sans-serif;
		color: #9ca3af;
		white-space: nowrap;
	}
	.step-title small {
		font-size: 11px;
		color: #b8b2c4;
	}
	.step-title.small {
		display: block;
		height: auto;
		font-size: 12px;
		line-height: 1.15;
		white-space: normal;
		max-width: 96px;
		margin-top: -6px;
	}
	.step-title.small b {
		color: #6b7280;
	}
	:global(.pair-note) {
		color: #6b7280;
		font-size: 13px;
		line-height: 1.5;
		margin: 6px 0 0;
		text-align: center;
	}
</style>
