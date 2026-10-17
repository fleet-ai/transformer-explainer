<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { blockIdx, attentionHeadIdx, moeTokenIndex, moeExample, tokens } from '~/store';
	import { t5Document, t5Example, t5Stack, t5Inspector } from '$lib/seq2seq/store';
	import TraceSelector from '../moe/TraceSelector.svelte';
	import type { T5Attention } from '$lib/seq2seq/types';
	const dispatch = createEventDispatcher();
	let selectedKey = 0,
		coordinate = 0,
		projection: 'up' | 'relu' | 'down' = 'up',
		row = 0;
	const fmt = (n: number | undefined) => (Number.isFinite(n) ? n!.toFixed(5) : '—');
	$: stage = $t5Inspector ?? 'self';
	$: example = $t5Example;
	$: token = Math.min($moeTokenIndex, $tokens.length - 1);
	$: tokenOptions = $tokens.map((label, value) => ({ value, label: `${value}: ${label}` }));
	$: ops = (
		stage === 'cross'
			? example?.crossAttention[$blockIdx]
			: example?.[$t5Stack].attentionOperations.layers[$blockIdx]
	) as T5Attention | undefined;
	$: keyTokens =
		stage === 'cross' ? example?.encoder.example.tokens : example?.[$t5Stack].example.tokens;
	$: keyOptions = (keyTokens ?? []).map((t) => ({
		value: t.index,
		label: `${t.index}: ${t.text}`
	}));
	$: safeKey = Math.min(selectedKey, (keyTokens?.length ?? 1) - 1);
	$: q = ops?.queryAfter[$attentionHeadIdx]?.[token] ?? [];
	$: k = ops?.keyAfter[$attentionHeadIdx]?.[safeKey] ?? [];
	$: dot = ops?.scores[$attentionHeadIdx]?.[token]?.[safeKey];
	$: bias = ops?.positionBias[$attentionHeadIdx]?.[token]?.[safeKey] ?? 0;
	$: probability = ops?.probabilities[$attentionHeadIdx]?.[token]?.[safeKey];
	$: dotFirst = q.slice(0, 4).reduce((sum, value, index) => sum + value * (k[index] ?? 0), 0);
	$: ff = example?.feedForward[$t5Stack][$blockIdx].tokens[token];
	$: operation = projection === 'down' ? ff?.down : ff?.up;
	$: input = projection === 'down' ? ff?.activation : ff?.inputPreview;
	$: layer = $moeExample?.layers[$blockIdx];
	function selectStage(next: typeof stage) {
		t5Inspector.set(next);
	}
</script>

<svelte:window
	on:keydown={(e) => {
		if (e.key === 'Escape') dispatch('close');
	}}
/>
<section class="t5-inspector" role="dialog" aria-modal="false" aria-label="Recorded T5 operations">
	<header>
		<div>
			<h2>
				T5 · {stage === 'cross'
					? 'Encoder → decoder attention'
					: stage === 'ffn'
						? 'Dense ReLU feed-forward'
						: stage === 'output'
							? 'Language model output'
							: 'Self-attention'}
			</h2>
			<p>
				{$t5Stack === 'encoder' ? 'Encoder' : 'Decoder'} block {$blockIdx + 1} of 6 · head {$attentionHeadIdx +
					1} of 8 · complete pretrained model
			</p>
		</div>
		<button class="close" aria-label="Close T5 operations" on:click={() => dispatch('close')}
			>×</button
		>
	</header>
	<nav>
		<button class:chosen={stage === 'self'} on:click={() => selectStage('self')}
			>Self-attention</button
		>{#if $t5Stack === 'decoder'}<button
				class:chosen={stage === 'cross'}
				on:click={() => selectStage('cross')}>Cross-attention</button
			>{/if}<button class:chosen={stage === 'ffn'} on:click={() => selectStage('ffn')}
			>Feed-forward</button
		><button class:chosen={stage === 'output'} on:click={() => selectStage('output')}
			>Output projection</button
		>
	</nav>
	<div class="body">
		{#if stage !== 'output'}<div class="selector-row">
				<span>Token</span><TraceSelector
					label="Token"
					options={tokenOptions}
					bind:value={$moeTokenIndex}
				/>
			</div>{/if}
		{#if stage === 'self' || stage === 'cross'}
			<p class="explanation">
				{stage === 'cross'
					? 'Queries come from the decoder; keys and values come from the final encoder memory. Every source token is visible.'
					: $t5Stack === 'encoder'
						? 'Every encoder token attends to every source token. A learned relative-position bias is added to QKᵀ.'
						: 'Decoder self-attention adds a learned relative-position bias and masks future positions.'}
				T5 uses unscaled dot products and no RoPE.
			</p>
			<div class="selector-row">
				<span>Key</span><TraceSelector
					label="Key token"
					options={keyOptions}
					bind:value={selectedKey}
				/>
			</div>
			<div class="equation">
				<strong>Q[{token}] · K[{safeKey}]</strong> = {#each q.slice(0, 4) as value, i}<span
						>({fmt(value)} × {fmt(k[i])}) +
					</span>{/each}<span class="omitted"
					>{fmt((dot ?? 0) - dotFirst)} <small>remaining 60 terms</small></span
				>
				≈ <b>{fmt(dot)}</b>
			</div>
			<div class="stat-grid">
				<div>Recorded dot product<b>{fmt(dot)}</b></div>
				<div>
					{stage === 'cross' ? 'Position bias' : 'Relative bias + mask'}<b
						>{Math.abs(bias) > 1e30 ? '−∞' : fmt(bias)}</b
					>
				</div>
				<div>
					Softmax over {keyTokens?.length} keys<b>{((probability ?? 0) * 100).toFixed(4)}%</b>
				</div>
			</div>
			<div class="coordinate">
				<label
					>Output coordinate <input
						type="number"
						min="0"
						max="63"
						bind:value={coordinate}
						on:change={() => (coordinate = Math.max(0, Math.min(63, Math.trunc(coordinate || 0))))}
					/></label
				><span
					>Σ A[{token}, j] × V[j, {coordinate}] =
					<b>{fmt(ops?.attentionContext?.[$attentionHeadIdx]?.[token]?.[coordinate])}</b></span
				>
			</div>
			<table>
				<thead
					><tr
						><th>Key token</th><th>Attention A</th><th>Value V[{coordinate}]</th><th
							>Contribution A × V</th
						></tr
					></thead
				><tbody
					>{#each keyTokens ?? [] as key, index}<tr
							><td>{key.index}: {key.text}</td><td
								>{fmt(ops?.probabilities[$attentionHeadIdx]?.[token]?.[index])}</td
							><td>{fmt(ops?.value[$attentionHeadIdx]?.[index]?.[coordinate])}</td><td
								>{fmt(
									(ops?.probabilities[$attentionHeadIdx]?.[token]?.[index] ?? 0) *
										(ops?.value[$attentionHeadIdx]?.[index]?.[coordinate] ?? 0)
								)}</td
							></tr
						>{/each}</tbody
				>
			</table>
			<p class="footnote">
				All key contributions are shown. Eight 64D head outputs are concatenated and multiplied by
				Wₒ to return a 512D residual update. Values come from the recorded forward pass.
			</p>
		{:else if stage === 'ffn' && ff}
			<div class="flow">
				<button class:chosen={projection === 'up'} on:click={() => (projection = 'up')}
					>Wᵢ × x · 512 → 2048</button
				><span>→</span><button
					class:chosen={projection === 'relu'}
					on:click={() => (projection = 'relu')}>ReLU: max(0, u)</button
				><span>→</span><button
					class:chosen={projection === 'down'}
					on:click={() => (projection = 'down')}>Wₒ × h · 2048 → 512</button
				>
			</div>
			{#if projection === 'relu'}<table>
					<thead
						><tr><th>Coordinate</th><th>Up projection u</th><th>ReLU(u) = max(0, u)</th></tr></thead
					><tbody
						>{#each ff.activation as value, i}<tr
								><td>{i}</td><td>{fmt(ff.up.output[i])}</td><td>{fmt(value)}</td></tr
							>{/each}</tbody
					>
				</table>
				<p class="footnote">
					First 4 of 2,048 recorded intermediate coordinates. This is a dense ReLU network: all
					intermediate units are evaluated for every token.
				</p>
			{:else if operation && input}<h3>
					{projection === 'up' ? 'Up' : 'Down'} projection · {projection === 'up'
						? '2048 × 512'
						: '512 × 2048'} weights
				</h3>
				<table class="weights">
					<thead
						><tr
							><th>Output</th>{#each [0, 1, 2, 3] as i}<th>W[{i}] × x[{i}]</th>{/each}<th
								>Remaining {projection === 'up' ? 508 : 2044} terms</th
							><th>Recorded result</th></tr
						></thead
					><tbody
						>{#each operation.weights as weights, i}<tr
								class:active={row === i}
								on:click={() => (row = i)}
								>{#each [i] as output}<td
										><button aria-label={`Select output ${i}`} on:click={() => (row = i)}
											>{i}</button
										></td
									>{/each}{#each weights as weight, j}<td
										>{fmt(weight)}<small>× {fmt(input[j])}</small></td
									>{/each}<td>{fmt(operation.omittedContribution[i])}</td><td
									><b>{fmt(operation.output[i])}</b></td
								></tr
							>{/each}</tbody
					>
				</table>
				<div class="equation">
					y[{row}] = {#each operation.weights[row] as weight, i}<span
							>({fmt(weight)} × {fmt(input[i])}) +
						</span>{/each}{fmt(operation.omittedContribution[row])} ≈
					<b>{fmt(operation.output[row])}</b>
				</div>{/if}
			<div class="residual">
				<h3>Residual stream · first four coordinates</h3>
				<table>
					<thead
						><tr
							><th>Coordinate</th><th>Feed-forward input after RMSNorm</th><th
								>Feed-forward output</th
							><th>Block output after residual addition</th></tr
						></thead
					><tbody
						>{#each ff.inputPreview as value, i}<tr
								><td>{i}</td><td>{fmt(value)}</td><td>{fmt(ff.down.output[i])}</td><td
									>{fmt(layer?.blockOutputPreview?.[token]?.[i])}</td
								></tr
							>{/each}</tbody
					>
				</table>
			</div>
		{:else if stage === 'output' && example}
			<p class="explanation">
				The full decoder generated <b>{example.translation}</b> from the source prompt. These logits
				predict the next token after that generated prefix. The tied vocabulary matrix scores all 32,128
				tokens.
			</p>
			<div class="equation">
				Final RMSNorm hidden state × 1/√512 ({fmt(example.lmHead.scale)}) → scaled 512D vector →
				vocabulary projection → softmax.
			</div>
			<table>
				<thead
					><tr
						><th>Candidate</th>{#each [0, 1, 2, 3] as i}<th>W[{i}] × x[{i}]</th>{/each}<th
							>Remaining 508</th
						><th>Recorded logit</th></tr
					></thead
				><tbody
					>{#each example.lmHead.candidateLabels as label, i}<tr
							><td>{label}</td>{#each example.lmHead.weights[i] as weight, j}<td
									>{fmt(weight)}<small>× {fmt(example.lmHead.inputPreview[j])}</small></td
								>{/each}<td>{fmt(example.lmHead.omittedContribution[i])}</td><td
								><b>{fmt(example.lmHead.logits[i])}</b></td
							></tr
						>{/each}</tbody
				>
			</table>
			<p class="footnote">
				Four vocabulary rows are shown with the exact recorded contribution of the other 508
				coordinates. The probability sidebar uses the complete vocabulary logits.
			</p>
		{/if}
		<footer>
			Recorded from <a href={$t5Document?.model.url} target="_blank" rel="noreferrer"
				>google-t5/t5-small</a
			> · all 6 encoder + 6 decoder blocks · all 8 heads. Rounded for display; actual tensors retain
			float32 precision.
		</footer>
	</div>
</section>

<style>
	.t5-inspector {
		position: fixed;
		z-index: 1300;
		right: 22px;
		bottom: 22px;
		width: min(1120px, calc(100vw - 44px));
		height: min(740px, calc(100vh - 44px));
		display: flex;
		flex-direction: column;
		background: #fff;
		border: 1px solid #ded2ec;
		border-radius: 12px;
		box-shadow: 0 15px 40px #21132d30;
		color: #806495;
		font:
			13px/1.5 Arial,
			sans-serif;
	}
	header {
		display: flex;
		justify-content: space-between;
		padding: 18px 24px;
		background: #fbf8fd;
		border-bottom: 1px solid #e9dfef;
	}
	h2 {
		font-size: 22px;
		color: #674887;
		margin: 0;
	}
	header p {
		margin: 3px 0 0;
		color: #a18eaf;
	}
	.close {
		border: 1px solid #e1d6eb;
		background: white;
		color: #816793;
		border-radius: 6px;
		font-size: 26px;
		width: 35px;
		height: 35px;
		line-height: 1;
		cursor: pointer;
	}
	nav {
		display: flex;
		padding: 10px 24px 0;
		gap: 8px;
	}
	nav button,
	.flow button {
		border: 1px solid #e3d6ed;
		border-radius: 4px;
		padding: 8px 14px;
		background: #fcf9fe;
		color: #9676ae;
		cursor: pointer;
	}
	.chosen {
		background: #ad83cc !important;
		color: white !important;
	}
	.body {
		padding: 14px 24px 20px;
		overflow: auto;
		min-height: 0;
	}
	.selector-row {
		display: flex;
		gap: 12px;
		align-items: center;
		margin-bottom: 10px;
	}
	.selector-row > span {
		width: 38px;
		flex-shrink: 0;
	}
	.selector-row :global(.selector) {
		min-width: 0;
		flex: 1;
	}
	.explanation {
		margin: 10px 0 14px;
		color: #a18eaf;
	}
	.equation {
		padding: 12px;
		background: #faf7fc;
		border: 1px solid #eee4f4;
		border-radius: 5px;
		font: 11px/2 monospace;
		overflow-wrap: anywhere;
	}
	.equation b {
		color: #7c4ca3;
	}
	.omitted {
		display: inline-block;
		background: #efe4f6;
		padding: 0 6px;
	}
	.omitted small {
		font: 9px Arial;
		display: block;
	}
	.stat-grid {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 12px;
		margin: 12px 0;
	}
	.stat-grid > div {
		background: #faf8fc;
		padding: 10px;
		border-radius: 5px;
	}
	.stat-grid b {
		display: block;
		font: 16px monospace;
		margin-top: 3px;
	}
	.coordinate {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin: 12px 0;
	}
	.coordinate input {
		width: 65px;
		border: 1px solid #dccbe9;
		border-radius: 4px;
		padding: 5px;
		color: #805c98;
	}
	table {
		width: 100%;
		border-collapse: separate;
		border-spacing: 3px;
		font-size: 11px;
		table-layout: fixed;
		text-align: center;
	}
	th {
		font-weight: normal;
		color: #ab95bb;
		padding: 5px;
	}
	td {
		background: #faf7fc;
		padding: 7px 4px;
		font-family: monospace;
		overflow-wrap: anywhere;
	}
	td small {
		display: block;
		color: #7697bd;
	}
	td:first-child {
		font: 11px Arial;
	}
	td button {
		border: 0;
		background: transparent;
		color: inherit;
		cursor: pointer;
	}
	.active td {
		background: #efe3f8;
	}
	.flow {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 14px;
		margin: 15px 0;
	}
	.footnote,
	footer {
		color: #aa99b8;
		font-size: 11px;
	}
	.footnote {
		margin: 12px 0;
	}
	h3 {
		font-size: 13px;
		margin: 12px 0 6px;
	}
	.residual {
		margin-top: 12px;
	}
	footer {
		border-top: 1px solid #eee6f4;
		padding-top: 10px;
		margin-top: 14px;
	}
	footer a {
		color: #9369ae;
	}
</style>
