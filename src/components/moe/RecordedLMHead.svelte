<script lang="ts">
	import { moeDocument, moeExample } from '~/store';
	import WeightPopoverCard from '../common/WeightPopoverCard.svelte';
	import { loadAttentionOperations, type RecordedLMHead } from '$lib/moe/attention';
	import { traceAssetUrl } from '$lib/moe/live';
	import { base } from '$app/paths';

	export let isOpen = true;
	let data: RecordedLMHead | null = null;
	let selected = 0;
	let error = '';
	let requested = '';
	$: if ($moeExample && requested !== $moeExample.id) load($moeExample.id);
	$: tokenId = data?.tokenIds[selected];
	$: tokenText = tokenId == null ? '' : ($moeDocument?.vocabulary[tokenId] ?? String(tokenId));
	$: reconstructed = data
		? data.weights[selected]
				.slice(0, 4)
				.reduce(
					(sum, weight, index) => sum + weight * data!.input[index],
					data.omittedContribution[selected]
				)
		: 0;

	async function load(id: string) {
		requested = id;
		data = null;
		error = '';
		selected = 0;
		try {
			const recording = await loadAttentionOperations(id);
			if (requested !== id) return;
			if (!recording.lmHead) throw new Error('The output projection recording is unavailable.');
			data = recording.lmHead;
		} catch (reason) {
			if (requested === id) error = reason instanceof Error ? reason.message : String(reason);
		}
	}
	function number(value: number) {
		return Number.isFinite(value) ? value.toFixed(5).replace('-', '−') : '—';
	}
</script>

<WeightPopoverCard
	id="logits"
	title="Logits · recorded output projection"
	className="recorded-lm-head"
	showReplay={false}
	timeline={undefined}
	bind:isOpen
>
	<div class="recorded-output">
		{#if error}<p role="alert">{error}</p>
		{:else if !data}<p role="status">Loading recorded output projection…</p>
		{:else}
			<p class="intro">
				After all {$moeDocument?.model.numLayers} blocks and final RMSNorm, the last token’s {data.inputSize.toLocaleString()}
				coordinates project into {$moeDocument?.model.vocabSize.toLocaleString()} vocabulary scores.
			</p>
			<div class="candidate-buttons" role="group" aria-label="Recorded vocabulary projection">
				{#each data.tokenIds as id, row}<button
						type="button"
						class:active={selected === row}
						aria-pressed={selected === row}
						on:click={() => (selected = row)}
						>{JSON.stringify($moeDocument?.vocabulary[id] ?? String(id))}<small>token {id}</small
						></button
					>{/each}
			</div>
			<div class="projection-grid">
				<div class="column">
					<h4>W<sub>vocab</sub>[{tokenId}]</h4>
					<small>first 4 weights</small>{#each data.weights[selected].slice(0, 4) as value}<code
							>{number(value)}</code
						>{/each}
				</div>
				<span class="operator">×</span>
				<div class="column">
					<h4>Hidden x</h4>
					<small>first 4 coordinates</small>{#each data.input.slice(0, 4) as value}<code
							>{number(value)}</code
						>{/each}
				</div>
				<span class="operator">+</span>
				<div class="column term">
					<h4>Remaining</h4>
					<small>{(data.inputSize - 4).toLocaleString()} inputs</small><code
						>{number(data.omittedContribution[selected])}</code
					>
				</div>
				<span class="operator">=</span>
				<div class="column term">
					<h4>Logit</h4>
					<small>{JSON.stringify(tokenText)}</small><code class="output"
						>{number(data.output[selected])}</code
					>
				</div>
			</div>
			<div class="numeric-formula" aria-label="Recorded logit formula">
				<strong>logit[{tokenId}]</strong> =
				{#each data.weights[selected].slice(0, 4) as weight, index}
					<span>{index ? '+' : ''} ({number(weight)} × {number(data.input[index])})</span>
				{/each}
				<span
					>+ {number(data.omittedContribution[selected])} ≈
					<strong>{number(data.output[selected])}</strong></span
				>
			</div>
			<p class="note">
				The remaining term is the recorded contribution of the other input coordinates. Values are
				rounded for display; reconstruction difference: {Math.abs(
					reconstructed - data.output[selected]
				).toExponential(2)}. Temperature and softmax follow this projection.
				<a href={traceAssetUrl(requested, 'attention', base)} target="_blank" rel="noreferrer"
					>Trace JSON ↗</a
				>
			</p>
		{/if}
	</div>
</WeightPopoverCard>

<style>
	:global(.recorded-lm-head) {
		width: min(64rem, calc(100vw - 3rem)) !important;
		max-width: calc(100vw - 3rem) !important;
	}
	.recorded-output {
		padding: 1.25rem;
		color: #73558b;
	}
	.intro,
	.note {
		font-size: 0.8rem;
		line-height: 1.5;
		margin: 0;
	}
	.candidate-buttons {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		margin: 1rem 0 1.5rem;
		border: 1px solid #dfd1ef;
		border-radius: 0.4rem;
		overflow: hidden;
	}
	button {
		padding: 0.6rem;
		border-right: 1px solid #dfd1ef;
		background: #faf7fc;
		font-weight: 600;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	button:last-child {
		border-right: 0;
	}
	button.active {
		background: #a47bd0;
		color: white;
	}
	button:focus-visible {
		outline: 2px solid #7549ad;
		outline-offset: -3px;
	}
	small {
		display: block;
		font-size: 0.7rem;
		font-weight: 400;
	}
	.projection-grid {
		display: grid;
		grid-template-columns: minmax(6rem, 1fr) 1rem minmax(6rem, 1fr) 1rem minmax(
				6rem,
				1fr
			) 1rem minmax(6rem, 1fr);
		gap: 0.5rem;
		align-items: center;
	}
	.column {
		text-align: center;
	}
	h4 {
		font-weight: 600;
	}
	.column > small {
		min-height: 1.6rem;
	}
	code {
		display: block;
		padding: 0.4rem 0.2rem;
		background: #f6f1fa;
		border-radius: 0.2rem;
		margin-top: 0.25rem;
		font-size: 0.8rem;
		font-variant-numeric: tabular-nums;
	}
	.term {
		align-self: stretch;
	}
	.term code {
		margin-top: 3.25rem;
	}
	.term .output {
		background: #ecdcf7;
		font-weight: 600;
	}
	.operator {
		font-size: 1.4rem;
	}
	.numeric-formula {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		background: #faf7fc;
		border: 1px solid #e5d8f1;
		border-radius: 0.4rem;
		padding: 1rem;
		margin: 1.25rem 0 0.75rem;
		font-family: monospace;
		font-size: 0.8rem;
	}
	.numeric-formula span {
		white-space: nowrap;
	}
	.note {
		color: #8e7c9d;
		font-size: 0.7rem;
	}
	a {
		color: #945bc2;
	}
	@media (max-width: 600px) {
		.recorded-output {
			padding: 0.7rem;
		}
		.projection-grid {
			gap: 0.2rem;
			grid-template-columns: 1fr 0.6rem 1fr 0.6rem 1fr 0.6rem 1fr;
		}
		code {
			font-size: 0.62rem;
		}
		h4,
		small {
			font-size: 0.62rem;
		}
	}
</style>
