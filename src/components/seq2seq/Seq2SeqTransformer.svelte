<script lang="ts">
	import RecordedTransformer from '../RecordedTransformer.svelte';
	import Seq2SeqAttention from './Seq2SeqAttention.svelte';
	import Seq2SeqFeedForward from './Seq2SeqFeedForward.svelte';
	import Seq2SeqOutput from './Seq2SeqOutput.svelte';
	import Seq2SeqInspector from './Seq2SeqInspector.svelte';
	import { loadT5Recording, selectT5Stack } from '$lib/seq2seq/adapter';
	import { t5Stack, t5Example, t5Inspector } from '$lib/seq2seq/store';
</script>

<div class="t5-transformer">
	<div class="stack-toolbar">
		<div class="stack-tabs" aria-label="Encoder or decoder stack">
			<button
				class:chosen={$t5Stack === 'encoder'}
				aria-pressed={$t5Stack === 'encoder'}
				on:click={() => selectT5Stack('encoder')}>Encoder · 6 blocks</button
			><button
				class:chosen={$t5Stack === 'decoder'}
				aria-pressed={$t5Stack === 'decoder'}
				on:click={() => selectT5Stack('decoder')}>Decoder · 6 blocks</button
			>
		</div>
		<span>Full model translation <b>{$t5Example?.translation ?? 'Loading…'}</b></span>
		<span class="scope"
			>All 8 heads · {$t5Stack === 'encoder'
				? 'bidirectional source attention'
				: 'causal prefix + encoder cross-attention'}</span
		>
	</div>
	<RecordedTransformer
		modelLabel="T5 · Encoder–decoder"
		customOutput={Seq2SeqOutput}
		customAttention={Seq2SeqAttention}
		customFeedForward={Seq2SeqFeedForward}
		loadRecording={loadT5Recording}
	/>
	{#if $t5Inspector}<Seq2SeqInspector on:close={() => t5Inspector.set(null)} />{/if}
</div>

<style>
	.t5-transformer {
		width: 100%;
		height: 100%;
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.stack-toolbar {
		position: relative;
		z-index: 150;
		height: 42px;
		flex: 0 0 42px;
		padding: 0 20px 7px;
		display: flex;
		align-items: center;
		gap: 16px;
		color: #9986ab;
		font:
			12px Arial,
			sans-serif;
	}
	.stack-tabs {
		display: flex;
		border: 1px solid #ddcbed;
		border-radius: 5px;
		overflow: hidden;
	}
	.stack-tabs button {
		border: 0;
		background: #fcf9ff;
		color: #9a7db2;
		padding: 8px 14px;
		font: inherit;
		cursor: pointer;
	}
	.stack-tabs button.chosen {
		background: #ab80c9;
		color: white;
	}
	.stack-toolbar b {
		color: #765889;
		margin-left: 8px;
		font-weight: 500;
	}
	@media (max-width: 1400px) {
		.scope {
			display: none;
		}
	}
	.scope {
		margin-left: auto;
		color: #a798b5;
	}
	.t5-transformer :global(.main-section) {
		flex: 1;
		min-height: 0;
	}
</style>
