<script lang="ts">
	import LinearSoftmax from '../LinearSoftmax.svelte';
	import { t5Stack, t5Example } from '$lib/seq2seq/store';
	import { selectT5Stack } from '$lib/seq2seq/adapter';
	export let className: string | undefined = undefined;
</script>

{#if $t5Stack === 'decoder'}<LinearSoftmax {className} compact />{:else}<div
		class={`encoder-memory ${className ?? ''}`}
	>
		<div class="title">Encoder memory</div>
		<div class="content">
			<p>
				All {$t5Example?.encoder.example.tokens.length} final encoder vectors become the keys and values
				in every decoder cross-attention block.
			</p>
			<div class="memory-shape">{$t5Example?.encoder.example.tokens.length} × 512</div>
			<button on:click={() => selectT5Stack('decoder')}
				>Continue to decoder →<small
					>6 blocks · causal self-attention<br />+ encoder cross-attention</small
				></button
			>
		</div>
	</div>{/if}

<style>
	.encoder-memory .content {
		padding: 0 22px 0 8px;
		color: #aa94ba;
		font:
			12px/1.5 Arial,
			sans-serif;
	}
	.memory-shape {
		border: 1px solid #ddcdec;
		padding: 20px;
		background: #faf7fd;
		border-radius: 6px;
		text-align: center;
		color: #9671b1;
		margin: 16px 0;
		font-size: 18px;
	}
	.content button {
		border: 1px solid #ddcdec;
		padding: 12px;
		color: #8b60ab;
		background: #fbf9fd;
		border-radius: 5px;
		cursor: pointer;
		width: 100%;
		white-space: normal;
	}
	.content button small {
		display: block;
		color: #ad99bc;
		margin-top: 6px;
		font-size: 10px;
	}
</style>
