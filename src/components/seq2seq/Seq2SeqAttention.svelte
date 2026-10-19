<script lang="ts">
	import Attention from '../Attention.svelte';
	import CrossAttention from './CrossAttention.svelte';
	import { t5Stack } from '$lib/seq2seq/store';
	export let className: string | undefined = undefined;
</script>

<Attention className={`${className ?? ''} t5-self-attention`} />
{#if $t5Stack === 'decoder'}<CrossAttention {className} />{/if}

<style>
	:global(.t5-self-attention .heads) {
		padding: 0 12px 0 22px !important;
	}
	:global(.t5-self-attention .head-block .qkv) {
		padding-left: 40px !important;
	}
	:global(.t5-self-attention .head-out) {
		margin-left: 6px !important;
		margin-right: 6px !important;
	}
	:global(.t5-self-attention .head-block) {
		height: clamp(280px, calc(100dvh - 370px), 650px) !important;
		min-height: 0 !important;
	}
	:global(.t5-self-attention .head-block > .qkv) {
		gap: 2.3rem;
	}
	:global(.t5-self-attention .head-flow-wrap) {
		display: flex;
		justify-content: flex-end;
		max-width: none;
	}
	:global(.t5-self-attention .column .label) {
		font-size: 10px;
	}
</style>
