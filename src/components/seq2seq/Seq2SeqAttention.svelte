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
		padding: 0 16px 0 16px !important;
	}
	:global(.t5-self-attention .head-block .qkv) {
		padding-left: 80px !important;
	}
	:global(.t5-self-attention .head-out) {
		margin-left: 6px !important;
		margin-right: 6px !important;
	}
	:global(.t5-self-attention .head-block) {
		height: calc(var(--t5-card-height) - 44px) !important;
		min-height: 0 !important;
	}
	:global(.t5-self-attention .head-block > .qkv) {
		gap: 20px;
		justify-content: space-between;
		padding-top: 20px;
		padding-bottom: 20px;
	}
	:global(.t5-self-attention .head-flow-wrap) {
		display: flex;
		justify-content: flex-end;
		max-width: none;
	}
	:global(.t5-self-attention .column .label) {
		font-size: 11px;
	}
	:global(.t5-self-attention .head-flow-wrap) {
		height: 44px;
		align-items: center;
		padding: 8px 12px;
	}
	:global(.t5-self-attention .column .title) {
		font-size: 12px !important;
		top: -18px !important;
		line-height: 14px;
	}
	:global(.t5-self-attention .column .title small) {
		display: inline;
		margin-left: 5px;
		font-size: 8px;
	}
	:global(.t5-self-attention .head-title) {
		font-size: 13px;
	}
</style>
