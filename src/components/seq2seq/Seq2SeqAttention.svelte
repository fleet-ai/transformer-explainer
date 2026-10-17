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
		padding: 0 1.5rem 0 4rem !important;
	}
	:global(.t5-self-attention .head-block .qkv) {
		padding-left: 4rem !important;
	}
	:global(.t5-self-attention .head-out) {
		margin-left: 1rem !important;
		margin-right: 1rem !important;
	}
</style>
