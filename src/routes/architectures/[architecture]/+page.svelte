<script lang="ts">
	import { base } from '$app/paths';
	import HybridTransformer from '~/components/hybrid/HybridTransformer.svelte';
	import VlmTransformer from '~/components/vlm/VlmTransformer.svelte';
	import Seq2SeqBlocks from '~/components/seq2seq/Seq2SeqBlocks.svelte';
	import VitTransformer from '~/components/vit/VitTransformer.svelte';
	import RecordedTransformer from '~/components/RecordedTransformer.svelte';
	import type { RecordedArchitectureId } from '~/lib/architecture-navigation';
	export let data: { architecture: RecordedArchitectureId };
</script>

<svelte:head>
	<title>Transformer Explainer · {data.architecture}</title>
	<meta
		name="description"
		content="Explore real pretrained transformer computations with interactive architecture diagrams and recorded tensors."
	/>
	<link rel="stylesheet" href={`${base}/fonts/font.css`} />
</svelte:head>

{#key data.architecture}
	{#if data.architecture === 'mla'}
		<RecordedTransformer tracePath="traces/deepseek-mla.json" modelLabel="DeepSeek · MLA" />
	{:else if data.architecture === 'encoder-decoder'}
		<Seq2SeqBlocks />
	{:else if data.architecture === 'hybrid'}
		<HybridTransformer />
	{:else if data.architecture === 'vision-language'}
		<VlmTransformer />
	{:else if data.architecture === 'vit'}
		<VitTransformer />
	{/if}
{/key}
