<script lang="ts">
	import RecordedTransformer from '../RecordedTransformer.svelte';
	import VitEmbedding from './VitEmbedding.svelte';
	import VitFeedForward from './VitFeedForward.svelte';
	import VitInspector from './VitInspector.svelte';
	import { loadVitRecording } from '$lib/vit/adapter';
	import { vitInspectorStage } from '$lib/vit/store';
</script>

<div class="vit-transformer">
	<RecordedTransformer
		modelLabel="Vision Transformer"
		customEmbedding={VitEmbedding}
		customFeedForward={VitFeedForward}
		loadRecording={loadVitRecording}
	/>
	{#if $vitInspectorStage}<VitInspector
			stage={$vitInspectorStage}
			on:close={() => vitInspectorStage.set(null)}
		/>{/if}
</div>

<style>
	.vit-transformer {
		display: flex;
		flex-direction: column;
		width: 100%;
		min-height: 100%;
	}
</style>
