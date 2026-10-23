<script lang="ts">
	import { fade } from 'svelte/transition';
	import RecordedTransformer from '../RecordedTransformer.svelte';
	import VitEmbedding from './VitEmbedding.svelte';
	import VitFeedForward from './VitFeedForward.svelte';
	import VitClassifierBox from './VitClassifierBox.svelte';
	import VitOperationPopup, { VIT_OPERATIONS, type VitOperation } from './VitOperationPopup.svelte';
	import VitProjectionPopup, {
		VIT_PROJECTIONS,
		type VitProjection
	} from './VitProjectionPopup.svelte';
	import { expandedBlock } from '~/store';
	import { loadVitRecording } from '$lib/vit/adapter';
	import { vitInspectorStage, vitRecording } from '$lib/vit/store';

	$: stage = $vitInspectorStage ?? '';
	$: projection = VIT_PROJECTIONS.includes(stage as VitProjection)
		? (stage as VitProjection)
		: null;
	$: operation = VIT_OPERATIONS.includes(stage as VitOperation) ? (stage as VitOperation) : null;
	// Dense dims the page while its Embedding ⊕ or Probabilities ⊕ is open; the compact layout does not.
	$: dimmed = $expandedBlock.id === 'embedding' || $expandedBlock.id === 'softmax';
</script>

<div class="vit-transformer">
	<RecordedTransformer
		modelLabel="Vision Transformer"
		customEmbedding={VitEmbedding}
		customFeedForward={VitFeedForward}
		loadRecording={loadVitRecording}
	/>
	{#if dimmed}<div
			class="vit-dim"
			role="presentation"
			transition:fade={{ duration: 100 }}
			on:click={() => expandedBlock.set({ id: null })}
		></div>{/if}
	{#if $vitRecording}<VitClassifierBox />{/if}
	{#if $vitRecording && projection}<VitProjectionPopup
			kind={projection}
			on:close={() => vitInspectorStage.set(null)}
		/>{:else if $vitRecording && operation}<VitOperationPopup
			kind={operation}
			on:close={() => vitInspectorStage.set(null)}
		/>{/if}
</div>

<style lang="scss">
	/* Not positioned: the band layer draws in page coordinates against the page origin. */
	.vit-transformer {
		display: flex;
		flex-direction: column;
		width: 100%;
		min-height: 100%;
	}
	.vit-dim {
		position: fixed;
		inset: 0;
		z-index: $DIM_INDEX;
		background: white;
		opacity: 0.7;
		user-select: none;
	}
	/* A wider final-block column leaves room for the Wcls box on the [CLS] band. */
	.vit-transformer :global(.main-section.squeeze .steps) {
		grid-template-columns: 120px minmax(0, 1fr) 62px 210px;
	}
	/* Dense's Probabilities ⊕ needs its logits column, and room above for its subtitles, while open. */
	.vit-transformer :global(.main-section.squeeze .steps.expanded .softmax.expanded > .title) {
		padding-bottom: 40px;
	}
	/* Dense's Probabilities ⊕ needs its logits column while it is open. */
	.vit-transformer :global(.main-section.squeeze .steps:has(.softmax.expanded)) {
		grid-template-columns: 120px minmax(0, 1fr) 62px 300px;
	}
</style>
