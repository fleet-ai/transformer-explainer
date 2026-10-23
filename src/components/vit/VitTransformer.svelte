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
	/* The head card is as tall as the Q/K/V bank (three vectors per row), as in Dense. */
	.vit-transformer :global(.main-section.squeeze) {
		--compact-card-height: max(230px, calc(var(--vector-height) * 18 + 42px));
	}
	/* Columns grow with the window, so the blocks do not absorb all of it. The final-block column
	   leaves room for the Wcls box on the [CLS] band. */
	.vit-transformer :global(.main-section.squeeze .nodes .steps) {
		grid-template-columns: clamp(124px, 8vw, 156px) minmax(0, 1fr) 62px clamp(210px, 13vw, 250px);
	}
	.vit-transformer
		:global(.main-section.squeeze:not(.attention-open) .nodes .steps .blocks .block-steps) {
		grid-template-columns: clamp(70px, 6vw, 120px) minmax(0, 1.4fr) minmax(235px, 0.9fr);
	}
	/* Dense's Probabilities ⊕ needs its logits column, and room above for its subtitles, while open. */
	.vit-transformer :global(.main-section.squeeze .nodes .steps:has(.softmax.expanded)) {
		grid-template-columns: clamp(124px, 8vw, 156px) minmax(0, 1fr) 62px clamp(300px, 18vw, 340px);
	}
	.vit-transformer :global(.main-section.squeeze .steps.expanded .softmax.expanded > .title) {
		padding-bottom: 40px;
	}
	/* Step titles sit above the Residual labels, as in Dense. */
	.vit-transformer :global(.main-section.squeeze .steps .step:not(.softmax) > .title) {
		padding-bottom: 38px;
	}
	/* Below the white "Transformer Block n" header, above the residual bracket. */
	.vit-transformer :global(.main-section.squeeze .residual-text) {
		top: -1.45rem;
		font-size: 0.8rem;
		line-height: 1;
	}
	/* Room between the Q/K/V bank and the head card for the bands that cross into it. */
	.vit-transformer :global(.main-section.squeeze .attention .heads) {
		padding-left: clamp(36px, 4.5vw, 84px) !important;
	}
	/* Room between each head list's title and its first token. */
	.vit-transformer :global(.main-section.squeeze .attention .head-block > .qkv) {
		padding-top: 32px !important;
	}
	.vit-transformer :global(.main-section.squeeze .attention .head-block > .qkv .column) {
		gap: 11px;
	}
	.vit-transformer :global(.main-section.squeeze .attention .column .title) {
		top: -27px;
	}
	.vit-transformer :global(.main-section.squeeze .attention .column .label) {
		font-size: 10.5px !important;
		line-height: 12px !important;
	}
	/* ImageNet class names need more than the compact 70px. */
	.vit-transformer :global(.main-section.squeeze .softmax:not(.expanded) .content) {
		grid-template-columns: 96px minmax(0, 1fr);
	}
	.vit-transformer :global(.main-section.squeeze .softmax:not(.expanded) .token-string) {
		width: 96px;
	}
</style>
