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
	/* Rows reach Dense's size (RecordedTransformer's height budget); labels and columns follow them. */
	.vit-transformer :global(.main-section.squeeze) {
		/* The head card is as tall as the Q/K/V bank (three vectors per row), as in Dense. */
		--compact-card-height: max(230px, calc(var(--vector-height) * 18 + 42px));
		--vit-label-size: clamp(11px, calc(var(--vector-height) * 0.72), 14px);
		--vit-token-column: clamp(132px, 8.5vw, 168px);
	}
	/* Room above the titles for "N more Transformer Blocks". */
	.vit-transformer :global(.main-section.squeeze .nodes) {
		padding-top: 22px;
	}
	/* Wider token and class columns so labels are not cut; a wider final-block column leaves room for
	   the Wcls box on the [CLS] band. The Q/K/V column grows a little for the bands into the bank, and
	   the MLP keeps a capped width instead of taking a third of the page. */
	.vit-transformer :global(.main-section.squeeze .nodes .steps) {
		grid-template-columns: var(--vit-token-column) minmax(0, 1fr) 62px clamp(232px, 15vw, 290px);
	}
	.vit-transformer
		:global(.main-section.squeeze:not(.attention-open) .nodes .steps .blocks .block-steps) {
		grid-template-columns: clamp(70px, 6vw, 120px) minmax(0, 1fr) minmax(
				235px,
				clamp(235px, 24vw, 420px)
			);
	}
	/* Dense's Probabilities ⊕ needs its logits column while it is open, and room above for its subtitles. */
	.vit-transformer :global(.main-section.squeeze .nodes .steps:has(.softmax.expanded)) {
		grid-template-columns: var(--vit-token-column) minmax(0, 1fr) 62px clamp(300px, 19vw, 360px);
	}
	.vit-transformer :global(.main-section.squeeze .steps.expanded .softmax.expanded > .title) {
		padding-bottom: 40px;
	}
	/* The column minus its padding, the gap and the vector strip. */
	.vit-transformer :global(.main-section.squeeze .embedding:not(.expanded) .token-string) {
		width: calc(var(--vit-token-column) - 40px);
	}
	.vit-transformer :global(.main-section.squeeze .softmax:not(.expanded) .content) {
		grid-template-columns: minmax(96px, 42%) minmax(0, 1fr);
	}
	.vit-transformer :global(.main-section.squeeze .softmax:not(.expanded) .token-string) {
		width: 100%;
		min-width: 0;
		overflow: hidden;
	}
	.vit-transformer :global(.main-section.squeeze .softmax .text-box) {
		font-size: var(--vit-label-size);
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
	/* The card keeps the bank's height, so K, Q and V spread top, middle and bottom as in Dense. */
	.vit-transformer :global(.main-section.squeeze .attention .head-block) {
		height: var(--compact-card-height) !important;
	}
	.vit-transformer :global(.main-section.squeeze .attention .head-block > .qkv) {
		justify-content: space-between;
	}
	.vit-transformer :global(.main-section.squeeze .attention .head-block > .qkv .column) {
		gap: 11px;
	}
	/* Keep each list's caption (Key 1 · Projected · 64D) clear of its first token label. */
	.vit-transformer :global(.main-section.squeeze .attention .head-block .column .title) {
		top: -28px;
	}
	.vit-transformer :global(.main-section.squeeze .attention .column .label) {
		font-size: clamp(9px, calc(var(--compact-card-height) / 52), 11.5px) !important;
		line-height: 12px !important;
	}
</style>
