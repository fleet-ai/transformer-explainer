<script lang="ts">
	import { recordedAssetUrl } from '$lib/recorded-json.js';
	import { onMount, type ComponentType } from 'svelte';
	import { base } from '$app/paths';
	import classNames from 'classnames';
	import { fade } from 'svelte/transition';
	import {
		tokens,
		expandedBlock,
		vectorHeight,
		rootRem,
		maxVectorHeight,
		minVectorHeight,
		maxVectorScale,
		headContentHeight,
		sampling,
		temperature,
		isTextbookOpen,
		moeModel,
		moeExample,
		blockIdx
	} from '~/store';
	import Sankey from '~/components/Sankey.svelte';
	import Attention from '~/components/Attention.svelte';
	import SubsequentBlocks from '~/components/SubsequentBlocks.svelte';
	import LinearSoftmax from '~/components/LinearSoftmax.svelte';
	import Embedding from '~/components/Embedding.svelte';
	import Mlp from '~/components/Mlp.svelte';
	import QKV from '~/components/QKV.svelte';
	import WeightPopovers from '~/components/WeightPopovers.svelte';
	import BlockTransition from '~/components/BlockTransition.svelte';
	import Textbook from '~/components/textbook/Textbook.svelte';
	import { hydrateRecordedExample, applyRecordedSampling } from '$lib/moe/adapter';
	import type { TraceDocument } from '$lib/moe/types';
	export let tracePath = 'traces/granite-moe.json';
	export let modelLabel = 'MoE';
	export let customEmbedding: ComponentType | null = null;
	export let customFeedForward: ComponentType | null = null;
	export let customAttention: ComponentType | null = null;
	export let customOutput: ComponentType | null = null;
	export let customQKV: ComponentType | null = null;
	export let loadRecording: (() => Promise<void | (() => void)>) | null = null;
	let cleanupRecording: (() => void) | void;
	let disposed = false;
	let active = false;
	let error = '';
	let vizHeight = 0;
	$: squeeze = ['hybrid', 'vision-language', 'vit'].includes($moeModel?.architecture ?? '');
	$: compact = squeeze || $moeModel?.architecture === 'encoder-decoder';
	$: titleHeight = compact ? 60 : rootRem * 5;
	const calculateVectorHeight = () => {
		const gaps = rootRem * 0.5 * ($tokens.length - 1);
		const height = Math.min(
			Math.max(
				((compact ? Math.min(vizHeight - 24, 480) : vizHeight) - titleHeight - gaps) /
					$tokens.length /
					(compact ? 4 : maxVectorScale),
				squeeze ? 6 : compact ? 8 : minVectorHeight
			),
			maxVectorHeight
		);
		vectorHeight.set(height);
		headContentHeight.set(Math.max($tokens.length * height * 3 + gaps, rootRem * 20));
	};
	$: if (vizHeight || $tokens.length) calculateVectorHeight();
	async function loadTraces() {
		error = '';
		try {
			if (loadRecording) {
				cleanupRecording = await loadRecording();
			} else {
				const response = await fetch(recordedAssetUrl(`${base}/${tracePath}`));
				if (!response.ok) throw new Error(`Trace file returned HTTP ${response.status}`);
				hydrateRecordedExample((await response.json()) as TraceDocument, 0);
			}
			if (disposed) {
				cleanupRecording?.();
				return;
			}
			active = true;
		} catch (reason) {
			error = reason instanceof Error ? reason.message : 'Could not load traces';
		}
	}
	onMount(() => {
		loadTraces();
		const unsubTemperature = temperature.subscribe(() => {
			if (active) applyRecordedSampling();
		});
		const unsubSampling = sampling.subscribe(() => {
			if (active) applyRecordedSampling();
		});
		return () => {
			disposed = true;
			cleanupRecording?.();
			unsubTemperature();
			unsubSampling();
		};
	});
</script>

<svelte:head>
	<title>Transformer Explainer · {modelLabel}</title>
	<meta
		name="description"
		content="Explore every transformer block, attention head, feed-forward operation, and next-token distribution from a recorded pretrained language model."
	/>
	<link rel="stylesheet" href={`${base}/fonts/font.css`} />
</svelte:head>
{#if error}<div class="trace-status" role="alert">
		{error} <button on:click={loadTraces}>Try again</button>
	</div>{:else if !active}<div class="trace-status" role="status">
		Loading recorded inference…
	</div>{/if}
{#if active}
	<div
		class:active
		class="main-section h-full w-full"
		class:hybrid-model={$moeModel?.architecture === 'hybrid'}
		class:mamba-only={$moeModel?.architecture === 'hybrid' &&
			!$moeExample?.layers[$blockIdx]?.queryPreview.length}
		class:expert-layout={!$moeModel?.architecture ||
			['moe', 'mla'].includes($moeModel.architecture)}
		class:attention-open={$expandedBlock.id === 'attention'}
		class:cross-open={$expandedBlock.id === 'cross-attention'}
		class:latent-model={$moeModel?.architecture === 'mla'}
		class:squeeze
		class:vision-language={$moeModel?.architecture === 'vision-language'}
		class:seq2seq={$moeModel?.architecture === 'encoder-decoder'}
		class:decoder={$moeModel?.architecture === 'encoder-decoder' && $moeModel?.stack === 'decoder'}
		style={`--vector-height: ${$vectorHeight}px;--title-height: ${titleHeight}px;--content-height:${vizHeight - titleHeight}px;`}
	>
		{#if !!$expandedBlock.id && !compact}
			<div
				class={classNames('dim', `${$expandedBlock.id || ''}`)}
				transition:fade={{ duration: 100 }}
			></div>
			<div
				class={classNames('dim-partial left', `${$expandedBlock.id || ''}`)}
				transition:fade={{ duration: 100 }}
			></div>
			<div
				class={classNames('dim-partial right', `${$expandedBlock.id || ''}`)}
				transition:fade={{ duration: 100 }}
			></div>
		{/if}
		<div class="sankey opacity-1" class:attention={$expandedBlock.id === 'attention'}>
			<Sankey />
		</div>
		<div class="nodes resize-watch">
			<div class="steps" class:expanded={!!$expandedBlock.id} bind:offsetHeight={vizHeight}>
				<svelte:component this={customEmbedding ?? Embedding} className="step" />
				<div class="blocks relative">
					<div class="block-steps main">
						<svelte:component this={customQKV ?? QKV} className="step" />
						<svelte:component this={customAttention ?? Attention} className="step" />
						<svelte:component this={customFeedForward ?? Mlp} className="step" />
					</div>
				</div>
				<SubsequentBlocks className="step" />
				<svelte:component this={customOutput ?? LinearSoftmax} className="step" />
			</div>
			<WeightPopovers />
			<BlockTransition />
			{#if !$moeModel?.architecture || $moeModel?.architecture === 'moe'}<Textbook
					showTextCard={$isTextbookOpen}
				/>{/if}
		</div>
	</div>
{/if}

<style lang="scss">
	.trace-status {
		padding: 8rem 3rem;
		color: #9ca3af;
	}
	.main-section {
		opacity: 0;
		&.active {
			opacity: 1;
		}
	}
	.nodes {
		height: 100%;
		width: 100%;
		padding: 2rem 0 3rem 0;
		position: relative;
	}
	.steps {
		position: relative;
		width: 100%;
		height: 100%;
		position: relative;
		display: grid;
		grid-template-columns: auto 5.8fr 0.6fr 0.85fr;

		&.expanded {
			:global(.step > .title) {
				padding-bottom: 3rem;
			}
		}

		.blocks {
			position: relative;
			width: 100%;
			height: 100%;

			.block-steps {
				height: 100%;
				width: 100%;
				position: absolute;
				display: grid;
				grid-template-columns: 0.5fr 2fr 1.8fr;
			}
			.block-steps.main {
				top: 0;
				left: 0;
			}
		}
	}

	/* Allocate attention's intrinsic width, then let the expert network shrink.
	   The absolute inner grid cannot contribute its minimum to the outer grid. */
	.expert-layout .steps .blocks .block-steps {
		grid-template-columns: minmax(40px, 0.35fr) auto minmax(0, 1.8fr);
	}
	.expert-layout.attention-open .steps {
		grid-template-columns: auto minmax(0, 1fr) 72px 155px;
	}
	.expert-layout.attention-open :global(.attention .heads) {
		padding-left: 2rem;
		padding-right: 4rem;
	}
	.expert-layout.attention-open .dim-partial.attention {
		display: none;
	}
	.expert-layout.attention-open :global(.attention .head-out) {
		margin-left: 1rem;
		margin-right: 1rem;
	}
	.expert-layout.attention-open :global(.transformer-blocks .content) {
		grid-template-columns: repeat(4, minmax(0, 1fr));
	}
	.expert-layout.attention-open :global(.transformer-blocks .guide) {
		width: 64px;
		left: -12px;
		white-space: normal;
		overflow-wrap: anywhere;
	}
	.expert-layout.attention-open :global(.transformer-blocks .guide .text) {
		font-size: 0.65rem;
		padding-left: 0;
	}

	.expert-layout.attention-open :global(.step.softmax > .title) {
		font-size: 0.8rem;
	}
	.expert-layout :global(.mlp.moe) {
		min-width: 0;
	}

	.seq2seq {
		--t5-card-height: clamp(320px, calc(100dvh - 310px), 480px);
	}
	.seq2seq .steps {
		grid-template-columns: clamp(105px, 6vw, 125px) minmax(0, 1fr) 40px 165px;
	}
	.seq2seq .steps .blocks .block-steps {
		grid-template-columns: 90px minmax(0, 1.5fr) minmax(230px, 1fr);
	}
	.seq2seq.decoder .steps .blocks .block-steps {
		grid-template-columns: 90px minmax(0, 1fr) minmax(0, 1.05fr) clamp(220px, 19vw, 320px);
	}

	.seq2seq.attention-open .steps .blocks .block-steps {
		grid-template-columns: 90px minmax(0, 1fr) clamp(220px, 24vw, 380px);
	}
	.seq2seq.decoder.attention-open .steps .blocks .block-steps {
		grid-template-columns: 90px minmax(0, 2.4fr) minmax(145px, 0.8fr) clamp(200px, 17vw, 290px);
	}
	.seq2seq.decoder.cross-open .steps .blocks .block-steps {
		grid-template-columns: 90px minmax(145px, 0.8fr) minmax(0, 2.4fr) clamp(200px, 17vw, 290px);
	}
	.seq2seq :global(.step > .title),
	.seq2seq .steps.expanded :global(.step > .title) {
		padding-bottom: 8px;
	}
	.seq2seq.attention-open :global(.cross-qkv),
	.seq2seq.attention-open :global(.cross-input),
	.seq2seq.attention-open :global(.cross-residual),
	.seq2seq.attention-open :global(.cross-head-out),
	.seq2seq.cross-open :global(.cross-input),
	.seq2seq.cross-open :global(.cross-residual),
	.seq2seq.cross-open :global(.t5-self-attention .head-block > .qkv),
	.seq2seq.cross-open :global(.t5-self-attention .head-out) {
		display: none;
	}
	.seq2seq.attention-open :global(.cross-flow > span),
	.seq2seq.cross-open :global(.t5-self-attention .head-flow-wrap) {
		visibility: hidden;
	}
	.seq2seq.attention-open :global(.cross-head-block) {
		padding: 24px 10px 48px;
	}
	.seq2seq.attention-open :global(.cross-matrix) {
		margin: 0;
	}
	.seq2seq.cross-open :global(.t5-self-attention .head-block) {
		padding: 24px 10px 48px;
	}
	.seq2seq.attention-open :global(.mlp .title small),
	.seq2seq.cross-open :global(.mlp .title small) {
		display: none;
	}
	.seq2seq.attention-open :global(.mlp .label),
	.seq2seq.cross-open :global(.mlp .label) {
		display: none;
	}
	.seq2seq.attention-open :global(.mlp .subtitle),
	.seq2seq.cross-open :global(.mlp .subtitle) {
		font-size: 10px;
	}
	.seq2seq :global(.step > .title) {
		font-size: 14px;
		white-space: normal;
		padding-bottom: 8px;
	}
	.seq2seq .nodes {
		padding-top: 14px;
		padding-bottom: 16px;
	}
	.seq2seq :global(.embedding:not(.expanded) .content) {
		padding-left: 6px;
	}
	.seq2seq :global(.embedding:not(.expanded) .token-string) {
		width: calc(clamp(105px, 6vw, 125px) - 18px);
		padding: 0 10px 0 4px;
	}
	.seq2seq :global(.embedding:not(.expanded) .token-string .label) {
		font-size: 12px;
		line-height: 1;
		height: auto;
		overflow: visible;
	}
	.seq2seq :global(.embedding:not(.expanded) .title) {
		padding-left: 20px;
	}
	.seq2seq :global(.softmax:not(.expanded) .content) {
		padding-right: 16px;
	}
	.seq2seq :global(.attention-bounding) {
		display: none;
	}
	.seq2seq :global(.attention > .title > div) {
		width: auto;
		white-space: normal;
	}
	.seq2seq.attention-open :global(.mlp .residual-text),
	.seq2seq.cross-open :global(.mlp .residual-text) {
		visibility: hidden;
	}
	.seq2seq :global(.step) {
		min-width: 0;
	}
	.seq2seq :global(.prob-dim) {
		bottom: 0;
	}
	.hybrid-model .steps {
		grid-template-columns: auto 6.5fr 0.5fr 0.9fr;
	}
	.hybrid-model .steps .blocks .block-steps {
		grid-template-columns: 0.4fr 2fr 2.1fr;
	}
	.hybrid-model.mamba-only .steps .blocks .block-steps {
		grid-template-columns: 0.4fr 3fr 1.2fr;
	}
	/* A fixed page budget: the active attention card borrows width from the MLP. */
	.squeeze {
		--compact-card-height: clamp(230px, calc(100dvh - 300px), 460px);
	}
	.squeeze :global(.sankey-top),
	.squeeze :global(.sankey-back) {
		pointer-events: none;
	}
	.squeeze :global(.sankey-top path),
	.squeeze :global(.sankey-back path) {
		pointer-events: visiblePainted;
	}
	.squeeze :global(.hybrid-computation) {
		position: relative;
		z-index: 40;
	}
	.squeeze .blocks {
		min-height: 0;
	}
	.squeeze.vision-language {
		--compact-card-height: clamp(200px, calc(100dvh - 410px), 410px);
	}
	.squeeze.hybrid-model {
		--compact-card-height: clamp(230px, calc(100dvh - 360px), 430px);
	}
	.squeeze .nodes {
		padding: 8px 0 16px;
	}
	.squeeze .steps {
		grid-template-columns: 120px minmax(0, 1fr) 35px 210px;
	}
	.squeeze .steps .blocks .block-steps {
		grid-template-columns: 70px minmax(0, 1.5fr) minmax(235px, 1fr);
	}
	.squeeze.attention-open .steps .blocks .block-steps {
		grid-template-columns: 70px minmax(0, 1fr) 210px;
	}
	.squeeze.mamba-only .steps .blocks .block-steps {
		grid-template-columns: 60px minmax(0, 1fr) 230px;
	}
	.squeeze :global(.step) {
		min-width: 0;
	}
	.squeeze :global(.step > .title),
	.squeeze .steps.expanded :global(.step > .title) {
		font-size: 13px;
		white-space: normal;
		padding-bottom: 8px;
	}
	.squeeze :global(.attention > .title > div) {
		width: auto;
		white-space: normal;
	}
	.squeeze :global(.attention .content) {
		display: block;
		width: 100%;
	}
	.squeeze :global(.attention .heads) {
		padding: 0 14px !important;
		min-width: 0;
		width: 100%;
	}
	.squeeze :global(.attention-bounding) {
		display: none;
	}
	.squeeze :global(.attention .head-block) {
		height: var(--compact-card-height) !important;
		min-height: 0 !important;
	}
	.squeeze :global(.attention .head-block > .qkv) {
		padding: 24px 0 24px 70px !important;
		gap: 24px;
		justify-content: space-between;
		flex-shrink: 0;
	}
	.squeeze :global(.attention.rotary .head-block > .qkv) {
		padding-left: 155px !important;
	}
	.squeeze :global(.attention .column .title) {
		font-size: 13px;
		top: -20px;
		line-height: 13px;
	}
	.squeeze :global(.attention .column .title small) {
		font-size: 8px;
	}
	.squeeze :global(.attention .column .label) {
		font-size: 9px !important;
		height: auto !important;
		line-height: 10px !important;
		padding: 0 !important;
	}
	.squeeze :global(.attention .head-block > .qkv .column) {
		gap: 9px;
	}
	.squeeze :global(.attention .head-block > .qkv .label) {
		top: -4px;
	}
	.squeeze :global(.attention .head-out) {
		margin: 0 6px !important;
		flex-shrink: 0;
	}
	.squeeze :global(.attention .head-block > .attention-matrix) {
		flex: 1;
		min-width: 0;
		margin-left: 16px;
	}
	.squeeze :global(.head-title) {
		font-size: 12px;
	}
	.squeeze :global(.embedding:not(.expanded) .content) {
		padding-left: 6px;
	}
	.squeeze :global(.embedding:not(.expanded) .token-string) {
		width: 102px;
		padding-right: 8px;
	}
	.squeeze :global(.embedding .label) {
		font-size: 11px;
		height: auto;
		line-height: 1;
	}
	.squeeze :global(.softmax:not(.expanded) .content) {
		padding-right: 12px;
		grid-template-columns: 70px minmax(0, 1fr);
	}
	.squeeze :global(.softmax:not(.expanded) .token-string) {
		width: 70px;
	}
	.squeeze :global(.softmax:not(.expanded) .probability-col) {
		min-width: 0;
	}
	.squeeze :global(.softmax:not(.expanded) .text-box span) {
		max-width: 100%;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.squeeze :global(.prob-dim) {
		bottom: 0;
	}
	.squeeze.attention-open :global(.mlp .label),
	.squeeze.attention-open :global(.mlp .title small),
	.squeeze.attention-open :global(.mlp .residual-text) {
		visibility: hidden;
	}
	.squeeze :global(.hybrid-computation) {
		min-width: 0;
	}
	.squeeze :global(.hybrid-output .content) {
		gap: 6px;
	}
	.squeeze :global(.mamba-step .content) {
		padding: 0 22px;
	}
	.squeeze :global(.mamba-core) {
		min-width: 0;
		padding: 12px 16px;
	}
	.squeeze :global(.compact .mamba-caption),
	.squeeze :global(.compact .state-formula),
	.squeeze :global(.compact p),
	.squeeze :global(.compact .scan-head span) {
		display: none;
	}
	.squeeze :global(.compact .mamba-heading) {
		flex-wrap: wrap;
	}
	.squeeze :global(.compact .token-row) {
		min-height: 20px;
	}
	.squeeze :global(.compact .scan-head) {
		margin: 8px 0;
	}
	.squeeze :global(.compact .operation-labels) {
		font-size: 7px;
	}

	.squeeze :global(.mlp .label) {
		display: none;
	}
	.squeeze :global(.softmax .text-box) {
		font-size: 11px;
	}
	@media (max-height: 650px) {
		.squeeze :global(.attention .head-block > .qkv .column) {
			gap: 5px;
		}
		.squeeze
			:global(.attention .head-block > .qkv .cell:not(:first-of-type):not(:last-of-type) .label) {
			visibility: hidden;
		}
		.squeeze :global(.vit-patch-controls) {
			gap: 4px;
			padding-top: 4px;
		}
		.squeeze :global(.vit-patch-controls .image-grid) {
			width: 42px;
			height: 42px;
		}
		.squeeze :global(.vit-patch-controls .patch-actions) {
			gap: 3px;
		}
		.squeeze :global(.vit-patch-controls .selection) {
			gap: 3px;
		}
		.squeeze :global(.vit-patch-controls .selection button) {
			padding: 2px;
		}
	}
	.squeeze :global(.hybrid-toolbar) {
		padding-left: 12px;
		padding-right: 12px;
	}
	.squeeze :global(.ffn-chain) {
		padding: 8px;
		margin-bottom: 8px;
	}
	.squeeze :global(.ffn-projections) {
		gap: 3px;
	}
	.squeeze :global(.compact .token-row) {
		margin-bottom: 3px;
		min-height: 12px;
		height: 12px;
	}
	.squeeze :global(.compact .token-row .vector),
	.squeeze :global(.compact .state-vector) {
		min-height: 12px;
	}
	.squeeze :global(.compact .operation-labels) {
		display: none;
	}
	.squeeze :global(.compact .mamba-heading) {
		gap: 3px;
		font-size: 9px;
	}
	.squeeze :global(.compact .mamba-heading button) {
		padding: 3px;
		font-size: 8px;
	}
	.squeeze :global(.compact .scan-head) {
		font-size: 8px;
		margin: 5px 0;
	}
	.squeeze :global(.compact .scan-head select) {
		padding: 2px;
		font-size: 9px;
	}
	.squeeze :global(.compact .state-vector) {
		width: 20px;
	}
	.squeeze :global(.compact .state-vector small) {
		display: none;
	}
	:global(.step) {
		height: 100%;
		display: grid;
		grid-template-rows: var(--title-height) 1fr;
	}

	:global(.step > .title) {
		z-index: $COLUMN_TITLE_INDEX;
		display: flex;
		flex-direction: column;
		justify-content: end;
		grid-row: 1;
		color: theme('colors.gray.400');
		white-space: nowrap;
		padding-bottom: 2rem;
		overflow: visible;
		min-width: 0;
		transition: all 0.5s;
		cursor: default;

		&:hover {
			color: theme('colors.gray.600');
		}
	}

	:global(.step > .title.expandable) {
		cursor: pointer;
	}

	:global(.step .content) {
		grid-row: 2;
		height: fit-content;
	}

	:global(.column) {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		position: relative;

		:global(.cell) {
			height: var(--vector-height);
			display: flex;
			gap: 1rem;
			align-items: center;
			position: relative;
		}

		:global(.subtitle) {
			position: absolute;
			top: 0;
			transform: translateY(calc(-100% - 1rem));
			text-align: center;
			font-size: 0.8rem;
			color: theme('colors.gray.400');
			width: 100%;
			z-index: $COLUMN_TITLE_INDEX;
		}
	}

	:global(.vector),
	:global(.sub-vector) {
		position: relative;
		z-index: $VECTOR_INDEX;
		width: 12px;
		height: var(--vector-height);
		flex-shrink: 0;
		justify-content: start;
	}
	:global(.cell.x1-12),
	:global(.vector.x1-12),
	:global(.sub-vector.x1-12) {
		height: calc(var(--vector-height) / 16);
	}

	:global(.cell.x3),
	:global(.vector.x3),
	:global(.sub-vector.x3) {
		height: calc(var(--vector-height) * 3);
	}
	:global(.cell.x4),
	:global(.vector.x4),
	:global(.sub-vector.x4) {
		height: calc(var(--vector-height) * 3.1);
	}

	:global(.vector.vocab),
	:global(.sub-vector.vocab) {
		height: 100%;
		width: 0;
	}

	:global(.sub-vector.head-rest) {
		flex: 1 0 0;
	}

	:global(.label) {
		font-size: 0.9rem;
		color: theme('colors.gray.700');
		z-index: $VECTOR_INDEX;
		display: inline;
		max-width: 7rem;
		overflow: hidden;
		text-overflow: ellipsis;
		text-align: right;
		line-height: var(--vector-height);
		height: var(--vector-height);
		flex-shrink: 0;
	}
	:global(.label.float) {
		position: absolute;
		left: -0.8rem;
		transform: translateX(-100%);
	}
	:global(.label.float-right) {
		position: absolute;
		left: -0.8rem;
	}

	:global(.ellipsis) {
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	:global(.bounding) {
		position: absolute;
		box-sizing: content-box;
		top: -0.5rem;
		padding: 0.5rem 0;
		left: 0;
		height: 100%;
		border: 2px dashed theme('colors.gray.300');
		border-radius: 0.5rem;
		transition: opacity 0.5s;
		opacity: 0;
		pointer-events: none;
	}
	:global(.bounding.active) {
		opacity: 0.8;
	}

	:global(.popover) {
		z-index: $POPOVER_INDEX;
		width: max-content;
	}

	:global(.tooltip) {
		z-index: $TOOLTIP_INDEX;
		background-color: white !important;
		color: theme('colors.gray.600') !important;
		border: 1px solid theme('colors.gray.200') !important;
		padding: 0.2rem 0.5rem !important;
		font-size: 0.8rem !important;
		white-space: nowrap;
		font-weight: 300 !important;
		border-color: theme('colors.gray.200') !important;
	}
	.dim {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		z-index: $DIM_INDEX;
		background-color: white;
		opacity: 0.7;
		user-select: none;

		&.attention {
			z-index: 0;
		}
	}
	.dim-partial {
		user-select: none;
		z-index: $PARTIAL_DIM_INDEX;
		position: absolute;
		top: 0;
		height: 100%;

		&.right {
			right: 0;
			background: linear-gradient(90deg, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 1) 80%);
		}
		&.left {
			left: 0;
			background: linear-gradient(-90deg, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 1) 80%);
		}

		&.embedding {
			&.left {
				display: none;
			}
			&.right {
				width: 60%;
			}
		}
		&.attention {
			&.left {
				width: 20%;
			}
			&.right {
				width: 20%;
			}
		}
		&.softmax {
			&.left {
				width: 60%;
			}
			&.right {
				display: none;
			}
		}
	}
	.sankey {
		position: absolute;
		left: 0;
		top: 0;
		width: 100%;
		height: 100%;

		&.attention {
			:global(.sankey-top) {
				z-index: $EXPANDED_ATTENTION_INDEX !important;
				pointer-events: none;
			}
			// :global(.sankey-top > g) {
			// 	opacity: 0.3;
			// }
			// :global(.sankey-top > g.attention) {
			// 	opacity: 1;
			// }
		}
	}

	:global(svg g.path-group) {
		transition: opacity 0.5s;
	}
	:global(div.step > div) {
		transition: opacity 0.5s;
	}
	:global(div.step .column) {
		transition: opacity 0.5s;
	}
</style>
