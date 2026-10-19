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
	$: titleHeight = $moeModel?.architecture === 'encoder-decoder' ? 60 : rootRem * 5;
	const calculateVectorHeight = () => {
		const gaps = rootRem * 0.5 * ($tokens.length - 1);
		const height = Math.min(
			Math.max(
				(vizHeight - titleHeight - gaps) / $tokens.length / maxVectorScale,
				$moeModel?.architecture === 'encoder-decoder' ? 8 : minVectorHeight
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
		class:latent-model={$moeModel?.architecture === 'mla'}
		class:seq2seq={$moeModel?.architecture === 'encoder-decoder'}
		class:decoder={$moeModel?.architecture === 'encoder-decoder' && $moeModel?.stack === 'decoder'}
		style={`--vector-height: ${$vectorHeight}px;--title-height: ${titleHeight}px;--content-height:${vizHeight - titleHeight}px;`}
	>
		{#if !!$expandedBlock.id}
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

	.seq2seq .steps {
		grid-template-columns: 145px minmax(0, 1fr) 60px 160px;
	}
	.seq2seq .steps .blocks .block-steps {
		grid-template-columns: 50px minmax(0, 2.1fr) minmax(170px, 1.2fr);
	}
	.seq2seq.decoder .steps .blocks .block-steps {
		grid-template-columns: 50px minmax(0, 1fr) minmax(0, 1fr) 170px;
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
		width: 125px;
	}
	.seq2seq :global(.embedding:not(.expanded) .title) {
		padding-left: 20px;
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
