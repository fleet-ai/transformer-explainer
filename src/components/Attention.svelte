<script lang="ts">
	import HeadFlow from './moe/HeadFlow.svelte';
	import PlainHeadFlow from './vit/PlainHeadFlow.svelte';
	import { hybridHeadOperations, hybridInspector } from '$lib/hybrid/store';
	import { vitHeadOperations } from '$lib/vit/store';
	import { vlmHeadOperations, vlmInspectorStage } from '$lib/vlm/store';
	import LatentHeadFlow from './mla/LatentHeadFlow.svelte';
	import LatentInspector from './mla/LatentInspector.svelte';
	import RoPEInspector from './moe/RoPEInspector.svelte';
	import VectorCanvas from './common/VectorCanvas.svelte';
	import HeadStack from '~/components/HeadStack.svelte';
	import {
		tokens,
		modelMeta,
		headContentHeight,
		expandedBlock,
		headGap,
		hoveredMatrixCell,
		attentionHeadIdx,
		moeExample,
		moeModel,
		moeAttentionOperations,
		blockIdx
	} from '~/store';
	import classNames from 'classnames';
	import AttentionMatrix from '~/components/AttentionMatrix.svelte';

	import { setContext, getContext } from 'svelte';
	import { Tooltip } from 'flowbite-svelte';
	import { onClickReadMore } from '~/utils/event';
	import TextbookTooltip from './common/TextbookTooltip.svelte';

	export let className: string | undefined = undefined;

	setContext('block-id', 'attention');
	const blockId = getContext('block-id');
	$: isAttentionExpanded = $expandedBlock.id === blockId;

	const queryHeadVectorColor = 'bg-blue-400';
	const keyHeadVectorColor = 'bg-red-400';
	const valHeadVectorColor = 'bg-green-400';

	const outputVectorColor = 'bg-purple-500';

	let isHovered = false;
	let showRoPE = false;
	let mlaStage: string | null = null;
	$: isMLA = $moeModel?.architecture === 'mla';
	$: isVit = $moeModel?.architecture === 'vit';
	$: isHybrid = $moeModel?.architecture === 'hybrid';
	$: isVlm = $moeModel?.architecture === 'vision-language';
	$: isPlain =
		['vit', 'encoder-decoder'].includes($moeModel?.architecture ?? '') ||
		(isVlm && $moeModel?.maskMode === 'bidirectional');
	$: valueSize = $moeModel?.valueHeadSize ?? headSize;
	$: kvHead = Math.floor(
		$attentionHeadIdx / (($moeModel?.attentionHeads ?? 16) / ($moeModel?.keyValueHeads ?? 8))
	);
	$: headSize = $moeModel?.attentionHeadSize ?? 64;
	$: layer = $moeExample?.layers[$blockIdx];
	$: operations = isVit
		? $vitHeadOperations
		: isVlm
			? $vlmHeadOperations
			: isHybrid
				? $hybridHeadOperations
				: $moeAttentionOperations?.example.id === $moeExample?.id
					? $moeAttentionOperations?.layers[$blockIdx]
					: undefined;

	function handleMouseEnter() {
		isHovered = true;
	}

	function handleMouseLeave() {
		isHovered = false;
	}
</script>

<div
	class={classNames('attention', className, {
		expanded: isAttentionExpanded
	})}
	data-click="attention-step"
>
	<div
		class="title"
		on:mouseenter={handleMouseEnter}
		on:mouseleave={handleMouseLeave}
		role="group"
		data-click="attention-step-title"
	>
		<div class="w-max">
			{#if isMLA}Multi-head Latent Attention{:else if isPlain}{$moeModel?.maskMode ===
				'bidirectional'
					? 'Bidirectional'
					: 'Causal'} Self Attention{:else}<TextbookTooltip id="self-attention"
					>Multi-head Self Attention</TextbookTooltip
				>{/if}
		</div>
	</div>
	<div class="content relative">
		<div
			class="bounding attention-bounding"
			class:active={isHovered && !isAttentionExpanded}
			style={`padding-bottom:${$modelMeta.attention_head_num * headGap.y}px`}
		></div>
		<div class="heads">
			<HeadStack>
				<div class="head-flow-wrap">
					{#if isPlain}<PlainHeadFlow />{:else if isMLA}<LatentHeadFlow
							on:inspect={(event) => (mlaStage = event.detail)}
							on:rope={() => (showRoPE = true)}
						/>{:else}<HeadFlow
							queryHeads={$moeModel?.attentionHeads ?? 16}
							keyValueHeads={$moeModel?.keyValueHeads ?? 8}
							{headSize}
							on:inspect={() => {
								if (isVlm) vlmInspectorStage.set('rope');
								else if (isHybrid) hybridInspector.set('rope');
								else showRoPE = true;
							}}
						/>{/if}
				</div>
				<div
					class="head-block flex w-full items-center justify-between px-2"
					style={`height:${Math.max($headContentHeight - 110, 320)}px;min-height:calc(var(--vector-height) * ${($tokens.length * 3) / 16} + ${($tokens.length - 1) * 1.5 + 14}rem);`}
				>
					<div class="qkv flex h-full flex-col justify-center gap-[5rem] pl-[6rem]">
						<div class="column key">
							<div class="head1 title">
								<TextbookTooltip id="qkv">Key {kvHead + 1}</TextbookTooltip><small
									>{isPlain ? 'Projected' : isMLA ? 'Content + rotary' : 'RoPE'} · {headSize}D</small
								>
							</div>

							{#each $tokens as token, index}
								<div
									class="head1 key cell x1-12 text-xs"
									class:last={index === $tokens.length - 1}
									class:active={$hoveredMatrixCell.col === index}
								>
									<span class="label float">{token}</span>
									<div class={`vector x1-12 ${keyHeadVectorColor}`}>
										<VectorCanvas
											colorScale="red"
											data={layer?.keyPreview?.[$attentionHeadIdx]?.[index]}
										/>
									</div>
								</div>
							{/each}
							<Tooltip class="popover" triggeredBy={'.step.attention .key .cell'} placement="right"
								>Key, {isPlain ? 'head' : isMLA ? 'content head' : 'shared KV head'}
								{kvHead + 1}, {isMLA
									? '128 content + 64 rotary coordinates'
									: `${isPlain ? 'no RoPE' : 'after RoPE'}, vector(${headSize})`}</Tooltip
							>
						</div>
						<div class="column query">
							<div class="head1 title">
								<TextbookTooltip id="qkv">Query {$attentionHeadIdx + 1}</TextbookTooltip><small
									>{isPlain ? 'Projected' : isMLA ? 'Content + rotary' : 'RoPE'} · {headSize}D</small
								>
							</div>
							{#each $tokens as token, index}
								<div
									class="head1 cell x1-12 query text-xs"
									class:last={index === $tokens.length - 1}
									class:active={$hoveredMatrixCell.row === index}
								>
									<span class="label float">{token}</span>
									<div class={`vector x1-12 ${queryHeadVectorColor}`}>
										<VectorCanvas
											colorScale="blue"
											data={layer?.queryPreview?.[$attentionHeadIdx]?.[index]}
										/>
									</div>
								</div>
							{/each}
							<Tooltip
								class="popover"
								triggeredBy={'.step.attention .query .cell'}
								placement="right"
								>Query, Head {$attentionHeadIdx + 1}, {isMLA
									? '128 content + 64 rotary coordinates'
									: `${isPlain ? 'no RoPE' : 'after RoPE'}, vector(${headSize})`}</Tooltip
							>
						</div>
						<div class="column value">
							<div class="head1 title">
								<TextbookTooltip id="qkv">Value {kvHead + 1}</TextbookTooltip><small
									>{isMLA ? 'Expanded latent' : 'unchanged'} · {valueSize}D</small
								>
							</div>
							{#each $tokens as token, index}
								<div class="head1 cell x1-12 text-xs" class:last={index === $tokens.length - 1}>
									<span class="label float">{token}</span>
									<div class={`vector x1-12 ${valHeadVectorColor}`}>
										<VectorCanvas
											colorScale="green"
											data={layer?.valuePreview?.[$attentionHeadIdx]?.[index]}
										/>
									</div>
								</div>
							{/each}
							<Tooltip
								class="popover"
								triggeredBy={'.step.attention .value .cell'}
								placement="right"
								>Value, {isPlain ? 'head' : isMLA ? 'expanded latent head' : 'shared KV head'}
								{kvHead + 1}, no RoPE, vector({valueSize})</Tooltip
							>
						</div>
					</div>
					<div class="resize-watch attention-matrix flex">
						<AttentionMatrix />
					</div>
					<div class="head-out mx-[2rem]">
						<div class="column out">
							<div class="head1 title">
								<TextbookTooltip id="output-concatenation">Out</TextbookTooltip>
							</div>
							{#each $tokens as token, index}
								<div class="head1 cell x1-12" class:last={index === $tokens.length - 1}>
									<div class={`vector x1-12 ${outputVectorColor}`}>
										<VectorCanvas
											colorScale="purple"
											data={operations?.attentionContext?.[$attentionHeadIdx]?.[index]}
										/>
									</div>
								</div>
							{/each}
							<Tooltip class="popover" triggeredBy={'.step.attention .out .cell'} placement="right"
								>Attention Out, Head {$attentionHeadIdx + 1}, vector({valueSize})</Tooltip
							>
						</div>
					</div>
				</div>
			</HeadStack>
		</div>
	</div>
</div>
{#if showRoPE}<RoPEInspector
		on:close={() => (showRoPE = false)}
		on:attention={() => {
			showRoPE = false;
			$expandedBlock = { id: 'attention' };
		}}
	/>{/if}

{#if mlaStage}<LatentInspector
		stage={mlaStage}
		on:close={() => (mlaStage = null)}
		on:rope={() => {
			mlaStage = null;
			showRoPE = true;
		}}
		on:attention={() => {
			mlaStage = null;
			$expandedBlock = { id: 'attention' };
		}}
	/>{/if}

<style lang="scss">
	.head-flow-wrap {
		position: relative;
		z-index: 30;
		padding: 12px 14px 8px;
		max-width: 508px;
	}
	.column .title small {
		display: block;
		font-size: 9px;
		line-height: 1.2;
		white-space: nowrap;
		color: #9ca3af;
	}
	.attention-matrix,
	.head-title {
		z-index: $COLUMN_TITLE_INDEX;
	}
	.attention {
		> .title > div {
			// cursor: help;
		}
		&.expanded {
			.title,
			:global(.head-content) {
				z-index: $EXPANDED_CONTENT_INDEX;
			}
			:global(.multi-head .head-card:first-child) {
				z-index: $EXPANDED_CONTENT_INDEX !important;
			}
		}

		.attention-bounding {
			top: -0.5rem;
			padding: 0.5rem 0;
			left: -0.3rem;
			width: calc(100% + 1rem);
			height: calc(100%);
		}
		.column {
			.label {
				font-size: 0.7rem;
				white-space: nowrap;
				max-width: 6rem;
				overflow: hidden;
				text-overflow: ellipsis;
				color: theme('colors.gray.600');
			}
			.title {
				z-index: $COLUMN_TITLE_INDEX;
				position: absolute;
				top: -2.65rem;
				left: 50%;
				transform: translateX(-50%);
				font-size: 0.9rem;
				white-space: nowrap;
				transition: none;
			}
			&.query .title {
				color: theme('colors.blue.400');
			}
			&.key .title {
				color: theme('colors.red.400');
			}
			&.value .title {
				color: theme('colors.green.400');
			}
			&.out .title {
				color: theme('colors.purple.400');
			}
		}
		.content {
			display: grid;
			grid-template-columns: auto 0;

			.tokens {
				gap: 0.6rem;
			}
		}
		.heads {
			padding: 0 7rem 0 8rem;

			.head1.cell {
				.label {
					height: auto;
					line-height: 1;
				}
				&.active {
					&.query {
						.label {
							background-color: theme('colors.blue.100');
							color: theme('colors.blue.700');
							font-size: 1rem;
							z-index: 100;
							padding: 0.2rem;
						}
					}
					&.key {
						.label {
							background-color: theme('colors.red.100');
							color: theme('colors.red.700');
							font-size: 1rem;
							z-index: 100;
							padding: 0.2rem;
						}
					}
				}
			}
		}
	}
</style>
