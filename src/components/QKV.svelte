<script lang="ts">
	import {
		tokens,
		modelMeta,
		attentionHeadIdx,
		blockIdx,
		moeExample,
		moeModel,
		moeAttentionOperations
	} from '~/store';
	import classNames from 'classnames';
	import { onDestroy } from 'svelte';
	import type { AttentionOperationsLayer } from '$lib/moe/attention';
	import { vlmHeadOperations } from '$lib/vlm/store';
	import { hybridHeadOperations } from '$lib/hybrid/store';
	import { vitHeadOperations } from '$lib/vit/store';
	import { loadAttentionOperations } from '$lib/moe/attention';
	import VectorCanvas from './common/VectorCanvas.svelte';
	import OperationGroup from './OperationGroup.svelte';
	import { Tooltip } from 'flowbite-svelte';

	export let className: string | undefined = undefined;

	const embeddingVectorColor = 'bg-gray-300';

	let vectorHoverIdx: number | null = null;
	const queryVectorColor = 'bg-blue-300';
	const keyVectorColor = 'bg-red-300';
	const valVectorColor = 'bg-green-300';

	const queryHeadVectorColor = 'bg-blue-500';
	const keyHeadVectorColor = 'bg-red-500';
	const valHeadVectorColor = 'bg-green-500';

	$: recording = $moeAttentionOperations;
	let request = 0;
	$: queryHeads = $moeModel?.attentionHeads ?? 16;
	$: keyValueHeads = $moeModel?.keyValueHeads ?? 8;
	$: headSize = $moeModel?.attentionHeadSize ?? 64;
	$: isMLA = $moeModel?.architecture === 'mla';
	$: isPlain = ['vit', 'encoder-decoder'].includes($moeModel?.architecture ?? '');
	$: valueSize = $moeModel?.valueHeadSize ?? headSize;
	$: kvHead = Math.floor($attentionHeadIdx / (queryHeads / keyValueHeads));
	$: layer = (
		$moeModel?.architecture === 'vit'
			? $vitHeadOperations
			: $moeModel?.architecture === 'vision-language'
				? $vlmHeadOperations
				: $moeModel?.architecture === 'hybrid'
					? $hybridHeadOperations
					: recording?.example.id === $moeExample?.id
						? recording?.layers[$blockIdx]
						: undefined
	) as AttentionOperationsLayer | undefined;
	$: if (
		$moeExample &&
		!isPlain &&
		!['vision-language', 'hybrid'].includes($moeModel?.architecture ?? '')
	)
		load($moeExample.id);
	async function load(id: string) {
		const current = ++request;
		try {
			const data = await loadAttentionOperations(id);
			if (current === request) $moeAttentionOperations = data;
		} catch {
			// The inspector exposes a retry; never replace missing recorded values with generated ones.
			if (current === request) $moeAttentionOperations = null;
		}
	}
	onDestroy(() => request++);
</script>

<div
	class={classNames('qkv', className)}
	class:latent={isMLA}
	class:plain={isPlain}
	role="none"
	data-click="qkv-step"
	style={`--q-ratio:${(3 * queryHeads * headSize) / (queryHeads * headSize + keyValueHeads * (headSize + valueSize))};--k-ratio:${(3 * keyValueHeads * headSize) / (queryHeads * headSize + keyValueHeads * (headSize + valueSize))};--v-ratio:${(3 * keyValueHeads * valueSize) / (queryHeads * headSize + keyValueHeads * (headSize + valueSize))};`}
>
	<div class="content relative">
		<div
			class="vector-column block-start-column relative flex"
			class:initial-column={$blockIdx === 0}
		>
			<div class="column vectors embedding-column">
				{#each $tokens as token, index}
					<div
						class={`vector ${$blockIdx !== 0 ? 'bg-blue-200' : embeddingVectorColor}`}
						class:last={index === $tokens.length - 1}
					>
						<VectorCanvas
							colorScale={$blockIdx !== 0 ? 'blue' : 'gray'}
							data={$moeExample?.layers[$blockIdx]?.residualInputPreview?.[index]}
						/>
					</div>
				{/each}
			</div>
			<Tooltip class="popover" triggeredBy={'.qkv .embedding-column .vector'} placement="right"
				>vector({$modelMeta.dimension})</Tooltip
			>

			<div class="operations flex">
				<OperationGroup type="dropout" id={'embedding-dropout'} />
				<OperationGroup type="residual-start" id={'embedding-residual'} />
				<OperationGroup type="ln" id={'embedding-ln'} />
			</div>
		</div>
		<div class="column qkv-column">
			{#each $tokens as token, index}
				<div
					class="qkv-weighted vector x3 flex flex-col"
					class:last={index === $tokens.length - 1}
					on:mouseenter={() => {
						vectorHoverIdx = index;
					}}
					on:mouseleave={() => {
						vectorHoverIdx = null;
					}}
					role="group"
				>
					<div class={`sub-vector query relative flex flex-col ${queryVectorColor}`}>
						<VectorCanvas
							colorScale="blue"
							data={(isMLA ? layer?.queryProjection : layer?.queryBefore)?.flatMap(
								(head) => head[index]
							)}
							label={layer
								? `Recorded QUERY projection before head split, ${queryHeads * headSize} coordinates. Colors normalized within this vector.`
								: 'Loading recorded projection values.'}
							active={vectorHoverIdx === index}
						/>
						<div
							class={`sub-vector x1-12 head1 ${queryHeadVectorColor} absolute`}
							style={`top:${(100 * $attentionHeadIdx) / queryHeads}%;height:${100 / queryHeads}%;`}
						></div>
						<div class="sub-vector head-rest">
							{#if vectorHoverIdx !== index}<span>Q</span>{/if}
						</div>
					</div>
					<div class={`sub-vector key relative flex flex-col ${keyVectorColor}`}>
						<VectorCanvas
							colorScale="red"
							data={isMLA
								? layer?.keyContent?.flatMap((head) => [
										...head[index],
										...(layer?.rawKeyRotary?.[0]?.[index] ?? [])
									])
								: layer?.keyBefore.flatMap((head) => head[index])}
							label={layer
								? `Recorded ${isMLA ? 'expanded content KEY plus shared rotary key' : 'KEY projection before head split'}, ${keyValueHeads * headSize} coordinates. Colors normalized within this vector.`
								: 'Loading recorded projection values.'}
							active={vectorHoverIdx === index}
						/>
						<div
							class={`sub-vector x1-12 head1 ${keyHeadVectorColor} absolute`}
							style={`top:${(100 * kvHead) / keyValueHeads}%;height:${100 / keyValueHeads}%;`}
						></div>
						<div class="sub-vector head-rest">
							{#if vectorHoverIdx !== index}<span>K</span>{/if}
						</div>
					</div>
					<div class={`sub-vector value relative flex flex-col ${valVectorColor}`}>
						<VectorCanvas
							colorScale="green"
							data={layer?.value.flatMap((head) => head[index])}
							label={layer
								? `Recorded ${isMLA ? 'expanded latent VALUE' : 'VALUE projection before head split'}, ${keyValueHeads * valueSize} coordinates. Colors normalized within this vector.`
								: 'Loading recorded projection values.'}
							active={vectorHoverIdx === index}
						/>
						<div
							class={`sub-vector x1-12 head1 ${valHeadVectorColor} absolute`}
							style={`top:${(100 * kvHead) / keyValueHeads}%;height:${100 / keyValueHeads}%;`}
						></div>
						<div class="sub-vector head-rest">
							{#if vectorHoverIdx !== index}<span>V</span>{/if}
						</div>
					</div>
				</div>
			{/each}
			<Tooltip class="popover" triggeredBy={'.qkv .qkv-column .vector'} placement="right"
				>{isMLA
					? 'MLA projections: content and rotary Q/K; expanded latent V.'
					: 'Before head split:'} Q {queryHeads * headSize}, K {keyValueHeads * headSize}, V {keyValueHeads *
					valueSize}. Highlighted head: Q/K {headSize}D, V {valueSize}D.</Tooltip
			>
		</div>
	</div>
</div>

<style lang="scss">
	.qkv.plain .content .qkv-column .sub-vector.query,
	.qkv.plain .content .qkv-column .sub-vector.key,
	.qkv.plain .content .qkv-column .sub-vector.value {
		height: var(--vector-height);
	}
	.qkv.latent .content .qkv-column .sub-vector.query,
	.qkv.latent .content .qkv-column .sub-vector.key {
		height: calc(var(--vector-height) * 1.125);
	}
	.qkv.latent .content .qkv-column .sub-vector.value {
		height: calc(var(--vector-height) * var(--v-ratio));
	}
	.qkv {
		.content {
			display: grid;
			grid-template-columns: 1fr 1fr;

			.vector-column {
				position: relative;
				left: 3rem;

				&.initial-column {
					left: -12px;
				}
			}
			.qkv-column {
				position: relative;
				left: 12px;
				display: flex;
				flex-direction: column;
				align-items: end;

				&.hide {
					pointer-events: none;
					opacity: 0.2;
				}
				.sub-vector {
					user-select: none;
					font-size: 1rem;
					opacity: 0.8;
					span {
						opacity: 0.45;
					}
					&.query {
						height: calc(var(--vector-height) * var(--q-ratio));
						color: theme('colors.blue.600');
					}
					&.key {
						height: calc(var(--vector-height) * var(--k-ratio));
						color: theme('colors.red.600');
					}
					&.value {
						height: calc(var(--vector-height) * var(--v-ratio));
						color: theme('colors.green.600');
					}

					.head-rest {
						height: 100%;
						display: flex;
						justify-content: center;
						align-items: center;
						font-weight: 700;
						text-shadow:
							-1px -1px 0 white,
							1px -1px 0 white,
							-1px 1px 0 white,
							1px 1px 0 white;
					}
				}
			}
		}

		&.animate-forward {
			.vector-column {
				transition-delay: 900ms;
				transition: left 100ms;
			}
		}
	}
</style>
