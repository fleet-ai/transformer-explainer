<script lang="ts">
	import { onMount, setContext, tick } from 'svelte';
	import { scaleLinear } from 'd3';
	import classNames from 'classnames';
	import {
		tokens,
		modelMeta,
		blockIdx,
		attentionHeadIdx,
		vectorHeight,
		headGap,
		bandRedraw,
		moeExample,
		moeModel,
		moeTokenIndex
	} from '~/store';
	import OperationGroup from './OperationGroup.svelte';
	import VectorCanvas from './common/VectorCanvas.svelte';
	import ExpertSumPopup from './moe/ExpertSumPopup.svelte';
	import ExpertCard from './moe/ExpertCard.svelte';
	import { tokenLabel } from '$lib/moe/format';
	import type { ExpertEntry, TokenTrace } from '$lib/moe/types';

	const CARD_HEIGHT = 330;
	const MAX_GHOSTS = 5;
	const ROUTER_WIDTH = 20;
	const SUM_WIDTH = 36;
	const SUM_BOX = 30;
	const CARD_MARGIN = 8;

	export let className: string | undefined = undefined;
	setContext('block-id', 'mlp');
	let network: HTMLDivElement;
	let routingColumn: HTMLDivElement;
	let networkWidth = 310;
	let rowGap = 8;
	let shownIndex = 0;
	let showSum = false;
	let isHovered = false;
	let showHelp = false;
	$: layer = $moeExample?.layers[$blockIdx];
	$: trace = layer?.tokens[$moeTokenIndex];
	$: isMLA = $moeModel?.architecture === 'mla';
	$: isDense = layer?.feedForwardType === 'dense';
	$: hidden = $moeModel?.hiddenSize ?? 1024;
	$: expertSize = $moeModel?.intermediateSize ?? 512;
	$: topK = $moeModel?.topK ?? 8;
	$: numExperts = $moeModel?.numExperts ?? 32;
	$: sharedExperts = isMLA && !isDense ? ($moeModel?.sharedExperts ?? 0) : 0;
	$: sharedSize = sharedExperts
		? ($moeModel?.sharedIntermediateSize ?? sharedExperts * expertSize)
		: 0;
	$: denseSize = $moeModel?.denseIntermediateSize ?? 10944;
	/** Each of these receives a full copy of the token vector; the card shows one, the rest stack behind it. */
	$: entries = buildEntries(
		trace,
		isDense,
		sharedExperts,
		expertSize,
		sharedSize,
		denseSize,
		numExperts
	);
	$: ghosts = Math.min(MAX_GHOSTS, Math.max(0, entries.length - 1));
	/** The weight slots stack to the box height, in the same order as the card's output taps. */
	$: totalWeight = entries.reduce((sum, entry) => sum + entry.weight, 0) || 1;
	$: slotShares = entries.map((_, order) => {
		const entry = entries[(order + shownIndex) % entries.length];
		return (entry.weight / totalWeight) * SUM_BOX;
	});
	$: if (shownIndex >= 0 && entries) tick().then(() => bandRedraw.update((count) => count + 1));
	$: cardWidth = Math.max(
		110,
		networkWidth - ROUTER_WIDTH - SUM_WIDTH - CARD_MARGIN * 2 - ghosts * headGap.x
	);
	$: rowHeight = $tokens.length * $vectorHeight + Math.max(0, $tokens.length - 1) * rowGap;
	$: networkHeight = Math.max(rowHeight, CARD_HEIGHT + ghosts * headGap.y);
	$: if ($moeExample || $blockIdx || $moeTokenIndex) {
		shownIndex = 0;
		showSum = false;
	}
	$: if (network && ($vectorHeight || $tokens)) tick().then(measure);

	function measure() {
		if (!network) return;
		networkWidth = network.clientWidth;
		if (routingColumn) rowGap = parseFloat(getComputedStyle(routingColumn).gap) || 8;
	}
	onMount(() => {
		const observer = new ResizeObserver(measure);
		observer.observe(network);
		measure();
		return () => observer.disconnect();
	});
	function buildEntries(
		token: TokenTrace | undefined,
		dense: boolean,
		shared: number,
		expertDim: number,
		sharedDim: number,
		denseDim: number,
		experts: number
	): ExpertEntry[] {
		if (dense)
			return [{ id: -1, kind: 'dense', label: 'Dense feed-forward', weight: 1, size: denseDim }];
		const list: ExpertEntry[] = (token?.selectedExperts ?? []).map((expert, rank) => ({
			id: expert,
			kind: 'routed',
			label: `Expert ${expert}`,
			weight: token!.routingWeights[rank],
			size: expertDim
		}));
		if (shared)
			list.push({
				id: experts,
				kind: 'shared',
				label: `${shared} shared experts`,
				weight: 1,
				size: sharedDim
			});
		return list;
	}
	function vectorColor(value: number, values: number[]) {
		const max = Math.max(...values.map(Math.abs), 0.00001);
		return scaleLinear<string>().domain([-max, 0, max]).range(['#c2a8e9', '#f7f1ff', '#9671ca'])(
			value
		);
	}
	async function chooseSum(index: number) {
		chooseToken(index);
		await tick();
		showSum = true;
	}
	function chooseToken(index: number) {
		moeTokenIndex.set(index);
	}
</script>

<svelte:window on:resize={measure} />

<div class={classNames('mlp', 'moe', 'mlpUp', 'mlpDown', className)} data-click="mlp-step">
	<div
		class="title"
		on:mouseenter={() => (isHovered = true)}
		on:mouseleave={() => (isHovered = false)}
		role="group"
		data-click="mlp-step-title"
	>
		<button class="moe-title" aria-expanded={showHelp} on:click={() => (showHelp = !showHelp)}
			>{isDense ? 'Dense Feed-forward' : 'Mixture of Experts'}
			<span aria-hidden="true">ⓘ</span></button
		>
		<span class="moe-subtitle"
			>{isDense
				? 'SwiGLU · every token uses the same MLP'
				: `${$moeModel?.topK ?? 8} of ${$moeModel?.numExperts ?? 32} routed experts${sharedExperts ? ` + ${sharedExperts} shared` : ' per token'}`}</span
		>
	</div>
	<!-- Keep the explanation outside the title's stacking context, above the diagram. -->
	{#if showHelp}
		<section class="moe-help" aria-label="MoE explanation">
			<button class="close" aria-label="Close MoE explanation" on:click={() => (showHelp = false)}
				>×</button
			>
			{#if isDense}This first DeepSeek block uses a dense SwiGLU network. Every token passes through
				the same up and gate projections, elementwise product, and down projection, followed by
				residual addition.
			{:else if isMLA}The router computes softmax over all 64 experts, then selects the top 6
				probabilities without renormalizing them. Every selected expert, and both always-active
				shared experts, receives the whole input vector. Their weighted outputs are added, then
				joined to the residual stream. Expert IDs are local to this layer.
			{:else}The router selects the top {$moeModel?.topK ?? 8} logits, then softmax gives their mixing
				weights. Every selected expert receives the whole input vector, compresses it to {$moeModel?.intermediateSize ??
					512} dimensions, and projects back. Only the outputs are weighted, added, scaled, and joined
				to the residual stream. Expert IDs are local to this layer.{/if}
		</section>
	{/if}

	<div class="content resize-watch relative">
		<div class="bounding mlp-bounding" class:active={isHovered}></div>
		<div class="first-layer layer flex">
			<div class="column initial">
				{#each $tokens as token, index}
					<div class="cell" class:last={index === $tokens.length - 1}>
						<span class="label float">{token}</span>
						<button
							class="vector initial-vector"
							class:chosen={index === $moeTokenIndex}
							aria-label={`Inspect ${isDense ? 'dense feed-forward' : 'MoE routing'} for token ${index}: ${token}`}
							aria-pressed={index === $moeTokenIndex}
							on:click={() => chooseToken(index)}
							title={`Attention output, token ${index}. Select to inspect its expert routes.`}
						>
							{#each layer?.attentionOutputPreview[index] ?? [] as value, dim}<span
									class="coordinate"
									style={`top:${dim * 12.5}%;background:${vectorColor(value, layer!.attentionOutputPreview[index])}`}
									title={`Attention output coordinate ${dim}: ${value}`}
								></span>{/each}
							<div
								class="sub-vector head1"
								style={`top:${($vectorHeight / $modelMeta.attention_head_num) * $attentionHeadIdx}px;height:${$vectorHeight / $modelMeta.attention_head_num}px;`}
							></div>
							<div class="sub-vector head-rest"></div>
						</button>
					</div>
				{/each}
			</div>
			<OperationGroup type="residual-end" id="embedding-residual" />
			<OperationGroup type="residual-start" id="mlp-residual" />
			<div class="column rmsnorm">
				{#each $tokens as _, index}<div class="cell" class:last={index === $tokens.length - 1}>
						<span class="normalization-dot" title="RMS normalization before routing"></span>
					</div>{/each}
			</div>
		</div>

		<div class="moe-network" bind:this={network} style={`height:${networkHeight}px;`}>
			<div class="column routing-column" bind:this={routingColumn}>
				{#if !isDense}<span class="subtitle">Router</span>{/if}{#each $tokens as token, index}<div
						class="cell"
						class:last={index === $tokens.length - 1}
					>
						{#if !isDense}<button
								class="router-dot"
								class:chosen={index === $moeTokenIndex}
								aria-label={`Select router token ${index}: ${token}`}
								aria-pressed={index === $moeTokenIndex}
								on:click={() => chooseToken(index)}>R</button
							>{/if}
					</div>{/each}
			</div>
			<div class="expert-card-slot" style={`left:${ROUTER_WIDTH + CARD_MARGIN}px;`}>
				<ExpertCard
					{entries}
					bind:index={shownIndex}
					hiddenSize={hidden}
					token={tokenLabel($tokens[$moeTokenIndex] ?? 'x')}
					width={cardWidth}
					height={CARD_HEIGHT}
					maxGhosts={MAX_GHOSTS}
				/>
			</div>
			<div class="column sum-column">
				{#if !isDense}<span class="subtitle">Sum</span>{/if}{#each $tokens as _, index}<div
						class="cell"
						class:last={index === $tokens.length - 1}
					>
						{#if !isDense}{#if index === $moeTokenIndex}<button
									class="sum-box"
									class:open={showSum}
									aria-label={`Open the weighted sum for token ${index}: ${$tokens[index]}`}
									aria-pressed={showSum}
									title="Σ · scale each expert output by its router weight, then add them"
									on:click={() => (showSum = true)}
									>Σ<span class="sum-slots" aria-hidden="true"
										>{#each slotShares as share}<span class="sum-slot" style={`height:${share}px`}
											></span>{/each}</span
									></button
								>{:else}<button
									class="sum-dot"
									aria-label={`Inspect weighted sum for token ${index}: ${$tokens[index]}`}
									aria-pressed={false}
									on:click={() => chooseSum(index)}
									title="Select this token and open its weighted sum">Σ</button
								>{/if}{/if}
					</div>{/each}
			</div>
			<div class="selection-note" style={`top:${networkHeight + 10}px;`}>
				<strong>{tokenLabel($tokens[$moeTokenIndex] ?? '')}</strong> → {isDense
					? 'dense SwiGLU'
					: `top-${topK}${sharedExperts ? ' + shared' : ''}`}<span
					>{isDense
						? '· click a projection to multiply'
						: '· arrows step through its experts'}</span
				>
			</div>
		</div>

		<div class="out-layer layer flex">
			<div class="column raw-sum" class:unscaled={($moeModel?.residualMultiplier ?? 0.22) === 1}>
				<span class="subtitle"
					>{#if ($moeModel?.residualMultiplier ?? 0.22) === 1}Output{:else if isDense}×{$moeModel?.residualMultiplier ??
							0.22}{:else}<button
							class="scale-box"
							aria-label={`Open the weighted sum and its ×${$moeModel?.residualMultiplier ?? 0.22} scaling`}
							title="Scale the weighted sum before it joins the residual stream"
							on:click={() => (showSum = true)}>×{$moeModel?.residualMultiplier ?? 0.22}</button
						>{/if}</span
				>{#each $tokens as _, index}<div class="cell" class:last={index === $tokens.length - 1}>
						<div
							class="raw-vector"
							class:chosen={index === $moeTokenIndex}
							title="Raw weighted MoE output, before residual scaling and addition"
						>
							{#each layer?.tokens[index].combinedOutputPreview ?? [] as value, dim}<span
									class="coordinate"
									style={`top:${dim * 12.5}%;background:${vectorColor(value, layer!.tokens[index].combinedOutputPreview)}`}
									title={`MoE sum coordinate ${dim}: ${value}`}
								></span>{/each}
						</div>
					</div>{/each}
			</div>
			<OperationGroup type="residual-end" id="mlp-residual" />
			<div class="column out mlp-out-column">
				{#each $tokens as token, index}<div class="cell" class:last={index === $tokens.length - 1}>
						<div
							class="vector output-vector"
							title={`Block output after residual addition, token ${index}${layer?.blockOutputPreview ? '; recorded coordinates' : '; schematic vector'}`}
						>
							<VectorCanvas colorScale="blue" data={layer?.blockOutputPreview?.[index]} />
						</div>
					</div>{/each}
			</div>
		</div>
	</div>
</div>

{#if showSum}
	<ExpertSumPopup
		{entries}
		hiddenSize={hidden}
		token={tokenLabel($tokens[$moeTokenIndex] ?? 'x')}
		residualMultiplier={$moeModel?.residualMultiplier ?? 0.22}
		shared={sharedExperts > 0}
		on:close={() => (showSum = false)}
	/>
{/if}

<style>
	.mlp.moe {
		position: relative;
		min-width: 420px;
	}
	.moe-title {
		padding: 0;
		border: 0;
		max-width: 100%;
		background: none;
		color: inherit;
		text-align: left;
		white-space: normal;
		font: inherit;
		cursor: pointer;
	}
	.moe-title > span {
		font-size: 0.7em;
		opacity: 0.65;
	}
	.moe-subtitle {
		white-space: normal;
		color: #aa99bb;
		font-size: 0.65rem;
		font-weight: 400;
		margin-top: 4px;
	}
	.mlp-bounding {
		top: -0.5rem;
		left: -0.2rem;
		width: calc(100% + 0.2rem);
		height: 100%;
		padding: 0.5rem 0;
	}
	.moe .content {
		display: grid;
		grid-template-columns: 50px minmax(0, 1fr) 43px;
		gap: 12px;
	}
	.first-layer,
	.out-layer {
		align-items: flex-start;
		position: relative;
	}
	.initial-vector {
		padding: 0;
		border: 0;
		background: #e9d5ff;
		cursor: pointer;
		overflow: hidden;
		display: block;
	}
	.initial-vector.chosen {
		box-shadow: 0 0 0 1px #b497d0;
	}
	.coordinate {
		position: absolute;
		height: 12.5%;
		left: 0;
		width: 100%;
		display: block;
	}
	.initial .head1 {
		position: absolute;
		background: #c084fc;
		left: 0;
		opacity: 0.65;
	}
	.rmsnorm {
		width: 13px;
		flex-shrink: 0;
	}
	.rmsnorm .subtitle {
		white-space: nowrap;
		font-size: 0.55rem;
		left: -12px;
		width: 44px;
	}
	.normalization-dot {
		height: 3px;
		width: 3px;
		border-radius: 50%;
		background: #bdb2c8;
		margin: auto;
	}
	.moe-network {
		position: relative;
		min-width: 0;
	}
	.routing-column,
	.sum-column {
		position: absolute;
		top: 0;
		z-index: 103;
	}
	.routing-column {
		left: 0;
		width: 20px;
	}
	.sum-column {
		right: 0;
		width: 36px;
	}
	.sum-column .cell {
		justify-content: flex-end;
	}
	.routing-column .subtitle,
	.sum-column .subtitle {
		font-size: 0.6rem;
		width: 46px;
		left: -13px;
	}
	.routing-column .subtitle {
		top: 100%;
		transform: translateY(6px);
	}
	.sum-column .subtitle {
		left: -2px;
	}
	.sum-box,
	.scale-box {
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font-family: Arial, sans-serif;
		cursor: pointer;
	}
	.sum-box:hover,
	.scale-box:hover {
		background: #f1e9f9;
		border-color: #b9a0d4;
	}
	.sum-box {
		position: relative;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 30px;
		height: 30px;
		padding: 0;
		flex-shrink: 0;
		font-size: 16px;
		line-height: 1;
	}
	.sum-box.open {
		background: #e0c8ef;
		border-color: #b28bce;
	}
	/* Anchors for the Sankey layer: slot heights follow the router weights and stack to the box height. */
	.sum-slots {
		position: absolute;
		left: -2px;
		top: -1.5px;
		width: 2px;
		display: flex;
		flex-direction: column;
		pointer-events: none;
	}
	.sum-slot {
		display: block;
		width: 2px;
		flex-shrink: 0;
	}
	.scale-box {
		display: block;
		margin: 0 auto;
		padding: 2px 6px;
		font-size: 10px;
		line-height: 1.3;
		white-space: nowrap;
	}
	.router-dot,
	.sum-dot {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 17px;
		height: 17px;
		border: 1px solid #dbcbe9;
		background: #faf6fe;
		color: #b79acc;
		border-radius: 50%;
		font-size: 9px;
		padding: 0;
		flex-shrink: 0;
	}
	.router-dot,
	.sum-dot {
		cursor: pointer;
	}
	.router-dot.chosen,
	.sum-dot.chosen {
		border-color: #aa7cc9;
		background: #d9bfea;
		color: #795298;
	}
	.expert-card-slot {
		position: absolute;
		top: 0;
		/* Below the Sankey layer, so bands into and out of the card draw over its stack. */
		z-index: 5;
	}
	.selection-note {
		position: absolute;
		left: 0;
		width: 100%;
		color: #ad99bf;
		text-align: center;
		font-size: 0.6rem;
		white-space: nowrap;
	}
	.selection-note strong {
		font: inherit;
		color: #9773ae;
	}
	.selection-note > span {
		margin-left: 0.5em;
		font-size: 0.55rem;
		color: #bcaacb;
	}
	.raw-sum {
		width: 12px;
	}
	.raw-sum .subtitle {
		width: 48px;
		left: -18px;
		font-size: 0.55rem;
	}
	.raw-sum.unscaled .subtitle {
		top: 100%;
		transform: translateY(8px);
	}
	.raw-vector {
		position: relative;
		height: var(--vector-height);
		width: 10px;
		background: #e9d5ff;
	}
	.output-vector {
		background: #bfdbfe;
	}
	.out-layer .column.out {
		margin-left: 7px;
	}
	.moe-help {
		position: absolute;
		z-index: 998;
		background: white;
		border: 1px solid #ddcfe8;
		border-radius: 5px;
		padding: 13px 26px 13px 13px;
		box-shadow: 0 4px 15px #5c417415;
		color: #9078a1;
		font-size: 0.67rem;
		line-height: 1.6;
		white-space: normal;
		font-weight: 400;
	}
	.moe-help {
		top: calc(var(--title-height) - 20px);
		width: 290px;
		left: 40px;
	}
	.close {
		border: 0;
		background: white;
		color: #af99bf;
		position: absolute;
		top: 4px;
		right: 6px;
		cursor: pointer;
		font-size: 16px;
		line-height: 1;
		padding: 3px;
	}
	button:focus-visible {
		outline: 2px solid #b596cf;
		outline-offset: 3px;
	}
</style>
