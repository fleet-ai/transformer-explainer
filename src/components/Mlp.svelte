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
		moeExample,
		moeModel,
		moeTokenIndex
	} from '~/store';
	import OperationGroup from './OperationGroup.svelte';
	import VectorCanvas from './common/VectorCanvas.svelte';
	import MoeInspector from './moe/MoeInspector.svelte';
	import { percent, tokenLabel } from '$lib/moe/format';

	export let className: string | undefined = undefined;
	setContext('block-id', 'mlp');
	let network: HTMLDivElement;
	let routingColumn: HTMLDivElement;
	let networkWidth = 310;
	let rowGap = 8;
	let selectedExpert: number | null = null;
	let showSum = false;
	let isHovered = false;
	let showHelp = false;
	$: layer = $moeExample?.layers[$blockIdx];
	$: trace = layer?.tokens[$moeTokenIndex];
	$: isMLA = $moeModel?.architecture === 'mla';
	$: isDense = layer?.feedForwardType === 'dense';
	$: expertColumns = isMLA ? 8 : 4;
	$: bankHeight = Math.ceil(($moeModel?.numExperts ?? 32) / expertColumns) * 30 - 5;
	$: sharedExperts = isMLA && !isDense ? ($moeModel?.sharedExperts ?? 0) : 0;
	$: rowHeight = $tokens.length * $vectorHeight + Math.max(0, $tokens.length - 1) * rowGap;
	$: networkHeight = Math.max(rowHeight, bankHeight + (sharedExperts ? 58 : 0));
	$: bankWidth = Math.min(
		isMLA
			? Math.min(264, Math.max(224, networkWidth * 0.67))
			: Math.min(184, Math.max(152, networkWidth * 0.56)),
		Math.max(40, networkWidth - 44)
	);
	$: bankLeft = (networkWidth - bankWidth) / 2;
	$: bankTop = Math.max(0, (rowHeight - bankHeight - (sharedExperts ? 58 : 0)) / 2);
	$: expertWidth = (bankWidth - (expertColumns - 1) * 5) / expertColumns;
	$: if ($moeExample || $blockIdx || $moeTokenIndex) {
		selectedExpert = null;
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
	function rowY(index: number, height: number, gap: number) {
		return height / 2 + index * (height + gap);
	}
	function expertX(expert: number, left: number, width: number) {
		return left + (expert % expertColumns) * (width + 5);
	}
	function expertY(expert: number, top: number) {
		return top + Math.floor(expert / expertColumns) * 30 + 12.5;
	}
	function wire(x1: number, y1: number, x2: number, y2: number) {
		const bend = Math.max(15, Math.abs(x2 - x1) * 0.5);
		return `M${x1},${y1} C${x1 + bend},${y1} ${x2 - bend},${y2} ${x2},${y2}`;
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
	async function chooseDense(index: number) {
		chooseToken(index);
		await tick();
		selectedExpert = -1;
	}
	function chooseToken(index: number) {
		moeTokenIndex.set(index);
		selectedExpert = null;
	}
</script>

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
				probabilities without renormalizing them. Their weighted outputs are added to two
				always-active shared experts, then to the residual stream. Expert IDs are local to this
				layer.
			{:else}The router selects the top {$moeModel?.topK ?? 8} logits, then softmax gives their mixing
				weights. Only those experts run. Their weighted outputs are added, scaled, and joined to the
				residual stream. Expert IDs are local to this layer.{/if}
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
			{#if isDense}
				<div class="dense-network" style={`height:${rowHeight}px;`}>
					<svg class="dense-wires" width={networkWidth} height={rowHeight} aria-hidden="true"
						>{#each $tokens as _, index}<path
								d={wire(
									0,
									rowY(index, $vectorHeight, rowGap),
									networkWidth / 2 - 52,
									rowY(index, $vectorHeight, rowGap)
								)}
							/><path
								d={wire(
									networkWidth / 2 + 52,
									rowY(index, $vectorHeight, rowGap),
									networkWidth,
									rowY(index, $vectorHeight, rowGap)
								)}
							/>{/each}</svg
					>
					<div class="column dense-column">
						{#each $tokens as token, index}<div class="cell">
								<button
									class="dense-operation"
									class:chosen={index === $moeTokenIndex}
									on:click={() => chooseDense(index)}
									aria-label={`Inspect dense SwiGLU for token ${index}: ${token}`}
									><span>SwiGLU</span><small
										>{$moeModel?.hiddenSize} → {$moeModel?.denseIntermediateSize ?? 10944} → {$moeModel?.hiddenSize}</small
									></button
								>
							</div>{/each}
					</div>
				</div>
			{:else}
				<svg class="moe-paths" width={networkWidth} height={networkHeight} aria-hidden="true">
					{#each layer?.tokens ?? [] as tokenTrace, index}
						{#if index !== $moeTokenIndex}
							{#each tokenTrace.selectedExperts as expert}<path
									d={wire(
										17,
										rowY(index, $vectorHeight, rowGap),
										expertX(expert, bankLeft, expertWidth),
										expertY(expert, bankTop)
									)}
									class="context-path"
								/><path
									d={wire(
										expertX(expert, bankLeft, expertWidth) + expertWidth,
										expertY(expert, bankTop),
										networkWidth - 12,
										rowY(index, $vectorHeight, rowGap)
									)}
									class="context-path"
								/>{/each}
						{/if}
					{/each}
					{#each trace?.selectedExperts ?? [] as expert, rank}<path
							d={wire(
								17,
								rowY($moeTokenIndex, $vectorHeight, rowGap),
								expertX(expert, bankLeft, expertWidth),
								expertY(expert, bankTop)
							)}
							class="selected-path"
							style={`stroke-width:${1.5 + trace!.routingWeights[rank] * 12};opacity:${selectedExpert === null || selectedExpert === expert ? 0.8 : 0.2}`}
						/><path
							d={wire(
								expertX(expert, bankLeft, expertWidth) + expertWidth,
								expertY(expert, bankTop),
								networkWidth - 12,
								rowY($moeTokenIndex, $vectorHeight, rowGap)
							)}
							class="selected-path"
							style={`stroke-width:${1.5 + trace!.routingWeights[rank] * 12};opacity:${selectedExpert === null || selectedExpert === expert ? 0.8 : 0.2}`}
						/>{/each}

					{#if sharedExperts}<path
							class="shared-path"
							d={wire(
								17,
								rowY($moeTokenIndex, $vectorHeight, rowGap),
								bankLeft + bankWidth / 2,
								bankTop + bankHeight + 32
							)}
						/><path
							class="shared-path"
							d={wire(
								bankLeft + bankWidth / 2,
								bankTop + bankHeight + 32,
								networkWidth - 12,
								rowY($moeTokenIndex, $vectorHeight, rowGap)
							)}
						/>{/if}
				</svg>
				<div class="column routing-column" bind:this={routingColumn}>
					<span class="subtitle">Router</span>{#each $tokens as token, index}<div
							class="cell"
							class:last={index === $tokens.length - 1}
						>
							<button
								class="router-dot"
								class:chosen={index === $moeTokenIndex}
								aria-label={`Select router token ${index}: ${token}`}
								aria-pressed={index === $moeTokenIndex}
								on:click={() => chooseToken(index)}>R</button
							>
						</div>{/each}
				</div>
				<div
					class="expert-bank"
					style={`left:${bankLeft}px;top:${bankTop}px;width:${bankWidth}px;grid-template-columns:repeat(${expertColumns},1fr);`}
				>
					<span class="bank-label">Experts</span
					>{#each Array($moeModel?.numExperts ?? 32) as _, expert}{@const rank =
							trace?.selectedExperts.indexOf(expert) ?? -1}<button
							class="expert"
							class:active={rank >= 0}
							class:inspected={selectedExpert === expert}
							aria-label={`Expert ${expert}, ${rank >= 0 ? `selected, mixing weight ${percent(trace!.routingWeights[rank], 2)}` : 'skipped'}. Inspect expert.`}
							aria-pressed={selectedExpert === expert}
							on:click={() => {
								showSum = false;
								selectedExpert = selectedExpert === expert ? null : expert;
							}}
							title={`Expert ${expert} · ${rank >= 0 ? percent(trace!.routingWeights[rank], 2) : 'not selected'}`}
							><span>E{expert.toString().padStart(2, '0')}</span><small
								>{rank >= 0 ? percent(trace!.routingWeights[rank], 0) : '—'}</small
							></button
						>{/each}
				</div>
				{#if sharedExperts}<button
						class="shared-experts"
						class:inspected={selectedExpert === $moeModel?.numExperts}
						style={`left:${bankLeft}px;top:${bankTop + bankHeight + 12}px;width:${bankWidth}px;`}
						on:click={() => {
							showSum = false;
							selectedExpert = $moeModel?.numExperts ?? 64;
						}}
						aria-label="Inspect the two always-active shared experts"
						>{sharedExperts} shared experts <small>always active · weight 1</small></button
					>{/if}
				<div class="column sum-column">
					<span class="subtitle">Sum</span>{#each $tokens as _, index}<div
							class="cell"
							class:last={index === $tokens.length - 1}
						>
							<button
								class="sum-dot"
								class:chosen={index === $moeTokenIndex}
								aria-label={`Inspect weighted sum for token ${index}: ${$tokens[index]}`}
								aria-pressed={showSum && index === $moeTokenIndex}
								on:click={() => chooseSum(index)}
								title="Inspect the weighted sum of expert outputs">Σ</button
							>
						</div>{/each}
				</div>
			{/if}
			<div class="selection-note" style={`top:${networkHeight + 10}px;`}>
				<strong>{tokenLabel($tokens[$moeTokenIndex] ?? '')}</strong> → {isDense
					? 'dense SwiGLU'
					: `top-${$moeModel?.topK ?? 8}${sharedExperts ? ' + shared' : ''}`}<span
					>{isDense
						? 'Click a SwiGLU block to see its multiplications'
						: 'Click an expert or Σ to see its operations'}</span
				>
			</div>
		</div>

		<div class="out-layer layer flex">
			<div class="column raw-sum" class:unscaled={($moeModel?.residualMultiplier ?? 0.22) === 1}>
				<span class="subtitle"
					>{($moeModel?.residualMultiplier ?? 0.22) === 1
						? 'Output'
						: `×${$moeModel?.residualMultiplier ?? 0.22}`}</span
				>{#each $tokens as _, index}<div class="cell" class:last={index === $tokens.length - 1}>
						<div
							class="raw-vector"
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

{#if selectedExpert !== null || showSum}
	<MoeInspector
		expert={showSum ? null : selectedExpert}
		on:close={() => {
			selectedExpert = null;
			showSum = false;
		}}
		on:expert={(event) => {
			selectedExpert = event.detail;
			showSum = false;
		}}
		on:sum={() => {
			selectedExpert = null;
			showSum = true;
		}}
	/>
{/if}

<style>
	.dense-network {
		position: relative;
		min-height: 100px;
	}
	.dense-wires {
		position: absolute;
		inset: 0;
		fill: none;
		stroke: #d8c1eb;
		stroke-width: 2;
	}
	.dense-column {
		align-items: center;
	}
	.dense-operation {
		position: relative;
		z-index: 103;
		background: #fbf7fd;
		border: 1px solid #dfcdec;
		border-radius: 4px;
		width: 104px;
		height: calc(var(--vector-height) + 2px);
		min-height: 26px;
		color: #9672b2;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		font:
			10px Arial,
			sans-serif;
		cursor: pointer;
	}
	.dense-operation small {
		font:
			6px Arial,
			sans-serif;
		color: #a792b6;
		margin-top: 2px;
	}
	.dense-operation.chosen {
		background: #e9d8f4;
		border-color: #ba97d3;
	}
	.shared-experts {
		position: absolute;
		z-index: 104;
		border: 1px solid #b6a2cc;
		border-radius: 4px;
		background: #eee7f8;
		color: #8b70a4;
		height: 38px;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-direction: column;
		font:
			10px Arial,
			sans-serif;
		cursor: pointer;
	}
	.shared-experts small {
		font-size: 8px;
		color: #a08aaf;
		margin-top: 3px;
	}
	.shared-experts.inspected {
		border-color: #9670b6;
		box-shadow: 0 0 0 1px #e9ddf3;
	}
	.shared-path {
		fill: none;
		stroke: #ac90c6;
		stroke-width: 2;
		stroke-dasharray: 3 2;
	}

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
	.moe-paths {
		position: absolute;
		left: 0;
		top: 0;
		overflow: visible;
		pointer-events: none;
		z-index: 90;
	}
	.context-path {
		fill: none;
		stroke: #d6c4e6;
		stroke-width: 0.7;
		opacity: 0.3;
	}
	.selected-path {
		fill: none;
		stroke: #bb97d9;
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
		width: 17px;
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
	.expert-bank {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		grid-auto-rows: 25px;
		gap: 5px;
		position: absolute;
		z-index: 104;
	}
	.bank-label {
		position: absolute;
		top: -28px;
		width: 100%;
		text-align: center;
		color: #a89ab6;
		font-size: 0.6rem;
	}
	.expert {
		border: 1px solid #e7deef;
		border-radius: 3px;
		padding: 1px 2px;
		background: #fcfaff;
		color: #b7a9c3;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		cursor: pointer;
		font:
			9px/1.2 'SFMono-Regular',
			Consolas,
			monospace;
	}
	.expert small {
		font:
			8px/1.1 Arial,
			sans-serif;
		color: #c3b7ce;
		margin-top: 1px;
	}
	.expert.active {
		border-color: #c5a5de;
		background: #e5d3f3;
		color: #9366b3;
	}
	.expert.active small {
		color: #9366b3;
	}
	.expert.inspected {
		border-color: #9670b6;
		background: #cda9e6;
		box-shadow: 0 0 0 1px #e9ddf3;
		color: #6e448c;
	}
	.expert:hover {
		border-color: #b293ca;
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
		display: block;
		margin-top: 5px;
		font-size: 0.55rem;
		color: #bcaacb;
	}
	.raw-sum {
		width: 12px;
	}
	.raw-sum .subtitle {
		width: 40px;
		left: -13px;
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
