<script lang="ts">
	import { onMount, setContext, tick } from 'svelte';
	import {
		tokens,
		blockIdx,
		moeExample,
		moeTokenIndex,
		vectorHeight,
		attentionHeadIdx,
		modelMeta,
		moeModel,
		bandRedraw
	} from '~/store';
	import { vlmLayer, vlmPhase, vlmVisiblePositions, vlmSelectedPosition } from '$lib/vlm/store';
	import { row } from '$lib/vit/types';
	import { tokenLabel } from '$lib/moe/format';
	import type { ExpertEntry } from '$lib/moe/types';
	import OperationGroup from '../OperationGroup.svelte';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	import ExpertCard from '../moe/ExpertCard.svelte';

	/** Every token runs through the same dense MLP: GELU in the vision encoder, SwiGLU in the
	 *  language decoder. It is drawn as the expert card of the MoE and MLA tabs, with one entry. */
	export let className: string | undefined = undefined;
	setContext('block-id', 'mlp');
	const CARD_HEIGHT = 330;
	const MAX_CARD_WIDTH = 190;
	let network: HTMLDivElement;
	let networkWidth = 200;
	$: vision = $vlmPhase === 'vision';
	$: hidden = $moeModel?.hiddenSize ?? 768;
	$: intermediate = $moeModel?.intermediateSize ?? 3072;
	$: layer = $moeExample?.layers[$blockIdx];
	$: entries = [
		{ id: -1, kind: 'dense', label: 'Feed-forward', weight: 1, size: intermediate }
	] as ExpertEntry[];
	$: cardWidth = Math.max(110, Math.min(MAX_CARD_WIDTH, networkWidth - 16));
	$: if (network && ($vectorHeight || $tokens || $vlmPhase)) tick().then(measure);
	$: if (cardWidth) tick().then(() => bandRedraw.update((count) => count + 1));

	function measure() {
		if (network) networkWidth = network.clientWidth;
	}
	function choose(index: number) {
		moeTokenIndex.set(index);
		vlmSelectedPosition.set($vlmVisiblePositions[index] ?? index);
	}
	onMount(() => {
		const observer = new ResizeObserver(measure);
		observer.observe(network);
		measure();
		return () => observer.disconnect();
	});
</script>

<div class={`mlp vlm-mlp ${className ?? ''}`} data-click="mlp-step">
	<div class="title">
		Feed-forward <small
			>{vision ? 'GELU' : 'SwiGLU'} · {hidden} → {intermediate} → {hidden} · every token</small
		>
	</div>
	<div class="content resize-watch relative">
		<div class="first-stage">
			<div class="column initial">
				{#each $tokens as token, index}<div class="cell">
						<span class="label float">{token}</span><button
							class="vector initial-vector bg-purple-200"
							class:chosen={index === $moeTokenIndex}
							aria-label={`Select ${token} for the feed-forward card`}
							aria-pressed={index === $moeTokenIndex}
							on:click={() => choose(index)}
							><VectorCanvas colorScale="purple" data={layer?.attentionOutputPreview[index]} />
							<div
								class="sub-vector head1"
								style={`position:absolute;top:${($vectorHeight / $modelMeta.attention_head_num) * $attentionHeadIdx}px;height:${$vectorHeight / $modelMeta.attention_head_num}px`}
							></div></button
						>
					</div>{/each}
			</div>
			<OperationGroup type="residual-end" id="embedding-residual" /><OperationGroup
				type="residual-start"
				id="mlp-residual"
			/><OperationGroup type="ln" id="mlp-ln" />
		</div>
		<div class="card-network" bind:this={network} style={`height:${CARD_HEIGHT}px;`}>
			<div class="card-slot" style={`width:${cardWidth}px;`}>
				<ExpertCard
					{entries}
					hiddenSize={hidden}
					token={tokenLabel($tokens[$moeTokenIndex] ?? 'x')}
					width={cardWidth}
					height={CARD_HEIGHT}
					gated={!vision}
					activation={vision ? 'GELU' : 'SiLU'}
					bias={vision}
				/>
			</div>
			<div class="selection-note" style={`top:${CARD_HEIGHT + 8}px;`}>
				<strong>{tokenLabel($tokens[$moeTokenIndex] ?? '')}</strong> → {vision
					? 'GELU MLP'
					: 'SwiGLU MLP'}<span>· click a box to see its formula</span>
			</div>
		</div>
		<div class="out-stage">
			<div class="column raw-sum">
				<span class="subtitle">Output</span>{#each $tokens as _, index}<div class="cell">
						<div
							class="raw-vector"
							class:chosen={index === $moeTokenIndex}
							title="Feed-forward output, before the residual addition"
						>
							<VectorCanvas
								colorScale="purple"
								data={$vlmLayer
									? row($vlmLayer, 'down', $vlmVisiblePositions[index], 8)
									: undefined}
							/>
						</div>
					</div>{/each}
			</div>
			<OperationGroup type="residual-end" id="mlp-residual" />
			<div class="column out mlp-out-column">
				{#each $tokens as token, index}<div class="cell">
						<div class="vector bg-blue-200" title={`Block output · ${token}`}>
							<VectorCanvas colorScale="blue" data={layer?.blockOutputPreview?.[index]} />
						</div>
					</div>{/each}
			</div>
		</div>
	</div>
</div>

<style>
	.vlm-mlp {
		position: relative;
	}
	.title {
		pointer-events: none;
	}
	.title small {
		display: block;
		font-size: 9px;
		color: #aa98ba;
		/* Below the residual label that the operation group draws over the title row. */
		margin-top: 30px;
		white-space: normal;
	}
	.content {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		gap: 10px;
		align-items: start;
	}
	.first-stage,
	.out-stage {
		display: flex;
		align-items: flex-start;
		position: relative;
	}
	.first-stage {
		margin-left: 10px;
	}
	.initial-vector {
		padding: 0;
		border: 0;
		cursor: pointer;
		display: block;
		overflow: hidden;
	}
	.initial-vector.chosen {
		box-shadow: 0 0 0 1px #b497d0;
	}
	.label.float {
		white-space: nowrap;
		max-width: 5rem;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.card-network {
		position: relative;
		min-width: 0;
	}
	.card-slot {
		position: absolute;
		top: 0;
		left: 50%;
		transform: translateX(-50%);
		/* Below the Sankey layer, so bands into and out of the card draw over its edges. */
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
		width: 10px;
	}
	.raw-sum .subtitle {
		width: 48px;
		left: -18px;
		font-size: 0.55rem;
		top: 100%;
		transform: translateY(8px);
	}
	.raw-vector {
		position: relative;
		height: var(--vector-height);
		width: 10px;
		background: #e9d5ff;
	}
	.raw-vector.chosen {
		box-shadow: 0 0 0 1px #b497d0;
	}
	.out-stage .column.out {
		margin-left: 7px;
	}
	button:focus-visible {
		outline: 2px solid #b596cf;
		outline-offset: 3px;
	}
</style>
