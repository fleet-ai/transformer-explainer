<script lang="ts">
	import { setContext } from 'svelte';
	import {
		tokens,
		blockIdx,
		moeExample,
		moeTokenIndex,
		vectorHeight,
		attentionHeadIdx,
		modelMeta
	} from '~/store';
	import {
		vitRecording,
		vitVisiblePositions,
		vitSelectedPosition,
		vitInspectorStage
	} from '$lib/vit/store';
	import { row } from '$lib/vit/types';
	import OperationGroup from '../OperationGroup.svelte';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	export let className: string | undefined = undefined;
	setContext('block-id', 'mlp');
	/** Dense's MLP: 192 → a 4× taller 768 column → 192, with GELU as a box on the wide column. */
	let geluHot = false;
	$: layer = $vitRecording?.layers[$blockIdx];
	$: hidden = $vitRecording?.document.model.hiddenSize ?? 192;
	$: inner = $vitRecording?.document.model.intermediateSize ?? 768;
	function inspect(stage: string, index: number) {
		moeTokenIndex.set(index);
		vitSelectedPosition.set($vitVisiblePositions[index]);
		vitInspectorStage.set(stage);
	}
</script>

<div class={`mlp ${className ?? ''}`} data-click="mlp-step">
	<div class="title"><span>MLP <small>{hidden} → {inner} → {hidden} · GELU</small></span></div>
	<div class="content resize-watch relative">
		<div class="first-layer">
			<div class="column initial">
				{#each $tokens as token, index}<div class="cell">
						<span class="label float">{token}</span><button
							class="vector bg-purple-200"
							aria-label={`Up projection for ${token}`}
							on:click={() => inspect('up', index)}
							><VectorCanvas
								colorScale="purple"
								data={$moeExample?.layers[$blockIdx].attentionOutputPreview[index]}
							/>
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
		<div class="second-layer">
			<div class="column" class:hot={geluHot}>
				<span class="subtitle">{inner}D</span>{#each $tokens as token, index}<button
						class="vector expanded-vector bg-indigo-200"
						aria-label={`Down projection for ${token}`}
						on:click={() => inspect('down', index)}
						><VectorCanvas
							colorScale="purple"
							data={layer ? row(layer, 'gelu', $vitVisiblePositions[index], 8) : undefined}
						/></button
					>{/each}
				<button
					class="gelu-box"
					class:hot={geluHot}
					title={`GELU · applied to each of the ${inner} coordinates`}
					aria-label={`Open the GELU activation for ${$tokens[$moeTokenIndex] ?? 'this token'}`}
					on:click={() => inspect('gelu', $moeTokenIndex)}
					on:mouseenter={() => (geluHot = true)}
					on:mouseleave={() => (geluHot = false)}
					on:focus={() => (geluHot = true)}
					on:blur={() => (geluHot = false)}>GELU</button
				>
			</div>
		</div>
		<div class="out-layer">
			<OperationGroup type="residual-end" id="mlp-residual" />
			<div class="column out mlp-out-column">
				{#each $tokens as token, index}<button
						class="vector bg-blue-200"
						aria-label={`Down projection for ${token}`}
						on:click={() => inspect('down', index)}
						><VectorCanvas
							colorScale="blue"
							data={$moeExample?.layers[$blockIdx].blockOutputPreview?.[index]}
						/></button
					>{/each}
			</div>
		</div>
	</div>
</div>

<style>
	.title small {
		margin-left: 4px;
		font-size: 9px;
		color: #aa98ba;
	}
	.content {
		display: grid;
		grid-template-columns: minmax(0, 1.6fr) minmax(12px, 1fr) auto;
		align-items: start;
	}
	.first-layer,
	.out-layer {
		display: flex;
	}
	.first-layer {
		margin-left: 10px;
	}
	.label.float {
		white-space: nowrap;
		max-width: 5rem;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.vector {
		border: 0;
		padding: 0;
	}
	.second-layer {
		position: relative;
		z-index: 20;
		pointer-events: none;
	}
	.second-layer .column {
		align-items: center;
	}
	.expanded-vector {
		pointer-events: auto;
		height: calc(var(--vector-height) * 4);
		transition: filter 0.12s;
	}
	.column.hot .expanded-vector {
		filter: brightness(0.94) saturate(1.4);
	}
	.subtitle {
		position: absolute;
		top: calc(100% + 10px);
		left: 50%;
		transform: translateX(-50%);
		font-size: 10px;
		white-space: nowrap;
	}
	/* The activation sits on the wide column, between W₁'s band and W₂'s. */
	.gelu-box {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		/* Above the column's vectors. */
		z-index: 110;
		pointer-events: auto;
		width: 42px;
		height: 24px;
		padding: 0;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			10.5px Arial,
			sans-serif;
		cursor: pointer;
		user-select: none;
	}
	.gelu-box.hot {
		background: #f1e9f9;
		border-color: #b9a0d4;
	}
	.gelu-box:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: 1px;
	}
</style>
