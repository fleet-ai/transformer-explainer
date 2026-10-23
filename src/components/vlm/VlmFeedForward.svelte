<script lang="ts">
	import { setContext } from 'svelte';
	import {
		tokens,
		blockIdx,
		moeExample,
		moeTokenIndex,
		vectorHeight,
		attentionHeadIdx,
		modelMeta,
		moeModel
	} from '~/store';
	import {
		vlmLayer,
		vlmPhase,
		vlmVisiblePositions,
		vlmSelectedPosition,
		vlmInspectorStage
	} from '$lib/vlm/store';
	import { row } from '$lib/vit/types';
	import OperationGroup from '../OperationGroup.svelte';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	export let className: string | undefined = undefined;
	setContext('block-id', 'mlp');
	function inspect(stage: string, index: number) {
		moeTokenIndex.set(index);
		vlmSelectedPosition.set($vlmVisiblePositions[index]);
		vlmInspectorStage.set(stage);
	}
</script>

<div class={`mlp ${className ?? ''}`} data-click="mlp-step">
	<div class="title">
		MLP <small
			>{$vlmPhase === 'vision' ? 'GELU' : 'SwiGLU'} · {$moeModel?.hiddenSize} → {$moeModel?.intermediateSize}
			→ {$moeModel?.hiddenSize}</small
		>
	</div>
	{#if $vlmPhase === 'language'}<button
			class="gate-projection"
			on:click={() => inspect('gate-projection', $moeTokenIndex)}>Wgate · gate projection ↗</button
		>{/if}
	<div class="content resize-watch relative">
		<div class="first-layer layer">
			<div class="column initial">
				{#each $tokens as token, index}<div class="cell">
						<span class="label float">{token}</span><button
							class="vector bg-purple-200"
							aria-label={`Up projection multiplication for ${token}`}
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
			<div class="column">
				<span class="subtitle">{$vlmPhase === 'vision' ? 'GELU' : 'SiLU(g) ⊙ up'}</span
				>{#each $tokens as token, index}<button
						class="vector expanded-vector bg-indigo-200"
						style={`height:calc(var(--vector-height) * ${($moeModel?.intermediateSize ?? 3072) / ($moeModel?.hiddenSize ?? 768)})`}
						aria-label={`Down projection multiplication for ${token}`}
						on:click={() => inspect('down', index)}
						><VectorCanvas
							colorScale="purple"
							data={$vlmLayer
								? row(
										$vlmLayer,
										$vlmPhase === 'vision' ? 'gelu' : 'product',
										$vlmVisiblePositions[index],
										8
									)
								: undefined}
						/></button
					>{/each}
			</div>
		</div>
		<div class="out-layer layer">
			<OperationGroup type="residual-end" id="mlp-residual" />
			<div class="column out mlp-out-column">
				{#each $tokens as token, index}<button
						class="vector bg-blue-200"
						aria-label={`Block output ${token}`}
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
	.mlp {
		position: relative;
	}
	.gate-projection {
		position: absolute;
		top: 34px;
		right: 0;
		z-index: 101;
		font-size: 9px;
		color: #8b72ad;
		border: 1px solid #e5dcf1;
		border-radius: 3px;
		padding: 3px 5px;
	}
	.expanded-vector {
		pointer-events: auto;
		height: calc(var(--vector-height) * 4);
	}
	.second-layer {
		pointer-events: none;
	}
	.label.float {
		white-space: nowrap;
		max-width: 5rem;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.title {
		pointer-events: none;
	}
	.title small {
		display: block;
		font-size: 9px;
		color: #aa98ba;
		margin-top: 34px;
	}
	.content {
		display: grid;
		grid-template-columns: minmax(0, 1.6fr) minmax(12px, 1fr) auto;
		align-items: start;
	}
	.layer {
		display: flex;
	}
	.operation {
		font-size: 9px;
		border: 1px solid #dbc9eb;
		border-radius: 3px;
		background: #fbf8fd;
		color: #9465b0;
		white-space: nowrap;
		padding: 0 7px;
	}
	.operation:hover {
		background: #eee0f8;
	}
	.second-layer {
		position: relative;
		z-index: 20;
	}
	.first-layer {
		margin-left: 10px;
	}
	.subtitle {
		position: absolute;
		top: calc(100% + 10px);
		left: 50%;
		transform: translateX(-50%);
		font-size: 10px;
		white-space: nowrap;
	}
	.vector {
		border: 0;
		padding: 0;
	}
	.projection-down,
	.feedforward-input {
		position: relative;
		z-index: 20;
	}
</style>
