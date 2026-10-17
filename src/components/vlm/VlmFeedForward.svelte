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
		Dense Feed-forward <small
			>{$vlmPhase === 'vision' ? 'GELU' : 'SwiGLU'} · {$moeModel?.hiddenSize} → {$moeModel?.intermediateSize}
			→ {$moeModel?.hiddenSize}</small
		>
	</div>
	<div class="content resize-watch relative">
		<div class="first-layer layer">
			<div class="column initial">
				{#each $tokens as token, index}<div class="cell">
						<span class="label float">{token}</span><button
							class="vector bg-purple-200"
							aria-label={`Attention output ${token}`}
							on:click={() => inspect('attention', index)}
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
		<div class="column feedforward-input">
			{#each $tokens as token, index}<button
					class="cell operation"
					on:click={() => inspect('up', index)}
					>{$vlmPhase === 'vision' ? 'W₁x + b₁' : 'Wup · x'}</button
				>{/each}
		</div>
		<div class="second-layer">
			<div class="column">
				<span class="subtitle">{$vlmPhase === 'vision' ? 'GELU' : 'SiLU(g) ⊙ up'}</span
				>{#each $tokens as token, index}<button
						class="vector bg-purple-200"
						aria-label={`Inspect activation for ${token}`}
						on:click={() => inspect($vlmPhase === 'vision' ? 'gelu' : 'gate', index)}
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
		<div class="column projection-down">
			{#each $tokens as token, index}<button
					class="cell operation"
					on:click={() => inspect('down', index)}
					>{$vlmPhase === 'vision' ? 'W₂h + b₂' : 'Wdown · h'}</button
				>{/each}
		</div>
		<div class="out-layer layer">
			<OperationGroup type="residual-end" id="mlp-residual" />
			<div class="column out mlp-out-column">
				{#each $tokens as token, index}<button
						class="vector bg-blue-200"
						aria-label={`Block output ${token}`}
						on:click={() => inspect('residual', index)}
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
	.label.float {
		white-space: nowrap;
		max-width: 5rem;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.title small {
		display: block;
		font-size: 9px;
		color: #aa98ba;
		margin-top: 8px;
	}
	.content {
		display: flex;
		gap: 10px;
		align-items: flex-start;
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
