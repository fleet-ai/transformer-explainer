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
	import { t5Example, t5Stack, t5Inspector } from '$lib/seq2seq/store';
	import OperationGroup from '../OperationGroup.svelte';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	export let className: string | undefined = undefined;
	setContext('block-id', 'mlp');
	$: layer = $moeExample?.layers[$blockIdx];
	$: feedforward = $t5Example?.feedForward[$t5Stack][$blockIdx];
	$: attentionOutput =
		$t5Stack === 'decoder'
			? $t5Example?.crossAttention[$blockIdx].attentionOutput
			: layer?.attentionOutputPreview;
	function inspect(index: number) {
		moeTokenIndex.set(index);
		t5Inspector.set('ffn');
	}
</script>

<div class={`mlp ${className ?? ''}`} data-click="mlp-step">
	<div class="title">MLP <small>512 → 2048 → 512 · ReLU</small></div>
	<div class="content resize-watch relative">
		<div class="first-layer layer">
			<div class="column initial">
				{#each $tokens as token, index}<div class="cell">
						<span class="label float">{token}</span><button
							class="vector bg-purple-200"
							aria-label={`Inspect feed-forward input ${token}`}
							on:click={() => inspect(index)}
							><VectorCanvas colorScale="purple" data={attentionOutput?.[index]} />
							<div
								class="sub-vector head1"
								style={`position:absolute;top:${($vectorHeight / $modelMeta.attention_head_num) * $attentionHeadIdx}px;height:${$vectorHeight / $modelMeta.attention_head_num}px`}
							></div></button
						>
					</div>{/each}
			</div>
			<OperationGroup
				type="residual-end"
				id={$t5Stack === 'decoder' ? 'cross-residual' : 'embedding-residual'}
			/><OperationGroup type="residual-start" id="mlp-residual" /><OperationGroup
				type="ln"
				id="mlp-ln"
			/>
		</div>
		<div class="second-layer">
			<div class="column">
				<span class="subtitle">2048D · ReLU</span>{#each $tokens as token, index}<button
						class="vector expanded-vector bg-indigo-200"
						aria-label={`Inspect MLP process for ${token}`}
						on:click={() => inspect(index)}
						><VectorCanvas
							colorScale="indigo"
							data={feedforward?.tokens[index].activation}
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
						on:click={() => inspect(index)}
						><VectorCanvas colorScale="blue" data={layer?.blockOutputPreview?.[index]} /></button
					>{/each}
			</div>
		</div>
	</div>
</div>

<style>
	.title small {
		display: block;
		font-size: 9px;
		color: #aa98ba;
		margin-top: 8px;
	}
	.content {
		display: grid;
		grid-template-columns: minmax(0, 1.6fr) minmax(12px, 1fr) auto;
		align-items: start;
	}
	.layer {
		display: flex;
	}
	.expanded-vector {
		height: calc(var(--vector-height) * 4);
	}

	.second-layer {
		position: relative;
		z-index: 20;
	}
	.first-layer {
		margin-left: 10px;
	}
	.subtitle {
		top: 100%;
		transform: translateY(6px);
		font-size: 9px;
		white-space: nowrap;
	}
	.vector {
		border: 0;
		padding: 0;
	}
	.first-layer .label {
		font-size: 11px;
		max-width: 60px;
	}
</style>
