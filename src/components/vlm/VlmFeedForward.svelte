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
	/** Dense's MLP: the hidden vector widens into a taller column and narrows back. The vision
	 *  encoder's GELU and the decoder's SwiGLU gate sit as boxes on the wide column. */
	let hot: string | null = null;
	$: vision = $vlmPhase === 'vision';
	$: hidden = $moeModel?.hiddenSize ?? 768;
	$: inner = $moeModel?.intermediateSize ?? 3072;
	$: ratio = inner / hidden;
	$: boxes = vision
		? [{ id: 'gelu', label: 'GELU', title: `GELU · applied to each of the ${inner} coordinates` }]
		: [
				{
					id: 'gate-projection',
					label: 'Wgate',
					title: `Gate projection · ${hidden} → ${inner}, alongside Wup`
				},
				{ id: 'silu', label: 'SiLU', title: `SiLU of the gate · each of the ${inner} coordinates` },
				{
					id: 'gate',
					label: '⊙',
					title: 'Multiply SiLU(gate) by the up projection, coordinate-wise'
				}
			];
	function inspect(stage: string, index: number) {
		moeTokenIndex.set(index);
		vlmSelectedPosition.set($vlmVisiblePositions[index] ?? index);
		vlmInspectorStage.set(stage);
	}
</script>

<div class={`mlp ${className ?? ''}`} data-click="mlp-step">
	<div class="title">
		MLP <small
			>{vision ? 'LayerNorm' : 'RMSNorm'} · {hidden} → {inner} → {hidden} · {vision
				? 'GELU'
				: 'SwiGLU'}</small
		>
	</div>
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
			<div class="column" class:hot={!!hot}>
				<span class="subtitle">{inner}D</span>{#each $tokens as token, index}<button
						class="vector expanded-vector bg-indigo-200"
						style={`height:calc(var(--vector-height) * ${ratio})`}
						aria-label={`Down projection for ${token}`}
						on:click={() => inspect('down', index)}
						><VectorCanvas
							colorScale="purple"
							data={$vlmLayer
								? row($vlmLayer, vision ? 'gelu' : 'product', $vlmVisiblePositions[index], 8)
								: undefined}
						/></button
					>{/each}
				<div class="ops">
					{#each boxes as box}<button
							type="button"
							class:hot={hot === box.id}
							class:glyph={box.id === 'gate'}
							title={box.title}
							aria-label={`Open ${box.title} for ${$tokens[$moeTokenIndex] ?? 'this token'}`}
							on:click={() => inspect(box.id, $moeTokenIndex)}
							on:mouseenter={() => (hot = box.id)}
							on:mouseleave={() => (hot = null)}
							on:focus={() => (hot = box.id)}
							on:blur={() => (hot = null)}>{box.label}</button
						>{/each}
				</div>
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
		display: block;
		font-size: 9px;
		color: #aa98ba;
		/* Below the residual label that the operation group draws over the title row. */
		margin-top: 30px;
		white-space: normal;
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
	/* The activation sits on the wide column, between the up band and the down band. */
	.ops {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		/* Above the column's vectors. */
		z-index: 110;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 6px;
		pointer-events: none;
	}
	.ops button {
		pointer-events: auto;
		width: 44px;
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
	.ops button.glyph {
		width: 26px;
		height: 26px;
		font-size: 15px;
		line-height: 1;
	}
	.ops button.hot {
		background: #f1e9f9;
		border-color: #b9a0d4;
	}
	.ops button:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: 1px;
	}
</style>
