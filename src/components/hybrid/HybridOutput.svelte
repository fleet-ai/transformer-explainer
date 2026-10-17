<script lang="ts">
	import { setContext } from 'svelte';
	import {
		tokens,
		blockIdx,
		moeTokenIndex,
		vectorHeight,
		attentionHeadIdx,
		modelMeta
	} from '~/store';
	import { hybridRecording, hybridLayer, hybridInspector } from '$lib/hybrid/store';
	import { row } from '$lib/vit/types';
	import MambaCore from './MambaCore.svelte';
	import OperationGroup from '../OperationGroup.svelte';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	export let className: string | undefined = undefined;
	setContext('block-id', 'mlp');
	$: hasAttention = $hybridRecording?.document.layers[$blockIdx].type === 'hybrid';
	function inspect(stage: string, index: number) {
		moeTokenIndex.set(index);
		hybridInspector.set(stage);
	}
</script>

<div class={`mlp hybrid-output ${className ?? ''}`} class:with-attention={hasAttention}>
	<div class="title">
		{hasAttention ? 'Gated FFN → Mamba' : 'Output projection'}<small
			>{hasAttention
				? 'Shared transformer · 8192D GELU gate · then state-space mixer'
				: '4096 → 2048 · residual addition'}</small
		>
	</div>
	<div class="content resize-watch relative">
		<div class="column initial">
			{#each $tokens as token, index}<div class="cell">
					<span class="label float">{token}</span><button
						class="vector"
						aria-label={`${hasAttention ? 'Attention output' : 'Gated state'} ${token}`}
						on:click={() => inspect(hasAttention ? 'attention' : 'output', index)}
						><VectorCanvas
							colorScale="purple"
							data={$hybridLayer
								? row($hybridLayer, hasAttention ? 'attentionOutput' : 'gated', index, 8)
								: undefined}
						/>
						<div
							class="sub-vector head1"
							style={`position:absolute;top:${($vectorHeight / $modelMeta.attention_head_num) * $attentionHeadIdx}px;height:${$vectorHeight / $modelMeta.attention_head_num}px`}
						></div></button
					>
				</div>{/each}
		</div>
		{#if hasAttention}<div class="hybrid-computation">
				<button class="ffn-chain" on:click={() => hybridInspector.set('ffn')}
					><b>RMSNorm → GELU(Wgate x) ⊙ Wup x → Wdown</b><small
						>Linear → add block input h → Mamba RMSNorm</small
					><span>Inspect real gated feed-forward ↗</span></button
				><MambaCore compact />
			</div>{:else}<div class="output-projection column">
				{#each $tokens as token, index}<button
						class="cell"
						on:click={() => inspect('output', index)}>Wout · z</button
					>{/each}
			</div>{/if}
		<div class="out-layer">
			<OperationGroup type="residual-end" id="embedding-residual" />
			<div class="column out mlp-out-column">
				{#each $tokens as token, index}<button
						class="vector"
						aria-label={`Hybrid block output ${token}`}
						on:click={() => inspect('output', index)}
						><VectorCanvas
							colorScale="blue"
							data={$hybridLayer ? row($hybridLayer, 'output', index, 8) : undefined}
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
		color: #ad99ba;
		margin-top: 8px;
	}
	.content {
		display: flex;
		gap: 14px;
		align-items: flex-start;
	}
	.column.initial {
		margin-left: 10px;
	}
	.initial .label {
		font-size: 11px;
		max-width: 70px;
	}
	.vector {
		padding: 0;
		border: 0;
		background: #e5d8f1;
		cursor: pointer;
	}
	.out-layer {
		display: flex;
		margin-left: auto;
	}
	.out .vector {
		background: #ddeafb;
	}
	.hybrid-computation {
		flex: 1;
		min-width: 260px;
		max-width: 440px;
	}
	.ffn-chain {
		display: block;
		background: #fbf8fd;
		border: 1px solid #e2d2ed;
		border-radius: 5px;
		padding: 10px 14px;
		margin-bottom: 10px;
		color: #936aac;
		text-align: left;
		width: 100%;
		cursor: pointer;
	}
	.ffn-chain b {
		font:
			10px/1.5 Arial,
			sans-serif;
	}
	.ffn-chain small {
		display: block;
		font:
			9px/1.5 Arial,
			sans-serif;
		color: #a88cb9;
		margin: 5px 0;
	}
	.ffn-chain span {
		font-size: 9px;
		color: #b398c6;
	}
	.output-projection {
		flex: 1;
		z-index: 30;
		position: relative;
	}
	.output-projection button {
		font-size: 11px;
		padding: 0 15px;
		border: 1px solid #d9c4e8;
		background: #fbf7fd;
		color: #9b76b4;
		border-radius: 4px;
		cursor: pointer;
	}
	.with-attention .content {
		gap: 10px;
	}
</style>
