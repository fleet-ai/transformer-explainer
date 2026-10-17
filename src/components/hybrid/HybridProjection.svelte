<script lang="ts">
	import QKV from '../QKV.svelte';
	import OperationGroup from '../OperationGroup.svelte';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	import { tokens, blockIdx, moeTokenIndex } from '~/store';
	import { hybridRecording, hybridLayer, hybridInspector } from '$lib/hybrid/store';
	import { row } from '$lib/vit/types';
	export let className: string | undefined = undefined;
	$: hasAttention = $hybridRecording?.document.layers[$blockIdx].type === 'hybrid';
	function inspect(index: number) {
		moeTokenIndex.set(index);
		hybridInspector.set('projection');
	}
</script>

{#if hasAttention}<QKV {className} />{:else}<div class={`qkv hybrid-projection ${className ?? ''}`}>
		<div class="title"></div>
		<div class="content relative">
			<div class="block-start-column vector-column">
				<div class="column vectors">
					{#each $tokens as token, index}<button
							class="vector"
							aria-label={`Mamba input ${token}`}
							on:click={() => inspect(index)}
							><VectorCanvas
								colorScale="blue"
								data={$hybridLayer ? row($hybridLayer, 'input', index, 8) : undefined}
							/></button
						>{/each}
				</div>
				<OperationGroup type="residual-start" id="embedding-residual" /><OperationGroup
					type="ln"
					id="embedding-ln"
				/>
			</div>
		</div>
	</div>{/if}

<style>
	.block-start-column {
		display: flex;
	}
	.vector {
		padding: 0;
		border: 0;
		background: #e4eefb;
		cursor: pointer;
	}
	.content {
		padding-left: 10px;
	}
</style>
