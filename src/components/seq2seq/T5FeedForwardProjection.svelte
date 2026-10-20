<script lang="ts">
	import AnimatedMatrixPopup from '../common/AnimatedMatrixPopup.svelte';
	import { blockIdx, moeTokenIndex, tokens } from '~/store';
	import { t5Example, t5Stack, t5Inspector } from '$lib/seq2seq/store';
	export let direction: 'up' | 'down' = 'up';
	$: tokenIndex = Math.max(0, Math.min($moeTokenIndex, $tokens.length - 1));
	$: ff = $t5Example?.feedForward[$t5Stack][$blockIdx]?.tokens[tokenIndex];
	$: operation = direction === 'down' ? ff?.down : ff?.up;
	$: input = direction === 'down' ? ff?.activation : ff?.inputPreview;
	$: inputSize = direction === 'down' ? 2048 : 512;
	$: outputSize = direction === 'down' ? 512 : 2048;
	const fmt = (value: number | undefined) => (Number.isFinite(value) ? value!.toFixed(6) : '—');
</script>

{#if operation && input}
	{#key `${$t5Stack}-${$blockIdx}-${tokenIndex}-${direction}`}
		<AnimatedMatrixPopup
			label={`T5 · ${$t5Stack === 'encoder' ? 'Encoder' : 'Decoder'} block ${$blockIdx + 1}`}
			projection={{
				label: direction === 'up' ? 'Up projection' : 'Down projection',
				input: direction === 'up' ? 'RMSNorm(X)' : 'ReLU(U)',
				weight: direction === 'up' ? 'Wᵢᵀ' : 'Wₒᵀ',
				output: direction === 'up' ? 'Up U' : 'Output Y',
				inputSize,
				outputSize
			}}
			token={$tokens[tokenIndex]}
			on:close={() => t5Inspector.set(null)}
		>
			<details class="recorded-products">
				<summary>Recorded values and products · token {tokenIndex}: {$tokens[tokenIndex]}</summary>
				<p>Recorded model values. Each animated cell represents a group of coordinates.</p>
				{#each operation.output as value, row}
					<p class="product">
						[{row}] = {#each operation.weights[row] ?? [] as weight, column}({fmt(input[column])} × {fmt(
								weight
							)}) +
						{/each}{fmt(operation.omittedContribution[row])}
						<span>(remaining {inputSize - (operation.weights[row]?.length ?? 0)} inputs)</span>
						≈ <strong>{fmt(value)}</strong>
					</p>
				{/each}
				<p>
					google-t5/t5-small · {$t5Stack} block {$blockIdx + 1} · bias-free projection · rounded for
					display.
				</p>
			</details>
		</AnimatedMatrixPopup>
	{/key}
{/if}

<style>
	.recorded-products {
		border-top: 1px solid #e5e7eb;
		padding-top: 12px;
		color: #6b7280;
		font-size: 12px;
	}
	summary {
		cursor: pointer;
	}
	p {
		margin-top: 10px;
	}
	.product {
		font-family: monospace;
		overflow-wrap: anywhere;
		line-height: 1.6;
		color: #59667a;
	}
	.product span {
		color: #9ca3af;
	}
</style>
