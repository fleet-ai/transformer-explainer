<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import AnimatedMatrixPopup from './AnimatedMatrixPopup.svelte';
	import { blockIdx, moeTokenIndex, moeModel } from '~/store';
	import { vitRecording, vitSelectedPosition } from '$lib/vit/store';
	import { vlmRecording, vlmPhase, vlmLayer, vlmSelectedPosition } from '$lib/vlm/store';
	import { hybridRecording, hybridLayer } from '$lib/hybrid/store';
	import { row, scalar } from '$lib/vit/types';
	export let stage: string;
	const dispatch = createEventDispatcher<{ close: void }>();
	let qkv = 'query';
	$: architecture = $moeModel?.architecture;
	$: hybrid = architecture === 'hybrid';
	$: vit = architecture === 'vit';
	$: data = hybrid ? $hybridLayer : vit ? $vitRecording?.layers[$blockIdx] : $vlmLayer;
	$: document = hybrid
		? $hybridRecording?.document
		: vit
			? $vitRecording?.document
			: $vlmRecording?.document.streams[$vlmPhase];
	$: position = hybrid ? $moeTokenIndex : vit ? $vitSelectedPosition : $vlmSelectedPosition;
	$: token = document?.tokens[position] ?? '';
	$: kind =
		stage === 'qkv'
			? qkv
			: stage === 'ffn' || stage === 'ffn-up'
				? 'up'
				: stage === 'ffn-down'
					? 'down'
					: stage === 'ffn-gate' || stage === 'gate-projection'
						? 'gate'
						: stage;
	$: isQkv = ['query', 'key', 'value'].includes(kind);
	$: inputName = isQkv
		? hybrid
			? 'attentionInput'
			: 'norm1'
		: kind === 'projection'
			? 'norm1'
			: kind === 'output'
				? 'gated'
				: kind === 'down'
					? hybrid || (!vit && $vlmPhase === 'language')
						? 'product'
						: 'gelu'
					: hybrid
						? 'ffnInput'
						: 'norm2';
	$: outputName = isQkv
		? kind !== 'value' && data?.descriptor.tensors[kind + 'Before']
			? kind + 'Before'
			: kind
		: kind === 'projection'
			? 'mambaProjection'
			: kind === 'output'
				? 'mambaOutput'
				: hybrid && kind === 'gate'
					? 'gateFFN'
					: kind;
	$: weightsName =
		kind === 'projection' ? 'inWeights' : kind === 'output' ? 'outWeights' : kind + 'Weights';
	$: input = data ? row(data, inputName, position) : [];
	$: outputSpec = data?.descriptor.tensors[outputName];
	$: output =
		data && outputSpec
			? isQkv
				? Array.from({ length: outputSpec.shape[0] }, (_, h) =>
						row(data, outputName, h * (document?.sequenceLength ?? 1) + position)
					).flat()
				: row(data, outputName, position)
			: [];
	$: inputSize = input.length;
	$: outputSize = output.length;
	$: biasName = kind + 'Bias';
	$: hasBias = !!data?.descriptor.tensors[biasName];
	$: productRows = data?.descriptor.tensors[weightsName]
		? Array.from({ length: Math.min(4, data.descriptor.tensors[weightsName].shape[0]) }, (_, c) => {
				const weights = row(data!, weightsName, c);
				const bias = hasBias ? scalar(data!, biasName, c) : 0;
				return {
					c,
					weights,
					bias,
					rest: weights.slice(4).reduce((sum, w, d) => sum + w * input[d + 4], 0),
					reconstructed: weights.reduce((sum, w, d) => sum + w * input[d], bias),
					recorded: output[c]
				};
			})
		: [];
	$: label = `${hybrid ? 'Zamba2' : vit ? 'DeiT-tiny' : 'SmolVLM · ' + $vlmPhase} · Block ${$blockIdx + 1}`;
	$: operationLabel = isQkv
		? `${kind[0].toUpperCase() + kind.slice(1)} projection`
		: kind === 'projection'
			? 'Mamba input projection'
			: kind === 'output'
				? 'Mamba output projection'
				: `${kind[0].toUpperCase() + kind.slice(1)} projection`;
	const fmt = (n: number) => (Number.isFinite(n) ? n.toFixed(6) : '—');
</script>

{#if data && inputSize && outputSize}
	{#key `${architecture}-${$vlmPhase}-${$blockIdx}-${position}-${kind}`}
		<AnimatedMatrixPopup
			{label}
			{token}
			bias={hasBias}
			projection={{
				label: operationLabel,
				input:
					kind === 'down'
						? 'Activated input H'
						: isQkv && hybrid
							? 'RMSNorm([hidden; embedding])'
							: 'Normalized input X',
				weight:
					hybrid && (isQkv || ['up', 'gate'].includes(kind))
						? `(W${isQkv ? kind[0] : kind} + adapter)ᵀ`
						: kind === 'projection'
							? 'Winᵀ'
							: kind === 'output'
								? 'Woutᵀ'
								: `W${isQkv ? kind[0] : kind}ᵀ`,
				output: isQkv
					? kind[0].toUpperCase()
					: kind === 'projection'
						? 'Gate + xBC + Δ'
						: 'Projected output',
				inputSize,
				outputSize
			}}
			on:close={() => dispatch('close')}
		>
			{#if stage === 'qkv'}<div class="projection-options" aria-label="Projection">
					{#each ['query', 'key', 'value'] as name}<button
							class:active={qkv === name}
							on:click={() => (qkv = name)}>{name[0].toUpperCase() + name.slice(1)}</button
						>{/each}
				</div>{/if}
			<details class="recorded-products">
				<summary>Recorded values and products · {token}</summary>
				<p>
					Recorded block {$blockIdx + 1}, position {position}. Each animated cell represents a group
					of coordinates.
				</p>
				{#each productRows as r}<p class="product">
						[{r.c}] = {#each r.weights.slice(0, 4) as w, d}({fmt(input[d])} × {fmt(w)}) +
						{/each}{fmt(r.rest)} <span>(remaining {inputSize - 4} products)</span>{hasBias
							? ' + bias ' + fmt(r.bias)
							: ''} = {fmt(r.reconstructed)}<br />Recorded output:
						<strong>{fmt(r.recorded)}</strong>
					</p>{/each}
				{#if hybrid && ['up', 'gate'].includes(kind)}<p>
						Weights above are the shared base projection. The recorded output also includes this
						block’s learned low-rank adapter; its contribution is recorded output minus the base
						product.
					</p>{/if}
				{#if !productRows.length}<p>Recorded input: {input.slice(0, 8).map(fmt).join(', ')}</p>
					<p>Recorded output: {output.slice(0, 8).map(fmt).join(', ')}</p>
					<p>
						This capture retains the projection outputs including the block adapter; its complete
						weight matrix is not stored.
					</p>{/if}
			</details>
		</AnimatedMatrixPopup>
	{/key}
{/if}

<style>
	.recorded-products {
		color: #6b7280;
		border-top: 1px solid #e5e7eb;
		padding-top: 12px;
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
	}
	.product span {
		color: #9ca3af;
	}
	.projection-options {
		display: flex;
		gap: 8px;
		margin-bottom: 16px;
	}
	.projection-options button {
		padding: 5px 14px;
		border: 1px solid #e5e7eb;
		border-radius: 4px;
		color: #6b7280;
	}
	.projection-options button.active {
		background: #eeeaff;
		border-color: #c4b5fd;
	}
</style>
