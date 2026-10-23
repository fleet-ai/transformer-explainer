<script lang="ts">
	import AnimatedMatrixPopup from '../common/AnimatedMatrixPopup.svelte';
	import HybridOperationPopup from './HybridOperationPopup.svelte';
	import { blockIdx, moeTokenIndex, attentionHeadIdx } from '~/store';
	import { tokenLabel } from '$lib/moe/format';
	import { hybridRecording, hybridInspector, hybridMambaView } from '$lib/hybrid/store';

	/** Every operation on the hybrid page opens a formula-only popup from here. */
	const QKV = ['query', 'key', 'value'] as const;
	const FFN = ['up', 'gate'] as const;
	let qkv: (typeof QKV)[number] = 'query';
	let ffn: (typeof FFN)[number] = 'up';
	$: document = $hybridRecording?.document;
	$: model = document?.model;
	$: tokens = (document?.tokens ?? []).map(tokenLabel);
	$: token = tokens[$moeTokenIndex] ?? '';
	$: isHybridBlock = document?.layers[$blockIdx]?.type === 'hybrid';
	$: hidden = model?.hiddenSize ?? 2048;
	$: inner = (model?.mambaHeads ?? 64) * (model?.mambaHeadSize ?? 64);
	$: state = model?.stateSize ?? 128;
	$: heads = model?.mambaHeads ?? 64;
	$: projected = 2 * inner + 2 * state + heads;
	$: stage = $hybridInspector ?? '';
	$: [name, detail] = stage.split(':');
	/** The same band or column means a different operation in each half of a hybrid block. */
	$: kind = $hybridMambaView
		? ({
				attention: 'scan',
				'attention-out': 'scan',
				'ffn-up': 'gate',
				'ffn-down': 'out-proj',
				'residual-start': 'residual',
				'residual-end': 'residual'
			}[name] ?? name)
		: ({ 'residual-start': 'residual', 'residual-end': 'residual' }[name] ?? name);
	$: label = `Zamba2 · Block ${$blockIdx + 1}`;
	$: mixerInput = isHybridBlock ? 'RMSNorm(x + t)' : 'RMSNorm(x)';
	function close() {
		hybridInspector.set(null);
	}
	type Projection = {
		label: string;
		input: string;
		weight: string;
		output: string;
		inputSize: number;
		outputSize: number;
	};
	$: projection = ((): Projection | null => {
		switch (kind) {
			case 'in-proj':
				return {
					label: 'Mamba input projection',
					input: mixerInput,
					weight: 'Winᵀ',
					output: 'z · x · B · C · dt',
					inputSize: hidden,
					outputSize: projected
				};
			case 'out-proj':
				return {
					label: 'Mamba output projection',
					input: 'Normalized gated y',
					weight: 'Woutᵀ',
					output: 'Mixer output',
					inputSize: inner,
					outputSize: hidden
				};
			case 'qkv':
				return {
					label: `${qkv[0].toUpperCase()}${qkv.slice(1)} projection`,
					input: 'RMSNorm([x ; e₀])',
					weight: `(W${qkv[0]} + adapter)ᵀ`,
					output: qkv[0].toUpperCase(),
					inputSize: 2 * hidden,
					outputSize: (model?.attentionHeads ?? 32) * (model?.attentionHeadSize ?? 128)
				};
			case 'attention-out':
				return {
					label: 'Attention output projection',
					input: 'Concatenated heads',
					weight: 'Woᵀ',
					output: 'Attention output',
					inputSize: (model?.attentionHeads ?? 32) * (model?.valueHeadSize ?? 128),
					outputSize: hidden
				};
			case 'ffn-up':
				return {
					label: ffn === 'up' ? 'Up projection' : 'Gate projection',
					input: 'RMSNorm(attention output)',
					weight: ffn === 'up' ? '(Wup + adapter)ᵀ' : '(Wgate + adapter)ᵀ',
					output: ffn === 'up' ? 'u' : 'g',
					inputSize: hidden,
					outputSize: model?.intermediateSize ?? 8192
				};
			case 'ffn-down':
				return {
					label: 'Down projection',
					input: 'Hidden h = GELU(g) ⊙ u',
					weight: 'Wdownᵀ',
					output: 'FFN output',
					inputSize: model?.intermediateSize ?? 8192,
					outputSize: hidden
				};
			case 'linear':
				return {
					label: 'Shared-transformer linear',
					input: 'FFN output',
					weight: 'Wlinearᵀ',
					output: 't',
					inputSize: hidden,
					outputSize: hidden
				};
			case 'classifier':
				return {
					label: 'Vocabulary projection',
					input: 'Final RMSNorm',
					weight: 'Wembeddingᵀ (tied)',
					output: 'Logits',
					inputSize: hidden,
					outputSize: model?.vocabSize ?? 32000
				};
		}
		return null;
	})();
	$: notes = {
		'in-proj': `One projection, split in its real order: z ${inner} (the gate), x ${inner}, B ${state}, C ${state}, and dt ${heads}, one per head. Only x, B and C go through the convolution.`,
		'out-proj': `The ${inner} gated values of all ${heads} heads are mixed back to the ${hidden}D residual stream.`,
		qkv: `The hidden state x and the original embedding e₀ are joined (${2 * hidden}D) and normalized. Each of the six hybrid blocks adds its own low-rank adapter to the shared weights. Q and K are rotated by RoPE next; V is not.`,
		'attention-out': `The ${model?.attentionHeads ?? 32} head contexts are joined and projected back to ${hidden}D. There is no residual inside the shared transformer.`,
		'ffn-up': `Wup and Wgate both read the same normalized vector; the block's adapter is added to the shared weights.`,
		'ffn-down': `There is no residual here: the result goes on to the linear layer.`,
		linear: `Each hybrid block has its own linear layer. Its output t is added to x only on the way into this block's Mamba mixer; the residual stream keeps x.`,
		classifier: `The output matrix is the token embedding matrix itself (tied weights). Softmax over the ${(model?.vocabSize ?? 32000).toLocaleString('en-US')} logits gives the probabilities.`
	} as Record<string, string>;
</script>

{#if document && projection}
	{#key `${stage}-${$blockIdx}-${$moeTokenIndex}-${qkv}-${ffn}`}
		<AnimatedMatrixPopup {projection} {token} {label} on:close={close}>
			{#if kind === 'qkv'}<div class="options" aria-label="Projection">
					{#each QKV as option}<button
							class:active={qkv === option}
							aria-pressed={qkv === option}
							on:click={() => (qkv = option)}>{option[0].toUpperCase() + option.slice(1)}</button
						>{/each}
				</div>{:else if kind === 'ffn-up'}<div class="options" aria-label="Projection">
					{#each FFN as option}<button
							class:active={ffn === option}
							aria-pressed={ffn === option}
							on:click={() => (ffn = option)}>{option === 'up' ? 'Wup' : 'Wgate'}</button
						>{/each}
				</div>{/if}
			{#if notes[kind]}<p class="popup-note">{notes[kind]}</p>{/if}
		</AnimatedMatrixPopup>
	{/key}
{:else if document && kind}
	{#key `${stage}-${$blockIdx}-${$moeTokenIndex}`}
		{#if kind === 'conv'}<HybridOperationPopup
				kind="conv"
				title={`${label} · Causal convolution (conv1d)`}
				{token}
				{tokens}
				position={$moeTokenIndex}
				size={inner + 2 * state}
				note={`Shown for ${detail === 'key' ? 'B' : detail === 'query' ? 'C' : 'x'}. The same convolution runs on x (${inner}), B (${state}) and C (${state}), each channel with its own kernel, then SiLU.`}
				on:close={close}
			/>{:else if kind === 'delta'}<HybridOperationPopup
				kind="delta"
				title={`${label} · Step size Δ and decay`}
				{token}
				size={heads}
				inputLabel="dt + bias"
				otherLabel="one Δ per head"
				outputLabel="Δ"
				note={`Each of the ${heads} heads gets one Δ per token and has one learned A < 0, so e^(ΔA) is between 0 and 1: a large Δ forgets more of the old state and writes more of the new token.`}
				on:close={close}
			/>{:else if kind === 'scan'}<HybridOperationPopup
				kind="scan"
				title={`${label} · Selective scan · head ${$attentionHeadIdx + 1}`}
				{token}
				{tokens}
				position={$moeTokenIndex}
				size={model?.mambaHeadSize ?? 64}
				{heads}
				note="The circles on the head card are M for this head, computed from the recorded B, C, Δ and A. Multiplying M by x gives the same y as running the recurrence token by token."
				on:close={close}
			/>{:else if kind === 'gate'}<HybridOperationPopup
				kind="gate"
				title={`${label} · Gate y ⊙ SiLU(z)`}
				{token}
				size={inner}
				inputLabel="Scan output y"
				otherLabel="z comes from in_proj and skips the scan"
				outputLabel="Gated y"
				on:close={close}
			/>{:else if kind === 'gated-norm'}<HybridOperationPopup
				kind="norm"
				title={`${label} · Gated RMSNorm`}
				{token}
				size={inner}
				epsilon={model?.rmsNormEps ?? 1e-5}
				inputLabel="Gated y"
				outputLabel="Normalized"
				note={`The norm runs over all ${inner} values of the gated vector, after the gate.`}
				on:close={close}
			/>{:else if kind === 'ln'}<HybridOperationPopup
				kind="norm"
				title={`${label} · ${detail === 'mlp-ln' ? 'Pre-FFN RMSNorm' : $hybridMambaView ? 'Mamba input RMSNorm' : 'Shared-transformer RMSNorm'}`}
				{token}
				size={detail === 'mlp-ln' || $hybridMambaView ? hidden : 2 * hidden}
				epsilon={model?.rmsNormEps ?? 1e-5}
				inputLabel={detail === 'mlp-ln'
					? 'Attention output'
					: $hybridMambaView
						? isHybridBlock
							? 'x + t'
							: 'Hidden state x'
						: '[x ; e₀]'}
				outputLabel="Normalized"
				note={detail === 'mlp-ln'
					? 'The shared transformer has no residual; the attention output is normalized and fed to the FFN.'
					: $hybridMambaView
						? isHybridBlock
							? 'In a hybrid block the shared transformer output t is added to x first; the residual stream itself keeps x.'
							: ''
						: `The hidden state x and the original embedding e₀ are joined into ${2 * hidden} values before the norm.`}
				on:close={close}
			/>{:else if kind === 'residual'}<HybridOperationPopup
				kind="residual"
				title={`${label} · Residual addition`}
				{token}
				size={hidden}
				inputLabel="Block input x"
				otherLabel="Mixer output"
				outputLabel="Block output"
				note={!$hybridMambaView
					? 'The residual stream x skips the whole shared transformer; the sum happens at the end of the Mamba half.'
					: isHybridBlock
						? 'Only the Mamba output joins the residual stream. The shared transformer output t was used for the mixer input but is not added here.'
						: ''}
				on:close={close}
			/>{:else if kind === 'gelu'}<HybridOperationPopup
				kind="gelu"
				title={`${label} · GELU`}
				{token}
				size={model?.intermediateSize ?? 8192}
				inputLabel="Gate projection g"
				otherLabel={`applied to each of ${(model?.intermediateSize ?? 8192).toLocaleString('en-US')} coordinates`}
				outputLabel="GELU(g)"
				on:close={close}
			/>{:else if kind === 'ffn-product'}<HybridOperationPopup
				kind="product"
				title={`${label} · Gate product ⊙`}
				{token}
				size={model?.intermediateSize ?? 8192}
				inputLabel="Up projection u"
				otherLabel="GELU(g)"
				outputLabel="Hidden h"
				on:close={close}
			/>{:else if kind === 'attention'}<HybridOperationPopup
				kind="attention"
				title={`${label} · Attention · head ${$attentionHeadIdx + 1}`}
				{token}
				{tokens}
				position={$moeTokenIndex}
				size={model?.attentionHeadSize ?? 128}
				note={`Zamba2 scales scores by 1/8. The circles on the head card are this head's recorded attention weights.`}
				on:close={close}
			/>{/if}
	{/key}
{/if}

<style>
	.options {
		display: flex;
		justify-content: center;
		gap: 8px;
		margin-bottom: 12px;
	}
	.options button {
		padding: 5px 14px;
		border: 1px solid #e5e7eb;
		border-radius: 4px;
		background: white;
		color: #6b7280;
		cursor: pointer;
	}
	.options button.active {
		background: #eeeaff;
		border-color: #c4b5fd;
	}
	.popup-note {
		color: #6b7280;
		font-size: 13px;
		line-height: 1.5;
		margin: 6px 0 0;
		text-align: center;
	}
</style>
