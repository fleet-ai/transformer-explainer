<script lang="ts" context="module">
	export type VitProjection = 'qkv' | 'up' | 'down' | 'patch' | 'classifier';
	export const VIT_PROJECTIONS: VitProjection[] = ['qkv', 'up', 'down', 'patch', 'classifier'];
</script>

<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import AnimatedMatrixPopup from '../common/AnimatedMatrixPopup.svelte';
	import { blockIdx } from '~/store';
	import { vitRecording, vitSelectedPosition, vitInspectorStage } from '$lib/vit/store';
	/** DeiT's learned matrix multiplications, as the shared formula-only multiplication popup. */
	export let kind: VitProjection;
	const dispatch = createEventDispatcher<{ close: void }>();
	let qkv: 'query' | 'key' | 'value' = 'query';

	$: document = $vitRecording?.document;
	$: model = document?.model;
	$: hidden = model?.hiddenSize ?? 192;
	$: inner = model?.intermediateSize ?? 768;
	$: heads = model?.attentionHeads ?? 3;
	$: headSize = model?.attentionHeadSize ?? 64;
	$: classes = document?.classLabels.length ?? 1000;
	$: patchSize = document?.image.patchSize ?? 16;
	$: pixels = patchSize * patchSize * 3;
	$: position = kind === 'classifier' ? 0 : $vitSelectedPosition;
	$: token = document?.tokens[position] ?? '';
	$: label =
		kind === 'patch' || kind === 'classifier' ? 'DeiT-tiny' : `DeiT-tiny · Block ${$blockIdx + 1}`;
	$: projection =
		kind === 'qkv'
			? {
					label: `${qkv[0].toUpperCase() + qkv.slice(1)} projection`,
					input: 'LayerNorm(x)',
					weight: `W${qkv[0]}ᵀ`,
					output: qkv[0].toUpperCase(),
					inputSize: hidden,
					outputSize: heads * headSize
				}
			: kind === 'up'
				? {
						label: 'MLP up projection',
						input: 'LayerNorm(x)',
						weight: 'W₁ᵀ',
						output: 'Up u',
						inputSize: hidden,
						outputSize: inner
					}
				: kind === 'down'
					? {
							label: 'MLP down projection',
							input: 'GELU(u)',
							weight: 'W₂ᵀ',
							output: 'MLP output',
							inputSize: inner,
							outputSize: hidden
						}
					: kind === 'patch'
						? {
								label: 'Patch projection',
								input: 'Flattened patch',
								weight: 'Wpatchᵀ',
								output: 'Patch embedding e',
								inputSize: pixels,
								outputSize: hidden
							}
						: {
								label: 'Classifier head',
								input: 'LayerNorm(x₁₂)[CLS]',
								weight: 'Wclsᵀ',
								output: 'Class logits z',
								inputSize: hidden,
								outputSize: classes
							};
</script>

{#key `${kind}-${qkv}-${$blockIdx}-${position}`}
	<AnimatedMatrixPopup {projection} {label} {token} bias on:close={() => dispatch('close')}>
		{#if kind === 'qkv'}<div class="projection-options" role="group" aria-label="Projection">
				{#each ['query', 'key', 'value'] as name}<button
						class:active={qkv === name}
						aria-pressed={qkv === name}
						on:click={() => (qkv = name === 'key' ? 'key' : name === 'value' ? 'value' : 'query')}
						>{name[0].toUpperCase() + name.slice(1)}</button
					>{/each}
			</div>{/if}
		<p class="vit-note">
			{#if kind === 'qkv'}The {heads * headSize}D output splits into {heads} heads of {headSize}D.
				Positions were added once at the input, so Q and K are not rotated.{:else if kind === 'up'}Every
				position uses the same W₁. GELU follows, coordinate by coordinate.{:else if kind === 'down'}Every
				position uses the same W₂. The result is added back to the residual stream.{:else if kind === 'patch'}{position ===
				0
					? '[CLS] skips this step: it is a learned vector, not a patch. '
					: ''}The same Wpatch projects all 196 patches; DeiT runs it as a {patchSize} × {patchSize}
				convolution with stride {patchSize}.{:else}Only position 0 ([CLS]) is read. The final
				LayerNorm runs on every position first, and softmax turns the {classes.toLocaleString(
					'en-US'
				)} logits into class probabilities.{/if}
		</p>
		{#if kind === 'classifier'}<div
				class="projection-options"
				role="group"
				aria-label="Related steps"
			>
				<button on:click={() => vitInspectorStage.set('final-norm')}>Final LayerNorm</button><button
					on:click={() => vitInspectorStage.set('softmax')}>Softmax</button
				>
			</div>{/if}
	</AnimatedMatrixPopup>
{/key}

<style>
	.projection-options {
		display: flex;
		justify-content: center;
		gap: 8px;
		margin: 4px 0 12px;
	}
	.projection-options button {
		width: auto;
		height: auto;
		padding: 5px 14px;
		border: 1px solid #e5e7eb;
		border-radius: 4px;
		color: #6b7280;
		font-size: 13px;
		background: #fff;
		cursor: pointer;
	}
	.projection-options button.active {
		background: #eeeaff;
		border-color: #c4b5fd;
	}
	.vit-note {
		color: #6b7280;
		font-size: 13px;
		line-height: 1.5;
		margin: 6px 0 10px;
		text-align: center;
	}
</style>
