<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { attentionHeadIdx, blockIdx, moeModel, moeTokenIndex, tokens } from '~/store';
	import { tokenLabel } from '$lib/moe/format';
	import { vlmPhase, vlmRecording, vlmVisiblePositions, vlmVisualToken } from '$lib/vlm/store';
	import AnimatedMatrixPopup from '../common/AnimatedMatrixPopup.svelte';
	import ExpertElementwisePopup from '../moe/ExpertElementwisePopup.svelte';
	import ExpertProjectionPopup from '../moe/ExpertProjectionPopup.svelte';
	import RoPEPopup from '../moe/RoPEPopup.svelte';
	import MlaNormPopup from '../mla/MlaNormPopup.svelte';
	import VlmShufflePopup from './VlmShufflePopup.svelte';
	import VlmMergePopup from './VlmMergePopup.svelte';
	import VlmAttentionPopup from './VlmAttentionPopup.svelte';

	/** Every vision-language operation opens a formula popup; no recorded-value tables. */
	export let stage: string;
	const dispatch = createEventDispatcher<{ close: void }>();
	const PROJECTIONS = ['query', 'key', 'value'];
	let qkv = 'query';
	const close = () => dispatch('close');
	$: projectionStep = (stage === 'gate-projection' ? 'gate' : stage === 'down' ? 'down' : 'up') as
		| 'up'
		| 'gate'
		| 'down';

	$: document = $vlmRecording?.document;
	$: vision = $vlmPhase === 'vision';
	$: model = $moeModel;
	$: hidden = model?.hiddenSize ?? (vision ? 768 : 576);
	$: intermediate = model?.intermediateSize ?? (vision ? 3072 : 1536);
	$: headSize = model?.attentionHeadSize ?? 64;
	$: heads = model?.attentionHeads ?? 12;
	$: kvHeads = model?.keyValueHeads ?? heads;
	$: token = tokenLabel($tokens[$moeTokenIndex] ?? 'x');
	$: label = `SmolVLM · ${vision ? 'vision' : 'language'} · Block ${$blockIdx + 1}`;
	$: image = document?.image;
	$: factor = image?.pixelShuffleFactor ?? 4;
	$: visionHidden = document?.streams.vision.model.hiddenSize ?? 768;
	$: languageHidden = document?.streams.language.model.hiddenSize ?? 576;
	$: packed = factor * factor * visionHidden;
	$: patchPixels = (image?.patchSize ?? 16) ** 2 * 3;
	$: language = document?.streams.language;
	$: qkvSize = (qkv === 'query' ? heads : kvHeads) * headSize;
	$: norm =
		stage === 'post-norm'
			? {
					kind: 'layernorm' as const,
					title: 'SmolVLM · vision · Post-LayerNorm',
					input: 'Block 12 output',
					output: 'Vision output',
					size: visionHidden
				}
			: {
					kind: vision ? ('layernorm' as const) : ('rmsnorm' as const),
					title: `${label} · ${vision ? 'LayerNorm' : 'RMSNorm'} ${stage === 'norm2' ? 2 : 1}`,
					input: stage === 'norm2' ? 'After attention r₁' : 'Block input x',
					output: stage === 'norm2' ? 'Feed-forward input' : 'Attention input',
					size: hidden
				};
</script>

{#if stage === 'embedding' && vision}
	<AnimatedMatrixPopup
		label="SmolVLM · vision"
		{token}
		bias
		projection={{
			label: 'Patch embedding',
			input: `Patch pixels ${image?.patchSize ?? 16}×${image?.patchSize ?? 16}×3`,
			weight: 'Wpatchᵀ',
			output: 'Patch embedding',
			inputSize: patchPixels,
			outputSize: visionHidden
		}}
		on:close={close}
	>
		<p class="vlm-popup-note">
			Then the learned position embedding of this patch is added: e<sub>i</sub> = W<sub>patch</sub>
			x<sub>i</sub> + b + p<sub>i</sub>. The image is resized to {image?.inputSize ?? 256} × {image?.inputSize ??
				256}, giving {(image?.gridSize ?? 16) ** 2} patches and no class token.
		</p>
	</AnimatedMatrixPopup>
{:else if (stage === 'embedding' || stage === 'merge') && language && document}
	<VlmMergePopup
		tokens={language.tokens}
		imagePositions={document.imageTokenPositions}
		hiddenSize={languageHidden}
		vocabSize={language.model.vocabSize}
		selected={vision ? -1 : ($vlmVisiblePositions[$moeTokenIndex] ?? -1)}
		on:close={close}
	/>
{:else if stage === 'connector' || stage === 'shuffle'}
	<VlmShufflePopup
		visualToken={$vlmVisualToken}
		gridSize={image?.gridSize ?? 16}
		{factor}
		hiddenSize={visionHidden}
		on:close={close}
	/>
{:else if stage === 'projector'}
	{#key $vlmVisualToken}
		<AnimatedMatrixPopup
			label="SmolVLM · Connector"
			token={`image ${$vlmVisualToken}`}
			projection={{
				label: 'Projection into the language width',
				input: 'Packed block v',
				weight: 'Wprojᵀ',
				output: 'Image token c',
				inputSize: packed,
				outputSize: languageHidden
			}}
			on:close={close}
		>
			<p class="vlm-popup-note">
				One bias-free matrix, shared by all 16 image tokens. Its output is the language input at
				position {document?.imageTokenPositions[$vlmVisualToken] ?? '—'}, unchanged.
			</p>
		</AnimatedMatrixPopup>
	{/key}
{:else if stage === 'post-norm' || stage === 'norm1' || stage === 'norm2'}
	<MlaNormPopup
		kind={norm.kind}
		title={norm.title}
		inputLabel={norm.input}
		outputLabel={norm.output}
		size={norm.size}
		epsilon={stage === 'post-norm' ? 1e-6 : (model?.rmsNormEps ?? 1e-6)}
		{token}
		on:close={close}
	>
		<p class="vlm-popup-note">
			{norm.kind === 'layernorm'
				? 'γ and β are a learned scale and shift per coordinate.'
				: 'γ is a learned scale per coordinate; RMSNorm subtracts no mean and adds no shift.'}
		</p>
	</MlaNormPopup>
{:else if stage === 'qkv'}
	{#key `${qkv}-${$moeTokenIndex}-${$blockIdx}`}
		<AnimatedMatrixPopup
			{label}
			{token}
			bias={vision}
			projection={{
				label: `${qkv[0].toUpperCase()}${qkv.slice(1)} projection`,
				input: 'Normalized input X',
				weight: `W${qkv[0]}ᵀ`,
				output: qkv[0].toUpperCase(),
				inputSize: hidden,
				outputSize: qkvSize
			}}
			on:close={close}
		>
			<div class="vlm-popup-options" role="group" aria-label="Projection">
				{#each PROJECTIONS as name}<button
						type="button"
						class:active={qkv === name}
						aria-pressed={qkv === name}
						on:click={() => (qkv = name)}>{name[0].toUpperCase() + name.slice(1)}</button
					>{/each}
			</div>
			<p class="vlm-popup-note">
				{qkvSize} = {qkv === 'query' ? heads : kvHeads} heads × {headSize}.{kvHeads < heads
					? ` ${heads} query heads share ${kvHeads} key/value heads.`
					: ''}{vision ? '' : ' Q and K are rotated by RoPE next; V is not.'}
			</p>
		</AnimatedMatrixPopup>
	{/key}
{:else if stage === 'rope' && !vision}
	<RoPEPopup on:close={close} />
{:else if stage === 'attention' && model}
	<VlmAttentionPopup
		{label}
		{token}
		sequenceLength={document?.streams[$vlmPhase].sequenceLength ?? $tokens.length}
		{headSize}
		head={$attentionHeadIdx}
		{heads}
		keyValueHeads={kvHeads}
		causal={!vision}
		rotary={!vision}
		on:close={close}
	/>
{:else if stage === 'residual'}
	<ExpertElementwisePopup
		kind="residual"
		intermediateSize={hidden}
		{token}
		{label}
		on:close={close}
	>
		<p class="vlm-popup-note">
			Twice per block: r₁ = x + Attention(Norm₁(x)), then y = r₁ + MLP(Norm₂(r₁)).
		</p>
	</ExpertElementwisePopup>
{:else if stage === 'up' || stage === 'down' || stage === 'gate-projection'}
	<ExpertProjectionPopup
		step={projectionStep}
		hiddenSize={hidden}
		intermediateSize={intermediate}
		{token}
		bias={vision}
		outputName="Feed-forward output"
		downInput={vision ? 'GELU output' : 'Gated input'}
		label={`${label} · Feed-forward`}
		on:close={close}
	/>
{:else if stage === 'gate' || stage === 'silu' || stage === 'gelu'}
	<ExpertElementwisePopup
		kind={stage === 'gate' ? 'product' : vision ? 'gelu' : 'silu'}
		intermediateSize={intermediate}
		{token}
		label={`${label} · Feed-forward`}
		on:close={close}
	/>
{:else if stage === 'output' && language}
	<AnimatedMatrixPopup
		label="SmolVLM · language"
		token={tokenLabel(language.tokens[language.sequenceLength - 1] ?? '')}
		projection={{
			label: 'Vocabulary projection',
			input: 'Final RMSNorm h',
			weight: 'Wvocabᵀ',
			output: 'Logits',
			inputSize: languageHidden,
			outputSize: language.model.vocabSize
		}}
		on:close={close}
	>
		<p class="vlm-popup-note">
			After all {language.model.numLayers} blocks and the final RMSNorm, the last position’s vector scores
			every vocabulary entry; softmax turns the logits into the probabilities on the page.
		</p>
	</AnimatedMatrixPopup>
{/if}

<style>
	:global(.vlm-popup-note) {
		color: #6b7280;
		font-size: 13px;
		line-height: 1.5;
		margin: 6px 0 0;
		text-align: center;
	}
	:global(.vlm-popup-options) {
		display: flex;
		justify-content: center;
		gap: 8px;
		margin-bottom: 12px;
	}
	:global(.vlm-popup-options button) {
		width: auto;
		height: auto;
		padding: 5px 14px;
		border: 1px solid #e5e7eb;
		border-radius: 4px;
		color: #6b7280;
		font-size: 13px;
	}
	:global(.vlm-popup-options button.active) {
		background: #eeeaff;
		border-color: #c4b5fd;
	}
</style>
