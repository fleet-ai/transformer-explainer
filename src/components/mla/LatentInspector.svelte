<script lang="ts">
	import TokenInspectorLayout from '../moe/TokenInspectorLayout.svelte';
	import TokenPicker from '../moe/TokenPicker.svelte';
	import { createEventDispatcher, onMount } from 'svelte';
	import { base } from '$app/paths';
	import {
		attentionHeadIdx,
		blockIdx,
		moeExample,
		moeModel,
		moeTokenIndex,
		moeAttentionOperations
	} from '~/store';
	import { loadAttentionOperations } from '$lib/moe/attention';
	import { traceAssetUrl } from '$lib/moe/live';
	import ProjectionOperation from '../common/ProjectionOperation.svelte';
	import { tokenLabel } from '$lib/moe/format';
	export let stage = 'compression';
	const dispatch = createEventDispatcher<{ close: void; rope: void; attention: void }>();
	const stages = [
		{ id: 'query', label: 'Query projection' },
		{ id: 'compression', label: 'KV compression' },
		{ id: 'expansion', label: 'K / V expansion' },
		{ id: 'attention', label: 'Attention output' }
	];
	let error = '';
	let loading = false;
	let request = 0;
	let coordinate = 0;
	let panel: HTMLDivElement;
	$: layer =
		$moeAttentionOperations?.example.id === $moeExample?.id
			? $moeAttentionOperations?.layers[$blockIdx]
			: undefined;
	$: model = $moeModel;
	$: token = $moeTokenIndex;
	$: head = $attentionHeadIdx;
	$: tokenOptions = ($moeExample?.tokens ?? []).map((t) => ({
		value: t.index,
		label: `${t.index}: ${tokenLabel(t.text)}`
	}));
	$: selected = stages.find((s) => s.id === stage) ?? stages[1];
	$: vectors =
		stage === 'query'
			? [
					{
						label: `Q content · head ${head + 1}`,
						values: layer?.queryContent?.[head]?.[token],
						color: 'query'
					},
					{
						label: 'Q rotary · before RoPE',
						values: layer?.queryBefore?.[head]?.[token],
						color: 'query'
					},
					{
						label: 'Q′ rotary · after RoPE',
						values: layer?.queryAfter?.[head]?.[token],
						color: 'query'
					}
				]
			: stage === 'compression'
				? [
						{
							label: 'Raw KV latent · down projection',
							values: layer?.rawLatent?.[token],
							color: 'latent'
						},
						{ label: 'KV latent · RMSNorm', values: layer?.kvLatent?.[token], color: 'latent' },
						{
							label: 'Shared rotary key · before RoPE',
							values: layer?.keyBefore?.[0]?.[token],
							color: 'key'
						},
						{
							label: 'K′ shared rotary · after RoPE',
							values: layer?.keyAfter?.[0]?.[token],
							color: 'key'
						}
					]
				: stage === 'expansion'
					? [
							{
								label: 'Shared normalized KV latent',
								values: layer?.kvLatent?.[token],
								color: 'latent'
							},
							{
								label: `K content · head ${head + 1}`,
								values: layer?.keyContent?.[head]?.[token],
								color: 'key'
							},
							{
								label: `V · head ${head + 1}`,
								values: layer?.value?.[head]?.[token],
								color: 'value'
							}
						]
					: [
							{
								label: `Attention-weighted V · head ${head + 1}`,
								values: layer?.attentionContext?.[head]?.[token],
								color: 'value'
							},
							{
								label: 'Projected attention output',
								values: layer?.attentionOutput?.[token],
								color: 'latent'
							}
						];
	$: projectionRows =
		stage === 'query'
			? [
					{
						label: 'Wq',
						weight: `${(model?.attentionHeads ?? 16) * (model?.attentionHeadSize ?? 192)} × ${model?.hiddenSize ?? 2048}`,
						input: `RMSNorm(x) · ${model?.hiddenSize ?? 2048}D`,
						output: `Q · ${(model?.attentionHeads ?? 16) * (model?.attentionHeadSize ?? 192)}D`,
						color: 'query'
					}
				]
			: stage === 'compression'
				? [
						{
							label: 'Wᴰᴷⱽ',
							weight: `${(model?.kvLatentSize ?? 512) + (model?.rotarySize ?? 64)} × ${model?.hiddenSize ?? 2048}`,
							input: `RMSNorm(x) · ${model?.hiddenSize ?? 2048}D`,
							output: `Latent ${model?.kvLatentSize}D + rotary key ${model?.rotarySize}D`
						}
					]
				: stage === 'expansion'
					? [
							{
								label: `Wᵁᴷⱽ · head ${head + 1}`,
								weight: `${(model?.contentKeySize ?? 128) + (model?.valueHeadSize ?? 128)} × ${model?.kvLatentSize ?? 512}`,
								input: `Normalized latent · ${model?.kvLatentSize}D`,
								output: `K content ${model?.contentKeySize}D + V ${model?.valueHeadSize}D`
							}
						]
					: [];
	$: maximum = Math.max(0, ...vectors.map((v) => (v.values?.length ?? 0) - 1));
	$: if (coordinate > maximum) coordinate = Math.max(0, maximum - 3);
	$: coords = [coordinate, coordinate + 1, coordinate + 2, coordinate + 3];
	$: if ($moeExample) load($moeExample.id);
	async function load(id: string) {
		const current = ++request;
		loading = true;
		error = '';
		try {
			const doc = await loadAttentionOperations(id);
			if (current === request) $moeAttentionOperations = doc;
		} catch (reason) {
			if (current === request)
				error = reason instanceof Error ? reason.message : 'Could not load MLA operations.';
		} finally {
			if (current === request) loading = false;
		}
	}
	function number(value: number | undefined) {
		return value === undefined
			? '—'
			: Math.abs(value) >= 10000 || (value !== 0 && Math.abs(value) < 0.0001)
				? value.toExponential(5)
				: value.toFixed(6);
	}
	function portal(element: HTMLElement) {
		document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	onMount(() => {
		const previous = document.activeElement;
		panel.focus({ preventScroll: true });
		return () => {
			request++;
			if (previous instanceof HTMLElement && previous.isConnected)
				previous.focus({ preventScroll: true });
		};
	});
</script>

<svelte:window
	on:keydown={(e) => {
		if (e.key === 'Escape') dispatch('close');
	}}
/>
<div
	use:portal
	bind:this={panel}
	class="latent-inspector"
	role="dialog"
	aria-modal="false"
	aria-label="Recorded MLA operations"
	tabindex="-1"
>
	<header>
		<div>
			<h2>Multi-head Latent Attention</h2>
			<p>
				DeepSeek-V2-Lite · Block {$blockIdx + 1} of {model?.numLayers} · Head {head + 1} of {model?.attentionHeads}
			</p>
		</div>
		<button class="close" aria-label="Close MLA inspector" on:click={() => dispatch('close')}
			>×</button
		>
	</header>
	<div class="inspector-body">
		<TokenInspectorLayout gap="16px"
			><TokenPicker
				slot="tokens"
				label="MLA token"
				options={tokenOptions}
				bind:value={$moeTokenIndex}
			/>
			<div class="stage-tabs" aria-label="MLA operation">
				{#each stages as item}<button
						class:chosen={stage === item.id}
						aria-pressed={stage === item.id}
						on:click={() => {
							stage = item.id;
							coordinate = 0;
						}}>{item.label}</button
					>{/each}
			</div>
			{#if loading}<p role="status">Loading recorded MLA tensors…</p>{:else if error}<p
					role="alert"
				>
					{error}
					<button class="text-button" on:click={() => $moeExample && load($moeExample.id)}
						>Retry</button
					>
				</p>{:else if layer}
				<div class="explanation">
					{#if projectionRows.length}<ProjectionOperation rows={projectionRows} />{/if}
					{#if stage === 'query'}
						<p>
							Each head splits into {model?.contentKeySize} content coordinates and {model?.rotarySize}
							positional coordinates. Only the positional branch receives RoPE. Concatenating both branches
							gives a {model?.attentionHeadSize}D query.
						</p>
						<button class="text-button" on:click={() => dispatch('rope')}
							>Inspect recorded RoPE rotations →</button
						>
					{:else if stage === 'compression'}
						<p>
							A {model?.hiddenSize}D hidden vector produces a {model?.kvLatentSize}D latent and a
							separate {model?.rotarySize}D rotary key. RMSNorm normalizes the latent. The cache
							stores the normalized latent and rotated key:
							<b>{(model?.kvLatentSize ?? 512) + (model?.rotarySize ?? 64)} coordinates per token</b
							> shared by every head.
						</p>
						<button class="text-button" on:click={() => dispatch('rope')}
							>Inspect the shared rotary key →</button
						>
					{:else if stage === 'expansion'}
						<p>
							Each head has its own learned expansion from the same {model?.kvLatentSize}D latent
							into {model?.contentKeySize}
							content-key and {model?.valueHeadSize} value coordinates. The {model?.rotarySize}D
							rotated key is shared across all heads. Content keys and values bypass RoPE.
						</p>
					{:else}
						<h3>
							o<sub>h</sub> = softmax((q<sup>C</sup>k<sup>Cᵀ</sup> + q<sup>R</sup>k<sup>Rᵀ</sup>) ×
							scale + mask) · V<sub>h</sub>
						</h3>
						<p>
							Content and positional dot products combine before softmax. Each head returns {model?.valueHeadSize}
							coordinates. The {model?.attentionHeads} heads concatenate and pass through W<sup
								>O</sup
							>
							to {model?.hiddenSize}D, followed by residual addition.
						</p>
						<button class="text-button" on:click={() => dispatch('attention')}
							>Inspect scores, causal mask, and softmax →</button
						>
					{/if}
				</div>
				<div class="recording-heading">
					<h3>Recorded values · {selected.label}</h3>
					<div class="coordinates">
						<span>Coordinates {coordinate}–{Math.min(coordinate + 3, maximum)}</span><button
							aria-label="Previous MLA coordinates"
							disabled={coordinate === 0}
							on:click={() => (coordinate = Math.max(0, coordinate - 4))}>‹</button
						><button
							aria-label="Next MLA coordinates"
							disabled={coordinate + 4 > maximum}
							on:click={() => (coordinate = Math.min(maximum, coordinate + 4))}>›</button
						>
					</div>
				</div>
				<div class="tensor-grid" style={`--columns:${vectors.length}`}>
					{#each vectors as vector}<section class={vector.color}>
							<h4>{vector.label}</h4>
							<small>{vector.values?.length ?? 0} recorded coordinates</small
							>{#each coords as index}<div class="value-row">
									<span>[{index}]</span><b>{number(vector.values?.[index])}</b>
								</div>{/each}
						</section>{/each}
				</div>
				<p class="footnote">
					Matrix boxes show shapes; values below each stage are recorded tensors. All {model?.numLayers}
					transformer blocks execute before the vocabulary prediction. The diagram reuses this block's
					recorded tensors; changing a head or token selects a different slice of that computation.
					<a
						href={traceAssetUrl($moeExample?.id ?? '', 'attention', base)}
						target="_blank"
						rel="noreferrer">Trace JSON ↗</a
					>
				</p>
			{/if}
		</TokenInspectorLayout>
	</div>
</div>

<style>
	.latent-inspector {
		position: fixed;
		right: 16px;
		bottom: 16px;
		width: min(1080px, calc(100vw - 32px));
		height: min(710px, calc(100dvh - 32px));
		z-index: 1200;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		border: 1px solid #ded5eb;
		border-radius: 8px;
		background: white;
		box-shadow: 0 8px 34px #33224b30;
		color: #665775;
		font:
			13px/1.5 Arial,
			Helvetica,
			sans-serif;
		outline: none;
		box-sizing: border-box;
	}
	header {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		flex: 0 0 auto;
		background: #faf8fc;
		border-bottom: 1px solid #e9e2ef;
		padding: 12px 18px;
	}
	h2 {
		font-size: 20px;
		font-weight: 600;
		color: #624685;
		margin: 0;
	}
	header p {
		margin: 3px 0 0;
		color: #91849e;
		font-size: 12px;
	}
	.close {
		width: 30px;
		height: 30px;
		background: white;
		color: #7b6c8e;
		border: 1px solid #ded5eb;
		border-radius: 6px;
		font-size: 24px;
		line-height: 1;
		cursor: pointer;
	}
	.inspector-body {
		padding: 16px 18px;
		flex: 1;
		min-height: 0;
		overflow: auto;
		display: flex;
		flex-direction: column;
		gap: 16px;
	}
	.stage-tabs {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		padding: 4px;
		border: 1px solid #e9dff0;
		border-radius: 6px;
		background: #fcfaff;
	}
	.stage-tabs button {
		padding: 9px 4px;
		color: #89729d;
		border: 1px solid transparent;
		border-radius: 4px;
		background: transparent;
		font: inherit;
		cursor: pointer;
	}
	.stage-tabs .chosen {
		background: #f1e6f8;
		border-color: #ddc7eb;
		color: #81549f;
	}
	.explanation {
		min-height: 130px;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.explanation h3 {
		font:
			18px Georgia,
			serif;
		color: #805a9b;
		margin: 0 0 6px;
	}
	.explanation p {
		margin: 0 0 6px;
		color: #8a7897;
	}
	.text-button,
	a {
		color: #8b59b6;
		text-decoration: none;
		border: 0;
		background: none;
		padding: 0;
		font: inherit;
		cursor: pointer;
	}
	.recording-heading {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
	}
	.recording-heading h3 {
		margin: 0;
		font-size: 14px;
		color: #7d6096;
	}
	.coordinates {
		display: flex;
		align-items: center;
		gap: 6px;
		color: #9986aa;
		font-size: 11px;
	}
	.coordinates button {
		border: 1px solid #ddc7eb;
		border-radius: 4px;
		background: #fcfaff;
		width: 24px;
		height: 24px;
		color: #8b59b6;
		cursor: pointer;
	}
	.coordinates button:disabled {
		opacity: 0.35;
		cursor: default;
	}
	.tensor-grid {
		display: grid;
		grid-template-columns: repeat(var(--columns), minmax(0, 1fr));
		gap: 14px;
	}
	.tensor-grid h4 {
		font-size: 12px;
		font-weight: 500;
		margin: 0;
		min-height: 36px;
		color: #8d729f;
	}
	.tensor-grid small {
		display: block;
		font-size: 10px;
		color: #a18fae;
		margin: 0 0 8px;
	}
	.value-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		background: #f7f1fa;
		padding: 9px 10px;
		border-radius: 3px;
		margin: 4px 0;
		color: #9169ac;
	}
	.value-row span {
		font-size: 10px;
		opacity: 0.7;
	}
	.value-row b {
		font:
			12px ui-monospace,
			SFMono-Regular,
			monospace;
	}
	.query .value-row {
		background: #eff5fd;
		color: #6d99cd;
	}
	.key .value-row {
		background: #fdf0f1;
		color: #bf8191;
	}
	.value .value-row {
		background: #eff8f3;
		color: #6ca58a;
	}
	.footnote {
		font-size: 11px;
		color: #a08eae;
		margin: auto 0 0;
	}
	button:focus-visible,
	a:focus-visible {
		outline: 2px solid #b596cf;
		outline-offset: 3px;
	}
	@media (max-height: 720px) {
		.inspector-body {
			gap: 10px;
			padding: 12px 16px;
		}
		.explanation {
			min-height: 100px;
		}
		.value-row {
			padding: 6px 8px;
		}
		.stage-tabs button {
			padding: 6px 4px;
		}
	}
</style>
