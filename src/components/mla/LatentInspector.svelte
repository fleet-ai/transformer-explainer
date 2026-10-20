<script lang="ts">
	import LatentProjectionFlow from './LatentProjectionFlow.svelte';
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
	import LatentProjectionPopup from './LatentProjectionPopup.svelte';
	import { mlaProjectionView } from '$lib/mla/projections.js';
	import { tokenLabel } from '$lib/moe/format';
	export let stage = 'compression';
	const dispatch = createEventDispatcher<{ close: void; rope: 'query' | 'key'; attention: void }>();
	let error = '';
	let loading = false;
	let request = 0;
	let coordinate = 0;
	let multiplicationOpen = false;
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
	$: projection = mlaProjectionView(layer, model, head, token, stage);
	$: vectors = projection.vectors;
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
		if (e.key === 'Escape' && !multiplicationOpen) dispatch('close');
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
			<h2>MLA projections</h2>
			<p>
				DeepSeek-V2-Lite · Block {$blockIdx + 1} of {model?.numLayers} · Before head split and RoPE
			</p>
		</div>
		<button class="close" aria-label="Close MLA inspector" on:click={() => dispatch('close')}
			>×</button
		>
	</header>
	<div class="inspector-body">
		<TokenInspectorLayout gap="6px"
			><TokenPicker
				slot="tokens"
				label="MLA token"
				options={tokenOptions}
				bind:value={$moeTokenIndex}
			/>
			<LatentProjectionFlow
				selected={stage}
				hiddenSize={model?.hiddenSize ?? 2048}
				heads={model?.attentionHeads ?? 16}
				contentSize={model?.contentKeySize ?? 128}
				rotarySize={model?.rotarySize ?? 64}
				latentSize={model?.kvLatentSize ?? 512}
				valueSize={model?.valueHeadSize ?? 128}
				on:attention={() => dispatch('attention')}
				on:select={(e) => {
					stage = e.detail;
					coordinate = 0;
					multiplicationOpen = true;
				}}
			/>
			{#if loading}<p role="status">Loading recorded MLA tensors…</p>{:else if error}<p
					role="alert"
				>
					{error}
					<button class="text-button" on:click={() => $moeExample && load($moeExample.id)}
						>Retry</button
					>
				</p>{:else if layer}
				{#if multiplicationOpen}
					{#key stage}
						<LatentProjectionPopup
							{projection}
							token={tokenLabel($moeExample?.tokens[token]?.text ?? '')}
							on:close={() => (multiplicationOpen = false)}
						>
							<details class="recorded-details">
								<summary>Recorded MLA values and details</summary>
								{#if stage === 'query'}
									<p>
										Each head splits into {model?.contentKeySize} content coordinates and {model?.rotarySize}
										positional coordinates. Only the positional branch receives RoPE. Concatenating both
										branches gives a {model?.attentionHeadSize}D query.
									</p>
								{:else if stage === 'compression'}
									<p>
										A {model?.hiddenSize}D hidden vector produces a {model?.kvLatentSize}D latent
										and a separate {model?.rotarySize}D rotary key. RMSNorm normalizes the latent.
										The cache stores the normalized latent and rotated key:
										<b
											>{(model?.kvLatentSize ?? 512) + (model?.rotarySize ?? 64)} coordinates per token</b
										> shared by every head.
									</p>
								{:else if stage === 'norm'}
									<p>
										RMSNorm preserves the {model?.kvLatentSize}D latent; the shared positional key
										bypasses this normalization.
									</p>
								{:else if ['expansion', 'key', 'value'].includes(stage)}
									<p>
										Each head has its own learned expansion from the same {model?.kvLatentSize}D
										latent into {model?.contentKeySize}
										content-key and {model?.valueHeadSize} value coordinates. The {model?.rotarySize}D
										rotated key is shared across all heads. Content keys and values bypass RoPE.
									</p>
								{:else}
									<p>
										Content and positional dot products combine before softmax. Each head returns {model?.valueHeadSize}
										coordinates. The {model?.attentionHeads} heads concatenate and pass through W<sup
											>O</sup
										>
										to {model?.hiddenSize}D, followed by residual addition.
									</p>
								{/if}
								<div class="recording-heading">
									<h3>Recorded values · {projection.label}</h3>
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
									Full-model recording · {model?.numLayers} blocks ·

									<a
										href={traceAssetUrl($moeExample?.id ?? '', 'attention', base)}
										target="_blank"
										rel="noreferrer">Trace JSON ↗</a
									>
								</p>
							</details>
						</LatentProjectionPopup>
					{/key}
				{/if}
			{/if}
		</TokenInspectorLayout>
	</div>
</div>

<style>
	.latent-inspector {
		position: fixed;
		right: 16px;
		bottom: 16px;
		width: min(1180px, calc(100vw - 32px));
		max-height: calc(100dvh - 32px);
		z-index: 1200;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		border: 1px solid #e5e7eb;
		border-radius: 8px;
		background: white;
		box-shadow: 0 8px 34px #00000026;
		color: #4b5563;
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
		background: #f3f4f6;
		border-bottom: 1px solid #e5e7eb;
		padding: 12px 18px;
	}
	h2 {
		font-size: 20px;
		font-weight: 600;
		color: #111827;
		margin: 0;
	}
	header p {
		margin: 3px 0 0;
		color: #9ca3af;
		font-size: 12px;
	}
	.close {
		width: 30px;
		height: 30px;
		background: white;
		color: #6b7280;
		border: 1px solid #e5e7eb;
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
	.recorded-details {
		border-top: 1px solid #e5e7eb;
		padding-top: 12px;
	}
	.recorded-details summary {
		color: #6b7280;
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
		.value-row {
			padding: 6px 8px;
		}
	}
</style>
