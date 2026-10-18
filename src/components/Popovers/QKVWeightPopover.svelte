<script lang="ts">
	import TokenInspectorLayout from '../moe/TokenInspectorLayout.svelte';
	import TokenPicker from '../moe/TokenPicker.svelte';
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import {
		attentionHeadIdx,
		blockIdx,
		moeExample,
		moeModel,
		moeTokenIndex,
		moeAttentionOperations,
		weightPopover
	} from '~/store';
	import { headMapping, loadAttentionOperations } from '$lib/moe/attention';
	import { traceAssetUrl } from '$lib/moe/live';
	import { tokenLabel } from '$lib/moe/format';
	import Katex from '~/utils/Katex.svelte';
	let panel: HTMLDivElement;
	let error = '';
	let loading = true;
	let request = 0;
	let coordinate = 0;
	$: coordinates = [coordinate, coordinate + 1, coordinate + 2, coordinate + 3];
	$: model = $moeModel;
	$: headSize = model?.attentionHeadSize ?? 64;
	$: mapping = headMapping(
		$attentionHeadIdx,
		model?.attentionHeads ?? 16,
		model?.keyValueHeads ?? 8
	);
	$: layer =
		$moeAttentionOperations?.example.id === $moeExample?.id
			? $moeAttentionOperations?.layers[$blockIdx]
			: undefined;
	$: tokenOptions = ($moeExample?.tokens ?? []).map((t) => ({
		value: t.index,
		label: `${t.index}: ${tokenLabel(t.text)}`
	}));
	$: hiddenSize = model?.hiddenSize ?? 1024;
	$: querySize = (model?.attentionHeads ?? 16) * headSize;
	$: kvSize = (model?.keyValueHeads ?? 8) * headSize;
	$: projectionSize = querySize + 2 * kvSize;
	$: traceTokens = $moeExample?.tokens ?? [];
	$: formula = `\\sum_{d=1}^{${hiddenSize}} X_{id} \\cdot W_{dj} = [Q\\;K\\;V]_{ij}`;
	$: vectors = [
		{
			label: `Q · head ${mapping.queryHead + 1}`,
			values: layer?.queryBefore[mapping.queryHead]?.[$moeTokenIndex],
			color: 'query'
		},
		{
			label: `K · shared head ${mapping.keyValueHead + 1}`,
			values: layer?.keyBefore[mapping.keyValueHead]?.[$moeTokenIndex],
			color: 'key'
		},
		{
			label: `V · shared head ${mapping.keyValueHead + 1}`,
			values: layer?.value[mapping.keyValueHead]?.[$moeTokenIndex],
			color: 'value'
		}
	];
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
				error = reason instanceof Error ? reason.message : 'Could not load recorded projections.';
		} finally {
			if (current === request) loading = false;
		}
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
		if (e.key === 'Escape') weightPopover.set(null);
	}}
/>
<div
	use:portal
	bind:this={panel}
	class="projection-inspector"
	role="dialog"
	aria-label="Recorded Q K V projections"
	aria-modal="false"
	tabindex="-1"
	on:click|stopPropagation
>
	<header>
		<div>
			<h2>Query Key Value</h2>
			<p>Granite · Block {$blockIdx + 1} · Query head {$attentionHeadIdx + 1} · Before RoPE</p>
		</div>
		<button
			class="close"
			aria-label="Close Q K V projections"
			on:click={() => weightPopover.set(null)}>×</button
		>
	</header>
	<div class="body">
		<p class="description">
			RMSNorm output × projection weights = Q · K · V. Granite uses no bias.
		</p>
		<div class="matrix-operation" style={`--q-share: ${querySize}fr; --kv-share: ${kvSize}fr`}>
			<section class="matrix-section input-matrix">
				<h3>RMSNorm output</h3>
				<div class="token-matrix">
					{#each traceTokens as token}
						<button
							class="matrix-row"
							class:selected={token.index === $moeTokenIndex}
							aria-label={`Select projection token ${token.index}: ${tokenLabel(token.text)}`}
							aria-pressed={token.index === $moeTokenIndex}
							on:click={() => ($moeTokenIndex = token.index)}
						>
							<span class="token-label">{tokenLabel(token.text)}</span><span
								class="input-cells"
								aria-hidden="true"
							></span>
						</button>
					{/each}
				</div>
				<p class="shape">({traceTokens.length}, {hiddenSize})</p>
			</section>
			<span class="operator">×</span>
			<section class="matrix-section weight-matrix">
				<h3>Q·K·V Weights</h3>
				<div
					class="weight-grid"
					aria-label={`Schematic concatenated projection weights, ${hiddenSize} by ${projectionSize}`}
				>
					<span class="q cells">Wqᵀ</span><span class="k cells">Wkᵀ</span><span class="v cells"
						>Wvᵀ</span
					>
				</div>
				<p class="shape">({hiddenSize}, {projectionSize})</p>
			</section>
			<span class="operator">=</span>
			<section class="matrix-section output-matrix">
				<h3>Q·K·V</h3>
				<div class="token-matrix">
					{#each traceTokens as token}
						<button
							class="matrix-row"
							class:selected={token.index === $moeTokenIndex}
							aria-label={`Select Q K V output for token ${token.index}: ${tokenLabel(token.text)}`}
							aria-pressed={token.index === $moeTokenIndex}
							on:click={() => ($moeTokenIndex = token.index)}
						>
							<span class="token-label">{tokenLabel(token.text)}</span>
							<span class="output-cells" aria-hidden="true"
								><span class="q cells"></span><span class="k cells"></span><span class="v cells"
								></span></span
							>
						</button>
					{/each}
				</div>
				<p class="shape">({traceTokens.length}, {projectionSize})</p>
			</section>
		</div>
		<div class="equation"><Katex math={formula} displayMode={true} /></div>
		<p class="footnote">
			Schematic matrix cells. W = [Wqᵀ | Wkᵀ | Wvᵀ] groups the three bias-free projections for
			display. Q has {querySize} coordinates; K and V each have {kvSize}.
		</p>
		<details class="recorded-details">
			<summary>Inspect recorded Q · K · V values</summary>
			<div class="recorded-body">
				<TokenInspectorLayout
					><TokenPicker
						slot="tokens"
						label="Projection token"
						options={tokenOptions}
						bind:value={$moeTokenIndex}
					/>
					<div class="recording-heading">
						<h3>Recorded outputs · before RoPE</h3>
						<div class="coordinates">
							<span>{coordinate}–{Math.min(coordinate + 3, headSize - 1)}</span><button
								aria-label="Previous projection coordinates"
								disabled={coordinate === 0}
								on:click={() => (coordinate = Math.max(0, coordinate - 4))}>‹</button
							><button
								aria-label="Next projection coordinates"
								disabled={coordinate + 4 >= headSize}
								on:click={() => (coordinate = Math.min(headSize - 4, coordinate + 4))}>›</button
							>
						</div>
					</div>
					{#if loading}<p role="status">Loading recorded projections…</p>{:else if error}<p
							role="alert"
						>
							{error} <button on:click={() => $moeExample && load($moeExample.id)}>Retry</button>
						</p>{:else if layer}
						<div class="tensor-grid">
							{#each vectors as vector}<section class={vector.color}>
									<h4>{vector.label}</h4>
									{#each coordinates as index}<div class="value-row">
											<span>[{index}]</span><b>{vector.values?.[index]?.toFixed(6) ?? '—'}</b>
										</div>{/each}
								</section>{/each}
						</div>
					{/if}
					<p class="footnote">
						Weights are shown by shape, not as recorded cells. The values above are captured
						projection outputs, indexed by head; splitting into heads does not change them. V
						bypasses RoPE. <a
							href={traceAssetUrl($moeExample?.id ?? '', 'attention', base)}
							target="_blank"
							rel="noreferrer">Trace JSON ↗</a
						>
					</p>
				</TokenInspectorLayout>
			</div>
		</details>
	</div>
</div>

<style>
	.projection-inspector {
		position: fixed;
		right: 16px;
		bottom: 16px;
		width: min(920px, calc(100vw - 32px));
		max-height: calc(100dvh - 32px);
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
			sans-serif;
		outline: none;
		box-sizing: border-box;
	}
	header {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		flex-shrink: 0;
		background: #f3f4f6;
		border-bottom: 1px solid #e9e2ef;
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
		color: #91849e;
		font-size: 12px;
	}
	button {
		font: inherit;
		cursor: pointer;
		color: #8b59b6;
		background: #faf7fc;
		border: 1px solid #ded5eb;
		border-radius: 4px;
		padding: 4px 10px;
	}
	button:disabled {
		opacity: 0.35;
		cursor: default;
	}
	button:focus-visible,
	a:focus-visible {
		outline: 2px solid #a07ac0;
		outline-offset: 2px;
	}
	.close {
		width: 30px;
		height: 30px;
		font-size: 24px;
		line-height: 1;
		padding: 0;
		background: white;
	}
	.body {
		padding: 14px 18px;
		overflow: auto;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.description {
		margin: 0;
	}
	.matrix-operation {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 28px minmax(0, 1.2fr) 28px minmax(0, 1.35fr);
		gap: 12px;
		align-items: center;
		padding: 10px 0;
	}
	.matrix-section {
		min-width: 0;
		text-align: center;
	}
	.matrix-section h3 {
		color: #111827;
		font-size: 16px;
		margin: 0 0 16px;
	}
	.token-matrix {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.matrix-row {
		display: flex;
		width: 100%;
		align-items: center;
		gap: 6px;
		border: 0;
		padding: 2px;
		background: transparent;
		color: #6b7280;
		border-radius: 2px;
		min-height: 24px;
	}
	.matrix-row.selected {
		background: #f3f0fa;
		outline: 1px solid #ded5eb;
	}
	.token-label {
		width: 44%;
		flex-shrink: 0;
		text-align: right;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: pre;
	}
	.input-cells {
		height: 17px;
		flex: 1;
		background: repeating-linear-gradient(90deg, #d1d5db 0 3px, #f3f4f6 3px 4px);
	}
	.output-cells,
	.weight-grid {
		display: grid;
		grid-template-columns: var(--q-share) var(--kv-share) var(--kv-share);
	}
	.output-cells {
		height: 17px;
		flex: 1;
	}
	.weight-grid {
		height: 90px;
	}
	.cells {
		display: flex;
		align-items: center;
		justify-content: center;
		background-image:
			linear-gradient(#ffffff70 1px, transparent 1px),
			linear-gradient(90deg, #ffffff70 1px, transparent 1px);
		background-size: 7px 7px;
		font-size: 14px;
	}
	.q {
		background-color: #bfdbfe;
		color: #3568a6;
	}
	.k {
		background-color: #fecaca;
		color: #ad5265;
	}
	.v {
		background-color: #a7f3d0;
		color: #397b5d;
	}
	.shape {
		color: #111827;
		font-size: 16px;
		margin: 10px 0 0;
	}
	.operator {
		color: #374151;
		font-size: 28px;
		text-align: center;
	}
	.recorded-details {
		border-top: 1px solid #e5e7eb;
		padding-top: 8px;
	}
	summary {
		cursor: pointer;
		color: #805a9b;
	}
	.recorded-body {
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding-top: 12px;
	}
	.equation {
		text-align: center;
		color: #6b7280;
		font:
			20px/1.5 Georgia,
			serif;
	}
	.recording-heading,
	.coordinates {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
	}
	h3 {
		font-size: 13px;
		font-weight: 500;
		margin: 0;
	}
	.coordinates {
		font-size: 11px;
	}
	.tensor-grid {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 14px;
	}
	h4 {
		font-size: 12px;
		font-weight: 500;
		margin: 0 0 6px;
	}
	.value-row {
		display: flex;
		justify-content: space-between;
		gap: 4px;
		padding: 6px 10px;
		border-radius: 3px;
		margin: 4px 0;
		background: #eff5fd;
		color: #527eaf;
	}
	.value-row span {
		font-size: 10px;
		opacity: 0.75;
	}
	.value-row b {
		font:
			12px ui-monospace,
			monospace;
	}
	.key .value-row {
		background: #fdf0f1;
		color: #ab6274;
	}
	.value .value-row {
		background: #eff8f3;
		color: #4f8a6d;
	}
	.footnote {
		font-size: 11px;
		color: #91849e;
		margin: 0;
	}
	a {
		color: #8b59b6;
	}
	@media (max-width: 600px) {
		.body {
			padding: 10px;
		}
		.tensor-grid {
			gap: 5px;
		}
		.value-row {
			padding: 6px 4px;
		}
		.value-row b {
			font-size: 10px;
		}
	}
</style>
