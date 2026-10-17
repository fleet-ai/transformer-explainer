<script lang="ts">
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
	import { attentionInspector } from '$lib/inspectors';
	import { headMapping, loadAttentionOperations } from '$lib/moe/attention';
	import { traceAssetUrl } from '$lib/moe/live';
	import { tokenLabel } from '$lib/moe/format';
	import TraceSelector from '../moe/TraceSelector.svelte';
	import ProjectionOperation from '../common/ProjectionOperation.svelte';
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
	$: rows = [
		{
			label: 'Wq',
			weight: `${(model?.attentionHeads ?? 16) * headSize} × ${model?.hiddenSize ?? 1024}`,
			input: `x · ${model?.hiddenSize ?? 1024}D`,
			output: `Q · ${(model?.attentionHeads ?? 16) * headSize}D`,
			color: 'query'
		},
		{
			label: 'Wk',
			weight: `${(model?.keyValueHeads ?? 8) * headSize} × ${model?.hiddenSize ?? 1024}`,
			input: `x · ${model?.hiddenSize ?? 1024}D`,
			output: `K · ${(model?.keyValueHeads ?? 8) * headSize}D`,
			color: 'key'
		},
		{
			label: 'Wv',
			weight: `${(model?.keyValueHeads ?? 8) * headSize} × ${model?.hiddenSize ?? 1024}`,
			input: `x · ${model?.hiddenSize ?? 1024}D`,
			output: `V · ${(model?.keyValueHeads ?? 8) * headSize}D`,
			color: 'value'
		}
	];
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
			<h2>Query · Key · Value</h2>
			<p>Granite · Block {$blockIdx + 1} · Query head {$attentionHeadIdx + 1} · Before RoPE</p>
		</div>
		<button
			class="close"
			aria-label="Close Q K V projections"
			on:click={() => weightPopover.set(null)}>×</button
		>
	</header>
	<div class="body">
		<div class="token-row">
			<span>Token</span><TraceSelector
				label="Projection token"
				options={tokenOptions}
				bind:value={$moeTokenIndex}
			/>
		</div>
		<p class="description">
			The same RMSNorm output <b>x</b> enters three separate, bias-free projections.
		</p>
		<ProjectionOperation {rows} />
		<div class="equation">
			Q[j] = Σ Wq[j, d] × x[d] <span
				>Sum over all {model?.hiddenSize ?? 1024} input coordinates; K and V use Wk and Wv.</span
			>
		</div>
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
		{#if loading}<p role="status">Loading recorded projections…</p>{:else if error}<p role="alert">
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
		<div class="next">
			<button on:click={() => attentionInspector.set({ kind: 'heads' })}>Head split ↗</button><span
				>→</span
			><button on:click={() => attentionInspector.set({ kind: 'rope', vector: 'query' })}
				>RoPE on Q / K ↗</button
			><span>→ Q′K′ᵀ · attention</span>
		</div>
		<p class="footnote">
			Weights are shown by shape, not as recorded cells. The values above are captured projection
			outputs, indexed by head; splitting into heads does not change them. V bypasses RoPE. <a
				href={traceAssetUrl($moeExample?.id ?? '', 'attention', base)}
				target="_blank"
				rel="noreferrer">Trace JSON ↗</a
			>
		</p>
	</div>
</div>

<style>
	.projection-inspector {
		position: fixed;
		right: 16px;
		bottom: 16px;
		width: min(980px, calc(100vw - 32px));
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
	.token-row {
		display: flex;
		align-items: center;
		gap: 12px;
		min-width: 0;
	}
	.description {
		margin: 0;
	}
	.equation {
		padding: 10px 12px;
		border-radius: 5px;
		background: #faf7fc;
		color: #805a9b;
		font:
			15px/1.5 Georgia,
			serif;
	}
	.equation span {
		display: block;
		font:
			11px/1.5 Arial,
			sans-serif;
		color: #91849e;
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
	.next {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 10px;
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
