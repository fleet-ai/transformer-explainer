<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { blockIdx, moeExample, moeModel, moeTokenIndex } from '~/store';
	import { percent, tokenLabel } from '$lib/moe/format';
	import { sumCoordinate } from '$lib/moe/arithmetic';
	import { loadExpertOperations, type ExpertOperationsDocument } from '$lib/moe/operations';
	import ExpertOperations from './ExpertOperations.svelte';

	export let expert: number | null = null;
	const dispatch = createEventDispatcher<{ close: void; expert: number; sum: void }>();
	let coordinate = 0;
	let recording: ExpertOperationsDocument | null = null;
	let loading = false;
	let error = '';
	let request = 0;
	let panel: HTMLDivElement;
	$: trace = $moeExample?.layers[$blockIdx]?.tokens[$moeTokenIndex];
	$: isMLA = $moeModel?.architecture === 'mla';
	$: isDense = $moeExample?.layers[$blockIdx]?.feedForwardType === 'dense';
	$: isShared = isMLA && expert === $moeModel?.numExperts;
	$: expertLabel = isDense
		? 'Dense feed-forward'
		: isShared
			? 'Shared experts'
			: `Expert ${expert}`;
	$: evaluated = isDense || isShared || rank >= 0;
	$: rank = expert !== null && trace ? trace.selectedExperts.indexOf(expert) : -1;
	$: sum =
		trace && !isDense
			? sumCoordinate(trace, coordinate, $moeModel?.residualMultiplier ?? 0.22)
			: null;
	$: operationToken =
		recording?.example.id === $moeExample?.id
			? recording?.layers[$blockIdx]?.tokens[$moeTokenIndex]
			: undefined;
	$: operation = operationToken?.experts.find((item) => item.id === expert) ?? null;
	$: if (expert !== null && evaluated && $moeExample) load($moeExample.id);

	async function load(exampleId: string) {
		const current = ++request;
		loading = true;
		error = '';
		try {
			const result = await loadExpertOperations(exampleId);
			if (current === request) recording = result;
		} catch (reason) {
			if (current === request)
				error = reason instanceof Error ? reason.message : 'Could not load expert operations.';
		} finally {
			if (current === request) loading = false;
		}
	}
	function number(value: number): string {
		return Math.abs(value) >= 10000 || (value !== 0 && Math.abs(value) < 0.0001)
			? value.toExponential(5)
			: value.toFixed(6);
	}
	// Keep the inspector inside the viewport even when the diagram pans horizontally.
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
	on:keydown={(event) => {
		if (event.key === 'Escape') dispatch('close');
	}}
/>

<div
	use:portal
	bind:this={panel}
	class="moe-inspector"
	role="dialog"
	aria-modal="false"
	aria-label={expert === null ? 'Weighted sum operations' : `${expertLabel} operations`}
	tabindex="-1"
>
	<header>
		<div>
			<h2>{expert === null ? 'Σ · Weighted sum' : `${expertLabel} · SwiGLU`}</h2>
			<p>
				Block {$blockIdx + 1} · token {$moeTokenIndex}
				<b>“{tokenLabel($moeExample?.tokens[$moeTokenIndex]?.text ?? '')}”</b>
			</p>
		</div>
		<button class="close" aria-label="Close operation inspector" on:click={() => dispatch('close')}
			>×</button
		>
	</header>
	<div class="inspector-body" class:expert-mode={expert !== null}>
		{#if expert !== null && trace}
			<div class="expert-meta">
				{#if isDense}<span>Block 1 · every token uses this dense network</span><span
						>{$moeModel?.denseIntermediateSize} intermediate coordinates</span
					>
				{:else if isShared}<span>{$moeModel?.sharedExperts} always-active shared experts</span><span
						>Added with weight <b>1</b></span
					><button class="text-button" on:click={() => dispatch('sum')}>See Σ weighted sum →</button
					>
				{:else}<span>Router probability <b>{percent(trace.routerProbabilities[expert], 2)}</b></span
					><span
						>Mixing weight <b>{rank >= 0 ? percent(trace.routingWeights[rank], 2) : '0%'}</b></span
					><button class="text-button" on:click={() => dispatch('sum')}>See Σ weighted sum →</button
					>{/if}
			</div>
			{#if !evaluated}
				<p class="message">
					This expert is outside the top-{$moeModel?.topK}. It was not evaluated for this token.
				</p>
			{:else if loading}
				<p class="message" role="status">Loading recorded multiplications…</p>
			{:else if error}
				<p class="message" role="alert">
					{error}
					<button class="text-button" on:click={() => $moeExample && load($moeExample.id)}
						>Try again</button
					>
				</p>
			{:else if operation && operationToken}
				{#key `${$moeExample?.id}-${$blockIdx}-${$moeTokenIndex}-${expert}`}
					<ExpertOperations
						{operation}
						{expert}
						label={expertLabel}
						inputPreview={operationToken.inputPreview}
						hiddenSize={$moeModel?.hiddenSize ?? 1024}
						intermediateSize={operation.intermediateSize ??
							(isDense
								? $moeModel?.denseIntermediateSize
								: isShared
									? $moeModel?.sharedIntermediateSize
									: $moeModel?.intermediateSize) ??
							512}
					/>
				{/key}
			{:else}<p class="message" role="alert">No operations recorded for this expert.</p>{/if}
		{:else if trace && sum}
			<div class="sum-formula">
				<span>y<sub>j</sub> =</span><strong>Σ</strong><span
					>w<sub>e</sub> × Expert<sub>e</sub>(x)<sub>j</sub>{#if isMLA}
						+ Shared(x)<sub>j</sub>{/if}</span
				>
			</div>
			<div class="coordinate-picker">
				<span>Output coordinate</span>
				<div>
					{#each trace.combinedOutputPreview as _, index}
						<button
							class:chosen={coordinate === index}
							aria-pressed={coordinate === index}
							aria-label={`Inspect sum coordinate ${index}`}
							on:click={() => (coordinate = index)}>{index}</button
						>
					{/each}
				</div>
				<small>First 8 of {$moeModel?.hiddenSize ?? 1024}</small>
			</div>
			<table>
				<thead
					><tr
						><th>Expert</th><th>Routing weight</th><th>Output [{coordinate}]</th><th
							>Weighted contribution</th
						></tr
					></thead
				>
				<tbody
					>{#each sum.terms as term}<tr
							><td
								><button class="text-button" on:click={() => dispatch('expert', term.expert)}
									>{term.shared ? 'Shared experts' : `E${term.expert.toString().padStart(2, '0')}`} ↗</button
								></td
							>
							<td>{number(term.weight)}</td><td
								><span class="operator">×</span> {number(term.value)}</td
							><td><span class="operator">=</span> {number(term.contribution)}</td></tr
						>{/each}</tbody
				>
				<tfoot
					><tr
						><th colspan="3">Σ · Add all {sum.terms.length} contributions</th><td
							>{number(sum.reconstructed)}</td
						></tr
					></tfoot
				>
			</table>
			<div class="sum-result">
				<span>Recorded MoE output [{coordinate}]</span><strong>{number(sum.recorded)}</strong>
			</div>
			<div class="residual-equation">
				<span
					>{($moeModel?.residualMultiplier ?? 0.22) === 1
						? 'Then add to the residual stream'
						: 'Then residual scaling'}</span
				><b
					>{number(sum.recorded)} × {$moeModel?.residualMultiplier ?? 0.22} = {number(
						sum.scaled
					)}</b
				>
			</div>
			<p class="footnote">
				{isMLA
					? 'Routed weights are selected from the full router softmax without renormalization. The shared branch is added once with weight 1.'
					: 'This scaled value is added to the residual stream.'} Values above are rounded for display;
				the sum uses recorded precision. Reconstruction difference: {Math.abs(
					sum.difference
				).toExponential(2)}.
			</p>
		{/if}
	</div>
</div>

<style>
	.moe-inspector {
		position: fixed;
		right: 16px;
		bottom: 16px;
		width: min(1080px, calc(100vw - 32px));
		height: min(680px, calc(100vh - 32px));
		height: min(680px, calc(100dvh - 32px));
		box-sizing: border-box;
		overflow: hidden;
		z-index: 1200;
		display: flex;
		flex-direction: column;
		background: white;
		border: 1px solid #ded5eb;
		border-radius: 8px;
		box-shadow: 0 8px 34px #33224b30;
		color: #665775;
		font:
			13px/1.5 Arial,
			sans-serif;
		outline: none;
	}
	header {
		flex: 0 0 auto;
		padding: 10px 16px;
		background: #faf8fc;
		border-bottom: 1px solid #e9e2ef;
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		border-radius: 8px 8px 0 0;
	}
	header > div {
		min-width: 0;
	}
	h2 {
		margin: 0;
		font-size: 17px;
		font-weight: 600;
		color: #624685;
	}
	header p {
		margin: 3px 0 0;
		font-size: 11px;
		color: #91849e;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	header b {
		color: #776286;
		font-weight: 500;
	}
	.close {
		flex-shrink: 0;
		border: 1px solid #ddd5e5;
		border-radius: 5px;
		background: white;
		color: #79698a;
		font-size: 21px;
		line-height: 22px;
		width: 28px;
		height: 28px;
		cursor: pointer;
	}
	.inspector-body {
		flex: 1;
		padding: 10px 16px;
		overflow: auto;
		min-height: 0;
		overscroll-behavior: contain;
		scrollbar-gutter: stable;
	}
	.inspector-body.expert-mode {
		display: flex;
		flex-direction: column;
		overflow: hidden;
		scrollbar-gutter: auto;
	}
	.expert-meta {
		flex: 0 0 auto;
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
		gap: 8px 18px;
		font-size: 11px;
		margin-bottom: 8px;
		font-variant-numeric: tabular-nums;
	}
	.expert-meta b {
		color: #7954a1;
	}
	.text-button {
		background: none;
		border: 0;
		padding: 0;
		color: #8758bd;
		font: inherit;
		cursor: pointer;
	}
	.text-button:hover {
		text-decoration: underline;
	}
	.message {
		margin: 24px 0;
		overflow: auto;
	}
	.sum-formula {
		display: flex;
		gap: 9px;
		align-items: center;
		justify-content: center;
		color: #8154ac;
		font-size: 18px;
		padding: 4px 0 15px;
	}
	.sum-formula strong {
		font-size: 30px;
		font-weight: 500;
	}
	.coordinate-picker {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
		font-size: 11px;
		margin-bottom: 15px;
	}
	.coordinate-picker > div {
		display: flex;
		gap: 4px;
	}
	.coordinate-picker button {
		width: 25px;
		height: 25px;
		background: #faf8fd;
		border: 1px solid #e1d7ed;
		border-radius: 4px;
		color: #9377ad;
		cursor: pointer;
	}
	.coordinate-picker button.chosen {
		background: #a379ce;
		border-color: #a379ce;
		color: white;
	}
	.coordinate-picker small {
		color: #a292af;
	}
	table {
		border-collapse: collapse;
		width: 100%;
		font:
			11px/1.5 'SFMono-Regular',
			Consolas,
			monospace;
	}
	th {
		font:
			10px/1.4 Arial,
			sans-serif;
		color: #9b8aa9;
		font-weight: 500;
		padding: 6px 5px;
		text-align: right;
	}
	th:first-child,
	td:first-child {
		text-align: left;
	}
	td {
		text-align: right;
		padding: 7px 5px;
		border-bottom: 1px solid #f0ebf4;
		white-space: nowrap;
	}
	.operator {
		color: #bbaac8;
		margin-right: 6px;
	}
	tfoot th,
	tfoot td {
		padding-top: 12px;
		font-weight: 600;
		color: #8359a9;
		border-bottom: 0;
	}
	.sum-result {
		display: flex;
		justify-content: space-between;
		padding: 12px 10px;
		margin-top: 10px;
		background: #f5effb;
		border-radius: 5px;
		font-size: 12px;
		color: #8054a6;
	}
	.sum-result strong {
		font-family: 'SFMono-Regular', Consolas, monospace;
	}
	.residual-equation {
		display: flex;
		flex-wrap: wrap;
		gap: 7px;
		justify-content: space-between;
		padding-top: 13px;
		font-size: 11px;
	}
	.residual-equation b {
		font-family: 'SFMono-Regular', Consolas, monospace;
		font-weight: 500;
		color: #8c6caa;
	}
	.footnote {
		margin: 12px 0 0;
		color: #a093ab;
		font-size: 10px;
	}
	button:focus-visible {
		outline: 2px solid #a87bd2;
		outline-offset: 3px;
	}
	@media (max-width: 560px) {
		.expert-meta {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
		.expert-meta .text-button {
			grid-column: 1 / -1;
			justify-self: start;
		}
	}
	@media (max-height: 540px) {
		.moe-inspector {
			height: calc(100vh - 40px);
			height: calc(100dvh - 40px);
		}
		.inspector-body.expert-mode {
			overflow: auto;
			scrollbar-gutter: stable;
		}
	}
</style>
