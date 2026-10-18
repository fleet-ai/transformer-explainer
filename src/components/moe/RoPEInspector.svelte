<script lang="ts">
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
	import {
		headMapping,
		loadAttentionOperations,
		ropePairWindow,
		ropeCoordinate
	} from '$lib/moe/attention';
	import { tokenLabel } from '$lib/moe/format';
	import { traceAssetUrl } from '$lib/moe/live';
	import RoPEUnitCircle from './RoPEUnitCircle.svelte';
	import RoPEValueCircle from './RoPEValueCircle.svelte';
	import TraceSelector from './TraceSelector.svelte';

	const dispatch = createEventDispatcher<{ close: void; attention: void }>();
	export let kind: 'query' | 'key' = 'query';
	let selectedCoordinate = 0;
	$: isMLA = $moeModel?.architecture === 'mla';
	$: headSize = isMLA ? ($moeModel?.rotarySize ?? 64) : ($moeModel?.attentionHeadSize ?? 64);
	$: pairCount = headSize / 2;
	$: if (!Number.isInteger(selectedCoordinate) || selectedCoordinate < 0) selectedCoordinate = 0;
	else if (selectedCoordinate >= pairCount) selectedCoordinate %= pairCount;
	$: tokenOptions = ($moeExample?.tokens ?? []).map((token) => ({
		value: token.index,
		label: `${token.index}: ${tokenLabel(token.text)}`
	}));
	$: coordinateOptions = Array.from({ length: pairCount }, (_, index) => ({
		value: index,
		label: String(index)
	}));
	$: coordinate =
		selectedCoordinate !== undefined &&
		Number.isInteger(selectedCoordinate) &&
		selectedCoordinate >= 0 &&
		selectedCoordinate < pairCount
			? ropeCoordinate(selectedCoordinate, headSize)
			: null;
	$: recording = $moeAttentionOperations;
	let loading = false;
	let error = '';
	let request = 0;
	let panel: HTMLDivElement;
	$: mapping = headMapping(
		$attentionHeadIdx,
		$moeModel?.attentionHeads ?? 16,
		(isMLA ? 1 : $moeModel?.keyValueHeads) ?? 8
	);
	$: layer = recording?.example.id === $moeExample?.id ? recording?.layers[$blockIdx] : undefined;
	$: before =
		kind === 'query'
			? layer?.queryBefore[$attentionHeadIdx][$moeTokenIndex]
			: layer?.keyBefore[mapping.keyValueHead][$moeTokenIndex];
	$: after =
		kind === 'query'
			? layer?.queryAfter[$attentionHeadIdx][$moeTokenIndex]
			: layer?.keyAfter[mapping.keyValueHead][$moeTokenIndex];
	$: position = recording?.positions[$moeTokenIndex];
	$: cosine = recording?.rotary.cos[$moeTokenIndex];
	$: sine = recording?.rotary.sin[$moeTokenIndex];
	$: pairs =
		before && after && cosine && sine && coordinate
			? ropePairWindow(before, after, cosine, sine, coordinate.coordinate)
			: null;
	$: value = layer?.value[isMLA ? $attentionHeadIdx : mapping.keyValueHead][$moeTokenIndex];
	$: if ($moeExample) load($moeExample.id);
	async function load(id: string) {
		const current = ++request;
		loading = true;
		error = '';
		try {
			const data = await loadAttentionOperations(id);
			if (current === request) $moeAttentionOperations = data;
		} catch (reason) {
			if (current === request)
				error = reason instanceof Error ? reason.message : 'Could not load recorded attention.';
		} finally {
			if (current === request) loading = false;
		}
	}
	function number(value: number): string {
		return Math.abs(value) >= 10000 || (value !== 0 && Math.abs(value) < 0.0001)
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
	on:keydown={(event) => {
		if (event.key === 'Escape') dispatch('close');
	}}
/>

<div
	use:portal
	bind:this={panel}
	class="rope-inspector"
	role="dialog"
	aria-modal="false"
	aria-label="Recorded RoPE operations"
	tabindex="-1"
>
	<header>
		<div>
			<h2>RoPE · Rotate Q and K</h2>
			<p>
				{isMLA ? 'DeepSeek · MLA' : 'Granite'} · Block {$blockIdx + 1} · Query head {$attentionHeadIdx +
					1} · {isMLA ? 'Shared rotary key' : `Shared K/V head ${mapping.keyValueHead + 1}`}
			</p>
		</div>
		<button class="close" aria-label="Close RoPE inspector" on:click={() => dispatch('close')}
			>×</button
		>
	</header>
	<div class="inspector-body">
		<aside class="token-sidebar" aria-label="Token selection">
			<span class="token-heading">Token</span>
			<TraceSelector
				label="RoPE token"
				options={tokenOptions}
				bind:value={$moeTokenIndex}
				vertical
			/>
		</aside>
		<div class="rotation-content">
			<div class="controls">
				<div class="kind-picker" aria-label="Vector to inspect">
					<button
						class:chosen={kind === 'query'}
						aria-pressed={kind === 'query'}
						on:click={() => (kind = 'query')}>Query {$attentionHeadIdx + 1}</button
					>
					<button
						class:chosen={kind === 'key'}
						aria-pressed={kind === 'key'}
						on:click={() => (kind = 'key')}
						>{isMLA ? 'Shared rotary key' : `Shared Key ${mapping.keyValueHead + 1}`}</button
					>
				</div>

				<div class="selector-row coordinate-selector">
					<span class="selector-label"
						>Coordinate
						<small id="rope-coordinate-range">0–{pairCount - 1}</small>
					</span>
					<TraceSelector
						label="RoPE coordinate"
						options={coordinateOptions}
						rows={2}
						bind:value={selectedCoordinate}
					/>
				</div>
			</div>
			{#if loading}<p class="message" role="status">Loading recorded rotations…</p>
			{:else if error}<p class="message" role="alert">
					{error}
					<button class="text-button" on:click={() => $moeExample && load($moeExample.id)}
						>Try again</button
					>
				</p>
			{:else if !coordinate}<p class="message" role="status">
					Choose a whole coordinate from 0 to {pairCount - 1}.
				</p>
			{:else if pairs && value}
				<div class="rotation-summary">
					<span
						>{kind === 'query' ? 'Q' : 'K'}
						{isMLA ? 'rotary branch' : 'head'} · {headSize}D → RoPE · position {position} → {headSize}D</span
					>
					<span aria-live="polite"
						>Selected {coordinate.coordinate} ↔ fixed partner {coordinate.partner} · j = i + {pairCount}</span
					>
					<span class="position-note"
						>{position === 0
							? 'Position 0: no rotation; before/after lines overlap.'
							: 'Four coordinates · two pairs · recorded rotations'}</span
					>
				</div>
				<div class="rotation-formula" aria-label="RoPE rotation formula">
					<span
						>x′<sub>i</sub> = x<sub>i</sub> cos θ<sub>i</sub> − x<sub>j</sub> sin θ<sub>i</sub
						></span
					>
					<span
						>x′<sub>j</sub> = x<sub>j</sub> cos θ<sub>i</sub> + x<sub>i</sub> sin θ<sub>i</sub
						></span
					>
				</div>
				<div class="pair-grid">
					{#each pairs as pair}
						<section
							class="rotation-pair"
							aria-label={`Rotation pair ${pair.indices[0]} and ${pair.indices[1]}`}
						>
							<div class="pair-heading">
								<strong>Pair {pair.indices[0]} ↔ {pair.indices[1]}</strong><span
									>θ<sub>{pair.indices[0]}</sub> · position {position}</span
								>
							</div>
							<div class="rotation" class:key={kind === 'key'}>
								<div class="pair-column">
									<span>Before RoPE</span>
									<div class:focused-coordinate={coordinate.coordinate === pair.indices[0]}>
										<small>[{pair.indices[0]}]</small><b>{number(pair.before[0])}</b>
									</div>
									<div class:focused-coordinate={coordinate.coordinate === pair.indices[1]}>
										<small>[{pair.indices[1]}]</small><b>{number(pair.before[1])}</b>
									</div>
								</div>
								<div class="circle-comparison">
									<div class="circle-view">
										<h3>{kind === 'query' ? 'Q' : 'K'} coordinate pair</h3>
										<RoPEValueCircle
											compact
											before={pair.before}
											after={pair.recorded}
											indices={pair.indices}
											cosine={pair.cosine}
											sine={pair.sine}
											selectedCoordinate={coordinate.coordinate}
										/>
									</div>
									<div class="circle-view">
										<h3>cos / sin · unit circle</h3>
										<RoPEUnitCircle
											compact
											cosine={pair.cosine}
											sine={pair.sine}
											pairIndex={pair.indices[0]}
											selected={coordinate.pairIndex === pair.indices[0]}
										/>
									</div>
								</div>
								<div class="pair-column">
									<span>Recorded after RoPE</span>
									<div class:focused-coordinate={coordinate.coordinate === pair.indices[0]}>
										<small>[{pair.indices[0]}]</small><b>{number(pair.recorded[0])}</b>
									</div>
									<div class:focused-coordinate={coordinate.coordinate === pair.indices[1]}>
										<small>[{pair.indices[1]}]</small><b>{number(pair.recorded[1])}</b>
									</div>
								</div>
							</div>
							<div class="equations">
								<p class:focused-equation={coordinate.coordinate === pair.indices[0]}>
									<span>x′[{pair.indices[0]}]</span> = {number(pair.before[0])} × {number(
										pair.cosine
									)} − ({number(pair.before[1])} × {number(pair.sine)})
									<b>≈ {number(pair.recorded[0])}</b>
								</p>
								<p class:focused-equation={coordinate.coordinate === pair.indices[1]}>
									<span>x′[{pair.indices[1]}]</span> = {number(pair.before[1])} × {number(
										pair.cosine
									)} + ({number(pair.before[0])} × {number(pair.sine)})
									<b>≈ {number(pair.recorded[1])}</b>
								</p>
							</div>
						</section>
					{/each}
				</div>
				<div class="model-notes">
					<div class="shared-note">
						{#if isMLA}<strong>All {$moeModel?.attentionHeads} heads share this rotary key</strong
							><span
								>Only 64 rotary coordinates rotate; 128 content coordinates bypass RoPE. DeepSeek
								reorders adjacent projection pairs into this half-split view before rotation.</span
							>{:else}<strong
								>Q {mapping.sharedQueryHeads.map((head) => head + 1).join(' + ')} share K/V {mapping.keyValueHead +
									1}</strong
							><span>Both query heads use the same rotated K and unchanged V.</span>{/if}
					</div>
					<div class="value-bypass">
						<strong>V bypasses RoPE</strong><span
							>V[{coordinate.coordinate}] = {number(value[coordinate.coordinate])} → {number(
								value[coordinate.coordinate]
							)}</span
						>
					</div>
				</div>
				<div class="next-operation">
					<span>After RoPE</span>
					<p>
						{isMLA ? '(QᶜKᶜᵀ + QʳKʳᵀ)' : 'Q′K′ᵀ'} × {$moeModel?.attentionMultiplier?.toPrecision(
							5
						) ?? '1/64'} → causal mask → softmax → × V
					</p>
					<button class="text-button" on:click={() => dispatch('attention')}
						>See attention operations →</button
					>
				</div>
				<p class="footnote">
					Recorded forward-pass values · rounded for display · max float32 reconstruction
					difference: {Math.max(
						...pairs.flatMap((pair) => pair.difference.map(Math.abs))
					).toExponential(2)}.
					<a
						href={traceAssetUrl(recording?.example.id ?? '', 'attention', base)}
						target="_blank"
						rel="noreferrer">Trace JSON ↗</a
					>
				</p>
			{/if}
		</div>
	</div>
</div>

<style>
	.rope-inspector {
		position: fixed;
		right: 16px;
		bottom: 16px;
		width: min(1180px, calc(100vw - 32px));
		height: min(680px, calc(100vh - 32px));
		height: min(680px, calc(100dvh - 32px));
		box-sizing: border-box;
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
		overflow: hidden;
	}
	header {
		flex: 0 0 auto;
		padding: 8px 14px;
		background: #faf8fc;
		border-bottom: 1px solid #e9e2ef;
		display: flex;
		justify-content: space-between;
		gap: 12px;
	}
	h2 {
		margin: 0;
		font-size: 17px;
		font-weight: 600;
		color: #624685;
	}
	header p {
		margin: 2px 0 0;
		font-size: 11px;
		color: #91849e;
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
		padding: 8px 14px;
		display: grid;
		grid-template-columns: 112px minmax(0, 1fr);
		gap: 14px;
		min-height: 0;
		flex: 1;
		overflow: hidden;
	}
	.token-sidebar {
		display: flex;
		flex-direction: column;
		gap: 8px;
		min-height: 0;
		font-size: 12px;
	}
	.token-heading {
		color: #91849e;
		padding: 5px 10px 0;
	}
	.rotation-content {
		min-width: 0;
		min-height: 0;
		overflow: auto;
		overscroll-behavior: contain;
	}
	.controls {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 6px 16px;
		align-items: center;
		font-size: 12px;
	}
	.selector-row {
		display: flex;
		min-width: 0;
		gap: 10px;
		align-items: center;
	}
	.selector-label {
		flex: 0 0 60px;
	}
	.selector-label small {
		display: block;
		font-size: 11px;
		color: #a694b7;
	}
	.coordinate-selector {
		grid-column: 1 / -1;
	}
	.kind-picker {
		display: flex;
	}
	.kind-picker button {
		background: #faf8fd;
		border: 1px solid #e1d7ed;
		padding: 6px 10px;
		font: inherit;
		color: #9377ad;
		cursor: pointer;
		white-space: nowrap;
	}
	.kind-picker button:first-child {
		border-radius: 4px 0 0 4px;
	}
	.kind-picker button:last-child {
		border-radius: 0 4px 4px 0;
	}
	.kind-picker button.chosen {
		background: #a379ce;
		border-color: #a379ce;
		color: white;
	}
	.rotation-summary {
		display: flex;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 2px 12px;
		margin: 6px 0 4px;
		color: #927aa4;
		font-size: 11px;
	}
	.position-note {
		color: #927e69;
	}
	.rotation-formula {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: 2px 22px;
		margin-bottom: 6px;
		color: #87669e;
		font-size: 12px;
	}
	.pair-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 10px;
	}
	.rotation-pair {
		min-width: 0;
		border: 1px solid #eee8f3;
		border-radius: 5px;
		padding: 6px 8px;
	}
	.pair-heading {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 4px;
		font-size: 11px;
		color: #a08eaf;
	}
	.pair-heading strong {
		font-size: 11px;
		font-weight: 600;
		color: #87669e;
	}
	.rotation {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		grid-template-areas: 'before after' 'circles circles';
		gap: 4px 12px;
		color: #6b97ce;
	}
	.rotation.key {
		color: #ce8990;
	}
	.pair-column {
		min-width: 0;
	}
	.pair-column:first-child {
		grid-area: before;
	}
	.pair-column:last-child {
		grid-area: after;
	}
	.pair-column > span {
		display: block;
		font-size: 11px;
		text-align: center;
		margin-bottom: 2px;
	}
	.pair-column > div {
		background: #eff5fd;
		padding: 2px 8px;
		margin-bottom: 2px;
		display: flex;
		justify-content: space-between;
		border-radius: 3px;
		gap: 6px;
	}
	.pair-column b {
		font:
			12px/1.5 'SFMono-Regular',
			Consolas,
			monospace;
	}
	.pair-column small {
		font-size: 11px;
		opacity: 0.7;
	}
	.pair-column > div.focused-coordinate {
		box-shadow: inset 3px 0 currentColor;
		background: #e1edfc;
	}
	.key .pair-column > div {
		background: #fdf1f2;
	}
	.key .pair-column > div.focused-coordinate {
		background: #fbe2e5;
	}
	.circle-comparison {
		grid-area: circles;
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		justify-items: center;
		align-items: start;
		gap: 4px;
	}
	.circle-view {
		min-width: 0;
	}
	.circle-view h3 {
		margin: 0;
		font-size: 11px;
		font-weight: 500;
		text-align: center;
		color: #87669e;
	}
	.equations {
		background: #faf8fc;
		border-radius: 5px;
		padding: 6px 8px;
		margin-top: 4px;
		font:
			11px/1.65 'SFMono-Regular',
			Consolas,
			monospace;
		overflow-wrap: anywhere;
	}
	.equations p {
		margin: 0;
	}
	.equations span,
	.equations b {
		color: #9468b7;
		font-weight: 500;
	}
	.equations p.focused-equation {
		font-weight: 600;
		color: #72518d;
	}
	.model-notes {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 10px;
		margin-top: 6px;
		font-size: 11px;
	}
	.shared-note {
		display: flex;
		flex-direction: column;
		padding-left: 8px;
		border-left: 2px solid #c5a8e0;
		color: #9a8da6;
	}
	.shared-note strong {
		color: #9066b5;
		font-weight: 500;
	}
	.value-bypass {
		display: flex;
		flex-direction: column;
		color: #6cae93;
		background: #f1faf5;
		padding: 2px 8px;
		border-radius: 5px;
	}
	.value-bypass strong {
		font-weight: 500;
	}
	.value-bypass span {
		font-family: 'SFMono-Regular', Consolas, monospace;
	}
	.next-operation {
		display: flex;
		flex-wrap: wrap;
		gap: 3px 12px;
		align-items: center;
		margin-top: 6px;
		font-size: 11px;
	}
	.next-operation > span {
		color: #a596b2;
	}
	.next-operation p {
		margin: 0;
		color: #88739a;
	}
	.text-button {
		border: 0;
		background: none;
		padding: 0;
		color: #8758bd;
		font: inherit;
		cursor: pointer;
	}
	.text-button:hover {
		text-decoration: underline;
	}
	.footnote {
		margin: 4px 0 0;
		color: #a093ab;
		font-size: 11px;
	}
	a {
		color: #956aba;
	}
	.message {
		margin: 24px 0;
	}
	button:focus-visible {
		outline: 2px solid #a87bd2;
		outline-offset: 3px;
	}
	@media (max-width: 900px) {
		.pair-grid {
			grid-template-columns: minmax(0, 1fr);
		}
		.rotation-summary {
			justify-content: flex-start;
		}
	}
	@media (max-width: 560px) {
		.inspector-body {
			grid-template-columns: 88px minmax(0, 1fr);
			gap: 8px;
			padding: 8px;
		}
		.controls {
			grid-template-columns: minmax(0, 1fr);
		}
		.model-notes {
			grid-template-columns: minmax(0, 1fr);
		}
		.circle-comparison {
			grid-template-columns: minmax(0, 1fr);
		}
		.rotation-formula {
			justify-content: flex-start;
		}
	}
</style>
