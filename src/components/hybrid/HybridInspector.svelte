<script lang="ts">
	import { setContext, tick } from 'svelte';
	import { writable } from 'svelte/store';
	const matrixPopup = writable(false);
	setContext('matrix-popup', matrixPopup);
	setContext('matrix-popup-label', 'Zamba2');
	async function openMatrix() {
		matrixPopup.set(false);
		await tick();
		matrixPopup.set(true);
	}

	import BlockChooser from '../common/BlockChooser.svelte';
	import TensorBandFlow from '../common/TensorBandFlow.svelte';
	import MatrixProduct from '../common/MatrixProduct.svelte';
	import FeedForwardFlow from '../common/FeedForwardFlow.svelte';
	import TokenInspectorLayout from '../moe/TokenInspectorLayout.svelte';
	import TokenPicker from '../moe/TokenPicker.svelte';
	import { recordedAssetUrl } from '$lib/recorded-json.js';

	import { createEventDispatcher } from 'svelte';
	import { base } from '$app/paths';
	import { blockIdx, attentionHeadIdx, attentionHeadIdxTemp, moeTokenIndex } from '~/store';
	import { hybridRecording, hybridLayer, hybridInspector } from '$lib/hybrid/store';
	import { tensor, row, scalar } from '$lib/vit/types';
	import {
		convolution,
		stateStep,
		readout,
		gatedNorm,
		projection,
		ropePair,
		dot
	} from '$lib/hybrid/math';
	export let stage = 'scan';
	const dispatch = createEventDispatcher<{ close: void }>();
	const stages = [
		['projection', 'Input projection'],
		['convolution', 'Causal convolution'],
		['scan', 'State + readout'],
		['gate', 'Gate + RMSNorm'],
		['output', 'Output + residual'],
		['qkv', 'Q · K · V'],
		['attention', 'Attention'],
		['rope', 'RoPE'],
		['ffn', 'Gated GELU'],
		['classifier', 'Next token']
	];
	let attentionPart = 'context';
	let qkvProjection = 'query';
	let mambaHead = 0,
		channel = 0,
		stateCoordinate = 0,
		attentionCoordinate = 0;
	let convStream = 'x',
		ffnStage = 'gelu',
		projectionIndex = 0;
	let ropeKind: 'query' | 'key' = 'query';
	let ropeCoordinate = 0;
	const f = (v: number | undefined) => (v == null ? '—' : Number.isFinite(v) ? v.toFixed(6) : '−∞');
	const clamp = (v: number, max: number) => Math.min(max, Math.max(0, Math.trunc(Number(v) || 0)));
	$: data = $hybridLayer;
	$: recording = $hybridRecording;
	$: document = recording?.document;
	$: tokenOptions = (document?.tokens ?? []).map((text, value) => ({
		value,
		label: `${value}: ${text}`
	}));
	$: common = recording?.common;
	$: count = document?.sequenceLength ?? 0;
	$: token = clamp($moeTokenIndex, Math.max(0, count - 1));
	$: head = clamp($attentionHeadIdx, 31);
	$: mambaHead = clamp(mambaHead, 63);
	$: channel = clamp(channel, 3);
	$: stateCoordinate = clamp(stateCoordinate, 127);
	$: attentionCoordinate = clamp(attentionCoordinate, 127);
	$: ropeCoordinate = clamp(ropeCoordinate, 63);
	$: isHybrid = !!data?.descriptor.tensors.query;
	$: current = stage === 'mamba' ? 'scan' : stage;
	$: selectedLabel = stages.find(([id]) => id === current)?.[1] ?? 'Recorded values';
	$: mambaStage = ['convolution', 'scan', 'gate'].includes(current);
	$: attentionStage = ['attention', 'rope'].includes(current);
	$: convIndex =
		convStream === 'x'
			? mambaHead * 64 + channel
			: (convStream === 'B' ? 4096 : 4224) + stateCoordinate;
	$: conv = data ? convolution(data, token, convIndex) : null;
	$: state = data ? stateStep(data, token, mambaHead, channel, stateCoordinate) : null;
	$: scan = data ? readout(data, token, mambaHead, channel) : null;
	$: gated = data ? gatedNorm(data, token, mambaHead * 64 + channel) : null;
	$: rotary =
		data && common && isHybrid
			? ropePair(data, common, token, head, ropeCoordinate, ropeKind)
			: null;
	$: projectionKind =
		current === 'projection'
			? ['inWeights', 'norm1', 'mambaProjection']
			: current === 'ffn' && !['gelu', 'activation'].includes(ffnStage)
				? ffnStage === 'down'
					? ['downWeights', 'product', 'down']
					: [
							ffnStage === 'up' ? 'upWeights' : 'gateWeights',
							'ffnInput',
							ffnStage === 'up' ? 'up' : 'gateFFN'
						]
				: ['outWeights', 'gated', 'mambaOutput'];
	$: projected =
		data && (current !== 'ffn' || isHybrid)
			? projection(
					data,
					token,
					projectionKind[0],
					projectionKind[1],
					projectionKind[2],
					projectionIndex
				)
			: null;
	$: logits = common ? tensor(common, 'logits') : new Float32Array();
	$: logitMax = logits.length ? Math.max(...logits) : 0;
	$: partition = logits.reduce((sum, value) => sum + Math.exp(value - logitMax), 0);
	function choose(id: string) {
		stage = id;
		hybridInspector.set(id);
	}
	function selectHead(value: number) {
		attentionHeadIdx.set(value);
		attentionHeadIdxTemp.set(value);
	}
	function close() {
		dispatch('close');
	}
</script>

<svelte:window
	on:keydown={(event) => {
		if (event.key === 'Escape') close();
	}}
/>
<div
	class="hybrid-inspector-overlay"
	role="presentation"
	on:click={(event) => {
		if (event.currentTarget === event.target) close();
	}}
>
	<div
		class="hybrid-inspector"
		role="dialog"
		aria-modal="true"
		aria-label={`Zamba2 ${selectedLabel}`}
		tabindex="-1"
	>
		<header>
			<div>
				<h2>Zamba2 · {selectedLabel}</h2>
				<p>
					Block {$blockIdx + 1} of 38 · {isHybrid ? 'Transformer + Mamba2' : 'Mamba2'} · token {token}
					“{document?.tokens[token]}”
				</p>
			</div>
			<button class="close" aria-label="Close recorded hybrid operation" on:click={close}>×</button>
		</header>
		<div class="body">
			<TokenInspectorLayout gap="14px"
				><TokenPicker
					pageSize={6}
					slot="tokens"
					label="Hybrid token"
					options={tokenOptions}
					bind:value={$moeTokenIndex}
				/>
				<BlockChooser
					items={stages.map(([id, label]) => ({
						id,
						label,
						disabled: ['attention', 'rope', 'ffn'].includes(id) && !isHybrid
					}))}
					selected={current}
					on:select={(e) => {
						((e) => choose(e.detail))(e);
						matrixPopup.set(false);
					}}
				/>
				<div class="controls">
					{#if mambaStage}<label
							>Mamba head <select bind:value={mambaHead}
								>{#each Array.from({ length: 64 }, (_, i) => i) as h}<option value={h}
										>{h + 1} / 64</option
									>{/each}</select
							></label
						>
						<div class="control">
							<span>Channel</span>
							<div class="segmented" aria-label="Recorded state channel">
								{#each [0, 1, 2, 3] as c}<button
										class:active={channel === c}
										on:click={() => (channel = c)}>{c}</button
									>{/each}
							</div>
						</div>{/if}
					{#if current === 'scan' || current === 'convolution'}<label
							>State coordinate <input
								type="number"
								min="0"
								max="127"
								step="1"
								bind:value={stateCoordinate}
							/></label
						>{/if}
					{#if attentionStage}<label
							>Attention head <select
								value={head}
								on:change={(e) => selectHead(Number(e.currentTarget.value))}
								>{#each Array.from({ length: 32 }, (_, i) => i) as h}<option value={h}
										>{h + 1} / 32</option
									>{/each}</select
							></label
						>{/if}
					{#if current === 'attention'}<label
							>V coordinate <input
								type="number"
								min="0"
								max="127"
								bind:value={attentionCoordinate}
							/></label
						>{/if}
					{#if current === 'rope'}<div class="segmented" aria-label="Rotated vector">
							<button class:active={ropeKind === 'query'} on:click={() => (ropeKind = 'query')}
								>Query</button
							><button class:active={ropeKind === 'key'} on:click={() => (ropeKind = 'key')}
								>Key</button
							>
						</div>
						<label
							>Coordinate <input type="number" min="0" max="63" bind:value={ropeCoordinate} /> ↔ {ropeCoordinate +
								64}</label
						>{/if}
				</div>
				<div class="content">
					{#if data && common && document}
						{#if ['projection', 'convolution', 'scan', 'gate', 'output'].includes(current)}<TensorBandFlow
								kind="mamba"
								label="Mamba2 mixer"
								selected={current}
								token={document.tokens[token]}
								on:select={(e) => {
									((e) => choose(e.detail))(e);
									openMatrix();
								}}
							/>
						{:else if current === 'qkv' && isHybrid}<TensorBandFlow
								kind="qkv"
								label="Query Key Value · base + adapter"
								hiddenSize={4096}
								heads={32}
								kvHeads={32}
								headSize={128}
								selected={qkvProjection}
								on:select={(e) => {
									qkvProjection = e.detail;
									openMatrix();
								}}
							/>
						{:else if current === 'attention'}<TensorBandFlow
								kind="attention"
								label="Transformer attention"
								headSize={128}
								sequenceLength={count}
								scale="÷ 8"
								causal
								rotary
								selected={attentionPart}
								on:select={(e) => {
									((e) => (attentionPart = e.detail))(e);
									openMatrix();
								}}
							/>
						{:else if current === 'rope'}<TensorBandFlow
								kind="rope"
								label="RoPE · Rotate Q and K"
								headSize={128}
								selected={ropeKind}
								on:select={(e) => {
									((e) => {
										if (e.detail !== 'value') ropeKind = e.detail as typeof ropeKind;
									})(e);
									openMatrix();
								}}
							/>
						{:else if current === 'classifier'}<TensorBandFlow
								kind="classifier"
								label="Wvocab"
								hiddenSize={2048}
								outputSize={logits.length}
								selected="classifier"
								on:select={() => openMatrix()}
							/>{/if}
						{#if current === 'ffn' && isHybrid}<FeedForwardFlow
								denseStyle
								projectionsOnly
								dimensional
								hiddenSize={2048}
								intermediateSize={8192}
								activation="GELU"
								selected={ffnStage === 'gelu' ? 'product' : ffnStage}
								token={document.tokens[token]}
								on:select={(e) => {
									((e) => (ffnStage = e.detail === 'product' ? 'gelu' : e.detail))(e);
									openMatrix();
								}}
							/>{/if}

						{#if current === 'convolution' && conv}<h3>Convolution + SiLU · 4,352 channels</h3>
							<div class="formula">
								Recorded channel {convIndex}: {f(conv.raw)} → SiLU → <b>{f(conv.output)}</b>
							</div>
							<details class="recorded-details">
								<summary>Recorded values and products</summary>
								<div class="subheading">
									<h3>Four-token causal depthwise convolution</h3>
									<div class="segmented" aria-label="Convolution stream">
										{#each ['x', 'B', 'C'] as stream}<button
												class:active={convStream === stream}
												on:click={() => (convStream = stream)}>{stream}</button
											>{/each}
									</div>
								</div>
								<p>
									Channel {convIndex} of 4,352 · {convStream === 'x'
										? `x for head ${mambaHead + 1}, channel ${channel}`
										: `${convStream} state coordinate ${stateCoordinate}, shared across heads`} · future
									tokens are never used.
								</p>
								<table>
									<thead
										><tr
											><th>Source token</th><th>Recorded input</th><th>Kernel weight</th><th
												>Product</th
											></tr
										></thead
									><tbody
										>{#each conv.terms as term}<tr
												><td
													>{term.position < 0
														? 'Left padding (zero)'
														: `${term.position}: ${document.tokens[term.position]}`}</td
												><td>{f(term.input)}</td><td>{f(term.weight)}</td><td>{f(term.product)}</td
												></tr
											>{/each}</tbody
									>
								</table>
								<div class="formula">
									Σ four products + bias {f(conv.bias)} = {f(conv.reconstructed)}<br />Recorded
									convolution = {f(conv.raw)} → SiLU(x) = x / (1 + e<sup>−x</sup>) →
									<strong>{f(conv.output)}</strong>
								</div>
								<p class="note">
									The same causal convolution supplies x, B and C. B and C each have 128
									coordinates; x has 64 heads × 64 channels.
								</p>
							</details>{:else if current === 'scan' && state && scan}<h3>
								Selective state scan · 4,096 outputs
							</h3>
							<div class="formula">
								sₜ = exp(ΔA) ⊙ sₜ₋₁ + ΔBx · yₜ = Csₜ + Dx<br />Recorded readout:
								<b>{f(scan?.recorded)}</b>
								· head {mambaHead + 1}, channel {channel}
							</div>
							<details class="recorded-details">
								<summary>Recorded values and products</summary>
								<div class="formula">
									Δ = max(0.001, softplus(dtRaw + dtBias)) · decay = e<sup>ΔA</sup><br />S<sub
										>t</sub
									>[h,c,s] = decay × S<sub>t−1</sub>[h,c,s] + Δ × B<sub>t</sub>[s] × x<sub>t</sub
									>[h,c]
								</div>
								<div class="facts">
									<div><span>dtRaw + bias</span><b>{f(state.dtRaw)} + {f(state.bias)}</b></div>
									<div><span>Recorded Δ</span><b>{f(state.dt)}</b></div>
									<div><span>A · decay</span><b>{f(state.A)} · {f(state.decay)}</b></div>
									<div><span>B[{stateCoordinate}] · x</span><b>{f(state.B)} · {f(state.x)}</b></div>
								</div>
								<div class="state-equation">
									<div><span>Previous state</span><b>{f(state.previous)}</b></div>
									<span>×</span>
									<div><span>Decay</span><b>{f(state.decay)}</b></div>
									<span>+</span>
									<div><span>Δ × B × x</span><b>{f(state.injection)}</b></div>
									<span>=</span>
									<div class="result"><span>Reconstructed state</span><b>{f(state.saved)}</b></div>
								</div>
								<div class="split">
									<div>
										<h3>State over the actual token sequence</h3>
										<table>
											<thead
												><tr><th>Token</th><th>S[h,c,{stateCoordinate}]</th><th>C × state</th></tr
												></thead
											><tbody
												>{#each document.tokens as text, t}{@const step = stateStep(
														data,
														t,
														mambaHead,
														channel,
														stateCoordinate
													)}<tr class:selected={t === token} on:click={() => moeTokenIndex.set(t)}
														><td
															><button class="text-button" on:click={() => moeTokenIndex.set(t)}
																>{t}: {text}</button
															></td
														><td>{f(step.saved)}</td><td>{f(step.contribution)}</td></tr
													>{/each}</tbody
											>
										</table>
									</div>
									<div>
										<h3>Read all 128 state coordinates</h3>
										<div class="formula small">
											y[h,c] = Σ<sub>s=0…127</sub> S[h,c,s] × C[s] + D[h] × x[h,c]
										</div>
										<dl>
											<dt>Selected contribution ({stateCoordinate})</dt>
											<dd>{f(scan.terms[stateCoordinate])}</dd>
											<dt>Other 127 contributions</dt>
											<dd>
												{f(scan.terms.reduce((a, b) => a + b, 0) - scan.terms[stateCoordinate])}
											</dd>
											<dt>Skip D × x</dt>
											<dd>{f(scan.skip)}</dd>
											<dt>Sequential reconstruction</dt>
											<dd>{f(scan.reconstructed)}</dd>
											<dt>Recorded scan-kernel output</dt>
											<dd class="strong">{f(scan.recorded)}</dd>
											<dt>Absolute difference</dt>
											<dd>{Math.abs(scan.reconstructed - scan.recorded).toExponential(2)}</dd>
										</dl>
									</div>
								</div>
								<p class="note">
									State values were reconstructed during capture from the recorded x, B, C, Δ and A.
									The independent model scan output is recorded directly. All 64 heads and all 128
									state coordinates are available for the first four channels of each head; the
									model computes all 64 channels.
								</p>
							</details>{:else if current === 'gate' && gated}<h3>
								Gate × readout → RMSNorm · 4,096D
							</h3>
							<div class="formula">
								{f(gated?.scan)} × SiLU({f(gated?.gate)}) → RMSNorm → <b>{f(gated?.recorded)}</b>
							</div>
							<details class="recorded-details">
								<summary>Recorded values and products</summary>
								<h3>Gate first, then RMSNorm over all 4,096 channels</h3>
								<div class="formula">
									z = y ⊙ SiLU(gate)<br />out[d] = z[d] / √(mean(z²) + 0.00001) × γ[d]
								</div>
								<div class="facts">
									<div><span>Mean square over 4,096 values</span><b>{f(gated.meanSquare)}</b></div>
									<div><span>RMS denominator</span><b>{f(gated.denominator)}</b></div>
								</div>
								<table>
									<thead
										><tr
											><th>Head channel</th><th>Recorded y</th><th>Gate</th><th>SiLU(gate)</th><th
												>y × SiLU</th
											><th>γ</th><th>Recorded norm output</th></tr
										></thead
									><tbody
										>{#each [0, 1, 2, 3] as c}{@const g = gatedNorm(
												data,
												token,
												mambaHead * 64 + c
											)}<tr class:selected={c === channel}
												><td
													><button class="text-button" on:click={() => (channel = c)}>{c}</button
													></td
												><td>{f(g.scan)}</td><td>{f(g.gate)}</td><td>{f(g.activation)}</td><td
													>{f(g.product)}</td
												><td>{f(g.weight)}</td><td>{f(g.recorded)}</td></tr
											>{/each}</tbody
									>
								</table>
								<div class="formula">
									Channel {channel}: {f(gated.product)} / {f(gated.denominator)} × {f(gated.weight)}
									≈
									<strong>{f(gated.recorded)}</strong>
								</div>
								<p class="note">
									The four displayed channels are a window into the 4,096-value vector. The
									denominator uses every recorded value, including the 4,092 values outside this
									table.
								</p>
							</details>{:else if current === 'rope' && rotary}
							<div class="subheading">
								<h3>{ropeKind === 'query' ? 'Q' : 'K'} head {head + 1} · 128 dimensions</h3>
								<span>Position {token} · coordinates {rotary.coordinate} and {rotary.partner}</span>
							</div>
							<div class="rope-view">
								<div>
									<h3>Before RoPE</h3>
									<div class="value">[{rotary.coordinate}] <b>{f(rotary.a)}</b></div>
									<div class="value">[{rotary.partner}] <b>{f(rotary.b)}</b></div>
								</div>
								<div class="circle">
									<svg
										viewBox="0 0 240 225"
										role="img"
										aria-label={`Unit circle: cosine ${f(rotary.cos)}, sine ${f(rotary.sin)}`}
										><circle
											cx="120"
											cy="105"
											r="78"
											fill="none"
											stroke="#d7c6e9"
											stroke-width="2"
										/><path d="M20 105H220M120 7V203" stroke="#e0d4eb" /><text x="207" y="122"
											>cos</text
										><text x="128" y="15">sin</text><line
											x1="120"
											y1="105"
											x2={120 + 78 * rotary.cos}
											y2={105 - 78 * rotary.sin}
											stroke="#9661bf"
											stroke-width="3"
										/><line
											x1="120"
											y1="105"
											x2={120 + 78 * rotary.cos}
											y2="105"
											stroke="#568ac9"
											stroke-width="3"
										/><line
											x1={120 + 78 * rotary.cos}
											y1="105"
											x2={120 + 78 * rotary.cos}
											y2={105 - 78 * rotary.sin}
											stroke="#d2799e"
											stroke-width="2"
											stroke-dasharray="4 3"
										/><circle
											cx={120 + 78 * rotary.cos}
											cy={105 - 78 * rotary.sin}
											r="5"
											fill="#9661bf"
										/><text x="120" y="223" text-anchor="middle"
											>θ ≈ {((rotary.angle * 180) / Math.PI).toFixed(2)}° · radius 1</text
										></svg
									>
									<p>cos θ = {f(rotary.cos)}<br />sin θ = {f(rotary.sin)}</p>
								</div>
								<div>
									<h3>Recorded after RoPE</h3>
									<div class="value">[{rotary.coordinate}] <b>{f(rotary.afterA)}</b></div>
									<div class="value">[{rotary.partner}] <b>{f(rotary.afterB)}</b></div>
								</div>
							</div>
							<div class="formula">
								x′[{rotary.coordinate}] = {f(rotary.a)} × {f(rotary.cos)} − ({f(rotary.b)} × {f(
									rotary.sin
								)}) ≈ <b>{f(rotary.afterA)}</b><br />x′[{rotary.partner}] = {f(rotary.b)} × {f(
									rotary.cos
								)} + ({f(rotary.a)} × {f(rotary.sin)}) ≈ <b>{f(rotary.afterB)}</b>
							</div>
							<p class="note">
								All 128 Q and K coordinates are rotated in 64 fixed pairs. V bypasses RoPE. These
								are the saved pretrained-model tensors and its actual cosine/sine values.
							</p>
						{:else if current === 'qkv' && isHybrid}
							{@const outputName =
								qkvProjection === 'query'
									? 'queryBefore'
									: qkvProjection === 'key'
										? 'keyBefore'
										: 'value'}
							<MatrixProduct
								input="RMSNorm([hidden; embedding])"
								weight={qkvProjection === 'query'
									? '(Wq + adapter)ᵀ'
									: qkvProjection === 'key'
										? '(Wk + adapter)ᵀ'
										: '(Wv + adapter)ᵀ'}
								output={qkvProjection === 'query' ? 'Q' : qkvProjection === 'key' ? 'K' : 'V'}
								inputSize={4096}
								outputSize={4096}
								token={document.tokens[token]}
								inputValues={row(data, 'attentionInput', token)}
								outputValues={Array.from({ length: 32 }, (_, h) =>
									row(data, outputName, h * count + token)
								).flat()}
							/>
							<details class="recorded-details">
								<summary>Recorded Q / K / V · before RoPE</summary>
								<p class="note">
									The current hidden state and original embedding are concatenated and normalized to
									4,096 coordinates. Each projection includes this Transformer block's learned
									adapter and produces 32 × 128 coordinates. Q and K receive RoPE inside attention;
									V stays unchanged. Matrix cells are schematic; the values below are recorded
									outputs including the adapter.
								</p>
								<table>
									<thead
										><tr
											><th>Coordinate</th><th>Q · head {head + 1}</th><th>K · head {head + 1}</th
											><th>V · head {head + 1}</th></tr
										></thead
									><tbody
										>{#each [0, 1, 2, 3] as c}<tr
												><td>{c}</td><td>{f(scalar(data, 'queryBefore', head, token, c))}</td><td
													>{f(scalar(data, 'keyBefore', head, token, c))}</td
												><td>{f(scalar(data, 'value', head, token, c))}</td></tr
											>{/each}</tbody
									>
								</table>
							</details>
						{:else if current === 'qkv'}<p>
								This block is a Mamba mixer. QKV projections are available in its Transformer
								blocks.
							</p>
						{:else if current === 'attention' && isHybrid}
							{#if attentionPart === 'scores'}<MatrixProduct
									input="Query Q′"
									weight="Key K′ᵀ"
									output="Scores"
									inputSize={128}
									outputSize={count}
									token={document.tokens[token]}
								/>
							{:else if attentionPart === 'softmax'}<MatrixProduct
									input="Scaled scores + mask"
									output="Attention A"
									activation="Softmax"
									inputSize={count}
									outputSize={count}
									token={document.tokens[token]}
									outputValues={row(data, 'attention', head * count + token)}
								/>
							{:else}<MatrixProduct
									input="Attention A"
									weight="Value V"
									output="Head context"
									inputSize={count}
									outputSize={128}
									token={document.tokens[token]}
								/>{/if}
							<details class="recorded-details">
								<summary>Recorded attention values and products</summary>
								<div class="formula">
									score(q,k) = Σ<sub>d=0…127</sub> Q′[q,d] × K′[k,d] · attention = softmax(score / 8
									+ causal mask)<br />context[q,{attentionCoordinate}] = Σ<sub>k≤q</sub>
									attention[q,k] × V[k,{attentionCoordinate}]
								</div>
								<table>
									<thead
										><tr
											><th>Key token</th><th>Recorded Q′K′ dot</th><th>Scaled + mask</th><th
												>Attention</th
											><th>V[{attentionCoordinate}]</th><th>Weighted value</th></tr
										></thead
									><tbody
										>{#each document.tokens as text, k}{@const score = scalar(
												data,
												'scores',
												head,
												token,
												k
											)}{@const p = scalar(data, 'attention', head, token, k)}{@const value =
												scalar(data, 'value', head, k, attentionCoordinate)}<tr
												class:masked={k > token}
												><td>{k}: {text}</td><td>{f(score)}</td><td
													>{k > token ? '−∞' : f(score / 8)}</td
												><td>{(p * 100).toFixed(3)}%</td><td>{f(value)}</td><td>{f(p * value)}</td
												></tr
											>{/each}</tbody
									>
								</table>
								<div class="formula">
									Σ weighted values ≈ <strong
										>{f(scalar(data, 'context', head, token, attentionCoordinate))}</strong
									>
									· recorded context, head {head + 1}
								</div>
								<p class="note">
									All 32 heads and all 128 value coordinates are selectable. Zamba2 uses a scale of
									1/8 for this attention. Each of the six Transformer blocks uses its own learned
									adapters with shared base weights.
								</p>
							</details>{:else if current === 'ffn' && isHybrid && ffnStage === 'activation'}<MatrixProduct
								input="Gate G"
								output="GELU(G)"
								activation="GELU"
								inputSize={8192}
								outputSize={8192}
								token={document.tokens[token]}
								inputValues={[0, 1, 2, 3].map((c) => scalar(data, 'gateFFN', token, c))}
								outputValues={[0, 1, 2, 3].map((c) => scalar(data, 'gelu', token, c))}
							/>{:else if current === 'ffn' && isHybrid && ffnStage === 'gelu'}<MatrixProduct
								input="GELU(G)"
								weight="Up U"
								output="Hidden H"
								inputSize={8192}
								outputSize={8192}
								elementwise
								token={document.tokens[token]}
							/>
							<div class="subheading">
								<h3>Gated GELU feed-forward network</h3>
							</div>
							<div class="formula">
								g = (Wgate + Wadapter)[x] · u = (Wup + Wadapter)[x]<br />GELU(g) = ½g × (1 + erf(g /
								√2)) · h = GELU(g) ⊙ u
							</div>
							<table>
								<thead
									><tr
										><th>Coordinate</th><th>Gate g</th><th>Recorded GELU(g)</th><th>Up u</th><th
											>Recorded product h</th
										></tr
									></thead
								><tbody
									>{#each [0, 1, 2, 3] as c}<tr
											><td>{c}</td><td>{f(scalar(data, 'gateFFN', token, c))}</td><td
												>{f(scalar(data, 'gelu', token, c))}</td
											><td>{f(scalar(data, 'up', token, c))}</td><td
												>{f(scalar(data, 'product', token, c))}</td
											></tr
										>{/each}</tbody
								>
							</table>
							<p class="note">
								First 4 of 8,192 coordinates. Gate and up projections include the selected block’s
								real low-rank adapter. The following down projection uses all 8,192 products.
							</p>
						{:else if ['projection', 'output', 'ffn'].includes(current) && projected}<MatrixProduct
								input={current === 'projection'
									? 'RMSNorm(X)'
									: current === 'output'
										? 'Gated H'
										: ffnStage === 'down'
											? 'Hidden H'
											: 'FFN input'}
								weight={current === 'projection'
									? 'Winᵀ'
									: current === 'output'
										? 'Woutᵀ'
										: ffnStage === 'down'
											? 'Wdownᵀ'
											: ffnStage === 'up'
												? '(Wup + adapter)ᵀ'
												: '(Wgate + adapter)ᵀ'}
								output={current === 'projection'
									? 'Gate + xBC + Δ'
									: current === 'output' || ffnStage === 'down'
										? 'Output Y'
										: ffnStage === 'up'
											? 'Up U'
											: 'Gate G'}
								inputSize={projected.input.length}
								outputSize={current === 'projection'
									? 8512
									: current === 'output' || ffnStage === 'down'
										? 2048
										: 8192}
								token={document.tokens[token]}
								inputValues={projected.input}
								outputValues={[projected.output]}
							/>
							<details class="recorded-details">
								<summary>Recorded values and products</summary>
								<div class="subheading">
									<h3>
										{current === 'projection'
											? '2,048D RMSNorm → 8,512 projected values'
											: current === 'output'
												? '4,096D gated output → 2,048D residual update'
												: `${ffnStage === 'down' ? 'Down' : ffnStage === 'up' ? 'Up' : 'Gate'} projection with actual weights`}
									</h3>
									{#if current === 'ffn'}{:else}<div
											class="segmented"
											aria-label="Projection output row"
										>
											{#each [0, 1, 2, 3] as c}<button
													class:active={projectionIndex === c}
													on:click={() => (projectionIndex = c)}>Row {c}</button
												>{/each}
										</div>{/if}
								</div>
								{#if current === 'projection'}<p>
										Split into gate (4,096), convolution inputs x/B/C (4,352), and dtRaw (64). These
										saved matrix rows produce gate coordinates 0–3.
									</p>{/if}
								{#if current === 'ffn'}<div class="control">
										<span>Output row</span>
										<div class="segmented" aria-label="Feed-forward output row">
											{#each [0, 1, 2, 3] as c}<button
													class:active={projectionIndex === c}
													on:click={() => (projectionIndex = c)}>{c}</button
												>{/each}
										</div>
									</div>{/if}
								<table>
									<thead
										><tr
											><th>Input coordinate</th><th>Recorded weight[{projectionIndex},d]</th><th
												>Recorded input[d]</th
											><th>Product</th></tr
										></thead
									><tbody
										>{#each [0, 1, 2, 3] as c}<tr
												><td>{c}</td><td>{f(projected.weights[c])}</td><td
													>{f(projected.input[c])}</td
												><td>{f(projected.weights[c] * projected.input[c])}</td></tr
											>{/each}</tbody
									>
								</table>
								<div class="formula">
									First 4 contributions {f(projected.first)} + remaining {projected.input.length -
										4} contributions
									{f(projected.remaining)} ≈ <b>{f(projected.output)}</b>
								</div>
								{#if current === 'output'}<div class="facts">
										{#each [0, 1, 2, 3] as c}<div>
												<span>Residual output [{c}]</span><b
													>{f(scalar(data, 'input', token, c))} + {f(
														scalar(data, 'mambaOutput', token, c)
													)}</b
												><strong>= {f(scalar(data, 'output', token, c))}</strong>
											</div>{/each}
									</div>
									<p class="note">
										{isHybrid
											? 'In this hybrid block the Transformer output is added before Mamba input RMSNorm. The final residual adds the Mamba result to the original block input.'
											: 'The block adds the projected Mamba output to its original residual input.'}
									</p>{:else if current === 'projection'}<p class="note">
										The input is the recorded RMSNorm result. {isHybrid
											? 'RMSNorm receives the residual stream plus this block’s Transformer output.'
											: 'RMSNorm receives the residual stream.'} Matrix rows 0–3 are captured in full;
										omitted contributions above are calculated from every saved input coordinate.
									</p>{:else}<p class="note">
										First four output rows are captured in full. Gate/up matrices combine shared
										weights and the selected block’s learned adapter; down projection uses all 8,192
										products.
									</p>{/if}
							</details>{:else if current === 'classifier'}<MatrixProduct
								input="Final RMSNorm"
								weight="Wvocabularyᵀ"
								output="Logits"
								inputSize={2048}
								outputSize={32000}
							/>
							<h3>All 38 blocks → final RMSNorm → 32,000 vocabulary logits</h3>
							<p>
								The prediction uses the last token ({count - 1}: {document.tokens[count - 1]}) after
								the complete model, independent of the block being inspected.
							</p>
							<div class="formula">
								logit[v] = Σ<sub>d=0…2047</sub> Wembedding[v,d] × finalHidden[d]<br />P(v) =
								exp(logit[v] − maxLogit) / Σ<sub>j=0…31999</sub> exp(logit[j] − maxLogit)
							</div>
							<table>
								<thead
									><tr
										><th>Candidate token</th><th>First 4 contributions</th><th>Remaining 2,044</th
										><th>Recorded logit</th><th>Probability</th></tr
									></thead
								><tbody
									>{#each Array.from(tensor(common, 'classifierTokenIds')) as id, i}{@const weights =
											row(common, 'classifierWeights', i)}{@const input = row(
											common,
											'finalHidden',
											count - 1
										)}<tr
											><td>{document.vocabulary[id]}</td><td
												>{f(dot(weights.slice(0, 4), input))}</td
											><td>{f(dot(weights, input, 4))}</td><td>{f(logits[id])}</td><td
												>{((Math.exp(logits[id] - logitMax) / partition) * 100).toFixed(3)}%</td
											></tr
										>{/each}</tbody
								>
							</table>
							<div class="formula">Recorded next token: <b>{document.nextToken}</b></div>
							<p class="note">
								The vocabulary projection is tied to the pretrained token embeddings. Probabilities
								are normalized over all 32,000 saved logits, including candidates outside this
								table.
							</p>
						{:else}<p>
								Select a recorded operation for this block. Attention, RoPE and gated GELU are
								available in Transformer + Mamba2 blocks 6, 12, 18, 24, 30 and 36.
							</p>{/if}
					{:else}<p class="loading" role="status">Loading recorded block values…</p>{/if}
				</div>
				<footer>
					Pretrained Zamba2-1.2B · complete 38-block forward pass · calculations use full saved
					precision; numbers are rounded for display. <a
						href={recordedAssetUrl(`${base}/traces/hybrid/manifest.json`)}
						target="_blank"
						rel="noreferrer">Trace provenance ↗</a
					>
				</footer>
			</TokenInspectorLayout>
		</div>
	</div>
</div>

<style>
	.hybrid-inspector-overlay {
		position: fixed;
		inset: 0;
		z-index: 1500;
		display: flex;
		justify-content: center;
		align-items: center;
		padding: 20px;
		background: #ffffff55;
	}
	.hybrid-inspector {
		width: min(1240px, calc(100vw - 40px));
		height: auto;
		max-height: calc(100dvh - 32px);
		display: flex;
		flex-direction: column;
		overflow: hidden;
		background: white;
		border: 1px solid #deceed;
		border-radius: 12px;
		box-shadow: 0 15px 40px #30204022;
		color: #806493;
		font-size: 13px;
	}
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-shrink: 0;
		padding: 18px 24px;
		background: #faf7fc;
		border-bottom: 1px solid #e7def0;
	}
	h2 {
		font-size: 23px;
		color: #68488a;
		margin: 0;
		font-weight: 700;
	}
	header p {
		margin: 6px 0 0;
		color: #94849f;
	}
	.close {
		width: 35px;
		height: 35px;
		line-height: 28px;
		font-size: 27px;
		border: 1px solid #decfea;
		border-radius: 7px;
		background: white;
	}
	.body {
		padding: 16px 24px;
		display: flex;
		flex-direction: column;
		gap: 14px;
		min-height: 0;
		flex: 1;
		overflow: auto;
		scrollbar-gutter: stable;
	}
	.content {
		flex: 1 0 auto;
	}
	nav {
		display: flex;
		flex-wrap: wrap;
		gap: 5px;
		flex-shrink: 0;
	}
	button {
		cursor: pointer;
	}
	button:disabled {
		opacity: 0.36;
		cursor: default;
	}
	button:focus-visible,
	input:focus-visible,
	select:focus-visible {
		outline: 2px solid #aa7dcc;
		outline-offset: 2px;
	}
	nav button,
	.segmented button {
		border: 1px solid #ddcbea;
		background: #fbf8fd;
		padding: 6px 9px;
		font-size: 12px;
	}
	nav button {
		border-radius: 5px;
	}
	button.active {
		background: #aa7dcc;
		border-color: #aa7dcc;
		color: white;
	}
	.controls {
		display: flex;
		align-items: center;
		gap: 12px 18px;
		flex-wrap: wrap;
		flex-shrink: 0;
	}
	.control,
	label {
		display: flex;
		gap: 7px;
		align-items: center;
	}
	.segmented {
		display: flex;
		flex-wrap: wrap;
	}
	.segmented button:first-child {
		border-radius: 5px 0 0 5px;
	}
	.segmented button:last-child {
		border-radius: 0 5px 5px 0;
	}
	select,
	input {
		border: 1px solid #ddcbea;
		border-radius: 5px;
		padding: 5px 8px;
		background: white;
		color: #725a86;
	}
	input {
		width: 65px;
	}
	h3 {
		font-weight: 600;
		font-size: 15px;
		color: #765293;
		margin: 8px 0 12px;
	}
	p {
		line-height: 1.5;
		margin: 8px 0 12px;
	}
	.subheading {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
	}
	.subheading h3 {
		margin: 0;
	}
	.subheading {
		margin-bottom: 12px;
	}
	.formula {
		background: #faf7fc;
		border: 1px solid #ede3f3;
		border-radius: 7px;
		padding: 12px 14px;
		line-height: 1.8;
		margin: 12px 0;
		color: #8055a0;
		font-family: ui-monospace, SFMono-Regular, monospace;
		font-size: 13px;
	}
	.formula.small {
		font-size: 12px;
		padding: 8px 10px;
	}
	table {
		border-collapse: separate;
		border-spacing: 3px;
		width: 100%;
		font-size: 12px;
		table-layout: auto;
	}
	th {
		font-weight: 500;
		padding: 8px 6px;
		color: #a18ab0;
		font-size: 11px;
	}
	td {
		padding: 9px 8px;
		text-align: center;
		font-family: ui-monospace, SFMono-Regular, monospace;
		font-variant-numeric: tabular-nums;
		background: #faf7fc;
		border-radius: 3px;
		color: #765689;
	}
	td:first-child {
		font-family: inherit;
	}
	td:last-child {
		background: #f1e8f8;
	}
	.selected td {
		background: #eadcf4;
	}
	.masked {
		opacity: 0.5;
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 8px;
		margin: 14px 0;
	}
	.facts > div {
		background: #faf7fc;
		border: 1px solid #eee4f4;
		border-radius: 6px;
		padding: 10px;
		display: flex;
		flex-direction: column;
		gap: 7px;
	}
	.facts span {
		font-size: 11px;
		color: #9b88a8;
	}
	.facts b,
	.facts strong {
		font:
			12px ui-monospace,
			SFMono-Regular,
			monospace;
		font-variant-numeric: tabular-nums;
	}
	.facts strong {
		color: #8354a7;
	}
	.state-equation {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 9px;
		margin: 15px 0;
	}
	.state-equation > div {
		display: flex;
		flex: 1;
		flex-direction: column;
		gap: 8px;
		padding: 12px;
		background: #f8f5fb;
		border-radius: 6px;
	}
	.state-equation span {
		font-size: 11px;
	}
	.state-equation b {
		font:
			15px ui-monospace,
			SFMono-Regular,
			monospace;
	}
	.state-equation .result {
		background: #ebddf5;
		color: #7e4da1;
	}
	.split {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 22px;
	}
	.split td {
		padding: 6px;
	}
	dl {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 9px;
		font-size: 12px;
	}
	dt {
		color: #9986a6;
	}
	dd {
		margin: 0;
		font-family: ui-monospace, SFMono-Regular, monospace;
	}
	.strong {
		font-weight: 700;
		color: #7e4da1;
	}
	.note {
		font-size: 12px;
		color: #9b88a8;
		line-height: 1.6;
	}
	.text-button {
		color: inherit;
		font: inherit;
		padding: 0;
		background: none;
		border: 0;
	}
	.rope-view {
		display: grid;
		grid-template-columns: 1fr 260px 1fr;
		align-items: center;
		gap: 20px;
		margin: 12px 0;
	}
	.rope-view h3 {
		text-align: center;
	}
	.value {
		background: #eef5ff;
		color: #608fcb;
		padding: 12px;
		margin: 5px 0;
		display: flex;
		justify-content: space-between;
		border-radius: 5px;
	}
	.value b {
		font-family: ui-monospace, SFMono-Regular, monospace;
	}
	.circle svg {
		width: 240px;
		height: 225px;
		display: block;
		margin: auto;
	}
	.circle svg text {
		fill: #a48bb6;
		font-size: 12px;
	}
	.circle p {
		text-align: center;
		font:
			12px/1.7 ui-monospace,
			SFMono-Regular,
			monospace;
		margin: 5px 0;
	}
	footer {
		font-size: 10px;
		line-height: 1.5;
		color: #a48fb4;
		flex-shrink: 0;
		margin-top: 8px;
	}
	footer a {
		color: #945bba;
	}
	.loading {
		padding: 50px;
		text-align: center;
	}
	@media (max-height: 850px) {
		.hybrid-inspector {
			height: auto;
			max-height: calc(100dvh - 32px);
		}
		.hybrid-inspector-overlay {
			padding: 12px;
		}
		header {
			padding: 12px 20px;
		}
		.body {
			padding: 12px 20px;
			gap: 8px;
		}
		.formula {
			margin: 8px 0;
			padding: 7px 12px;
			line-height: 1.5;
		}
		.facts {
			margin: 8px 0;
		}
		.facts > div {
			padding: 6px;
			gap: 4px;
		}
		.state-equation {
			margin: 8px 0;
		}
		.state-equation > div {
			padding: 7px;
			gap: 5px;
		}
		h3 {
			margin: 6px 0 8px;
			font-size: 14px;
		}
		th {
			padding: 5px;
		}
		dl {
			gap: 6px;
		}
		.split td {
			padding: 4px;
		}
		.note {
			font-size: 11px;
			margin: 7px 0;
		}
		.rope-view {
			margin: 5px 0;
		}
		.circle svg {
			width: 210px;
			height: 197px;
		}
	}
	@media (max-width: 850px) {
		.hybrid-inspector {
			width: calc(100vw - 20px);
		}
		.body {
			overflow: auto;
		}
		.content {
			flex: none;
			min-height: 350px;
		}
		.controls {
			gap: 10px;
		}
		.split {
			grid-template-columns: 1fr;
		}
		.facts {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
		.rope-view {
			grid-template-columns: 1fr 1fr;
		}
		.circle {
			grid-column: 1/-1;
			grid-row: 2;
		}
		.subheading {
			align-items: flex-start;
			flex-direction: column;
		}
		td {
			padding: 7px 4px;
			font-size: 10px;
		}
		.state-equation {
			flex-wrap: wrap;
		}
		header {
			padding: 12px 16px;
		}
		.body {
			padding: 12px;
		}
		h2 {
			font-size: 20px;
		}
	}

	.hybrid-inspector {
		border-color: #e5e7eb;
		border-radius: 8px;
		color: #6b7280;
	}
	header {
		background: #f3f4f6;
		border-color: #e5e7eb;
	}
	h2 {
		color: #111827;
		font-weight: 600;
	}
	header p {
		color: #9ca3af;
	}
	.close {
		color: #6b7280;
		border-color: #e5e7eb;
	}
</style>
