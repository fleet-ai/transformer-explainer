<script lang="ts">
	import { recordedAssetUrl } from '$lib/recorded-json.js';

	import { createEventDispatcher } from 'svelte';
	import { base } from '$app/paths';
	import { blockIdx, attentionHeadIdx, attentionHeadIdxTemp } from '~/store';
	import { vlmRecording, vlmPhase, vlmLayer, vlmSelectedPosition } from '$lib/vlm/store';
	import { row, scalar, tensor } from '$lib/vit/types';
	import VitAttentionMap from '../vit/VitAttentionMap.svelte';
	import RoPEUnitCircle from '../moe/RoPEUnitCircle.svelte';
	import RoPEValueCircle from '../moe/RoPEValueCircle.svelte';
	export let stage = 'attention';
	const dispatch = createEventDispatcher<{ close: void }>();
	let coordinate = 0,
		visualToken = 0,
		ropeKind = 'query';
	$: document = $vlmRecording?.document;
	$: stream = document?.streams[$vlmPhase];
	$: model = stream?.model;
	$: data = $vlmLayer;
	$: common = $vlmRecording?.common[$vlmPhase];
	$: position = Math.min($vlmSelectedPosition, (stream?.sequenceLength ?? 1) - 1);
	$: head = $attentionHeadIdx;
	$: kvHead = Math.floor(head / ((model?.attentionHeads ?? 1) / (model?.keyValueHeads ?? 1)));
	$: stages = [
		['embedding', 'Embedding'],
		['connector', 'Image connector'],
		['norm1', 'Norm 1'],
		['qkv', 'Q · K · V'],
		...($vlmPhase === 'language' ? [['rope', 'RoPE']] : []),
		['attention', 'Attention × V'],
		['norm2', 'Norm 2'],
		['up', 'Up projection'],
		[
			$vlmPhase === 'language' ? 'gate' : 'gelu',
			$vlmPhase === 'language' ? 'SiLU gate × up' : 'GELU'
		],
		['down', 'Down projection'],
		['residual', 'Residuals'],
		['output', 'Language output']
	];
	$: maxCoordinate = ['qkv', 'attention'].includes(stage)
		? 60
		: stage === 'rope'
			? 31
			: stage === 'up' || stage === 'down'
				? 0
				: stage === 'gate' || stage === 'gelu'
					? (model?.intermediateSize ?? 4) - 4
					: (model?.hiddenSize ?? 4) - 4;
	$: if (coordinate > maxCoordinate) coordinate = 0;
	$: coords = Array.from(
		{ length: 4 },
		(_, i) => Math.min(Math.max(0, coordinate), maxCoordinate) + i
	);
	const f = (v: number | undefined) => (v == null ? '—' : v.toFixed(6));
	const sum = (a: number[], b: number[], start = 0) =>
		a.slice(start).reduce((total, v, i) => total + v * b[start + i], 0);
	function selectHead(index: number) {
		attentionHeadIdx.set(index);
		attentionHeadIdxTemp.set(index);
	}
	function dotRow(name: string, c: number) {
		if (!data) return null;
		const input = row(
			data,
			name === 'down' ? ($vlmPhase === 'language' ? 'product' : 'gelu') : 'norm2',
			position
		);
		const weights = row(data, name + 'Weights', c);
		const bias = data.descriptor.tensors[name + 'Bias'] ? scalar(data, name + 'Bias', c) : 0;
		return {
			input,
			weights,
			bias,
			rest: sum(input, weights, 4),
			output: scalar(data, name, position, c)
		};
	}
	function attentionRow(c: number) {
		if (!data || !stream) return null;
		const n = stream.sequenceLength;
		const probabilities = row(data, 'attention', head * n + position);
		const values = Array.from({ length: n }, (_, k) => scalar(data, 'value', kvHead, k, c));
		return {
			probabilities,
			values,
			rest: sum(probabilities, values, 4),
			output: scalar(data, 'context', head, position, c)
		};
	}
	function normalized(c: number) {
		if (!data || !model) return null;
		const name = stage === 'norm2' ? 'norm2' : 'norm1',
			input = row(data, name === 'norm1' ? 'input' : 'residualAfterAttention', position);
		const mean =
			model.normalization === 'layernorm' ? input.reduce((a, b) => a + b, 0) / input.length : 0;
		const variance = input.reduce((a, b) => a + (b - mean) ** 2, 0) / input.length;
		return {
			input: input[c],
			mean,
			variance,
			weight: scalar(data, name + 'Weight', c),
			bias: data.descriptor.tensors[name + 'Bias'] ? scalar(data, name + 'Bias', c) : 0,
			output: scalar(data, name, position, c)
		};
	}
	$: rotaryIndex = Math.min(31, Math.max(0, Math.floor(coordinate)));
	$: cosine = $vlmRecording
		? scalar(
				$vlmRecording.common.language,
				'cos',
				Math.min(position, ($vlmRecording?.document.streams.language.sequenceLength ?? 1) - 1),
				rotaryIndex
			)
		: 1;
	$: sine = $vlmRecording
		? scalar(
				$vlmRecording.common.language,
				'sin',
				Math.min(position, ($vlmRecording?.document.streams.language.sequenceLength ?? 1) - 1),
				rotaryIndex
			)
		: 0;
</script>

<svelte:window
	on:keydown={(e) => {
		if (e.key === 'Escape') dispatch('close');
	}}
/>
<div
	class="vlm-inspector-overlay"
	role="presentation"
	on:click={(e) => {
		if (e.target === e.currentTarget) dispatch('close');
	}}
>
	<div
		class="vlm-inspector"
		role="dialog"
		aria-modal="true"
		aria-label="Recorded SmolVLM operation"
		tabindex="-1"
	>
		<header>
			<div>
				<h2>SmolVLM · {stages.find(([id]) => id === stage)?.[1]}</h2>
				<p>
					{#if stage === 'connector'}After all 12 vision blocks · 256 patches → 16 visual tokens{:else if stage === 'output'}After
						all 30 language blocks · final token → vocabulary{:else}{$vlmPhase === 'vision'
							? 'Vision encoder'
							: 'Language decoder'} · block {$blockIdx + 1} of
						{model?.numLayers} · head {head + 1} · {stream?.tokens[position]}{/if}
				</p>
			</div>
			<button
				class="close"
				aria-label="Close recorded VLM operation"
				on:click={() => dispatch('close')}>×</button
			>
		</header>
		<div class="body">
			<nav>
				{#each stages as [id, label]}<button
						class:active={stage === id}
						on:click={() => {
							stage = id;
							coordinate = 0;
						}}>{label}</button
					>{/each}
			</nav>
			{#if !['connector', 'output'].includes(stage)}<div class="controls">
					<label
						>Position <select aria-label="Position" bind:value={$vlmSelectedPosition}
							>{#each stream?.tokens ?? [] as token, p}<option value={p}>{p}: {token}</option
								>{/each}</select
						></label
					>{#if ['qkv', 'rope', 'attention'].includes(stage)}<label
							>Head <select
								aria-label="Head"
								value={head}
								on:change={(e) => selectHead(Number(e.currentTarget.value))}
								>{#each Array(model?.attentionHeads ?? 0) as _, h}<option value={h}>{h + 1}</option
									>{/each}</select
							></label
						>{/if}{#if maxCoordinate > 0 && !['connector', 'output'].includes(stage)}<label
							>Coordinate <input
								type="number"
								min="0"
								max={maxCoordinate}
								bind:value={coordinate}
							/></label
						>{/if}
				</div>{/if}
			{#if document && stream && model && common && data && $vlmRecording}
				{#key `${$vlmPhase}:${$blockIdx}:${position}:${head}:${coordinate}:${visualToken}:${ropeKind}:${stage}`}
					{#if stage === 'embedding'}{#if $vlmPhase === 'vision'}<div class="formula">
								16 × 16 RGB patches → learned 768D projection + learned position embeddings
							</div>
							<div class="image-equation">
								<img src={`${base}/${document.image.src}`} alt={document.image.alt} />
								<p>
									The recorded processor resizes the image to 256 × 256. All 256 patches enter all
									12 vision blocks. This vision encoder has no CLS token.
								</p>
							</div>
							<table>
								<thead
									><tr
										><th>d</th><th>Projected patch</th><th>Learned position</th><th
											>Encoder input</th
										></tr
									></thead
								><tbody
									>{#each coords as c}<tr
											><td>{c}</td><td>{f(scalar(common, 'patchTokens', position, c))}</td><td
												>{f(scalar(common, 'positionEmbedding', position, c))}</td
											><td>{f(scalar(common, 'embedding', position, c))}</td></tr
										>{/each}</tbody
								>
							</table>{:else}<div class="formula">
								Language input = text token embeddings with 16 projected image embeddings inserted
								at their placeholder positions
							</div>
							<table>
								<thead><tr><th>d</th><th>Recorded input coordinate</th><th>Source</th></tr></thead
								><tbody
									>{#each coords as c}<tr
											><td>{c}</td><td>{f(scalar(common, 'embedding', position, c))}</td><td
												>{document.imageTokenPositions.includes(position)
													? `visual token ${document.imageTokenPositions.indexOf(position)}`
													: 'text embedding lookup'}</td
											></tr
										>{/each}</tbody
								>
							</table>
							<p>
								Visual and text positions participate in the same causal language self-attention.
								Image processing and projection run before these 30 decoder blocks.
							</p>{/if}
					{:else if stage === 'connector'}<div class="formula">
							256 × 768D vision outputs → 4 × 4 pixel shuffle → 16 × 12,288D → learned projection →
							16 × 576D
						</div>
						<label class="controls"
							>Visual token <select bind:value={visualToken}
								>{#each Array(16) as _, i}<option value={i}
										>{i} → language position {document.imageTokenPositions[i]}</option
									>{/each}</select
							></label
						>
						<p>
							The connector rearranges each group of 16 spatial patch vectors, then applies a
							learned matrix. It does not average the patches or apply cross-attention.
						</p>
						{#each [0, 1, 2, 3] as c}{@const input = row(
								$vlmRecording.connector,
								'packed',
								visualToken
							)}{@const weights = row($vlmRecording.connector, 'weights', c)}
							<div class="dot-product">
								<strong>visual[{visualToken},{c}]</strong>
								<div class="terms">
									{#each [0, 1, 2, 3] as d}<span>({f(weights[d])} × {f(input[d])})</span
										>{#if d < 3}<span>+</span>{/if}{/each}
								</div>
								<div>
									+ <span class="remaining"
										>{f(sum(weights, input, 4))}<small>remaining 12,284 products</small></span
									>
									≈
									<strong>{f(scalar($vlmRecording.connector, 'projected', visualToken, c))}</strong>
								</div>
							</div>{/each}
					{:else if stage === 'norm1' || stage === 'norm2'}<div class="formula">
							{model.normalization === 'layernorm'
								? 'LayerNorm(x)[d] = (x[d] − mean(x)) / √(variance(x) + ε) × γ[d] + β[d]'
								: 'RMSNorm(x)[d] = x[d] / √(mean(x²) + ε) × γ[d]'} · ε={model.rmsNormEps}
						</div>
						<table>
							<thead
								><tr
									><th>d</th><th>x</th><th
										>{model.normalization === 'layernorm' ? 'mean' : 'subtracted mean'}</th
									><th>{model.normalization === 'layernorm' ? 'variance' : 'mean(x²)'}</th><th>γ</th
									><th>β</th><th>Recorded result</th></tr
								></thead
							><tbody
								>{#each coords as c}{@const r = normalized(c)}{#if r}<tr
											><td>{c}</td><td>{f(r.input)}</td><td>{f(r.mean)}</td><td>{f(r.variance)}</td
											><td>{f(r.weight)}</td><td>{f(r.bias)}</td><td>{f(r.output)}</td></tr
										>{/if}{/each}</tbody
							>
						</table>
					{:else if stage === 'qkv'}<div class="formula">
							Q = Wq x{model.normalization === 'layernorm' ? ' + bq' : ''} · K = Wk x{model.normalization ===
							'layernorm'
								? ' + bk'
								: ''} · V = Wv x{model.normalization === 'layernorm' ? ' + bv' : ''}
						</div>
						<p>
							{model.attentionHeads} query heads; {model.keyValueHeads} key/value heads; 64 coordinates
							per head. {$vlmPhase === 'language'
								? `Query heads ${kvHead * 3 + 1}–${kvHead * 3 + 3} share KV head ${kvHead + 1}. Q and K below are before RoPE.`
								: 'Q/K/V have independent heads, with learned position information already included in the input.'}
						</p>
						<table>
							<thead
								><tr
									><th>d</th><th>Query head {head + 1}</th><th>Key head {kvHead + 1}</th><th
										>Value head {kvHead + 1}</th
									></tr
								></thead
							><tbody
								>{#each coords as c}<tr
										><td>{c}</td><td
											>{f(
												scalar(
													data,
													$vlmPhase === 'language' ? 'queryBefore' : 'query',
													head,
													position,
													c
												)
											)}</td
										><td
											>{f(
												scalar(
													data,
													$vlmPhase === 'language' ? 'keyBefore' : 'key',
													kvHead,
													position,
													c
												)
											)}</td
										><td>{f(scalar(data, 'value', kvHead, position, c))}</td></tr
									>{/each}</tbody
							>
						</table>
					{:else if stage === 'rope' && $vlmPhase === 'language'}<div class="segmented">
							{#each ['query', 'key'] as kind}<button
									class:active={ropeKind === kind}
									on:click={() => (ropeKind = kind)}
									>{kind === 'query' ? `Query ${head + 1}` : `Shared key ${kvHead + 1}`}</button
								>{/each}
						</div>
						{@const rotaryHead = ropeKind === 'query' ? head : kvHead}
						<div class="rotary">
							<div>
								<h3>{ropeKind === 'query' ? 'Q' : 'K'} coordinate pair</h3>
								<RoPEValueCircle
									compact
									before={[
										scalar(data, ropeKind + 'Before', rotaryHead, position, rotaryIndex),
										scalar(data, ropeKind + 'Before', rotaryHead, position, rotaryIndex + 32)
									]}
									after={[
										scalar(data, ropeKind, rotaryHead, position, rotaryIndex),
										scalar(data, ropeKind, rotaryHead, position, rotaryIndex + 32)
									]}
									indices={[rotaryIndex, rotaryIndex + 32]}
									{cosine}
									{sine}
									selectedCoordinate={rotaryIndex}
								/>
							</div>
							<div>
								<h3>cos / sin · unit circle</h3>
								<RoPEUnitCircle compact {cosine} {sine} pairIndex={rotaryIndex} selected />
							</div>
						</div>
						{#each [rotaryIndex, Math.min(31, rotaryIndex === 31 ? 30 : rotaryIndex + 1)] as c}{@const h =
								ropeKind === 'query' ? head : kvHead}{@const a = scalar(
								data,
								ropeKind + 'Before',
								h,
								position,
								c
							)}{@const b = scalar(data, ropeKind + 'Before', h, position, c + 32)}{@const co =
								scalar(common, 'cos', position, c)}{@const si = scalar(common, 'sin', position, c)}
							<div class="dot-product">
								x′[{c}] = {f(a)} × {f(co)} − ({f(b)} × {f(si)}) ≈
								<strong>{f(scalar(data, ropeKind, h, position, c))}</strong><br />x′[{c + 32}] = {f(
									b
								)} × {f(co)} + ({f(a)} × {f(si)}) ≈
								<strong>{f(scalar(data, ropeKind, h, position, c + 32))}</strong>
							</div>{/each}
						<p>
							V bypasses RoPE. Query heads sharing this KV head use the same rotated K and the same
							original V.
						</p>
					{:else if stage === 'attention'}<div class="attention-detail">
							<VitAttentionMap
								layer={data}
								{head}
								query={position}
								count={stream.sequenceLength}
								causal={$vlmPhase === 'language'}
								on:query={(e) => vlmSelectedPosition.set(e.detail)}
							/>
							<div>
								<div class="formula">
									A = softmax(QKᵀ / 8{$vlmPhase === 'language' ? ' + causal mask' : ''})<br
									/>context[q,d] = Σₖ A[q,k] × V[k,d]
								</div>
								<p>
									The matrix contains all {stream.sequenceLength} queries and keys for head {head +
										1}. {$vlmPhase === 'language'
										? 'The same mask applies to text and visual token positions.'
										: 'Every patch attends to the whole image.'}
								</p>
								<table>
									<thead><tr><th>Key</th><th>Q · K</th><th>Recorded attention</th></tr></thead
									><tbody
										>{#each [0, 1, 2, 3] as k}<tr
												><td>{stream.tokens[k]}</td><td
													>{f(scalar(data, 'scores', head, position, k))}</td
												><td>{f(scalar(data, 'attention', head, position, k))}</td></tr
											>{/each}</tbody
									>
								</table>
							</div>
						</div>
						{#each coords as c}{@const r = attentionRow(c)}{#if r}<div class="dot-product">
									<strong>context[{position},{c}]</strong>
									<div class="terms">
										{#each [0, 1, 2, 3] as k}<span
												>({f(r.probabilities[k])} × {f(r.values[k])})</span
											>{#if k < 3}<span>+</span>{/if}{/each}
									</div>
									<div>
										+ <span class="remaining"
											>{f(r.rest)}
											<small>remaining {stream.sequenceLength - 4} positions</small></span
										>
										≈ <strong>{f(r.output)}</strong>
									</div>
								</div>{/if}{/each}
					{:else if stage === 'up' || stage === 'down'}<div class="formula">
							{stage === 'up'
								? 'u = Wup · normalized input'
								: 'y = Wdown · activation'}{$vlmPhase === 'vision' ? ' + bias' : ''}
						</div>
						{#each [0, 1, 2, 3] as c}{@const r = dotRow(stage, c)}{#if r}<div class="dot-product">
									<strong>{stage}[{c}]</strong>
									<div class="terms">
										{#each r.weights.slice(0, 4) as w, d}<span>({f(w)} × {f(r.input[d])})</span
											>{#if d < 3}<span>+</span>{/if}{/each}
									</div>
									<div>
										+ <span class="remaining"
											>{f(r.rest)}<small>remaining {r.input.length - 4} products</small></span
										>{#if $vlmPhase === 'vision'}
											+ {f(r.bias)} <small>bias</small>{/if} ≈ <strong>{f(r.output)}</strong>
									</div>
								</div>{/if}{/each}
					{:else if stage === 'gate' && $vlmPhase === 'language'}<div class="formula">
							g = Wgate · x; u = Wup · x<br />h[d] = SiLU(g[d]) × u[d] = g[d] / (1 + exp(−g[d])) ×
							u[d]
						</div>
						<table>
							<thead
								><tr><th>d</th><th>Gate g</th><th>SiLU(g)</th><th>× Up u</th><th>Product h</th></tr
								></thead
							><tbody
								>{#each coords as c}<tr
										><td>{c}</td><td>{f(scalar(data, 'gate', position, c))}</td><td
											>{f(scalar(data, 'siluGate', position, c))}</td
										><td>{f(scalar(data, 'up', position, c))}</td><td
											>{f(scalar(data, 'product', position, c))}</td
										></tr
									>{/each}</tbody
							>
						</table>
					{:else if stage === 'gelu' && $vlmPhase === 'vision'}<div class="formula">
							GELU(u) = ½u · (1 + tanh(√(2/π) · (u + 0.044715u³)))
						</div>
						<table>
							<thead><tr><th>d</th><th>Up projection</th><th>Recorded GELU</th></tr></thead><tbody
								>{#each coords as c}<tr
										><td>{c}</td><td>{f(scalar(data, 'up', position, c))}</td><td
											>{f(scalar(data, 'gelu', position, c))}</td
										></tr
									>{/each}</tbody
							>
						</table>
						<p>This vision encoder uses the pretrained model's tanh GELU approximation.</p>
					{:else if stage === 'residual'}<div class="formula">
							r₁ = block input + attention output<br />block output = r₁ + feed-forward output
						</div>
						<table>
							<thead
								><tr
									><th>d</th><th>Input</th><th>+ Attention</th><th>= r₁</th><th>+ FFN</th><th
										>Output</th
									></tr
								></thead
							><tbody
								>{#each coords as c}<tr
										><td>{c}</td><td>{f(scalar(data, 'input', position, c))}</td><td
											>{f(scalar(data, 'attentionOutput', position, c))}</td
										><td>{f(scalar(data, 'residualAfterAttention', position, c))}</td><td
											>{f(scalar(data, 'down', position, c))}</td
										><td>{f(scalar(data, 'output', position, c))}</td></tr
									>{/each}</tbody
							>
						</table>
					{:else if stage === 'output'}{@const lang = $vlmRecording.common.language}{@const n =
							document.streams.language.sequenceLength}{@const input = row(
							lang,
							'finalHidden',
							n - 1
						)}
						<div class="formula">
							logits = Wvocabulary · RMSNorm(language block 30)[last position]<br />p(token) =
							softmax(all 49,280 vocabulary logits)
						</div>
						<table>
							<thead
								><tr
									><th>Token</th><th>First 4 products</th><th>Remaining 572</th><th
										>Recorded logit</th
									></tr
								></thead
							><tbody
								>{#each Array.from(tensor(lang, 'classifierTokenIds')) as id, i}{@const weights =
										row(lang, 'classifierWeights', i)}<tr
										><td>{document.vocabulary[id]}</td><td
											>{f(sum(weights.slice(0, 4), input.slice(0, 4)))}</td
										><td>{f(sum(weights, input, 4))}</td><td>{f(scalar(lang, 'logits', id))}</td
										></tr
									>{/each}</tbody
							>
						</table>
						<p><strong>Recorded greedy continuation:</strong> {document.completion}</p>
						<p>
							These are the model's recorded predictions for this test image, including any
							incorrect color claims.
						</p>
					{/if}
					<footer>
						Full pretrained SmolVLM inference: 12 vision blocks + connector + 30 language blocks.
						Values are float32 recordings, rounded for display. <a
							href={recordedAssetUrl(`${base}/traces/vlm/manifest.json`)}
							target="_blank"
							rel="noreferrer">Trace provenance ↗</a
						>
					</footer>
				{/key}
			{:else}<p>Loading the selected block's recorded tensors…</p>{/if}
		</div>
	</div>
</div>

<style>
	.vlm-inspector-overlay {
		position: fixed;
		inset: 0;
		z-index: 1500;
		display: flex;
		align-items: center;
		justify-content: center;
		background: #ffffff55;
		padding: 20px;
	}
	.vlm-inspector {
		width: min(1100px, calc(100vw - 40px));
		max-height: calc(100vh - 40px);
		display: flex;
		flex-direction: column;
		background: white;
		border: 1px solid #deceed;
		border-radius: 12px;
		box-shadow: 0 15px 40px #30204022;
		color: #8b6d9f;
		overflow: hidden;
	}
	header {
		padding: 18px 24px;
		background: #faf7fc;
		border-bottom: 1px solid #e7def0;
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-shrink: 0;
	}
	h2 {
		font-size: 22px;
		font-weight: 700;
		color: #68488a;
		margin: 0;
	}
	header p {
		font-size: 12px;
		margin: 7px 0 0;
	}
	.close {
		width: 34px;
		height: 34px;
		border: 1px solid #decfea;
		border-radius: 7px;
		background: white;
		font-size: 26px;
		line-height: 26px;
	}
	.body {
		padding: 18px 24px;
		overflow: auto;
		min-height: 0;
	}
	nav,
	.segmented {
		display: flex;
		flex-wrap: wrap;
		gap: 0;
	}
	nav {
		gap: 4px;
		margin-bottom: 14px;
	}
	nav button,
	.segmented button {
		border: 1px solid #ddcbea;
		padding: 7px 10px;
		background: #fbf8fd;
		font-size: 11px;
	}
	button.active {
		background: #aa7dcc;
		color: white;
		border-color: #aa7dcc;
	}
	.controls {
		display: flex;
		align-items: center;
		gap: 18px;
		flex-wrap: wrap;
		margin-bottom: 16px;
		font-size: 12px;
	}
	.controls label {
		display: flex;
		align-items: center;
		gap: 7px;
	}
	.controls select,
	.controls input {
		border: 1px solid #ddcbea;
		background: white;
		color: #836595;
		padding: 6px;
		border-radius: 4px;
		font-size: 12px;
	}
	.controls input {
		width: 70px;
	}
	.formula {
		font-family: Georgia, serif;
		font-size: 19px;
		color: #81549f;
		background: #faf7fc;
		padding: 12px 14px;
		margin: 10px 0 14px;
		border-radius: 5px;
	}
	table {
		width: 100%;
		font-size: 12px;
		border-collapse: separate;
		border-spacing: 3px 4px;
		margin: 10px 0;
	}
	th {
		font-weight: 500;
		padding: 7px;
		color: #a48ab6;
		text-align: left;
	}
	td {
		background: #faf7fc;
		padding: 9px;
		font-family: monospace;
		color: #816296;
	}
	td:last-child {
		background: #eee1f8;
		color: #8552a8;
	}
	p {
		font-size: 12px;
		line-height: 1.6;
		margin: 10px 0;
	}
	.image-equation {
		display: flex;
		align-items: center;
		gap: 24px;
		font-size: 13px;
		line-height: 2;
	}
	.image-equation img {
		width: 150px;
		border: 1px solid #eee3f5;
	}
	.dot-product {
		border: 1px solid #e3d4ee;
		padding: 10px 14px;
		border-radius: 5px;
		margin: 7px 0;
		font-family: monospace;
		font-size: 12px;
	}
	.dot-product > strong {
		font-size: 13px;
		color: #8050a4;
	}
	.terms {
		display: flex;
		gap: 7px;
		flex-wrap: wrap;
		margin: 7px 0;
	}
	.remaining {
		display: inline-block;
		background: #f2e8f9;
		border-radius: 3px;
		padding: 4px 8px;
	}
	.remaining small {
		display: block;
		font-family: Arial, sans-serif;
		color: #a68db7;
	}
	.dot-product small {
		font-size: 10px;
	}
	.attention-detail {
		display: flex;
		gap: 24px;
	}
	.attention-detail > div:last-child {
		flex: 1;
		min-width: 0;
	}
	footer {
		font-size: 10px;
		margin-top: 15px;
		padding-top: 12px;
		border-top: 1px solid #eee5f5;
	}
	a {
		color: #9a61bf;
	}
	@media (max-width: 800px) {
		.attention-detail {
			flex-direction: column;
		}
		.vlm-inspector-overlay {
			padding: 8px;
		}
		.vlm-inspector {
			width: calc(100vw - 16px);
			max-height: calc(100vh - 16px);
		}
	}
	.rotary {
		display: flex;
		align-items: center;
		gap: 20px;
		justify-content: center;
	}
	.rotary > div {
		flex: 1;
		max-width: 380px;
		min-width: 0;
		text-align: center;
	}
</style>
