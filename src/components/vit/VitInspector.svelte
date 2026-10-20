<script lang="ts">
	import { setContext, tick } from 'svelte';
	import { writable } from 'svelte/store';
	const matrixPopup = writable(false);
	setContext('matrix-popup', matrixPopup);
	setContext('matrix-popup-label', 'DeiT');
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
	import { attentionHeadIdx, attentionHeadIdxTemp, blockIdx } from '~/store';
	import { vitRecording, vitSelectedPosition } from '$lib/vit/store';
	import { tensor, row, scalar, type TensorData } from '$lib/vit/types';
	import VitAttentionMap from './VitAttentionMap.svelte';
	export let stage = 'attention';
	const dispatch = createEventDispatcher<{ close: void }>();
	const stages = [
		['embedding', 'Patches + position'],
		['norm1', 'LayerNorm 1'],
		['qkv', 'Q · K · V'],
		['attention', 'Attention × V'],
		['norm2', 'LayerNorm 2'],
		['up', 'Up projection'],
		['gelu', 'GELU'],
		['down', 'Down projection'],
		['residual', 'Residuals'],
		['classifier', 'CLS classifier']
	];
	let attentionPart = 'context';
	let coordinate = 0,
		projection = 'query';
	$: data = $vitRecording?.layers[$blockIdx];
	$: common = $vitRecording?.common;
	$: document = $vitRecording?.document;
	$: tokenOptions = (document?.tokens ?? []).map((text, value) => ({
		value,
		label: `${value}: ${text}`
	}));
	$: position = $vitSelectedPosition;
	$: head = $attentionHeadIdx;
	$: selectedStage = stages.find((s) => s[0] === stage)?.[1] ?? 'Recorded operation';
	$: maxCoordinate =
		stage === 'qkv' || stage === 'attention'
			? 60
			: stage === 'up' || stage === 'down'
				? 0
				: stage === 'gelu'
					? 764
					: 188;
	$: if (coordinate > maxCoordinate) coordinate = 0;
	$: coords = Array.from(
		{ length: 4 },
		(_, i) => Math.max(0, Math.min(maxCoordinate, coordinate)) + i
	);
	const f = (n: number | undefined) => (n == null ? '—' : n.toFixed(6));
	function selectHead(h: number) {
		attentionHeadIdx.set(h);
		attentionHeadIdxTemp.set(h);
	}
	function dot(a: number[], b: number[], start = 0) {
		return a.slice(start).reduce((sum, v, i) => sum + v * b[i + start], 0);
	}
	function projectionRow(which: string, c: number) {
		if (!data) return null;
		const isQkv = ['query', 'key', 'value'].includes(which),
			outputIndex = isQkv ? head * 64 + c : c;
		const input = row(
			data,
			which === 'down' ? 'gelu' : which === 'up' ? 'norm2' : 'norm1',
			position
		);
		const weights = row(data, which + 'Weights', outputIndex);
		const bias = scalar(data, which + 'Bias', outputIndex);
		const output = isQkv
			? scalar(data, which, head, position, c)
			: scalar(data, which, position, c);
		return {
			input,
			weights,
			bias,
			output,
			remaining: dot(weights, input, 4),
			reconstructed: dot(weights, input) + bias
		};
	}
	function contextRow(c: number) {
		if (!data) return null;
		const probs = row(data, 'attention', head * 197 + position);
		const values = Array.from({ length: 197 }, (_, k) => scalar(data, 'value', head, k, c));
		return {
			probs,
			values,
			remaining: dot(probs, values, 4),
			output: scalar(data, 'context', head, position, c),
			mass: probs.reduce((a, b) => a + b, 0)
		};
	}
	function normRows(name: string) {
		if (!data) return [];
		const input = row(data, name === 'norm1' ? 'input' : 'residualAfterAttention', position);
		const mean = input.reduce((a, b) => a + b, 0) / 192;
		const variance = input.reduce((a, b) => a + (b - mean) ** 2, 0) / 192;
		return coords.map((c) => ({
			c,
			input: input[c],
			mean,
			variance,
			weight: scalar(data, name + 'Weight', c),
			bias: scalar(data, name + 'Bias', c),
			output: scalar(data, name, position, c)
		}));
	}
	$: classIndices = common
		? Array.from(tensor(common, 'logits'))
				.map((value, id) => ({ value, id }))
				.sort((a, b) => b.value - a.value)
				.slice(0, 4)
		: [];
</script>

<svelte:window
	on:keydown={(e) => {
		if (e.key === 'Escape') dispatch('close');
	}}
/>
<div
	class="vit-inspector-overlay"
	role="presentation"
	on:click={(e) => {
		if (e.target === e.currentTarget) dispatch('close');
	}}
>
	<section
		class="vit-inspector"
		role="dialog"
		aria-modal="true"
		aria-label={`DeiT ${selectedStage}`}
		tabindex="-1"
	>
		<header>
			<div>
				<h2>DeiT · {selectedStage}</h2>
				<p>Block {$blockIdx + 1} of 12 · Head {head + 1} of 3 · {document?.tokens[position]}</p>
			</div>
			<button
				class="close"
				aria-label="Close recorded ViT operation"
				on:click={() => dispatch('close')}>×</button
			>
		</header>
		<div class="body">
			<TokenInspectorLayout gap="0px"
				><TokenPicker
					pageSize={6}
					slot="tokens"
					label="ViT token"
					options={tokenOptions}
					bind:value={$vitSelectedPosition}
				/>
				<BlockChooser
					items={stages.map(([id, label]) => ({ id, label }))}
					selected={stage}
					on:select={(e) => {
						((e) => {
							stage = e.detail;
							coordinate = 0;
						})(e);
						matrixPopup.set(false);
					}}
				/>
				<div class="controls">
					{#if ['qkv', 'attention'].includes(stage)}<div class="segmented" aria-label="Head">
							{#each [0, 1, 2] as h}<button class:active={head === h} on:click={() => selectHead(h)}
									>Head {h + 1}</button
								>{/each}
						</div>{/if}{#if maxCoordinate > 0 && stage !== 'classifier'}<label
							>Coordinates <input
								type="number"
								min="0"
								max={maxCoordinate}
								step="4"
								bind:value={coordinate}
							/>–{coordinate + 3}</label
						>{/if}
				</div>
				{#if data && common && document}
					{#if stage === 'attention'}<TensorBandFlow
							kind="attention"
							label="ViT attention"
							selected={attentionPart}
							headSize={64}
							sequenceLength={197}
							scale="÷ √64"
							on:select={(e) => {
								((e) => (attentionPart = e.detail))(e);
								openMatrix();
							}}
						/>
					{:else if stage === 'embedding'}<TensorBandFlow
							kind="embedding"
							label="Patch embeddings"
							hiddenSize={768}
							outputSize={192}
							selected="embedding"
							on:select={() => openMatrix()}
						/>
					{:else if ['norm1', 'norm2'].includes(stage)}<TensorBandFlow
							kind="norm"
							label="Layer normalization"
							normalization="LayerNorm"
							hiddenSize={192}
							outputSize={192}
							selected="norm"
							on:select={() => openMatrix()}
						/>
					{:else if stage === 'residual'}<TensorBandFlow
							kind="residual"
							label="Residual additions"
							hiddenSize={192}
							selected="residual"
							on:select={(e) => {
								((e) => (stage = e.detail))(e);
								openMatrix();
							}}
						/>
					{:else if stage === 'classifier'}<TensorBandFlow
							kind="classifier"
							label="Wclassifier"
							hiddenSize={192}
							outputSize={1000}
							selected="classifier"
							on:select={() => openMatrix()}
						/>{/if}
					{#if ['up', 'gelu', 'down'].includes(stage)}<FeedForwardFlow
							denseStyle
							projectionsOnly
							dimensional
							hiddenSize={192}
							intermediateSize={768}
							gated={false}
							activation="GELU"
							token={document.tokens[position]}
							selected={stage === 'gelu' ? 'activation' : stage}
							on:select={(e) => {
								((e) => {
									stage = e.detail === 'activation' ? 'gelu' : e.detail;
									coordinate = 0;
								})(e);
								openMatrix();
							}}
						/>{/if}{#key `${$blockIdx}:${position}:${head}:${coordinate}:${projection}:${stage}`}
						{#if stage === 'embedding'}<MatrixProduct
								input="Patch pixels"
								weight="Wpatchᵀ"
								output="Patch embedding"
								inputSize={768}
								outputSize={192}
								rows={196}
								bias
							/>
							<div class="formula">x₀ = [CLS; patch₀; …; patch₁₉₅] + Eposition</div>
							<div class="image-equation">
								<img src={`${base}/${document.image.src}`} alt={document.image.alt} /><span
									>224 × 224 RGB → 196 patches of 16 × 16<br />Learned projection: 768 pixel
									coordinates → 192D<br />Prepend learned [CLS] → 197 positions</span
								>
							</div>
							<table>
								<thead
									><tr
										><th>Coordinate</th><th>{position === 0 ? 'Learned CLS' : 'Projected patch'}</th
										><th>+</th><th>Learned position</th><th>= Encoder input</th></tr
									></thead
								><tbody
									>{#each coords as c}<tr
											><td>{c}</td><td>{f(scalar(common, 'patchTokens', position, c))}</td><td>+</td
											><td>{f(scalar(common, 'positionEmbedding', position, c))}</td><td
												>{f(scalar(common, 'embedding', position, c))}</td
											></tr
										>{/each}</tbody
								>
							</table>
						{:else if stage === 'norm1' || stage === 'norm2'}<div class="formula">
								LayerNorm(x)[d] = (x[d] − mean(x)) / √(variance(x) + {document.model.rmsNormEps}) ×
								γ[d] + β[d]
							</div>
							<table>
								<thead
									><tr
										><th>d</th><th>x[d]</th><th>mean</th><th>variance</th><th>γ[d]</th><th>β[d]</th
										><th>Recorded result</th></tr
									></thead
								><tbody
									>{#each normRows(stage) as r}<tr
											><td>{r.c}</td><td>{f(r.input)}</td><td>{f(r.mean)}</td><td
												>{f(r.variance)}</td
											><td>{f(r.weight)}</td><td>{f(r.bias)}</td><td>{f(r.output)}</td></tr
										>{/each}</tbody
								>
							</table>
							<p>
								Mean and variance use all 192 coordinates of this position. Each of the 12 blocks
								has its own learned normalization parameters.
							</p>
						{:else if stage === 'qkv' || stage === 'up' || stage === 'down'}
							{#if stage === 'qkv'}<TensorBandFlow
									kind="qkv"
									label="Query Key Value"
									hiddenSize={192}
									heads={3}
									kvHeads={3}
									headSize={64}
									selected={projection}
									on:select={(e) => {
										((e) => (projection = e.detail))(e);
										openMatrix();
									}}
								/>{/if}
							{@const which = stage === 'qkv' ? projection : stage}<MatrixProduct
								input={stage === 'down' ? 'GELU(U)' : 'LayerNorm(X)'}
								weight={stage === 'up'
									? 'Wupᵀ'
									: stage === 'down'
										? 'Wdownᵀ'
										: which === 'query'
											? 'Wqᵀ'
											: which === 'key'
												? 'Wkᵀ'
												: 'Wvᵀ'}
								output={stage === 'up'
									? 'Up U'
									: stage === 'down'
										? 'Output Y'
										: which.toUpperCase()}
								inputSize={stage === 'down' ? 768 : 192}
								outputSize={stage === 'up' ? 768 : 192}
								bias
								token={document.tokens[position]}
								inputValues={projectionRow(which, 0)?.input ?? []}
								outputValues={coords.map((c) => projectionRow(which, c)?.output ?? 0)}
							/>
							<details class="recorded-details">
								<summary>Recorded values and products</summary>
								{#if stage === 'qkv'}
									<p>
										Each 192D projection splits into three independent 64D heads. Learned position
										embeddings are already in x; this model has no RoPE or shared K/V heads.
									</p>{/if}
								<div class="formula">
									{stage === 'qkv'
										? `${projection.toUpperCase()} = W · LayerNorm(x) + b`
										: stage === 'up'
											? 'u = W₁ · LayerNorm(x) + b₁'
											: 'y = W₂ · GELU(u) + b₂'}
								</div>
								{#each coords as c}{@const r = projectionRow(which, c)}{#if r}<div
											class="dot-product"
										>
											<strong>{which}[{stage === 'qkv' ? head * 64 + c : c}]</strong>
											<div class="terms">
												{#each r.weights.slice(0, 4) as w, d}<span>({f(w)} × {f(r.input[d])})</span
													>{#if d < 3}<span>+</span>{/if}{/each}
											</div>
											<div>
												+ <span class="remaining"
													>{f(r.remaining)}
													<small>remaining {r.input.length - 4} products</small></span
												>
												+ {f(r.bias)} <small>bias</small> ≈ <strong>{f(r.output)}</strong>
											</div>
										</div>{/if}{/each}
								<p>
									{stage === 'up' || stage === 'down'
										? 'Four captured output rows are shown, using every input coordinate. '
										: ''}Numbers are recorded float32 values; displayed products are rounded to six
									decimals.
								</p>
							</details>{:else if stage === 'attention'}
							{#if attentionPart === 'scores'}<MatrixProduct
									input="Query Q"
									weight="Key Kᵀ"
									output="Scores"
									inputSize={64}
									outputSize={197}
									token={document.tokens[position]}
									inputValues={coords.map((c) => scalar(data, 'query', head, position, c))}
								/>
							{:else if attentionPart === 'softmax'}<MatrixProduct
									input="Scaled scores"
									output="Attention A"
									activation="Softmax"
									inputSize={197}
									outputSize={197}
									token={document.tokens[position]}
									outputValues={row(data, 'attention', head * 197 + position)}
								/>
							{:else}<MatrixProduct
									input="Attention A"
									weight="Value V"
									output="Head context"
									inputSize={197}
									outputSize={64}
									token={document.tokens[position]}
									inputValues={row(data, 'attention', head * 197 + position)}
								/>{/if}
							<details class="recorded-details">
								<summary>Recorded attention values and products</summary>
								<div class="attention-detail">
									<VitAttentionMap
										layer={data}
										{head}
										query={position}
										on:query={(e) => vitSelectedPosition.set(e.detail)}
									/>
									<div>
										<div class="formula">
											A = softmax(QKᵀ / √64)<br />context[q,d] = Σₖ₌₀¹⁹⁶ A[q,k] × V[k,d]
										</div>
										<p>
											Every query can attend to [CLS] and every patch. There is no causal mask. The
											complete 197 × 197 recorded matrix is shown at left.
										</p>
										<table>
											<thead
												><tr><th>Key</th><th>Raw Q · K</th><th>÷ 8</th><th>Attention</th></tr
												></thead
											><tbody
												>{#each [0, 1, 2, 3] as k}<tr
														><td>{document.tokens[k]}</td><td
															>{f(scalar(data, 'scores', head, position, k))}</td
														><td>{f(scalar(data, 'scores', head, position, k) / 8)}</td><td
															>{f(scalar(data, 'attention', head, position, k))}</td
														></tr
													>{/each}</tbody
											>
										</table>
									</div>
								</div>
								{#each coords as c}{@const r = contextRow(c)}{#if r}<div class="dot-product">
											<strong>context[{position},{c}]</strong>
											<div class="terms">
												{#each [0, 1, 2, 3] as k}<span>({f(r.probs[k])} × {f(r.values[k])})</span
													>{#if k < 3}<span>+</span>{/if}{/each}
											</div>
											<div>
												+ <span class="remaining"
													>{f(r.remaining)} <small>remaining 193 positions</small></span
												>
												≈ <strong>{f(r.output)}</strong>
											</div>
										</div>{/if}{/each}
								<p>
									All three 64D head outputs concatenate to 192D and pass through Woutput + bias
									before residual addition. Every attention row retains its full softmax
									denominator.
								</p>
							</details>{:else if stage === 'gelu'}<MatrixProduct
								input="Up U"
								output="GELU(U)"
								inputSize={768}
								outputSize={768}
								activation="GELU"
							/>
							<div class="formula">GELU(u) = ½ u · (1 + erf(u / √2))</div>
							<table>
								<thead
									><tr><th>Coordinate</th><th>Up projection u</th><th>Recorded GELU(u)</th></tr
									></thead
								><tbody
									>{#each coords as c}<tr
											><td>{c}</td><td>{f(scalar(data, 'up', position, c))}</td><td
												>{f(scalar(data, 'gelu', position, c))}</td
											></tr
										>{/each}</tbody
								>
							</table>
							<p>
								The same dense 192 → 768 → 192 feed-forward network acts independently on every
								position. These are the model's exact GELU outputs, without a gate or expert router.
							</p>
						{:else if stage === 'residual'}<div class="formula">
								r₁ = block input + attention output<br />block output = r₁ + feed-forward output
							</div>
							<table>
								<thead
									><tr
										><th>d</th><th>Block input</th><th>+ Attention</th><th>= r₁</th><th>+ FFN</th
										><th>= Block output</th></tr
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
						{:else if stage === 'classifier'}<MatrixProduct
								input="Final CLS"
								weight="Wclassifierᵀ"
								output="Class logits"
								inputSize={192}
								outputSize={1000}
								bias
								inputValues={row(common, 'finalHidden', 0)}
							/>
							<div class="formula">
								logits = Wclassifier · LayerNorm(xblock12)[CLS] + bclassifier<br />p(class) =
								softmax(logits over all 1,000 classes)
							</div>
							<p>
								Classification reads the final [CLS] position (0), after all 12 blocks. Patch
								positions are not separate class predictions. This synthetic shape image is outside
								ImageNet's usual photographs.
							</p>
							<table>
								<thead
									><tr
										><th>Class</th><th>First 4 coordinate products</th><th>Remaining 188</th><th
											>Bias</th
										><th>Recorded logit</th></tr
									></thead
								><tbody
									>{#each classIndices as candidate}{@const weights = row(
											common,
											'classifierWeights',
											candidate.id
										)}{@const input = row(common, 'finalHidden', 0)}<tr
											><td>{document.classLabels[candidate.id]}</td><td
												>{f(dot(weights.slice(0, 4), input.slice(0, 4)))}</td
											><td>{f(dot(weights, input, 4))}</td><td
												>{f(scalar(common, 'classifierBias', candidate.id))}</td
											><td>{f(candidate.value)}</td></tr
										>{/each}</tbody
								>
							</table>
						{/if}
						<footer>
							Recorded from pretrained {document.model.id} · all 12 blocks / 3 heads / 197 positions.
							<a
								href={recordedAssetUrl(`${base}/traces/vit/manifest.json`)}
								target="_blank"
								rel="noreferrer">Trace provenance ↗</a
							>
						</footer>
					{/key}
				{/if}
			</TokenInspectorLayout>
		</div>
	</section>
</div>

<style>
	.vit-inspector-overlay {
		position: fixed;
		inset: 0;
		z-index: 1500;
		display: flex;
		align-items: center;
		justify-content: center;
		background: #ffffff55;
		padding: 20px;
	}
	.vit-inspector {
		width: min(1440px, calc(100vw - 24px));
		max-height: calc(100dvh - 24px);
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
		padding: 12px 20px;
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
		padding: 8px 20px;
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
		margin-bottom: 8px;
		font-size: 12px;
	}
	.controls label {
		display: flex;
		align-items: center;
		gap: 7px;
	}
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
		margin-top: 6px;
		padding-top: 6px;
		border-top: 1px solid #eee5f5;
	}
	a {
		color: #9a61bf;
	}
	@media (max-width: 800px) {
		.attention-detail {
			flex-direction: column;
		}
		.vit-inspector-overlay {
			padding: 8px;
		}
		.vit-inspector {
			width: calc(100vw - 16px);
			max-height: calc(100vh - 16px);
		}
	}

	.vit-inspector {
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
