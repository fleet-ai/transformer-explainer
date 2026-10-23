<script lang="ts">
	import HybridPopupShell from './HybridPopupShell.svelte';
	/**
	 * Formula-only popups for the Zamba2 operations that are not matrix projections.
	 * Cells and sweeps are schematic; no recorded values are shown.
	 */
	type Kind =
		| 'gate'
		| 'gelu'
		| 'product'
		| 'delta'
		| 'norm'
		| 'residual'
		| 'conv'
		| 'scan'
		| 'attention';
	export let kind: Kind;
	export let title: string;
	export let token = '';
	/** Every token of the prompt, for the operations that read other positions. */
	export let tokens: string[] = [];
	export let position = 0;
	/** Width of the vector the operation works on. */
	export let size = 2048;
	export let inputLabel = '';
	export let outputLabel = '';
	export let otherLabel = '';
	export let epsilon = 1e-5;
	export let heads = 64;
	export let note = '';

	const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
	// Abramowitz–Stegun erf, close enough to draw the exact GELU curve.
	const erf = (v: number) => {
		const s = Math.sign(v),
			a = Math.abs(v),
			t = 1 / (1 + 0.3275911 * a);
		const y =
			1 -
			((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
				t *
				Math.exp(-a * a);
		return s * y;
	};
	const curves = {
		gate: (v: number) => v * sigmoid(v),
		gelu: (v: number) => 0.5 * v * (1 + erf(v / Math.SQRT2)),
		delta: (v: number) => Math.log1p(Math.exp(v))
	} as const;
	$: curveKind = kind === 'gate' || kind === 'gelu' || kind === 'delta' ? kind : null;
	const W = 280;
	const H = 170;
	const PAD = 18;
	type Range = { x0: number; x1: number; y0?: number; y1?: number };
	$: range = (kind === 'delta' ? { x0: -6, x1: 4, y0: -0.3, y1: 4.2 } : { x0: -5, x1: 5 }) as Range;
	$: fn = curveKind ? curves[curveKind] : null;
	$: points = fn
		? Array.from({ length: 121 }, (_, k) => {
				const x = range.x0 + ((range.x1 - range.x0) * k) / 120;
				return [x, fn!(x)];
			})
		: [];
	$: yLow = range.y0 ?? Math.min(...points.map(([, y]) => y)) - 0.15;
	$: yHigh = range.y1 ?? (fn ? fn(range.x1) : 1);
	$: sx = (x: number) => PAD + ((x - range.x0) / (range.x1 - range.x0)) * (W - PAD * 2);
	$: sy = (y: number) => H - PAD - ((y - yLow) / (yHigh - yLow)) * (H - PAD * 2);
	$: curve = points
		.map(([x, y], k) => `${k === 0 ? 'M' : 'L'}${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`)
		.join(' ');
	$: plotLabel =
		kind === 'gate'
			? 'SiLU(z) = z · σ(z)'
			: kind === 'gelu'
				? 'GELU(g) = ½ g (1 + erf(g / √2))'
				: 'softplus(v) = ln(1 + eᵛ)';
	$: axis = kind === 'gate' ? 'z' : kind === 'gelu' ? 'g' : 'v';
	$: sweep = kind === 'conv' || kind === 'scan' || kind === 'attention' ? Math.max(1, size) : size;

	// Causal convolution: the window of four positions ending at the selected token.
	$: taps = [3, 2, 1, 0].map((back) => ({
		index: position - back,
		label: position - back < 0 ? '0 pad' : (tokens[position - back] ?? '')
	}));
	$: working =
		kind === 'conv'
			? 'Sliding the window across the channels'
			: kind === 'scan'
				? 'Carrying the state token by token'
				: kind === 'attention'
					? 'Weighting the visible tokens'
					: 'Working across the vector';
	$: elementwise = kind === 'gate' || kind === 'gelu' || kind === 'product' || kind === 'delta';
</script>

<HybridPopupShell {title} size={sweep} on:close let:progress>
	{@const j = progress.coordinate}
	{#if elementwise}
		<p class="notice">
			Element-wise · every coordinate is computed on its own, with no sum across coordinates ·
			schematic animation
		</p>
		<div class="equation">
			<div class="tensor">
				<div>{inputLabel} <span class="token">“{token}”</span></div>
				<div class="cells input" class:working={!progress.complete} aria-hidden="true">
					{#each Array(32) as _}<i></i>{/each}
				</div>
				<strong>(1, {size})</strong>
			</div>
			<span class="operator">{kind === 'gate' || kind === 'product' ? '⊙' : '→'}</span>
			{#if curveKind}
				<div class="plot">
					<div>{plotLabel}</div>
					<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={plotLabel}>
						<line class="axis" x1={sx(range.x0)} x2={sx(range.x1)} y1={sy(0)} y2={sy(0)} />
						<line class="axis" x1={sx(0)} x2={sx(0)} y1={sy(yLow)} y2={sy(yHigh)} />
						<path class="curve" d={curve} />
						<text class="axis-label" x={W - PAD} y={sy(0) + 14} text-anchor="end">{axis}</text>
					</svg>
					<strong>{otherLabel}</strong>
				</div>
			{:else}
				<div class="tensor">
					<div>{otherLabel}</div>
					<div class="cells activated" aria-hidden="true">
						{#each Array(32) as _}<i></i>{/each}
					</div>
					<strong>(1, {size})</strong>
				</div>
			{/if}
			<span class="operator">=</span>
			<div class="tensor">
				<div>{outputLabel}</div>
				<div class="cells output" aria-hidden="true">
					{#each Array(32) as _, i}<i
							class:done={i < progress.completed}
							class:active={i === progress.active}
						></i>{/each}
				</div>
				<strong>(1, {size})</strong>
			</div>
		</div>
		<div class="formula-lines" aria-hidden="true">
			{#if kind === 'gate'}
				<div>a<sub>{j}</sub> = z<sub>{j}</sub> · σ(z<sub>{j}</sub>)</div>
				<div>h<sub>{j}</sub> = y<sub>{j}</sub> · a<sub>{j}</sub></div>
			{:else if kind === 'gelu'}
				<div>a<sub>{j}</sub> = ½ g<sub>{j}</sub> (1 + erf(g<sub>{j}</sub> / √2))</div>
			{:else if kind === 'product'}
				<div>h<sub>{j}</sub> = u<sub>{j}</sub> · GELU(g<sub>{j}</sub>)</div>
			{:else}
				<div>Δ<sub>{j}</sub> = max(0.001, softplus(dt<sub>{j}</sub> + b<sub>{j}</sub>))</div>
				<div>
					decay<sub>{j}</sub> = e<sup>Δ<sub>{j}</sub> · A<sub>{j}</sub></sup>, A<sub>{j}</sub> &lt; 0
				</div>
			{/if}
		</div>
	{:else if kind === 'norm' || kind === 'residual'}
		<p class="notice">
			{kind === 'norm'
				? 'No weights except one learned scale γ per coordinate · schematic animation'
				: 'Element-wise addition · the sizes must match · schematic animation'}
		</p>
		<div class="equation">
			<div class="tensor">
				<div>{inputLabel} <span class="token">“{token}”</span></div>
				<div class="cells input" class:working={!progress.complete} aria-hidden="true">
					{#each Array(32) as _}<i></i>{/each}
				</div>
				<strong>(1, {size})</strong>
			</div>
			<span class="operator">{kind === 'norm' ? '→' : '+'}</span>
			{#if kind === 'norm'}
				<div class="norm-box" aria-hidden="true">
					<strong>RMSNorm</strong>
					<span>÷ RMS · × γ</span>
				</div>
			{:else}
				<div class="tensor">
					<div>{otherLabel}</div>
					<div class="cells activated" aria-hidden="true">
						{#each Array(32) as _}<i></i>{/each}
					</div>
					<strong>(1, {size})</strong>
				</div>
			{/if}
			<span class="operator">=</span>
			<div class="tensor">
				<div>{outputLabel}</div>
				<div class="cells output" aria-hidden="true">
					{#each Array(32) as _, i}<i
							class:done={i < progress.completed}
							class:active={i === progress.active}
						></i>{/each}
				</div>
				<strong>(1, {size})</strong>
			</div>
		</div>
		<div class="formula-lines" aria-hidden="true">
			{#if kind === 'norm'}
				<div>
					RMS(x) = √( (1/{size}) ∑<sub>d=1…{size}</sub> x<sub>d</sub>² + ε ), ε = {epsilon.toExponential(
						0
					)}
				</div>
				<div>y<sub>{j}</sub> = γ<sub>{j}</sub> · x<sub>{j}</sub> / RMS(x)</div>
			{:else}
				<div>out<sub>{j}</sub> = x<sub>{j}</sub> + u<sub>{j}</sub></div>
			{/if}
		</div>
	{:else if kind === 'conv'}
		<p class="notice">
			Depthwise and causal · each of the {size.toLocaleString('en-US')} channels has its own 4 weights,
			and a token never reads a later one · schematic animation
		</p>
		<div class="token-strip" aria-label="Four positions read for the selected token">
			{#each tokens as text, index}<span
					class="token-cell"
					class:in-window={index <= position && index > position - 4}
					class:selected={index === position}>{text}</span
				>{/each}
		</div>
		<div class="window-taps" aria-hidden="true">
			{#each taps as tap, k}<div class="tap" class:pad={tap.index < 0}>
					<span>w<sub>{k}</sub></span><small>{tap.label}</small>
				</div>{/each}
			<span class="operator">→</span>
			<div class="tap result"><span>Σ + b → SiLU</span><small>“{token}”</small></div>
		</div>
		<div class="formula-lines" aria-hidden="true">
			<div>
				c<sub>t</sub>[{j}] = b<sub>{j}</sub> + ∑<sub>k=0…3</sub> w<sub>{j},k</sub> · u<sub
					>t−3+k</sub
				>[{j}]
			</div>
			<div>
				x, B, C = SiLU(c<sub>t</sub>) split as 4096 + 128 + 128 · positions before the first token
				are 0
			</div>
		</div>
	{:else if kind === 'scan'}
		<p class="notice">
			One head shown · {heads} heads run side by side · B and C are shared by all heads, Δ, A, D and
			x belong to each head · schematic animation
		</p>
		<div class="state-strip" aria-label="The state carried from token to token">
			{#each tokens as text, index}{#if index}<span class="carry" aria-hidden="true">→</span>{/if}
				<div class="state-cell" class:selected={index === position} class:done={index <= position}>
					<span>s<sub>{index}</sub></span><small>{text}</small>
				</div>{/each}
		</div>
		<div class="formula-lines" aria-hidden="true">
			<div>
				s<sub>t</sub> = e<sup>Δ<sub>t</sub>A</sup> · s<sub>t−1</sub> + Δ<sub>t</sub> · x<sub>t</sub>
				B<sub>t</sub><sup>⊤</sup>
			</div>
			<div>y<sub>t</sub> = s<sub>t</sub> C<sub>t</sub> + D · x<sub>t</sub></div>
			<div class="small">
				unrolled: y<sub>t</sub> = ∑<sub>s≤t</sub> M<sub>t,s</sub> x<sub>s</sub> + D x<sub>t</sub>, M<sub
					>t,s</sub
				>
				= Δ<sub>s</sub> e<sup>A (Δ<sub>s+1</sub> + … + Δ<sub>t</sub>)</sup> (C<sub>t</sub> · B<sub
					>s</sub
				>)
			</div>
		</div>
		<p class="status">
			State per head: 64 channels × 128. All {heads} heads: {(heads * 64 * 128).toLocaleString(
				'en-US'
			)} numbers, whatever the prompt length.
		</p>
	{:else if kind === 'attention'}
		<p class="notice">
			One head shown · each head has its own Q, K and V · causal mask, then softmax over the visible
			keys · schematic animation
		</p>
		<div class="formula-lines" aria-hidden="true">
			<div>
				score<sub>t,s</sub> = Q′<sub>t</sub> · K′<sub>s</sub> / 8 · masked to −∞ when s &gt; t
			</div>
			<div>a<sub>t,s</sub> = softmax<sub>s</sub>(score<sub>t,s</sub>)</div>
			<div>context<sub>t</sub> = ∑<sub>s≤t</sub> a<sub>t,s</sub> · V<sub>s</sub></div>
		</div>
		<div class="state-strip" aria-label="Keys the selected token can read">
			{#each tokens as text, index}<div
					class="state-cell"
					class:selected={index === position && progress.complete}
					class:done={index <= Math.min(position, (progress.completed / 32) * (position + 1))}
					class:masked={index > position}
				>
					<span>{index <= position ? 'k' : '−∞'}<sub>{index}</sub></span><small>{text}</small>
				</div>{/each}
		</div>
	{/if}
	<p class="status" role="status">
		{progress.complete ? 'Complete' : working} · schematic sweep
	</p>
	{#if note}<p class="note-text">{note}</p>{/if}
	<slot />
</HybridPopupShell>

<style>
	.notice {
		color: #9ca3af;
		font-size: 12px;
		margin: 0;
	}
	.equation {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 22px minmax(0, 1.15fr) 22px minmax(0, 1fr);
		gap: 10px;
		align-items: center;
		margin-top: 24px;
	}
	.tensor,
	.plot {
		min-width: 0;
		display: grid;
		align-items: center;
		text-align: center;
	}
	.tensor {
		grid-template-rows: 34px 110px 34px;
	}
	.plot {
		grid-template-rows: 34px auto 34px;
	}
	.plot svg {
		width: 100%;
		height: auto;
		max-height: 170px;
	}
	.tensor strong,
	.plot strong {
		font-size: 18px;
		font-weight: 400;
		white-space: nowrap;
	}
	.plot strong {
		font-size: 13px;
		color: #6b7280;
		white-space: normal;
	}
	.token {
		font-size: 12px;
		color: #9ca3af;
	}
	.cells {
		display: grid;
		grid-template-columns: repeat(32, minmax(0, 1fr));
		gap: 1px;
		height: 26px;
	}
	i {
		background: #c8ccd3;
	}
	.input i {
		background: #cdc0fc;
	}
	.input.working i {
		background: #b7a3f1;
	}
	.activated i {
		background: #b9c8fd;
	}
	.output i {
		background: #f3f4f6;
	}
	.output i.done {
		background: #b9c8fd;
	}
	.output i.active {
		background: #8da6f5;
	}
	.operator {
		font-size: 26px;
		color: #374151;
		text-align: center;
	}
	.axis {
		stroke: #d1d5db;
		stroke-width: 1;
	}
	.curve {
		fill: none;
		stroke: #8da6f5;
		stroke-width: 2.5;
	}
	.axis-label {
		fill: #9ca3af;
		font:
			italic 12px Georgia,
			serif;
	}
	.norm-box {
		display: grid;
		place-items: center;
		align-content: center;
		gap: 4px;
		height: 90px;
		border: 1.5px solid #cbb4df;
		border-radius: 6px;
		background: #faf7fd;
		color: #694684;
	}
	.norm-box strong {
		font-size: 18px;
		font-weight: 600;
	}
	.norm-box span {
		font-size: 13px;
		color: #8a6aa3;
	}
	.formula-lines {
		display: grid;
		gap: 10px;
		font:
			italic 24px Georgia,
			serif;
		color: #6b7280;
		text-align: center;
		margin: 24px 0 18px;
	}
	.formula-lines .small {
		font-size: 18px;
	}
	sub,
	sup {
		font-size: 12px;
	}
	.token-strip,
	.state-strip {
		display: flex;
		justify-content: center;
		align-items: center;
		flex-wrap: wrap;
		gap: 6px;
		margin: 24px 0 16px;
	}
	.token-cell {
		padding: 4px 10px;
		border: 1px solid #e5e7eb;
		border-radius: 4px;
		color: #9ca3af;
		font-size: 13px;
	}
	.token-cell.in-window {
		border-color: #cbb4df;
		background: #faf7fd;
		color: #694684;
	}
	.token-cell.selected {
		background: #e0c8ef;
		border-color: #b28bce;
	}
	.window-taps {
		display: flex;
		justify-content: center;
		align-items: center;
		gap: 10px;
		margin-top: 18px;
	}
	.tap {
		display: grid;
		justify-items: center;
		min-width: 64px;
		padding: 6px 10px;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			italic 18px Georgia,
			serif;
	}
	.tap small {
		font:
			11px ui-sans-serif,
			system-ui,
			sans-serif;
		color: #8a6aa3;
	}
	.tap.pad {
		border-style: dashed;
		opacity: 0.6;
	}
	.tap.result {
		font-size: 15px;
	}
	.state-cell {
		display: grid;
		justify-items: center;
		min-width: 58px;
		padding: 6px 8px;
		border: 1px solid #e5e7eb;
		border-radius: 4px;
		color: #9ca3af;
		font:
			italic 18px Georgia,
			serif;
	}
	.state-cell small {
		font:
			11px ui-sans-serif,
			system-ui,
			sans-serif;
	}
	.state-cell.done {
		border-color: #cbb4df;
		background: #faf7fd;
		color: #694684;
	}
	.state-cell.selected {
		background: #e0c8ef;
		border-color: #b28bce;
	}
	.state-cell.masked {
		background: #f3f4f6;
	}
	.carry {
		color: #b28bce;
		font-size: 18px;
	}
	.status {
		color: #9ca3af;
		font-size: 12px;
		text-align: center;
		margin: 0 0 12px;
	}
	.note-text {
		color: #6b7280;
		font-size: 13px;
		line-height: 1.5;
		margin: 6px 0 0;
		text-align: center;
	}
	@media (max-width: 600px) {
		.equation {
			gap: 5px;
			grid-template-columns: minmax(0, 1fr) 12px minmax(0, 1.1fr) 12px minmax(0, 1fr);
		}
		.tensor {
			font-size: 11px;
			grid-template-rows: 42px 70px 30px;
		}
		.tensor strong {
			font-size: 12px;
		}
		.operator {
			font-size: 18px;
		}
		.formula-lines {
			font-size: 19px;
		}
	}
</style>
