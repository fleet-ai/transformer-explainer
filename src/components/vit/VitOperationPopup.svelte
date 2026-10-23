<script lang="ts" context="module">
	export type VitOperation =
		| 'patchify'
		| 'cls'
		| 'position'
		| 'norm1'
		| 'norm2'
		| 'final-norm'
		| 'attention'
		| 'gelu'
		| 'residual'
		| 'softmax';
	export const VIT_OPERATIONS: VitOperation[] = [
		'patchify',
		'cls',
		'position',
		'norm1',
		'norm2',
		'final-norm',
		'attention',
		'gelu',
		'residual',
		'softmax'
	];
</script>

<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { base } from '$app/paths';
	import { CloseOutline, RefreshOutline } from 'flowbite-svelte-icons';
	import { projectionSweep } from '$lib/moe/projection-sweep';
	import { blockIdx } from '~/store';
	import { vitRecording, vitSelectedPosition } from '$lib/vit/store';
	/** Element-wise and sequence operations of DeiT: a formula plus a schematic sweep, never recorded values. */
	export let kind: VitOperation;
	const dispatch = createEventDispatcher<{ close: void }>();
	let panel: HTMLDivElement;
	let frame = 0;
	let reduced = false;

	$: document = $vitRecording?.document;
	$: model = document?.model;
	$: hidden = model?.hiddenSize ?? 192;
	$: inner = model?.intermediateSize ?? 768;
	$: headSize = model?.attentionHeadSize ?? 64;
	$: positions = document?.sequenceLength ?? 197;
	$: classes = document?.classLabels.length ?? 1000;
	$: patch = document?.image.patchSize ?? 16;
	$: grid = document?.image.gridSize ?? 14;
	$: epsilon = model?.rmsNormEps ?? 1e-12;
	$: position = $vitSelectedPosition;
	$: token = document?.tokens[position] ?? '';
	$: size =
		kind === 'gelu'
			? inner
			: kind === 'attention'
				? positions
				: kind === 'softmax'
					? classes
					: kind === 'patchify'
						? patch * patch * 3
						: hidden;
	let progress = projectionSweep(0, 192);
	$: j = progress.coordinate;
	$: block = $blockIdx + 1;
	let titles: Record<VitOperation, string>;
	$: titles = {
		patchify: 'Image → 16 × 16 patches',
		cls: 'Prepend [CLS]',
		position: '+ Position embedding',
		norm1: `Block ${block} · LayerNorm before attention`,
		norm2: `Block ${block} · LayerNorm before the MLP`,
		'final-norm': 'Final LayerNorm',
		attention: `Block ${block} · Bidirectional attention`,
		gelu: `Block ${block} · GELU activation`,
		residual: `Block ${block} · Residual additions`,
		softmax: `Softmax over ${classes.toLocaleString('en-US')} classes`
	};
	$: isNorm = kind === 'norm1' || kind === 'norm2' || kind === 'final-norm';
	// GELU drawn from its formula; no recorded values sit on the curve.
	const erf = (x: number) => {
		const t = 1 / (1 + 0.3275911 * Math.abs(x));
		const y =
			1 -
			((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
				t *
				Math.exp(-x * x);
		return x >= 0 ? y : -y;
	};
	const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));
	const W = 280;
	const H = 170;
	const PAD = 18;
	const xMin = -4;
	const xMax = 4;
	const yMin = -0.4;
	const yMax = gelu(xMax);
	const sx = (x: number) => PAD + ((x - xMin) / (xMax - xMin)) * (W - PAD * 2);
	const sy = (y: number) => H - PAD - ((y - yMin) / (yMax - yMin)) * (H - PAD * 2);
	const curve = Array.from({ length: 121 }, (_, k) => {
		const x = xMin + ((xMax - xMin) * k) / 120;
		return `${k === 0 ? 'M' : 'L'}${sx(x).toFixed(1)} ${sy(gelu(x)).toFixed(1)}`;
	}).join(' ');
	$: selectedPatch = position > 0 ? position - 1 : -1;

	function portal(element: HTMLElement) {
		window.document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	function replay() {
		cancelAnimationFrame(frame);
		const start = performance.now();
		progress = projectionSweep(reduced ? 3500 : 0, size);
		if (reduced) return;
		const tick = (now: number) => {
			progress = projectionSweep(now - start, size);
			if (!progress.complete) frame = requestAnimationFrame(tick);
		};
		frame = requestAnimationFrame(tick);
	}
	function keydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			event.preventDefault();
			event.stopImmediatePropagation();
			dispatch('close');
		} else if (event.key === 'Tab') {
			const targets = Array.from(panel.querySelectorAll<HTMLElement>('button, summary, a[href]'));
			const first = targets[0],
				last = targets[targets.length - 1];
			if (
				event.shiftKey &&
				(window.document.activeElement === first || window.document.activeElement === panel)
			) {
				event.preventDefault();
				last?.focus();
			} else if (!event.shiftKey && window.document.activeElement === last) {
				event.preventDefault();
				first?.focus();
			}
		}
	}
	onMount(() => {
		const previous = window.document.activeElement;
		const media = matchMedia('(prefers-reduced-motion: reduce)');
		reduced = media.matches;
		const change = () => {
			reduced = media.matches;
			if (reduced) {
				cancelAnimationFrame(frame);
				progress = projectionSweep(3500, size);
			}
		};
		media.addEventListener('change', change);
		panel.focus({ preventScroll: true });
		replay();
		return () => {
			cancelAnimationFrame(frame);
			media.removeEventListener('change', change);
			if (previous instanceof HTMLElement && previous.isConnected)
				previous.focus({ preventScroll: true });
		};
	});
</script>

<svelte:window on:keydown|capture={keydown} />
<div class="popup-layer" use:portal>
	<div class="shade"></div>
	<div
		class="popup"
		bind:this={panel}
		role="dialog"
		aria-modal="true"
		aria-label={`DeiT-tiny · ${titles[kind]}`}
		tabindex="-1"
	>
		<header>
			<h3>DeiT-tiny · {titles[kind]}</h3>
			<div class="controls">
				<button aria-label="Replay animation" on:click={replay}><RefreshOutline /></button>
				<button aria-label="Close" on:click={() => dispatch('close')}><CloseOutline /></button>
			</div>
		</header>
		<div class="body">
			<p class="notice">
				{#if kind === 'patchify'}No weights · the image is cut into a {grid} × {grid} grid and each patch
					is read out as one vector{:else if kind === 'cls'}No arithmetic · one learned vector is
					placed in front of the {grid * grid} patch embeddings{:else if kind === 'position'}Element-wise
					· one learned {hidden}D row per position is added, coordinate by coordinate{:else if isNorm}No
					weights to multiply · mean and variance come from the whole vector, then every coordinate
					is shifted, scaled, and given its own γ and β{:else if kind === 'attention'}Bidirectional
					· no causal mask · every position attends to all {positions} positions, [CLS] included{:else if kind === 'gelu'}Element-wise
					· every coordinate is computed on its own, with no gate and no sum across coordinates{:else if kind === 'residual'}Element-wise
					· each sub-layer's output is added to the vector that entered it{:else}A full softmax over
					every class · nothing is filtered or renormalized{/if} · schematic animation
			</p>

			<div class="equation" class:wide={kind === 'attention'}>
				{#if kind === 'patchify'}
					<div class="tensor">
						<div>Input image</div>
						{#if document}<div
								class="image-grid"
								style={`background-image:url('${base}/${document.image.src}');--grid:${grid}`}
								aria-hidden="true"
							>
								{#each Array(grid * grid) as _, p}<i class:selected={p === selectedPatch}
									></i>{/each}
							</div>{/if}
						<strong>(3, {patch * grid}, {patch * grid})</strong>
					</div>
					<span class="operator">→</span>
					<div class="op-box" aria-hidden="true">
						<strong>{patch} × {patch} × 3</strong>
						<span>{position > 0 ? token : 'one patch'} · flatten</span>
					</div>
				{:else if kind === 'cls'}
					<div class="tensor">
						<div>Patch embeddings</div>
						<div class="rows" aria-hidden="true">
							{#each Array(12) as _}<i></i>{/each}
						</div>
						<strong>({grid * grid}, {hidden})</strong>
					</div>
					<span class="operator">→</span>
					<div class="op-box" aria-hidden="true">
						<strong>[CLS]</strong>
						<span>learned · ({1}, {hidden})</span>
					</div>
				{:else if kind === 'gelu'}
					<div class="tensor">
						<div>Up projection u <span class="token">“{token}”</span></div>
						<div class="cells input" class:working={!progress.complete} aria-hidden="true">
							{#each Array(32) as _}<i></i>{/each}
						</div>
						<strong>(1, {inner})</strong>
					</div>
					<span class="operator">→</span>
					<div class="plot">
						<div>GELU(u) = u · Φ(u)</div>
						<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The GELU curve">
							<line class="axis" x1={sx(xMin)} x2={sx(xMax)} y1={sy(0)} y2={sy(0)} />
							<line class="axis" x1={sx(0)} x2={sx(0)} y1={sy(yMin)} y2={sy(yMax)} />
							<path class="curve" d={curve} />
							<text class="axis-label" x={W - PAD} y={sy(0) + 14} text-anchor="end">u</text>
						</svg>
						<strong>applied to each of {inner} coordinates</strong>
					</div>
				{:else if kind === 'attention'}
					<div class="tensor">
						<div>Query q <span class="token">“{token}”</span></div>
						<div class="cells input short" class:working={!progress.complete} aria-hidden="true">
							{#each Array(16) as _}<i></i>{/each}
						</div>
						<strong>(1, {headSize})</strong>
					</div>
					<span class="operator">×</span>
					<div class="tensor">
						<div>Keys Kᵀ</div>
						<div class="cells keys" aria-hidden="true">
							{#each Array(32) as _, i}<i class:active={i === progress.active}></i>{/each}
						</div>
						<strong>({headSize}, {positions})</strong>
					</div>
					<span class="operator">→</span>
					<div class="op-box" aria-hidden="true">
						<strong>÷ √{headSize} · softmax</strong>
						<span>over all {positions} keys</span>
					</div>
				{:else}
					<div class="tensor">
						<div>
							{kind === 'position'
								? position === 0
									? 'Learned [CLS]'
									: 'Patch embedding e'
								: kind === 'residual'
									? 'Block input x'
									: kind === 'softmax'
										? 'Class logits z'
										: kind === 'final-norm'
											? 'Block 12 output'
											: 'Residual stream x'}
							{#if kind !== 'softmax'}<span class="token"
									>“{kind === 'final-norm' ? '[CLS]' : token}”</span
								>{/if}
						</div>
						<div class="cells input" class:working={!progress.complete} aria-hidden="true">
							{#each Array(32) as _}<i></i>{/each}
						</div>
						<strong>(1, {size})</strong>
					</div>
					<span class="operator">{kind === 'position' || kind === 'residual' ? '+' : '→'}</span>
					{#if kind === 'position' || kind === 'residual'}
						<div class="tensor">
							<div>
								{kind === 'position' ? `Epos[${position}]` : 'Sub-layer output f(x)'}
							</div>
							<div class="cells added" aria-hidden="true">
								{#each Array(32) as _}<i></i>{/each}
							</div>
							<strong>(1, {hidden})</strong>
						</div>
					{:else}
						<div class="op-box" aria-hidden="true">
							<strong>{kind === 'softmax' ? 'softmax' : 'LayerNorm'}</strong>
							<span>{kind === 'softmax' ? 'exp · ÷ Σ' : '− μ · ÷ σ · × γ + β'}</span>
						</div>
					{/if}
				{/if}
				<span class="operator">=</span>
				<div class="tensor">
					<div>
						{kind === 'patchify'
							? 'Flattened patch p'
							: kind === 'cls'
								? 'Encoder sequence'
								: kind === 'position'
									? 'Encoder input x₀'
									: kind === 'attention'
										? 'Attention weights a'
										: kind === 'gelu'
											? 'Activated h'
											: kind === 'residual'
												? 'Block output'
												: kind === 'softmax'
													? 'Class probabilities p'
													: 'Normalized x̂'}
					</div>
					{#if kind === 'cls'}
						<div class="rows" aria-hidden="true">
							<i class="cls-row"></i>{#each Array(12) as _}<i></i>{/each}
						</div>
						<strong>({positions}, {hidden})</strong>
					{:else}
						<div class="cells output" aria-hidden="true">
							{#each Array(32) as _, i}<i
									class:done={i < progress.completed}
									class:active={i === progress.active}
								></i>{/each}
						</div>
						<strong>(1, {size})</strong>
					{/if}
				</div>
			</div>

			<div class="formula-lines" aria-hidden="true">
				{#if kind === 'patchify'}
					<div>
						p<sub>r,c</sub> = image[ :, {patch}r … {patch}r+{patch - 1}, {patch}c … {patch}c+{patch -
							1} ] ∈ ℝ<sup>3×{patch}×{patch}</sup>
					</div>
					<div>vec(p<sub>r,c</sub>) ∈ ℝ<sup>{patch * patch * 3}</sup>, r, c = 0 … {grid - 1}</div>
				{:else if kind === 'cls'}
					<div>
						x = [ x<sub>cls</sub> ; e<sub>1</sub> ; … ; e<sub>{grid * grid}</sub> ] ∈ ℝ<sup
							>{positions}×{hidden}</sup
						>
					</div>
					<div>e<sub>i</sub> = vec(p<sub>i</sub>) · W<sub>patch</sub> + b</div>
				{:else if kind === 'position'}
					<div>
						x₀[{position}, {j}] = {position === 0 ? 'x' : 'e'}<sub
							>{position === 0 ? 'cls' : ''}</sub
						>[{position}, {j}] + E<sub>pos</sub>[{position}, {j}]
					</div>
				{:else if isNorm}
					<div>
						μ = (1/{hidden}) ∑<sub>d</sub> x<sub>d</sub> , σ² = (1/{hidden}) ∑<sub>d</sub> (x<sub
							>d</sub
						> − μ)²
					</div>
					<div>
						x̂<sub>{j}</sub> = γ<sub>{j}</sub> · (x<sub>{j}</sub> − μ) / √(σ² + ε) + β<sub>{j}</sub>
						, ε = {epsilon.toExponential(0)}
					</div>
				{:else if kind === 'attention'}
					<div>
						s<sub>k</sub> = q · k<sub>k</sub> / √{headSize} , a<sub>k</sub> = e<sup
							>s<sub>k</sub></sup
						>
						/ ∑<sub>k′=0…{positions - 1}</sub>
						e<sup>s<sub>k′</sub></sup>
					</div>
					<div>
						out<sub>d</sub> = ∑<sub>k=0…{positions - 1}</sub> a<sub>k</sub> · v<sub>k,d</sub>
					</div>
				{:else if kind === 'gelu'}
					<div>
						h<sub>{j}</sub> = u<sub>{j}</sub> · Φ(u<sub>{j}</sub>) = ½ u<sub>{j}</sub> (1 + erf(u<sub
							>{j}</sub
						> / √2))
					</div>
				{:else if kind === 'residual'}
					<div>r<sub>{j}</sub> = x<sub>{j}</sub> + Attention(LN₁(x))<sub>{j}</sub></div>
					<div>out<sub>{j}</sub> = r<sub>{j}</sub> + MLP(LN₂(r))<sub>{j}</sub></div>
				{:else}
					<div>
						p<sub>{j}</sub> = e<sup>z<sub>{j}</sub></sup> / ∑<sub>c=1…{classes}</sub> e<sup
							>z<sub>c</sub></sup
						>
					</div>
				{/if}
			</div>
			<p class="status" role="status">
				{progress.complete ? 'Complete' : 'Sweeping across the vector'} · schematic sweep
			</p>
			<p class="notice">
				{#if kind === 'patchify'}DeiT computes this as a {patch} × {patch} convolution with stride {patch},
					which is the same as flattening each patch and multiplying it by Wpatch. {grid * grid} patches
					in all.{:else if kind === 'cls'}x<sub>cls</sub> is the same learned vector for every image
					and has no pixels. After all {model?.numLayers ?? 12} blocks, only this position feeds the
					classifier. This DeiT has no distillation token: {positions} = 1 + {grid *
						grid}.{:else if kind === 'position'}Learned absolute positions, added once at the input.
					Attention later uses no RoPE and no relative bias.{:else if kind === 'final-norm'}It runs
					on every position after block {model?.numLayers ?? 12}, but the classifier reads only
					[CLS].{:else if isNorm}γ and β are learned per coordinate, and every block has its own.{:else if kind === 'attention'}Each
					of the {model?.attentionHeads ?? 3} heads does this with its own
					{headSize}D slice of Q, K and V. The {model?.attentionHeads ?? 3} outputs are joined back to
					{hidden}D and projected by Wout.{:else if kind === 'gelu'}Φ is the standard normal CDF.
					The same {hidden} → {inner} →
					{hidden} MLP acts on every position.{:else if kind === 'residual'}Both additions keep
					{hidden} coordinates; nothing is scaled.{:else}The list shows the top classes of the full
					distribution. The image is synthetic, so the ImageNet labels are the model's real output,
					not shape names.{/if}
			</p>
		</div>
	</div>
</div>

<style>
	.popup-layer {
		position: fixed;
		inset: 0;
		z-index: 1400;
		display: grid;
		place-items: center;
		padding: 16px;
		box-sizing: border-box;
		font:
			14px/1.5 ui-sans-serif,
			system-ui,
			sans-serif;
		color: #111827;
	}
	.shade {
		position: absolute;
		inset: 0;
		background: #ffffffc7;
	}
	.popup {
		position: relative;
		width: min(1000px, 100%);
		max-height: calc(100dvh - 32px);
		border: 1px solid #e5e7eb;
		background: #fff;
		border-radius: 8px;
		box-shadow:
			0 4px 6px -1px #0000001a,
			0 2px 4px -2px #0000001a;
		display: flex;
		flex-direction: column;
		outline: none;
	}
	header {
		background: #f3f4f6;
		border-bottom: 1px solid #e5e7eb;
		border-radius: 8px 8px 0 0;
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 10px 16px;
		gap: 8px;
	}
	h3 {
		font-size: 18px;
		font-weight: 600;
		margin: 0;
	}
	.controls {
		display: flex;
		gap: 8px;
	}
	button {
		display: grid;
		place-items: center;
		width: 30px;
		height: 30px;
		padding: 4px;
		border: 0;
		background: transparent;
		color: #6b7280;
		cursor: pointer;
	}
	button :global(svg) {
		width: 22px;
		height: 22px;
	}
	.body {
		padding: 16px 24px 20px;
		min-height: 0;
		overflow: auto;
	}
	.notice {
		color: #9ca3af;
		font-size: 12px;
		margin: 0;
	}
	.equation {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 26px minmax(0, 1.15fr) 26px minmax(0, 1fr);
		gap: 10px;
		align-items: center;
		margin-top: 24px;
	}
	.equation.wide {
		grid-template-columns: minmax(0, 0.8fr) 22px minmax(0, 1.1fr) 22px 150px 22px minmax(0, 1fr);
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
	.cells.short {
		grid-template-columns: repeat(16, minmax(0, 1fr));
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
	.added i,
	.keys i {
		background: #b9c8fd;
	}
	.keys i.active {
		background: #8da6f5;
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
	.rows {
		display: grid;
		gap: 2px;
		justify-self: center;
		width: 70%;
	}
	.rows i {
		height: 5px;
		background: #cdc0fc;
	}
	.rows i.cls-row {
		background: #8da6f5;
	}
	.image-grid {
		justify-self: center;
		width: 110px;
		height: 110px;
		display: grid;
		grid-template-columns: repeat(var(--grid), 1fr);
		background-size: cover;
		border: 1px solid #e5e7eb;
	}
	.image-grid i {
		background: transparent;
		border: 0.5px solid #ffffff88;
	}
	.image-grid i.selected {
		border: 1.5px solid #7c3aed;
		background: #c4b5fd66;
	}
	.op-box {
		display: grid;
		place-items: center;
		gap: 2px;
		min-height: 62px;
		padding: 6px;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			600 15px Arial,
			sans-serif;
		text-align: center;
	}
	.op-box span {
		font:
			400 11px Arial,
			sans-serif;
		color: #8a6aa3;
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
	.formula-lines {
		display: grid;
		gap: 10px;
		font:
			italic 22px Georgia,
			serif;
		color: #6b7280;
		text-align: center;
		margin: 24px 0 18px;
	}
	sub,
	sup {
		font-size: 12px;
	}
	.status {
		color: #9ca3af;
		font-size: 12px;
		text-align: center;
		margin: 0 0 12px;
	}
	@media (max-width: 600px) {
		.body {
			padding: 12px;
		}
		h3 {
			font-size: 16px;
		}
		.equation,
		.equation.wide {
			gap: 5px;
			grid-template-columns: minmax(0, 1fr) 12px minmax(0, 1.1fr) 12px minmax(0, 1fr);
		}
		.formula-lines {
			font-size: 17px;
		}
	}
</style>
