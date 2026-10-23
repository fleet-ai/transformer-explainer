<script lang="ts">
	import { setContext } from 'svelte';
	import {
		tokens,
		blockIdx,
		moeExample,
		moeTokenIndex,
		vectorHeight,
		attentionHeadIdx,
		modelMeta
	} from '~/store';
	import {
		hybridRecording,
		hybridLayer,
		hybridInspector,
		hybridMambaView,
		hybridStage,
		hybridHot
	} from '$lib/hybrid/store';
	import { row } from '$lib/vit/types';
	import OperationGroup from '../OperationGroup.svelte';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	export let className: string | undefined = undefined;
	setContext('block-id', 'mlp');

	/**
	 * The MLP column. The Mamba half gates the scan output with z, normalizes it and projects it
	 * back to the residual stream; the shared-transformer half is a gated GELU MLP and a linear
	 * layer whose output t feeds this block's Mamba half.
	 */
	$: model = $hybridRecording?.document.model;
	$: hidden = model?.hiddenSize ?? 2048;
	$: inner = (model?.mambaHeads ?? 64) * (model?.mambaHeadSize ?? 64);
	$: intermediate = model?.intermediateSize ?? 8192;
	$: layer = $moeExample?.layers[$blockIdx];
	$: tall = 2 * $vectorHeight;
	$: preview = (name: string, index: number) =>
		$hybridLayer?.descriptor.tensors[name] ? row($hybridLayer, name, index, 64) : undefined;
	// A box and the bands of its operation light up together, from either side.
	$: if (typeof document !== 'undefined')
		document
			.querySelectorAll<SVGPathElement>('path.sankey-path[class*="hybrid-op-"]')
			.forEach((path) => {
				path.style.opacity =
					$hybridHot && path.classList.contains(`hybrid-op-${$hybridHot}`) ? '0.6' : '';
			});
	const hover = (group: string) => ({
		enter: () => hybridHot.set(group),
		leave: () => hybridHot.set(null)
	});
	function open(stage: string, index?: number) {
		if (index !== undefined) moeTokenIndex.set(index);
		hybridInspector.set(stage);
	}
</script>

{#if $hybridMambaView}<div class={`mlp mamba-output ${className ?? ''}`} data-click="mlp-step">
		<div class="title">
			Gate · RMSNorm · Wout<small>z gates the scan output · {inner} → {hidden}</small>
		</div>
		<div class="content resize-watch relative">
			<div class="first-layer layer">
				<div class="column initial">
					<span class="subtitle">y · {inner}</span>
					{#each $tokens as token, index}<div class="cell" style={`height:${tall}px`}>
							<span class="label float">{token}</span><button
								class="vector y-vector bg-purple-200"
								style={`height:${tall}px`}
								class:chosen={index === $moeTokenIndex}
								aria-label={`Gate for token ${index}: ${token}`}
								on:click={() => open('gate', index)}
								><VectorCanvas colorScale="purple" data={preview('scanOutput', index)} />
								<div
									class="sub-vector head1"
									style={`top:${(tall / $modelMeta.attention_head_num) * $attentionHeadIdx}px;height:${Math.max(1, tall / $modelMeta.attention_head_num)}px`}
								></div></button
							>
						</div>{/each}
				</div>
			</div>
			<div class="ops">
				<button
					class="op-box gate-box"
					title="y ⊙ SiLU(z): z comes straight from in_proj"
					class:hot={$hybridHot === 'gate'}
					on:mouseenter={hover('gate').enter}
					on:mouseleave={hover('gate').leave}
					on:click={() => open('gate')}
					>⊙<small>SiLU(z)</small>{#each $tokens as _, index}<span
							class="gate-port"
							style={`top:${index * (tall + 8)}px;height:${tall}px`}
						></span>{/each}</button
				>
				<button
					class="op-box norm-box"
					title="RMSNorm over 4096"
					class:hot={$hybridHot === 'gate'}
					on:mouseenter={hover('gate').enter}
					on:mouseleave={hover('gate').leave}
					on:click={() => open('gated-norm')}>RMSNorm<small>{inner}</small></button
				>
			</div>
			<div class="second-layer">
				<div class="column gated">
					{#each $tokens as token, index}<div
							class="vector gated-vector bg-indigo-200"
							style={`height:${tall}px`}
						>
							<VectorCanvas colorScale="purple" data={preview('gated', index)} />
						</div>{/each}
					<span class="band-label">Wout ⊕</span>
				</div>
			</div>
			<div class="out-layer layer">
				<OperationGroup type="residual-end" id="embedding-residual" />
				<div class="column out mlp-out-column">
					{#each $tokens as token, index}<button
							class="vector bg-blue-200"
							aria-label={`Block output ${token}`}
							on:click={() => open('residual', index)}
							><VectorCanvas colorScale="blue" data={layer?.blockOutputPreview?.[index]} /></button
						>{/each}
				</div>
			</div>
		</div>
	</div>{:else}<div class={`mlp ffn-output ${className ?? ''}`} data-click="mlp-step">
		<div class="title">
			Gated MLP → Linear<small
				>GELU(g) ⊙ u · {hidden} → {intermediate} → {hidden} · shared weights + block adapter</small
			>
		</div>
		<div class="content resize-watch relative">
			<div class="first-layer layer">
				<div class="column initial">
					{#each $tokens as token, index}<div class="cell">
							<span class="label float">{token}</span><button
								class="vector bg-purple-200"
								aria-label={`Up and gate projections for ${token}`}
								on:click={() => open('ffn-up', index)}
								><VectorCanvas colorScale="purple" data={layer?.attentionOutputPreview[index]} />
								<div
									class="sub-vector head1"
									style={`position:absolute;top:${($vectorHeight / $modelMeta.attention_head_num) * $attentionHeadIdx}px;height:${$vectorHeight / $modelMeta.attention_head_num}px`}
								></div></button
							>
						</div>{/each}
				</div>
				<OperationGroup type="ln" id="mlp-ln" />
			</div>
			<div class="second-layer">
				<div class="column product">
					<div class="ffn-ops">
						<button
							class="op-box small"
							class:hot={$hybridHot === 'ffn-up'}
							on:mouseenter={hover('ffn-up').enter}
							on:mouseleave={hover('ffn-up').leave}
							on:click={() => open('gelu')}>GELU</button
						>
						<button
							class="op-box small glyph"
							class:hot={$hybridHot === 'ffn-up'}
							on:mouseenter={hover('ffn-up').enter}
							on:mouseleave={hover('ffn-up').leave}
							on:click={() => open('ffn-product')}>⊙</button
						>
					</div>
					{#each $tokens as token, index}<div class="vector x4 expanded-vector bg-indigo-200">
							<VectorCanvas colorScale="purple" data={preview('product', index)} />
						</div>{/each}
					<span class="band-label up">Wup, Wgate ⊕</span>
					<span class="band-label down">Wdown ⊕</span>
				</div>
			</div>
			<button
				class="op-box linear-box"
				title="This block's own linear layer, 2048 → 2048, after Wdown; its output is t"
				class:hot={$hybridHot === 'ffn-down'}
				on:mouseenter={hover('ffn-down').enter}
				on:mouseleave={hover('ffn-down').leave}
				on:click={() => open('linear')}
				>{#each $tokens as _, index}<span
						class="linear-in"
						style={`top:${index * ($vectorHeight + 8)}px;height:${$vectorHeight}px`}
					></span><span
						class="linear-out"
						style={`top:${index * ($vectorHeight + 8)}px;height:${$vectorHeight}px`}
					></span>{/each}Linear<small>{hidden} → {hidden}</small></button
			>
			<button
				class="op-box next-half"
				class:hot={$hybridHot === 'next'}
				title="This block's Mamba half: x + t → RMSNorm → Mamba2 mixer, added to x"
				on:mouseenter={hover('next').enter}
				on:mouseleave={hover('next').leave}
				on:click={() => hybridStage.set('mamba')}
				><span class="next-port" aria-hidden="true"></span><span
					class="next-out-port"
					aria-hidden="true"
				></span><span class="t-label" aria-hidden="true">t</span>Mamba ›<small>x + t · 2 of 2</small
				></button
			>
			<div class="out-layer layer">
				<OperationGroup type="residual-end" id="embedding-residual" />
				<div class="column out mlp-out-column">
					{#each $tokens as token, index}<button
							class="vector bg-blue-200"
							aria-label={`Block output ${token}, after the Mamba half`}
							on:click={() => hybridStage.set('mamba')}
							><VectorCanvas colorScale="blue" data={layer?.blockOutputPreview?.[index]} /></button
						>{/each}
				</div>
			</div>
		</div>
	</div>{/if}

<style>
	.title small {
		display: block;
		font-size: 9px;
		color: #ad99ba;
		margin-top: 6px;
		white-space: normal;
	}
	.content {
		display: flex;
		align-items: flex-start;
		gap: 0;
	}
	.layer {
		display: flex;
	}
	.first-layer {
		margin-left: 10px;
	}
	.label.float {
		white-space: nowrap;
		max-width: 5rem;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.vector {
		border: 0;
		padding: 0;
		display: block;
	}
	button.vector {
		cursor: pointer;
	}
	.y-vector.chosen {
		box-shadow: 0 0 0 1px #b497d0;
	}
	.head1 {
		position: absolute;
		left: 0;
		width: 100%;
		background: #c084fc;
		opacity: 0.8;
	}
	.column .subtitle {
		top: calc(100% + 8px);
		transform: none;
		font-size: 9px;
		white-space: nowrap;
	}
	/* The z bypass drawn by the band layer. */
	:global(path.hybrid-z) {
		stroke-dasharray: 6 4;
	}
	.mamba-output .first-layer {
		margin-right: 26px;
	}
	.ops {
		position: relative;
		z-index: 120;
		display: flex;
		gap: 10px;
		height: calc(12 * var(--vector-height) + 40px);
	}
	.op-box {
		position: relative;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			11px Arial,
			sans-serif;
		cursor: pointer;
		display: grid;
		place-items: center;
		align-content: center;
		padding: 0 4px;
	}
	.op-box:hover,
	.op-box.hot {
		background: #f1e9f9;
		border-color: #b9a0d4;
	}
	.mamba-output,
	.ffn-output {
		user-select: none;
		-webkit-user-select: none;
	}
	.op-box:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: 2px;
	}
	.op-box small {
		display: block;
		font-size: 8.5px;
		color: #8a6aa3;
	}
	.gate-box {
		width: 44px;
		font-size: 18px;
	}
	.norm-box {
		width: 58px;
	}
	.gate-port {
		position: absolute;
		left: -2px;
		top: 0;
		width: 2px;
		pointer-events: none;
	}
	.second-layer {
		position: relative;
		z-index: 20;
		pointer-events: none;
	}
	.mamba-output .second-layer {
		margin-left: 26px;
	}
	.band-label {
		position: absolute;
		top: calc(100% + 10px);
		left: 50%;
		transform: translateX(-50%);
		font-size: 11px;
		color: #9ca3af;
		white-space: nowrap;
	}
	.mamba-output .band-label {
		left: calc(100% + 60px);
	}
	.out-layer {
		margin-left: auto;
	}
	.out-layer .column.out .vector {
		cursor: pointer;
	}
	.ffn-output .first-layer {
		margin-right: 44px;
	}
	.ffn-output .out-layer {
		margin-left: 0;
	}
	.ffn-ops {
		position: absolute;
		right: calc(100% + 4px);
		top: 0;
		display: flex;
		flex-direction: column;
		gap: 6px;
		z-index: 120;
		pointer-events: auto;
	}
	.op-box.small {
		width: 38px;
		height: 24px;
		font-size: 10px;
	}
	.op-box.small.glyph {
		font-size: 15px;
	}
	.band-label.up {
		left: -34px;
	}
	.band-label.down {
		left: calc(100% + 34px);
	}
	.expanded-vector {
		pointer-events: auto;
	}
	/* The Wdown band runs into this box; its output t runs on into the Mamba step. */
	.linear-box {
		z-index: 120;
		height: calc(6 * var(--vector-height) + 40px);
		width: 46px;
		margin-left: auto;
		font-size: 10px;
	}
	.linear-box small {
		font-size: 7.5px;
		white-space: nowrap;
	}
	.linear-in,
	.linear-out {
		position: absolute;
		width: 2px;
		pointer-events: none;
	}
	.linear-in {
		left: -2px;
	}
	.linear-out {
		right: -2px;
	}
	/* The collapsed second half: t goes in, the block output comes out. */
	.next-half {
		z-index: 120;
		margin: calc(3 * var(--vector-height) - 6px) 18px 0 20px;
		height: 52px;
		padding: 4px 6px;
		white-space: nowrap;
	}
	.t-label {
		position: absolute;
		right: calc(100% + 6px);
		top: -14px;
		font-size: 10px;
		color: #a07cc0;
	}
	.next-out-port {
		position: absolute;
		right: -2px;
		top: 0;
		width: 2px;
		height: 100%;
		pointer-events: none;
	}
	.next-port {
		position: absolute;
		left: -2px;
		top: 0;
		width: 2px;
		height: 100%;
		pointer-events: none;
	}
</style>
