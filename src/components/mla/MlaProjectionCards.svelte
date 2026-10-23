<script lang="ts">
	import { onMount, tick } from 'svelte';
	import {
		attentionHeadIdx,
		bandRedraw,
		headGap,
		moeExample,
		moeModel,
		moeTokenIndex,
		tokens
	} from '~/store';
	import { attentionInspector } from '$lib/inspectors';
	import { mlaProjectionView } from '$lib/mla/projections.js';
	import { tokenLabel } from '$lib/moe/format';
	import AnimatedMatrixPopup from '../common/AnimatedMatrixPopup.svelte';
	import MlaNormPopup from './MlaNormPopup.svelte';

	/** Two cards in the QKV column: the query projection, and the latent KV path whose
	 *  normalized latent plus rotated key is what the cache stores for every token. */
	type Step = 'query' | 'compression' | 'norm' | 'key' | 'value';
	type Hot = Step | 'rope';
	const MAX_GHOSTS = 5;
	const BAND_PX = 36;
	const MIN_WIDTH = 150;
	const MAX_WIDTH = 240;
	let host: HTMLDivElement;
	let left = 56;
	let width = 160;
	let step: Step | null = null;
	let hovered: Hot | null = null;

	$: model = $moeModel;
	$: heads = model?.attentionHeads ?? 16;
	$: hidden = model?.hiddenSize ?? 2048;
	$: content = model?.contentKeySize ?? 128;
	$: rotary = model?.rotarySize ?? 64;
	$: latent = model?.kvLatentSize ?? 512;
	$: valueSize = model?.valueHeadSize ?? 128;
	$: querySize = heads * (content + rotary);
	$: keySize = heads * content;
	$: valueTotal = heads * valueSize;
	$: cached = latent + rotary;
	$: layers = model?.numLayers ?? 27;
	// Band thickness follows the dimension, with Q (the widest) at BAND_PX.
	$: px = (size: number) => Math.max(2.5, (size / querySize) * BAND_PX);
	$: xPx = px(hidden);
	$: latentPx = px(latent);
	$: tokenText = tokenLabel($moeExample?.tokens[$moeTokenIndex]?.text ?? 'x');
	$: ghosts = Math.min(MAX_GHOSTS, Math.max(0, $tokens.length - 1));
	$: fw = width - 20;
	$: projection =
		step && step !== 'norm'
			? mlaProjectionView(undefined, model, $attentionHeadIdx, $moeTokenIndex, step)
			: null;
	$: note = {
		query: `${querySize} = ${heads} heads × (${content} content + ${rotary} rotary). Only the rotary part is rotated by RoPE, and head ${$attentionHeadIdx + 1} reads its own ${content + rotary}D slice.`,
		compression: `${hidden}D becomes a ${latent}D latent plus a ${rotary}D rotary key, ${cached}D in all. This layer's cache keeps the normalized latent and the rotated key: ${cached} numbers per token, shared by every head. Each of the ${layers} layers has its own cache, ${cached * layers} numbers per token in all.`,
		key: `Every head expands the same ${latent}D latent into its own ${content}D content key. The ${rotary}D rotated key is shared by all heads.`,
		value: `Every head expands the same ${latent}D latent into its own ${valueSize}D value. Values never rotate.`,
		norm: ''
	};

	// Slice of the selected head inside a band, as a thin stripe at its place along the thickness.
	$: stripe = (thickness: number) => ({
		offset: (($attentionHeadIdx + 0.5) / heads - 0.5) * thickness,
		thickness: Math.max(1.8, thickness / heads)
	});
	$: qStripe = stripe(BAND_PX);
	$: kvStripe = stripe(px(keySize));

	function measure() {
		if (!host) return;
		const content = host.parentElement;
		const column = content?.querySelector('.vector-column');
		const headCard = document.querySelector('.head-card.first');
		if (!content || !column) return;
		const origin = content.getBoundingClientRect().left;
		left = Math.max(0, column.getBoundingClientRect().right - origin + 6);
		if (headCard) {
			const free =
				headCard.getBoundingClientRect().left - (origin + left) - ghosts * headGap.x - 12;
			width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, free));
		}
	}
	function select(id: Hot) {
		if (id === 'rope') {
			step = null;
			attentionInspector.set({ kind: 'rope', vector: 'key' });
		} else step = id;
	}
	const titles: Record<Hot, string> = {
		query: 'query projection (Wq)',
		compression: 'KV down-projection (W↓KV)',
		norm: 'latent RMSNorm',
		rope: 'RoPE on the shared position key',
		key: 'key up-projection (W↑K)',
		value: 'value up-projection (W↑V)'
	};
	onMount(() => {
		measure();
		const observer = new ResizeObserver(measure);
		observer.observe(document.body);
		return () => observer.disconnect();
	});
	$: if (width || left || ghosts >= 0) tick().then(() => bandRedraw.update((count) => count + 1));

	// Query card geometry.
	const qY = 40;
	const qBoxX = 52;
	const qBoxW = 66;
	const projH = 40;
	// KV card geometry.
	const inY = 34;
	const boxX = 46;
	const rowCache = 110;
	const ropeY = 115;
	const ropeH = 38;
	const rowK = 190;
	const rowV = 262;
	$: ropeX = fw - 32;
	// Between the latent band on the left and the rotary band on the right.
	$: cacheX = (22 + ropeX) / 2;
	$: kvBoxes = <
		{
			id: Hot;
			x: number;
			y: number;
			w: number;
			h?: number;
			label: string;
			sub: string;
			hint?: string;
			accent?: string;
		}[]
	>[
		{ id: 'compression', x: boxX, y: inY, w: 56, label: 'W↓KV', sub: `${hidden}→${cached}` },
		{ id: 'norm', x: 36, y: rowCache, w: 56, label: 'RMSNorm', sub: `${latent}` },
		{ id: 'rope', x: ropeX, y: ropeY, w: 58, h: ropeH, label: 'RoPE ↻', sub: `${rotary}D key` },
		{
			id: 'key',
			x: 64,
			y: rowK,
			w: 64,
			h: projH,
			label: 'W↑K',
			sub: `${latent}→${keySize}`,
			hint: `head: ${latent}→${content}`,
			accent: '#cf6a6a'
		},
		{
			id: 'value',
			x: 64,
			y: rowV,
			w: 64,
			h: projH,
			label: 'W↑V',
			sub: `${latent}→${valueTotal}`,
			hint: `head: ${latent}→${valueSize}`,
			accent: '#3f9a62'
		}
	];
	$: qBands = [
		{ id: 'query' as Step, w: xPx, d: `M8 ${qY} H${qBoxX - qBoxW / 2}` },
		{ id: 'query' as Step, w: BAND_PX, d: `M${qBoxX + qBoxW / 2} ${qY} H${fw - 8}` }
	];
	$: kvBands = <{ id: Hot; w: number; d: string; color?: string }[]>[
		{ id: 'compression' as Hot, w: xPx, d: `M8 ${inY} H18` },
		{ id: 'compression' as Hot, w: latentPx, d: `M34 ${inY + 14} V96` },
		{
			id: 'rope' as Hot,
			w: 3,
			color: '#f1c27a',
			d: `M64 ${inY + 14} C64 72 ${ropeX} 72 ${ropeX} 96`
		},
		{ id: 'norm' as Hot, w: latentPx, d: `M22 ${rowCache + 13} V${rowV}` },
		{ id: 'key' as Hot, w: latentPx, d: `M22 ${rowK} H32` },
		{ id: 'value' as Hot, w: latentPx, d: `M22 ${rowV} H32` },
		{ id: 'key' as Hot, w: px(keySize), d: `M96 ${rowK} H${fw - 8}` },
		{ id: 'value' as Hot, w: px(valueTotal), d: `M96 ${rowV} H${fw - 8}` },
		{
			id: 'rope' as Hot,
			w: 3,
			color: '#f1c27a',
			d: `M${ropeX} ${ropeY + ropeH / 2 - 1} V${rowK + 14}`
		}
	];
</script>

<div
	class="mla-cards"
	bind:this={host}
	style={`left:${left}px;width:${width + ghosts * headGap.x}px;`}
>
	<div class="stack query-stack" style={`width:${width}px;height:118px;`}>
		<div class="card query-card" style={`width:${width}px;height:118px;`}>
			<header>
				<strong>Query</strong>
				<span>{querySize}D · {heads} heads</span>
			</header>
			<svg
				width={fw}
				height="70"
				viewBox={`0 0 ${fw} 70`}
				role="group"
				aria-label="Query projection"
			>
				{#each qBands as band}<path
						d={band.d}
						style={`stroke-width:${band.w}px`}
						class:hot={hovered === band.id}
						role="button"
						tabindex="0"
						aria-label={`Open ${titles[band.id]}`}
						on:click={() => select(band.id)}
						on:mouseenter={() => (hovered = band.id)}
						on:mouseleave={() => (hovered = null)}
						on:keydown={(event) => {
							if (event.key === 'Enter' || event.key === ' ') {
								event.preventDefault();
								select(band.id);
							}
						}}
					/>{/each}
				<path
					class="stripe"
					d={`M${qBoxX + qBoxW / 2} ${qY + qStripe.offset} H${fw - 8}`}
					style={`stroke-width:${qStripe.thickness}px;stroke:#6fa0ee`}
				/>
				<rect class="bar mla-in mla-q-in" x="0" y={qY - xPx / 2} width="8" height={xPx} />
				<rect
					class="bar q-bar mla-q-out"
					x={fw - 8}
					y={qY - BAND_PX / 2}
					width="8"
					height={BAND_PX}
				/>
				<foreignObject x={qBoxX - qBoxW / 2} y={qY - projH / 2} width={qBoxW} height={projH}>
					<button
						type="button"
						class:hot={hovered === 'query'}
						title={titles.query}
						aria-label={`Open ${titles.query}`}
						on:click={() => select('query')}
						on:mouseenter={() => (hovered = 'query')}
						on:mouseleave={() => (hovered = null)}
						>Wq<small>{hidden}→{querySize}</small><small class="hint" style="color:#4f7fd0"
							>head: {hidden}→{content + rotary}</small
						></button
					>
				</foreignObject>
				<rect
					class="slice"
					x={fw - 8}
					y={qY + qStripe.offset - qStripe.thickness / 2}
					width="8"
					height={qStripe.thickness}
					fill="#6fa0ee"
				/>
				<path
					class="marker"
					d={`M${fw + 7} ${qY + qStripe.offset - 3.5} L${fw + 1} ${qY + qStripe.offset} L${fw + 7} ${qY + qStripe.offset + 3.5} Z`}
					fill="#6fa0ee"
				/>
				<text class="out-label q" x={fw - 12} y="15" text-anchor="end"
					>Q · head {$attentionHeadIdx + 1}: {content + rotary}</text
				>
			</svg>
		</div>
	</div>

	<div class="stack kv-stack" style={`width:${width}px;height:370px;margin-top:12px;`}>
		{#each Array(ghosts) as _, ghost}
			<div
				class="ghost"
				style={`width:${width}px;height:370px;transform:translate(${(ghost + 1) * headGap.x}px,${(ghost + 1) * headGap.y}px);z-index:${ghosts - ghost};opacity:${Math.max(0.2, 1 - (ghost + 1) * 0.16)};`}
			></div>
		{/each}
		<div class="card kv-card" style={`width:${width}px;height:370px;z-index:${ghosts + 1};`}>
			<header>
				<strong>Latent KV · cache</strong>
				<span>{cached} / token, per layer</span>
			</header>
			<svg
				width={fw}
				height="310"
				viewBox={`0 0 ${fw} 310`}
				role="group"
				aria-label="Latent KV projection and cache"
			>
				<rect class="cache" x="0" y="66" width={fw} height="92" rx="8" />
				<text class="cache-label" x={cacheX} y="146" text-anchor="middle">cache · {cached}</text>
				<text class="cache-note" x={cacheX} y="155" text-anchor="middle">shared by all heads</text>
				{#each kvBands as band}<path
						d={band.d}
						style={`stroke-width:${band.w}px;${band.color ? `stroke:${band.color}` : ''}`}
						class:hot={hovered === band.id}
						role="button"
						tabindex="0"
						aria-label={`Open ${titles[band.id]}`}
						on:click={() => select(band.id)}
						on:mouseenter={() => (hovered = band.id)}
						on:mouseleave={() => (hovered = null)}
						on:keydown={(event) => {
							if (event.key === 'Enter' || event.key === ' ') {
								event.preventDefault();
								select(band.id);
							}
						}}
					/>{/each}
				<path
					class="stripe"
					d={`M96 ${rowK + kvStripe.offset} H${fw - 8}`}
					style={`stroke-width:${kvStripe.thickness}px;stroke:#e98a8a`}
				/>
				<path
					class="stripe"
					d={`M96 ${rowV + kvStripe.offset} H${fw - 8}`}
					style={`stroke-width:${kvStripe.thickness}px;stroke:#6fc08a`}
				/>
				<rect class="bar mla-in mla-kv-in" x="0" y={inY - xPx / 2} width="8" height={xPx} />
				<rect
					class="bar k-bar mla-k-out"
					x={fw - 8}
					y={rowK - px(keySize) / 2}
					width="8"
					height={px(keySize)}
				/>
				<rect class="bar pos-tail" x={fw - 8} y={rowK + px(keySize) / 2} width="8" height="6" />
				<rect
					class="bar v-bar mla-v-out"
					x={fw - 8}
					y={rowV - px(valueTotal) / 2}
					width="8"
					height={px(valueTotal)}
				/>
				{#each [{ row: rowK, color: '#e98a8a' }, { row: rowV, color: '#6fc08a' }] as slice}
					<rect
						class="slice"
						x={fw - 8}
						y={slice.row + kvStripe.offset - kvStripe.thickness / 2}
						width="8"
						height={kvStripe.thickness}
						fill={slice.color}
					/>
					<path
						class="marker"
						d={`M${fw + 7} ${slice.row + kvStripe.offset - 3.5} L${fw + 1} ${slice.row + kvStripe.offset} L${fw + 7} ${slice.row + kvStripe.offset + 3.5} Z`}
						fill={slice.color}
					/>
				{/each}
				<text class="out-label k" x={fw - 12} y={rowK + 31} text-anchor="end"
					>K · head {$attentionHeadIdx + 1}</text
				>
				<text class="out-sub" x={fw - 12} y={rowK + 41} text-anchor="end"
					>{content} + {rotary} shared</text
				>
				<text class="out-label v" x={fw - 12} y={rowV + 31} text-anchor="end"
					>V · head {$attentionHeadIdx + 1}</text
				>
				<text class="out-sub" x={fw - 12} y={rowV + 41} text-anchor="end">{valueSize}D</text>
				{#each kvBoxes as box}<foreignObject
						x={box.x - box.w / 2}
						y={box.y - (box.h ?? 28) / 2}
						width={box.w}
						height={box.h ?? 28}
						><button
							type="button"
							class:hot={hovered === box.id}
							title={titles[box.id]}
							aria-label={`Open ${titles[box.id]}`}
							on:click={() => select(box.id)}
							on:mouseenter={() => (hovered = box.id)}
							class:big={box.id === 'rope'}
							on:mouseleave={() => (hovered = null)}
							>{box.label}<small>{box.sub}</small>{#if box.hint}<small
									class="hint"
									style={`color:${box.accent}`}>{box.hint}</small
								>{/if}</button
						></foreignObject
					>{/each}
			</svg>
			<footer>behind: other tokens' cache</footer>
		</div>
	</div>
</div>

{#if step === 'norm'}
	<MlaNormPopup
		size={latent}
		{rotary}
		epsilon={model?.rmsNormEps ?? 1e-6}
		token={tokenText}
		on:close={() => (step = null)}
	/>
{:else if step && projection}
	{#key `${step}-${$moeTokenIndex}-${$attentionHeadIdx}`}
		<AnimatedMatrixPopup {projection} token={tokenText} label="MLA" on:close={() => (step = null)}>
			<p class="mla-note">{note[step]}</p>
		</AnimatedMatrixPopup>
	{/key}
{/if}

<style>
	.mla-cards {
		position: absolute;
		top: 0;
		z-index: 5;
	}
	.stack {
		position: relative;
	}
	.card,
	.ghost {
		position: absolute;
		left: 0;
		top: 0;
		box-sizing: border-box;
		background: white;
		border: 1px solid #f3f4f6;
		border-radius: 4px;
		box-shadow:
			0 4px 6px -1px rgba(3, 7, 18, 0.15),
			0 2px 4px -2px rgba(3, 7, 18, 0.08);
	}
	.card {
		display: flex;
		flex-direction: column;
		padding: 8px 10px 6px;
		user-select: none;
		-webkit-user-select: none;
		font:
			11px/1.3 Arial,
			sans-serif;
		color: #9ca3af;
	}
	header {
		display: flex;
		flex-direction: column;
		margin-bottom: 4px;
	}
	header strong {
		color: #6b7280;
		font-weight: 500;
		font-size: 13px;
	}
	header span {
		color: #a898ba;
		font-size: 9.5px;
		white-space: nowrap;
	}
	svg {
		display: block;
		overflow: visible;
	}
	path {
		fill: none;
		stroke: #dfe3fb;
		pointer-events: stroke;
		cursor: pointer;
		transition: stroke 0.12s;
	}
	path.hot {
		stroke: #cfd5f7;
	}
	path.stripe {
		pointer-events: none;
		opacity: 0.85;
	}
	path:focus-visible {
		outline: 2px solid #a779c8;
	}
	.bar {
		fill: #e9d5ff;
	}
	.q-bar {
		fill: #bfd7fb;
	}
	.k-bar {
		fill: #f8c6c6;
	}
	.v-bar {
		fill: #c9ecd3;
	}
	.pos-tail {
		fill: #f1c27a;
	}
	.cache {
		fill: #fffaf0;
		stroke: #e5c07a;
		stroke-dasharray: 5 4;
	}
	.cache-label {
		fill: #9a6b13;
		font-size: 9px;
		font-weight: 600;
	}
	.cache-note {
		fill: #b08a3c;
		font-size: 8px;
	}
	.out-label {
		font-size: 9px;
		font-weight: 600;
	}
	.out-sub {
		fill: #b7a9c3;
		font-size: 8px;
	}
	.slice,
	.marker {
		pointer-events: none;
	}
	.out-label.q {
		fill: #6fa0ee;
	}
	.out-label.k {
		fill: #e98a8a;
	}
	.out-label.v {
		fill: #6fc08a;
	}
	foreignObject button {
		width: 100%;
		height: 100%;
		padding: 0;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			10.5px/1.05 Arial,
			sans-serif;
		cursor: pointer;
		display: grid;
		place-items: center;
		align-content: center;
	}
	foreignObject button small {
		display: block;
		font-size: 8px;
		color: #8a6aa3;
	}
	foreignObject button small.hint {
		font-weight: 600;
		margin-top: 1px;
	}
	foreignObject button.big {
		font-size: 13px;
	}
	foreignObject button.big small {
		font-size: 9.5px;
	}
	foreignObject button.hot {
		background: #f1e9f9;
		border-color: #b9a0d4;
	}
	foreignObject button:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: -2px;
	}
	footer {
		margin-top: auto;
		color: #b7a9c3;
		font-size: 8.5px;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	:global(.mla-note) {
		color: #6b7280;
		font-size: 13px;
		line-height: 1.5;
		margin: 6px 0 0;
		text-align: center;
	}
</style>
