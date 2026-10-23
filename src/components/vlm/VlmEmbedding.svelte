<script lang="ts">
	import { onMount, setContext, tick } from 'svelte';
	import { fade } from 'svelte/transition';
	import { base } from '$app/paths';
	import { ZoomInOutline } from 'flowbite-svelte-icons';
	import { gsap, Flip } from '~/utils/gsap';
	import {
		tokens,
		moeExample,
		moeTokenIndex,
		expandedBlock,
		blockIdx,
		isExpandOrCollapseRunning
	} from '~/store';
	import {
		vlmRecording,
		vlmPhase,
		vlmVisiblePositions,
		vlmSelectedPosition,
		vlmVisualToken,
		vlmInspectorStage
	} from '$lib/vlm/store';
	import { row } from '$lib/vit/types';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	import VlmControls from './VlmControls.svelte';
	export let className: string | undefined = undefined;
	setContext('block-id', 'embedding');

	/** Dense's Embedding ⊕: the column opens in place. Vision: patch → Wpatch, + learned position.
	 *  Language: text tokens look up a row; image tokens take the connector output unchanged. */
	let expanded = false;
	let isHovered = false;
	let root: HTMLDivElement;
	$: document = $vlmRecording?.document;
	$: vision = $vlmPhase === 'vision';
	$: common = $vlmRecording?.common.vision;
	$: grid = document?.image.gridSize ?? 16;
	$: pixels = (document?.image.patchSize ?? 16) ** 2 * 3;
	$: hidden = document?.streams[$vlmPhase].model.hiddenSize ?? 768;
	$: imagePositions = document?.imageTokenPositions ?? [];
	$: isImage = (index: number) =>
		!vision && imagePositions.includes($vlmVisiblePositions[index] ?? -1);
	$: patchRows = vision
		? $vlmVisiblePositions.map((p) => (common ? row(common, 'patchTokens', p, 8) : []))
		: [];
	$: positionRows = vision
		? $vlmVisiblePositions.map((p) => (common ? row(common, 'positionEmbedding', p, 8) : []))
		: [];
	$: if ($expandedBlock.id !== 'embedding' && expanded) toggle(false);
	$: if ($expandedBlock.id === 'embedding' && !expanded) toggle(true);

	async function toggle(open: boolean) {
		const state = Flip.getState('.embedding .token-column');
		expanded = open;
		await tick();
		isExpandOrCollapseRunning.set(true);
		Flip.from(state, {
			duration: 0.5,
			ease: 'power2.inOut',
			onComplete: () => isExpandOrCollapseRunning.set(false)
		});
		if (open) gsap.to('.embedding-detail', { opacity: 1, duration: 0.5, delay: 0.4 });
	}
	function select(index: number) {
		moeTokenIndex.set(index);
		vlmSelectedPosition.set($vlmVisiblePositions[index] ?? index);
	}
	function open(stage: string, index: number) {
		select(index);
		if (isImage(index)) vlmVisualToken.set(imagePositions.indexOf($vlmVisiblePositions[index]));
		vlmInspectorStage.set(stage);
	}
	/** The squeeze layout has no dim layer of its own, so the expansion brings Dense's. */
	function dimLayer(element: HTMLElement) {
		window.document.querySelector('.main-section')?.appendChild(element);
		return { destroy: () => element.remove() };
	}
	function patchStyle(position: number) {
		if (!document) return '';
		const step = 100 / (grid - 1);
		return `background-image:url('${base}/${document.image.src}');background-size:${grid * 100}%;background-position:${(position % grid) * step}% ${Math.floor(position / grid) * step}%;`;
	}
	onMount(() => {
		const outside = (event: Event) => {
			if (expanded && root && !root.contains(event.target as Node)) expandedBlock.set({ id: null });
		};
		const section = window.document.querySelector<HTMLElement>('.main-section');
		section?.addEventListener('click', outside);
		return () => section?.removeEventListener('click', outside);
	});
</script>

{#if expanded}<div class="vlm-dim" use:dimLayer transition:fade={{ duration: 100 }}></div>{/if}
<div
	class={`embedding vlm-embedding ${className ?? ''}`}
	class:expanded
	bind:this={root}
	data-click="embedding-step"
	role="none"
	on:click={() => {
		if (!expanded) expandedBlock.set({ id: 'embedding' });
	}}
	on:keydown={(event) => {
		if (event.key === 'Enter' && !expanded) expandedBlock.set({ id: 'embedding' });
	}}
>
	<div
		class="title expandable"
		role="none"
		on:click|stopPropagation={() => expandedBlock.set({ id: expanded ? null : 'embedding' })}
		on:keydown={() => {}}
		on:mouseenter={() => (isHovered = true)}
		on:mouseleave={() => (isHovered = false)}
	>
		<div class="title-text">Embedding <ZoomInOutline /></div>
	</div>
	<div class="content relative">
		<div class="bounding embedding-bounding" class:active={isHovered && !expanded}></div>
		<div class="embedding-inputs">
			<div class="token-column resize-watch">
				<div class="column token-string">
					{#if expanded}<div class="subtitle embedding-detail">
							{#if vision}<button
									class="step-link"
									on:click|stopPropagation={() => open('embedding', $moeTokenIndex)}
									>Patches<small>{grid} × {grid} · 16 × 16 px</small></button
								>{:else}<button
									class="step-link"
									on:click|stopPropagation={() => open('merge', $moeTokenIndex)}
									>Tokenization<small>text + image</small></button
								>{/if}
						</div>{/if}
					{#each $tokens as token, index}<div class="cell">
							<button
								class="token-label"
								class:chosen={index === $moeTokenIndex}
								class:image-token={isImage(index)}
								aria-pressed={index === $moeTokenIndex}
								aria-label={`Select ${token}`}
								on:click|stopPropagation={() => select(index)}
								>{#if vision}<span class="patch" style={patchStyle($vlmVisiblePositions[index])}
									></span>{/if}<span class="text">{token}</span></button
							>
						</div>{/each}
				</div>
				{#if expanded && vision}
					<div class="column patch-embedding embedding-detail">
						<div class="subtitle">Patch<br />Embedding<small>{pixels} → {hidden}</small></div>
						{#each $tokens as token, index}<div class="cell">
								<svg viewBox="0 0 31 9" class="arrow" aria-hidden="true"
									><path
										fill="currentColor"
										d="M30.39 4.74a.5.5 0 0 0 0-.7L27.2.85a.5.5 0 1 0-.7.71l2.83 2.83-2.83 2.83a.5.5 0 1 0 .7.7l3.19-3.18ZM.32 4.89h29.71v-1H.32v1Z"
									/></svg
								><button
									class="op"
									title={`Flatten ${token} to ${pixels} numbers, then multiply by Wpatch`}
									aria-label={`Open the patch projection for ${token}`}
									on:click|stopPropagation={() => open('embedding', index)}>Wpatch</button
								>
								<div class="vector bg-gray-300">
									<VectorCanvas active data={patchRows[index]} />
								</div>
							</div>{/each}
					</div>
					<div class="column symbol embedding-detail">
						{#each $tokens as _}<div class="cell">+</div>{/each}
					</div>
					<div class="column position-embedding embedding-detail">
						<div class="subtitle">
							Position<br />Embedding<small>learned · {grid * grid}</small>
						</div>
						{#each $tokens as _, index}<div class="cell">
								<div class="vector bg-gray-300">
									<VectorCanvas active data={positionRows[index]} />
								</div>
								<span class="value">{$vlmVisiblePositions[index]}</span>
							</div>{/each}
					</div>
					<div class="column symbol embedding-detail">
						{#each $tokens as _}<div class="cell">=</div>{/each}
					</div>
				{:else if expanded}
					<!-- Position is metadata for RoPE, not an embedding operand. -->
					<div class="column token-position embedding-detail">
						<div class="subtitle">Position<small>for RoPE</small></div>
						{#each $tokens as _, index}<div class="cell">
								<span class="value plain">{$vlmVisiblePositions[index]}</span>
							</div>{/each}
					</div>
					<div class="column token-embedding embedding-detail">
						<div class="subtitle">
							Token<br />Embedding<small>lookup · or Wproj</small>
						</div>
						{#each $tokens as token, index}<div class="cell">
								<svg viewBox="0 0 31 9" class="arrow" aria-hidden="true"
									><path
										fill="currentColor"
										d="M30.39 4.74a.5.5 0 0 0 0-.7L27.2.85a.5.5 0 1 0-.7.71l2.83 2.83-2.83 2.83a.5.5 0 1 0 .7.7l3.19-3.18ZM.32 4.89h29.71v-1H.32v1Z"
									/></svg
								>{#if isImage(index)}<button
										class="op image-op"
										title={`Image token ${imagePositions.indexOf($vlmVisiblePositions[index])}: the connector output, placed here unchanged`}
										aria-label={`Open the connector projection for ${token}`}
										on:click|stopPropagation={() => open('projector', index)}>Wproj</button
									>{:else}<span class="id"
										>{index === 0 ? 'id ' : ''}{document?.tokenIds[$vlmVisiblePositions[index]] ??
											''}</span
									>{/if}
							</div>{/each}
					</div>
					<div class="column symbol embedding-detail">
						{#each $tokens as _}<div class="cell">=</div>{/each}
					</div>
				{/if}
			</div>
			<div class="vector-column block-start-column">
				<div class="column vectors embedding-column">
					{#each $tokens as token, index}<div
							class="vector bg-gray-300"
							class:from-image={isImage(index)}
							title={isImage(index)
								? `Image token · the connector output for ${token}, unchanged`
								: `Embedding of ${token}`}
						>
							<VectorCanvas
								data={$moeExample?.embeddingPreview[index]}
								active={$blockIdx === 0 && (isHovered || expanded)}
							/>
						</div>{/each}
				</div>
			</div>
		</div>
		<!-- The tab's own controls; clicking them never expands the embedding. -->
		<div class="controls-slot" role="none" on:click|stopPropagation on:keydown|stopPropagation>
			<VlmControls />
		</div>
	</div>
</div>

<style lang="scss">
	.embedding.expanded {
		> .title,
		> .content {
			z-index: $EXPANDED_CONTENT_INDEX;
		}
		/* Room above the columns for their subtitles, as Dense has. */
		> .title {
			padding-bottom: 38px !important;
		}
		.token-column {
			background: white;
			box-shadow: 14px 0 14px white;
		}
	}
	.embedding-detail {
		opacity: 0;
	}
	:global(.vlm-dim) {
		position: absolute;
		inset: 0;
		z-index: $DIM_INDEX;
		background-color: white;
		opacity: 0.7;
		user-select: none;
	}
	.title {
		position: relative;
	}
	.title-text {
		display: flex;
		align-items: center;
		gap: 4px;
		width: max-content;
		cursor: pointer;
		font-size: 13px;
	}
	.title-text :global(svg) {
		width: 16px;
		height: 16px;
	}
	.content {
		display: flex;
		flex-direction: column;
	}
	.embedding-inputs {
		display: flex;
		gap: 12px;
	}
	.token-column {
		display: flex;
		position: relative;
	}
	.token-column .column {
		flex-shrink: 0;
	}
	.subtitle {
		transform: translateY(calc(-100% - 6px)) !important;
		font-size: 10px !important;
		line-height: 1.2;
		white-space: normal;
	}
	.subtitle small,
	.step-link small {
		display: block;
		font-size: 8.5px;
		color: #b7a9c3;
	}
	.step-link {
		width: 100%;
		color: inherit;
		cursor: pointer;
	}
	.step-link:hover {
		color: #6b7280;
	}
	.token-label {
		display: flex;
		align-items: center;
		justify-content: flex-end;
		gap: 6px;
		width: 100%;
		height: var(--vector-height);
		font-size: 11px;
		color: #4b5563;
		white-space: nowrap;
		cursor: pointer;
	}
	.token-label .text {
		max-width: 85px;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.token-label:hover .text {
		color: #111827;
	}
	.token-label.image-token .text {
		color: #9064b0;
	}
	.token-label.chosen .text {
		color: #7c3aed;
		font-weight: 600;
	}
	.patch {
		display: inline-block;
		width: 14px;
		height: 14px;
		border: 1px solid #dfd6e9;
		border-radius: 2px;
		flex-shrink: 0;
	}
	.token-label.chosen .patch {
		border-color: #a77fd0;
	}
	.patch-embedding .cell,
	.position-embedding .cell,
	.token-embedding .cell {
		gap: 6px;
		justify-content: flex-start;
	}
	.patch-embedding,
	.position-embedding,
	.token-embedding,
	.token-position {
		padding: 0 4px;
	}
	.token-position .cell {
		justify-content: center;
	}
	/* Wide enough for each column's subtitle, as in Dense. */
	.expanded .token-string {
		min-width: 76px;
	}
	.token-position {
		min-width: 48px;
	}
	.token-embedding {
		min-width: 84px;
	}
	.arrow {
		width: 26px;
		color: #d1d5db;
		flex-shrink: 0;
	}
	.op {
		width: 44px;
		height: min(18px, calc(var(--vector-height) + 2px));
		padding: 0;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			9.5px/1 Arial,
			sans-serif;
		cursor: pointer;
		flex-shrink: 0;
	}
	.op:hover {
		background: #f1e9f9;
		border-color: #b9a0d4;
	}
	.value {
		min-width: 26px;
		padding: 0 3px;
		border: 1px solid #e5e7eb;
		border-radius: 3px;
		font: 10px/1.3 monospace;
		color: #6b7280;
		text-align: center;
	}
	.value.plain {
		border: 0;
	}
	.id {
		font: 10px/1.3 monospace;
		color: #4b5563;
		white-space: nowrap;
	}
	.symbol {
		width: 16px;
		color: #9ca3af;
		font-size: 12px;
	}
	.symbol .cell {
		justify-content: center;
	}
	.embedding-bounding {
		left: -4px;
		width: calc(100% + 8px);
	}
	.vector.from-image {
		box-shadow: inset 2px 0 0 #a78bfa;
	}
	button:focus-visible {
		outline: 2px solid #b596cf;
		outline-offset: 2px;
	}
</style>
