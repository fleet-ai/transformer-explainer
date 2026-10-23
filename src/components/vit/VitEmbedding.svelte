<script lang="ts">
	import { setContext, tick } from 'svelte';
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
		vitRecording,
		vitVisiblePositions,
		vitSelectedPosition,
		vitInspectorStage
	} from '$lib/vit/store';
	import { row } from '$lib/vit/types';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	import VitPatchControls from './VitPatchControls.svelte';
	export let className: string | undefined = undefined;
	setContext('block-id', 'embedding');

	/** Dense's Embedding ⊕: the column opens in place to show patch → Wpatch (+ [CLS]) + position = x₀. */
	let expanded = false;
	let isHovered = false;
	$: common = $vitRecording?.common;
	$: document = $vitRecording?.document;
	$: hidden = document?.model.hiddenSize ?? 192;
	$: grid = document?.image.gridSize ?? 14;
	$: pixels = (document?.image.patchSize ?? 16) ** 2 * 3;
	$: patchRows = $vitVisiblePositions.map((p) => (common ? row(common, 'patchTokens', p, 8) : []));
	$: positionRows = $vitVisiblePositions.map((p) =>
		common ? row(common, 'positionEmbedding', p, 8) : []
	);
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
		vitSelectedPosition.set($vitVisiblePositions[index]);
	}
	function open(stage: string, index: number) {
		select(index);
		vitInspectorStage.set(stage);
	}
	function patchStyle(position: number) {
		if (!document) return '';
		const patch = position - 1;
		const step = 100 / (grid - 1);
		return `background-image:url('${base}/${document.image.src}');background-size:${grid * 100}%;background-position:${(patch % grid) * step}% ${Math.floor(patch / grid) * step}%;`;
	}
</script>

<div
	class={`embedding vit-embedding ${className ?? ''}`}
	class:expanded
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
		<div class="title-text">Patch embedding <ZoomInOutline /></div>
	</div>
	<div class="content relative">
		<div class="bounding embedding-bounding" class:active={isHovered && !expanded}></div>
		<div class="patch-inputs">
			<div class="token-column resize-watch">
				<div class="column token-string">
					{#if expanded}<div class="subtitle embedding-detail">
							<button
								class="step-link"
								on:click|stopPropagation={() => open('patchify', $moeTokenIndex)}
								>Patches<small>{grid} × {grid} · 16 × 16</small></button
							>
						</div>{/if}
					{#each $tokens as token, index}<div class="cell">
							<button
								class="patch-label"
								class:chosen={index === $moeTokenIndex}
								aria-pressed={index === $moeTokenIndex}
								aria-label={`Select ${token}`}
								on:click|stopPropagation={() => select(index)}
								>{#if $vitVisiblePositions[index] === 0}<span class="cls">CLS</span>{:else}<span
										class="patch"
										style={patchStyle($vitVisiblePositions[index])}
									></span>{/if}<span class="text">{token}</span></button
							>
						</div>{/each}
				</div>
				{#if expanded}
					<div class="column patch-embedding embedding-detail">
						<div class="subtitle">Patch<br />Embedding<small>{pixels} → {hidden}</small></div>
						{#each $tokens as token, index}<div class="cell">
								<svg viewBox="0 0 31 9" class="arrow" aria-hidden="true"
									><path
										fill="currentColor"
										d="M30.39 4.74a.5.5 0 0 0 0-.7L27.2.85a.5.5 0 1 0-.7.71l2.83 2.83-2.83 2.83a.5.5 0 1 0 .7.7l3.19-3.18ZM.32 4.89h29.71v-1H.32v1Z"
									/></svg
								>{#if $vitVisiblePositions[index] === 0}<button
										class="op"
										title="Learned [CLS] vector, prepended to the patches"
										aria-label="Open how [CLS] is prepended"
										on:click|stopPropagation={() => open('cls', index)}>[CLS]</button
									>{:else}<button
										class="op"
										title={`Flatten ${token} to ${pixels} numbers, then multiply by Wpatch`}
										aria-label={`Open the patch projection for ${token}`}
										on:click|stopPropagation={() => open('patch', index)}>Wpatch</button
									>{/if}
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
							Position<br />Embedding<small>learned · {document?.sequenceLength ?? 197}</small>
						</div>
						{#each $tokens as token, index}<div class="cell">
								<button
									class="position"
									title={`Add the learned position-${$vitVisiblePositions[index]} row`}
									aria-label={`Open the position embedding for ${token}`}
									on:click|stopPropagation={() => open('position', index)}
									><span class="vector bg-gray-300"
										><VectorCanvas active data={positionRows[index]} /></span
									><span class="value">{$vitVisiblePositions[index]}</span></button
								>
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
							title={`Encoder input x₀ for ${token}`}
						>
							<VectorCanvas
								data={$moeExample?.embeddingPreview[index]}
								active={$blockIdx === 0 && (isHovered || expanded)}
							/>
						</div>{/each}
				</div>
			</div>
		</div>
		<VitPatchControls />
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
	}
	.embedding-detail {
		opacity: 0;
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
	}
	.title-text :global(svg) {
		width: 16px;
		height: 16px;
	}
	.content {
		display: flex;
		flex-direction: column;
	}
	.patch-inputs {
		display: flex;
		gap: 12px;
	}
	.token-column {
		display: flex;
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
	.patch-label {
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
	.patch-label:hover .text {
		color: #111827;
	}
	.patch-label.chosen .text {
		color: #7c3aed;
		font-weight: 600;
	}
	.patch-label.chosen .patch,
	.patch-label.chosen .cls {
		border-color: #a77fd0;
	}
	.patch,
	.cls {
		display: inline-block;
		width: 16px;
		height: 16px;
		border: 1px solid #dfd6e9;
		border-radius: 2px;
		flex-shrink: 0;
	}
	.cls {
		font-size: 6.5px;
		background: #f2eaf9;
		color: #8f65ad;
		line-height: 14px;
		text-align: center;
	}
	.patch-embedding .cell,
	.position-embedding .cell {
		gap: 6px;
		justify-content: flex-start;
	}
	.patch-embedding,
	.position-embedding {
		padding: 0 4px;
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
	.op:hover,
	.position:hover .value {
		background: #f1e9f9;
		border-color: #b9a0d4;
	}
	.position {
		display: flex;
		align-items: center;
		gap: 6px;
		cursor: pointer;
	}
	.position .value {
		min-width: 26px;
		padding: 0 3px;
		border: 1px solid #e5e7eb;
		border-radius: 3px;
		font: 10px/1.3 monospace;
		color: #6b7280;
		text-align: center;
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
	button:focus-visible {
		outline: 2px solid #b596cf;
		outline-offset: 2px;
	}
</style>
