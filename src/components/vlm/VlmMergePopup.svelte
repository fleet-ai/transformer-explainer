<script lang="ts">
	import VlmFormulaPopup from './VlmFormulaPopup.svelte';
	import { tokenLabel } from '$lib/moe/format';
	/** The language input: a lookup for every text token, and the connector's output in place of
	 *  each image placeholder. */
	export let tokens: string[] = [];
	export let imagePositions: number[] = [];
	export let hiddenSize = 576;
	export let vocabSize = 49280;
	export let selected = -1;
	$: n = tokens.length;
	$: first = imagePositions[0] ?? 0;
	$: last = imagePositions[imagePositions.length - 1] ?? 0;
	$: textCount = n - imagePositions.length;
	$: isImage = (position: number) => imagePositions.includes(position);
</script>

<VlmFormulaPopup
	title="SmolVLM · Language input · text and image tokens"
	notice={`No weights to multiply · text positions look up a row of the embedding table; image positions take the connector output unchanged · schematic animation`}
	size={n}
	cells={n}
	let:progress
	on:close
>
	<div class="equation">
		<div class="tensor">
			<div>Text lookup W<sub>E</sub>[id]</div>
			<div class="cells input" style={`--cells:${n}`} aria-hidden="true">
				{#each tokens as _, position}<i class:empty={isImage(position)}></i>{/each}
			</div>
			<strong>({textCount}, {hiddenSize})</strong>
		</div>
		<span class="operator">+</span>
		<div class="tensor">
			<div>Connector output c</div>
			<div class="cells image" style={`--cells:${n}`} aria-hidden="true">
				{#each tokens as _, position}<i class:empty={!isImage(position)}></i>{/each}
			</div>
			<strong>({imagePositions.length}, {hiddenSize})</strong>
		</div>
		<span class="operator">=</span>
		<div class="tensor">
			<div>Language input e</div>
			<div class="cells output" style={`--cells:${n}`} aria-hidden="true">
				{#each tokens as _, position}<i
						class:done={position < progress.completed}
						class:active={position === progress.active}
						class:from-image={position < progress.completed && isImage(position)}
						class:chosen={position === selected}
					></i>{/each}
			</div>
			<strong>({n}, {hiddenSize})</strong>
		</div>
	</div>
	<ol class="sequence" aria-label="Language sequence">
		{#each tokens as token, position}<li
				class:image={isImage(position)}
				class:active={position === progress.active}
				class:chosen={position === selected}
				title={`${position}: ${token}`}
			>
				<span>{position}</span>{tokenLabel(token)}
			</li>{/each}
	</ol>
	<div class="formula-lines" aria-hidden="true">
		{#if progress.active >= 0 && isImage(progress.active)}
			<div>
				e<sub>{progress.active}</sub> = c<sub>{progress.active - first}</sub> · image token,
				position
				{first}…{last}
			</div>
		{:else if progress.active >= 0}
			<div>
				e<sub>{progress.active}</sub> = W<sub>E</sub>[id<sub>{progress.active}</sub>] · one of {vocabSize.toLocaleString(
					'en-US'
				)} rows
			</div>
		{:else}
			<div>
				e<sub>t</sub> = W<sub>E</sub>[id<sub>t</sub>] for text, e<sub>{first}+k</sub> = c<sub>k</sub
				>
				for k = 0…{imagePositions.length - 1}
			</div>
		{/if}
	</div>
	<p class="status" role="status">
		{progress.complete ? `Complete · input (${n}, ${hiddenSize})` : 'Filling the sequence'} · schematic
		sweep
	</p>
	<p class="vlm-note">
		The {imagePositions.length} image tokens replace placeholder tokens between the image markers, so
		every language block sees them as ordinary positions under the same causal mask.
	</p>
</VlmFormulaPopup>

<style>
	.cells i.empty {
		background: #f3f4f6;
	}
	.cells.image i {
		background: #e9d5ff;
	}
	.cells.image i.empty {
		background: #f3f4f6;
	}
	.cells.output i.from-image {
		background: #c4a6ee;
	}
	.cells.output i.chosen {
		box-shadow: inset 0 0 0 1.5px #7c5aa6;
	}
	.sequence {
		display: flex;
		flex-wrap: wrap;
		gap: 3px;
		list-style: none;
		padding: 0;
		margin: 18px 0 0;
		justify-content: center;
	}
	.sequence li {
		font-size: 11px;
		line-height: 1.2;
		color: #6b7280;
		background: #f3f4f6;
		border-radius: 3px;
		padding: 2px 5px;
		white-space: nowrap;
		max-width: 110px;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.sequence li span {
		color: #9ca3af;
		margin-right: 3px;
		font-size: 9px;
	}
	.sequence li.image {
		background: #f3e8ff;
		color: #7e57a8;
	}
	.sequence li.active {
		outline: 1.5px solid #8da6f5;
	}
	.sequence li.chosen {
		font-weight: 600;
	}
</style>
