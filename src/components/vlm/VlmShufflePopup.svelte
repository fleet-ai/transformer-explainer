<script lang="ts">
	import VlmFormulaPopup from './VlmFormulaPopup.svelte';
	/** Pixel shuffle: one 4 × 4 block of patch vectors is laid end to end as one long vector. */
	export let visualToken = 0;
	export let gridSize = 16;
	export let factor = 4;
	export let hiddenSize = 768;
	$: blocks = gridSize / factor;
	$: groupSize = factor * factor;
	$: packedSize = groupSize * hiddenSize;
	$: blockRow = Math.floor(visualToken / blocks);
	$: blockColumn = visualToken % blocks;
	$: patches = gridSize * gridSize;
	/** Raster order inside the block, which is the order the recording concatenates them in. */
	$: members = Array.from({ length: groupSize }, (_, i) => ({
		row: blockRow * factor + Math.floor(i / factor),
		column: blockColumn * factor + (i % factor)
	}));
	const inBlock = (patch: number, row: number, column: number) =>
		Math.floor(Math.floor(patch / gridSize) / factor) === row &&
		Math.floor((patch % gridSize) / factor) === column;
</script>

<VlmFormulaPopup
	title={`SmolVLM · Connector · Pixel shuffle · image ${visualToken}`}
	notice={`No weights · a rearrangement: the ${groupSize} patch vectors of one ${factor} × ${factor} block are laid end to end · nothing is averaged · schematic animation`}
	size={packedSize}
	cells={groupSize}
	let:progress
	on:close
>
	<div class="equation">
		<div class="tensor">
			<div>Vision output · {patches} patches</div>
			<div class="patch-grid" style={`--grid:${gridSize}`} aria-hidden="true">
				{#each Array(patches) as _, patch}<i class:member={inBlock(patch, blockRow, blockColumn)}
					></i>{/each}
			</div>
			<strong>({patches}, {hiddenSize})</strong>
		</div>
		<span class="operator">→</span>
		<div class="tensor">
			<div>Block {blockRow},{blockColumn} · {groupSize} patches</div>
			<div class="block-grid" style={`--factor:${factor}`} aria-hidden="true">
				{#each members as member, i}<i
						class:done={i < progress.completed}
						class:active={i === progress.active}
						title={`patch ${member.row},${member.column}`}
					></i>{/each}
			</div>
			<strong>({groupSize}, {hiddenSize})</strong>
		</div>
		<span class="operator">=</span>
		<div class="tensor">
			<div>Packed vector v<sub>{visualToken}</sub></div>
			<div class="cells output" style={`--cells:${groupSize}`} aria-hidden="true">
				{#each members as _, i}<i
						class:done={i < progress.completed}
						class:active={i === progress.active}
					></i>{/each}
			</div>
			<strong>(1, {packedSize.toLocaleString('en-US')})</strong>
		</div>
	</div>
	<div class="formula-lines" aria-hidden="true">
		<div>
			v<sub>{visualToken}</sub> = [ h<sub>{members[0].row},{members[0].column}</sub> ; h<sub
				>{members[1].row},{members[1].column}</sub
			>
			; … ; h<sub>{members[groupSize - 1].row},{members[groupSize - 1].column}</sub> ]
		</div>
		{#if !progress.complete}
			{@const member = members[Math.max(0, progress.active)]}
			<div>
				v<sub>{visualToken}</sub>[{Math.max(0, progress.active) * hiddenSize} … {(Math.max(
					0,
					progress.active
				) +
					1) *
					hiddenSize -
					1}] = h<sub>{member.row},{member.column}</sub>
			</div>
		{:else}
			<div>
				{patches} × {hiddenSize} = {blocks * blocks} × {packedSize.toLocaleString('en-US')} numbers
			</div>
		{/if}
	</div>
	<p class="status" role="status">
		{progress.complete
			? `Complete · packed (1, ${packedSize.toLocaleString('en-US')})`
			: 'Laying the block’s patch vectors end to end'} · schematic sweep
	</p>
	<p class="vlm-note">
		Each of the {blocks * blocks} image tokens packs its own {factor} × {factor} block, so 256 patches
		become
		{blocks * blocks} vectors of {packedSize.toLocaleString('en-US')} numbers. The learned projection
		then maps each one to the language model’s width.
	</p>
</VlmFormulaPopup>

<style>
	.patch-grid,
	.block-grid {
		display: grid;
		gap: 1px;
		margin: 0 auto;
		aspect-ratio: 1;
		height: 104px;
	}
	.patch-grid {
		grid-template-columns: repeat(var(--grid), 1fr);
	}
	.block-grid {
		grid-template-columns: repeat(var(--factor), 1fr);
		gap: 3px;
	}
	i {
		background: #e5e7eb;
	}
	.patch-grid i.member {
		background: #a78bfa;
	}
	.block-grid i {
		background: #cdc0fc;
	}
	.block-grid i.done {
		background: #b9c8fd;
	}
	.block-grid i.active {
		background: #8da6f5;
	}
</style>
