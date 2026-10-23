<script lang="ts">
	import Matrix from '../common/Matrix.svelte';
	import { attentionHeadIdx, blockIdx, expandedBlock, hoveredMatrixCell, modelData } from '~/store';
	import { t5Example, t5Stack, t5Inspector } from '$lib/seq2seq/store';
	import { scaleLinear } from 'd3';
	import { maskArray } from '~/utils/array';
	export let cross = false;
	let width = 0;
	$: id = cross ? 'cross-attention' : 'attention';
	$: expanded = $expandedBlock.id === id;
	$: layer = $t5Example?.crossAttention[$blockIdx];
	$: scores = cross
		? (layer?.scores?.[$attentionHeadIdx] ?? [])
		: ($modelData?.outputs?.[`block_${$blockIdx}_attn_head_${$attentionHeadIdx}_attn`]?.data ?? []);
	$: masked = cross
		? (layer?.maskedScores?.[$attentionHeadIdx] ?? [])
		: ($modelData?.outputs?.[`block_${$blockIdx}_attn_head_${$attentionHeadIdx}_attn_masked`]
				?.data ?? []);
	$: probabilities = cross
		? (layer?.probabilities?.[$attentionHeadIdx] ?? [])
		: ($modelData?.outputs?.[`block_${$blockIdx}_attn_head_${$attentionHeadIdx}_attn_dropout`]
				?.data ?? []);
	$: displayedProbabilities =
		!cross && $t5Stack === 'decoder' ? maskArray(probabilities) : probabilities;
	$: columns = probabilities[0]?.length ?? 1;
	$: stages = expanded
		? cross
			? [
					{ label: 'Dot product', data: scores, kind: 'scores' },
					{ label: 'Softmax', data: displayedProbabilities, kind: 'probabilities' }
				]
			: [
					{ label: 'Dot product', data: scores, kind: 'scores' },
					{
						label: $t5Stack === 'encoder' ? 'Relative bias' : 'Bias + causal mask',
						data: masked,
						kind: 'scores'
					},
					{ label: 'Softmax', data: displayedProbabilities, kind: 'probabilities' }
				]
		: [
				{
					label: cross ? 'Cross-attention' : 'Attention',
					data: displayedProbabilities,
					kind: 'probabilities'
				}
			];
	$: cellSize = Math.max(
		3,
		Math.min(26, ((width - (stages.length - 1) * 20) / stages.length - (columns - 1) * 2) / columns)
	);
	function color(data: number[][], probabilities: boolean) {
		const values = data.flat().filter(Number.isFinite);
		return scaleLinear<string>()
			.domain(probabilities ? [0, 1] : [Math.min(...values), Math.max(...values)])
			.range(['#faf8ff', '#7425d5']);
	}
	function toggle() {
		expandedBlock.set({ id: expanded ? null : id });
	}
</script>

<div class="t5-matrices resize-watch" class:expanded bind:clientWidth={width}>
	<div class="stages" style={`--stages:${stages.length}`}>
		{#each stages as stage, index (`${stage.label}-${$blockIdx}-${$attentionHeadIdx}-${$t5Stack}`)}
			<div
				class="stage attention-matrix"
				class:attention-initial={!cross && index === 0}
				class:attention-out={!cross && index === stages.length - 1}
				class:cross-initial={cross && index === 0}
				class:cross-result={cross && index === stages.length - 1}
			>
				{#if index > 0}<span class="stage-arrow" aria-hidden="true">→</span>{/if}
				<button
					class="matrix-button"
					on:click|stopPropagation={toggle}
					aria-label={`${expanded ? 'Collapse' : 'Expand'} ${cross ? 'cross-attention' : 'self-attention'}`}
					aria-expanded={expanded}
				>
					<Matrix
						className="main"
						data={stage.data}
						shape="circle"
						colorScale={color(stage.data, stage.kind === 'probabilities')}
						cellWidth={cellSize}
						cellHeight={cellSize}
						rowGap={2}
						colGap={2}
						showSize={false}
						onMouseOverCell={(_, d) => hoveredMatrixCell.set({ row: d.rowIndex, col: d.colIndex })}
						onMouseOutCell={() => hoveredMatrixCell.set({ row: null, col: null })}
						onMouseOutSvg={() => hoveredMatrixCell.set({ row: null, col: null })}
						showTooltip={(_, d) => (Number.isFinite(d) ? Number(d).toFixed(5) : 'Masked')}
						highlightRow={undefined}
						highlightCol={undefined}
					/>
				</button>
				<span class="stage-label">{stage.label}{!expanded ? ' ⊕' : ''}</span>
			</div>
		{/each}
	</div>
	{#if expanded}<div class="stage-actions">
			<button on:click|stopPropagation={() => t5Inspector.set(cross ? 'cross' : 'self')}
				>Recorded values ↗</button
			><button on:click|stopPropagation={toggle}>Collapse −</button>
		</div>{/if}
</div>

<style>
	.t5-matrices {
		width: 100%;
		min-width: 0;
		position: relative;
		z-index: 35;
	}
	.stages {
		display: grid;
		grid-template-columns: repeat(var(--stages), minmax(0, 1fr));
		gap: 20px;
		align-items: center;
	}
	.stage {
		position: relative;
		min-width: 0;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.matrix-button {
		border: 0;
		padding: 0;
		background: transparent;
		cursor: pointer;
		display: block;
		max-width: 100%;
	}
	.matrix-button:focus-visible {
		outline: 2px solid #b49bdd;
		outline-offset: 5px;
		border-radius: 4px;
	}
	.stage-label {
		position: absolute;
		top: calc(100% + 10px);
		width: 100%;
		text-align: center;
		font-size: clamp(10px, 1vw, 15px);
		line-height: 1.3;
		color: #8b93a1;
	}
	.stage-arrow {
		position: absolute;
		left: -17px;
		top: 50%;
		color: #bbc0ca;
		font-size: 18px;
	}
	.stage-actions {
		position: absolute;
		top: calc(100% + 54px);
		display: flex;
		justify-content: center;
		gap: 16px;
		width: 100%;
		font-size: 11px;
		color: #9272b2;
	}
	.stage-actions button {
		padding: 4px;
		background: #fff;
		border: 1px solid #e6dcf1;
		border-radius: 4px;
	}
</style>
