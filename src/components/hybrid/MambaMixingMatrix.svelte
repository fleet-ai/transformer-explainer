<script lang="ts">
	import Matrix from '../common/Matrix.svelte';
	import { attentionHeadIdx, blockIdx, expandedBlock, hoveredMatrixCell } from '~/store';
	import { hybridMambaOperations, hybridInspector } from '$lib/hybrid/store';
	import { scaleLinear } from 'd3';

	/** The scan of one SSM head as a causal token-mixing matrix, drawn like the attention circles. */
	let width = 0;
	$: expanded = $expandedBlock.id === 'attention';
	$: matrix = $hybridMambaOperations?.matrix;
	$: stages = matrix
		? expanded
			? [
					{ label: 'C · Bᵀ', data: matrix.scores, kind: 'signed', masked: false },
					{ label: 'Decay L(Δ, A) · causal', data: matrix.decay, kind: 'decay', masked: true },
					{ label: 'Mixing M = L ∘ C·Bᵀ', data: matrix.mixing, kind: 'signed', masked: true }
				]
			: [{ label: 'State-space mixing', data: matrix.mixing, kind: 'signed', masked: true }]
		: [];
	$: columns = matrix?.mixing.length ?? 1;
	$: cellSize = Math.max(
		3,
		Math.min(18, ((width - (stages.length - 1) * 20) / stages.length - (columns - 1) * 2) / columns)
	);
	function color(data: number[][], kind: string) {
		const values = data.flat().filter(Number.isFinite);
		const max = Math.max(...values.map(Math.abs), 1e-6);
		// Negative mixing weights exist (C·B can be negative); they are drawn orange.
		return kind === 'decay'
			? scaleLinear<string>().domain([0, max]).range(['#faf8ff', '#7425d5'])
			: scaleLinear<string>().domain([-max, 0, max]).range(['#ea8a3c', '#faf8ff', '#7425d5']);
	}
	function toggle() {
		expandedBlock.set({ id: expanded ? null : 'attention' });
	}
</script>

<div class="mamba-matrices resize-watch" class:expanded bind:clientWidth={width}>
	<button
		class="delta-box"
		title="Step size Δ and the decay e^(ΔA): one per head and token"
		on:click|stopPropagation={() => hybridInspector.set('delta')}
		>Δ · A<small>decay mask</small></button
	>
	<div class="stages" style={`--stages:${stages.length}`}>
		{#each stages as stage, index (`${stage.label}-${$blockIdx}-${$attentionHeadIdx}`)}
			<div
				class="stage attention-matrix"
				class:attention-initial={index === 0}
				class:attention-out={index === stages.length - 1}
			>
				{#if index > 0}<span class="stage-arrow" aria-hidden="true">→</span>{/if}
				<button
					class="matrix-button"
					on:click|stopPropagation={toggle}
					aria-label={`${expanded ? 'Collapse' : 'Expand'} the state-space mixing matrix`}
					aria-expanded={expanded}
				>
					<Matrix
						className="main"
						data={stage.data}
						shape="circle"
						colorScale={color(stage.data, stage.kind)}
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
			<button on:click|stopPropagation={() => hybridInspector.set('scan')}>Formula ↗</button
			><button on:click|stopPropagation={toggle}>Collapse −</button>
		</div>{/if}
</div>

<style>
	.mamba-matrices {
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
		white-space: nowrap;
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
		color: inherit;
		cursor: pointer;
	}
	.delta-box {
		border-radius: 3px;
		cursor: pointer;
		position: absolute;
		right: 0;
		bottom: calc(100% + 14px);
		display: grid;
		justify-items: center;
		padding: 2px 8px;
		border: 1.5px solid #e5c07a;
		background: #fffaf0;
		color: #9a6b13;
		font-size: 11px;
		font-weight: 600;
		line-height: 1.2;
		z-index: 36;
	}
	.delta-box small {
		font-size: 8px;
		font-weight: 400;
		color: #b08a3c;
	}
	.stage-actions button:hover {
		background: #eee0f7;
	}
	.delta-box:hover {
		background: #fdf0d8;
	}
</style>
