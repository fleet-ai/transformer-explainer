<script lang="ts">
	export let rows: {
		label: string;
		input: string;
		weight: string;
		output: string;
		color?: string;
	}[];
</script>

<div class="projection-operations" aria-label="Projection dimensions">
	<div class="column-labels">
		<span>Weight matrix</span><span></span><span>Input vector</span><span></span><span
			>Output vector</span
		>
	</div>
	{#each rows as row}
		<div
			class="projection-row"
			class:query={row.color === 'query'}
			class:key={row.color === 'key'}
			class:value={row.color === 'value'}
		>
			<div class="cell"><b>{row.label}</b><small>{row.weight}</small></div>
			<span class="operator">×</span>
			<div class="cell"><b>{row.input}</b></div>
			<span class="operator">=</span>
			<div class="cell"><b>{row.output}</b></div>
		</div>
	{/each}
</div>

<style>
	.projection-operations {
		display: grid;
		gap: 6px;
	}
	.column-labels,
	.projection-row {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 24px minmax(0, 1fr) 24px minmax(0, 1fr);
		align-items: stretch;
		gap: 6px;
	}
	.column-labels {
		color: #8a7897;
		font-size: 11px;
		text-align: center;
	}
	.cell {
		min-width: 0;
		padding: 9px 12px;
		border: 1px solid #e5d9ef;
		border-radius: 5px;
		background: #faf7fc;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-wrap: wrap;
		gap: 8px;
		text-align: center;
		color: #805a9b;
	}
	.cell b {
		font-size: 13px;
		font-weight: 500;
	}
	.cell small {
		font-size: 11px;
		white-space: nowrap;
	}
	.operator {
		align-self: center;
		text-align: center;
		color: #a58db8;
		font-size: 20px;
	}
	.query .cell {
		background: #eff5fd;
		border-color: #d8e6f8;
		color: #527eaf;
	}
	.key .cell {
		background: #fdf0f1;
		border-color: #f2dadd;
		color: #ab6274;
	}
	.value .cell {
		background: #eff8f3;
		border-color: #d7ebe0;
		color: #4f8a6d;
	}
	@media (max-width: 600px) {
		.cell {
			padding: 6px;
			gap: 2px;
		}
		.cell b {
			font-size: 11px;
		}
		.column-labels,
		.projection-row {
			grid-template-columns: minmax(0, 1fr) 12px minmax(0, 1fr) 12px minmax(0, 1fr);
			gap: 2px;
		}
	}
</style>
