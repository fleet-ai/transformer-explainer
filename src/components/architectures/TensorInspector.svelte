<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import {
		numeric,
		tensorColor,
		tensorRows,
		tensorRowForPosition,
		type RecordedOperation
	} from '$lib/architectures/types';
	export let operation: RecordedOperation;
	export let context = '';
	export let selectedToken = 0;
	export let selectedHead = 0;
	export let selectedTokenLabel = '';
	const dispatch = createEventDispatcher<{ close: void }>();
	let panel: HTMLDivElement;
	let tensorIndex = 0;
	let row = 0;
	let column = 0;
	let lastOperation: RecordedOperation;
	let lastHead = -1;
	let lastToken = -1;
	let lastTokenLabel = '';
	$: if (
		operation !== lastOperation ||
		selectedHead !== lastHead ||
		selectedToken !== lastToken ||
		selectedTokenLabel !== lastTokenLabel
	) {
		if (operation !== lastOperation || selectedHead !== lastHead)
			tensorIndex = Math.max(
				0,
				operation.tensors.findIndex(
					(tensor) => tensor.role === 'attention' && tensor.head === selectedHead
				)
			);
		lastOperation = operation;
		lastHead = selectedHead;
		lastToken = selectedToken;
		lastTokenLabel = selectedTokenLabel;
		row = preferredRow(operation.tensors[tensorIndex]);
		column = 0;
	}
	$: tensor = operation.tensors[tensorIndex];
	$: rows = tensorRows(tensor);
	$: row = Math.max(0, Math.min(row, rows.length - 1));
	$: column = Math.max(0, Math.min(column, (rows[row]?.length ?? 1) - 1));
	$: rowStart = Math.floor(row / 12) * 12;
	$: columnStart = Math.floor(column / 12) * 12;
	$: displayedRows = rows.slice(rowStart, rowStart + 12);
	$: maxWidth = Math.max(1, ...rows.map((values) => values.length));
	$: scale = rows.reduce(
		(maximum, values) =>
			values.reduce((result, value) => Math.max(result, Math.abs(value)), maximum),
		0.000001
	);
	$: probability = tensor?.role === 'attention' || tensor?.role === 'probabilities';
	$: value = rows[row]?.[column];
	$: rowLabel = tensor?.rowLabels?.[row] ?? String(row);
	$: columnLabel = tensor?.columnLabels?.[column] ?? String(column);
	$: selectionMissing =
		!!selectedTokenLabel && tensorRowForPosition(tensor, selectedToken, selectedTokenLabel) < 0;
	function preferredRow(item: typeof tensor): number {
		return Math.max(0, tensorRowForPosition(item, selectedToken, selectedTokenLabel));
	}
	function selectTensor(index: number) {
		tensorIndex = index;
		row = preferredRow(operation.tensors[index]);
		column = 0;
	}
	function portal(element: HTMLElement) {
		document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	onMount(() => {
		const previous = document.activeElement;
		panel.focus({ preventScroll: true });
		return () => {
			if (previous instanceof HTMLElement && previous.isConnected)
				previous.focus({ preventScroll: true });
		};
	});
</script>

<svelte:window
	on:keydown={(event) => {
		if (event.key === 'Escape') dispatch('close');
	}}
/>

<div
	use:portal
	bind:this={panel}
	class="tensor-inspector"
	role="dialog"
	aria-modal="false"
	aria-label={operation.label}
	tabindex="-1"
>
	<header>
		<div>
			<h2>{operation.label}</h2>
			<p>{context}</p>
		</div>
		<button class="close" on:click={() => dispatch('close')} aria-label="Close operation inspector"
			>×</button
		>
	</header>
	<div class="body">
		<p class="description">{operation.description}</p>
		<div class="formula" aria-label="Operation formula">{operation.formula}</div>
		{#if operation.tensors.length}
			<div class="tensor-tabs" role="group" aria-label="Recorded tensors">
				{#if operation.tensors.length > 6}
					<label for="inspector-tensor">Recorded tensor / head</label><select
						id="inspector-tensor"
						value={tensorIndex}
						on:change={(event) => selectTensor(Number(event.currentTarget.value))}
						>{#each operation.tensors as item, index}<option value={index}>{item.name}</option
							>{/each}</select
					>
				{:else}{#each operation.tensors as item, index}<button
							class:active={tensorIndex === index}
							aria-pressed={tensorIndex === index}
							on:click={() => selectTensor(index)}>{item.name}</button
						>{/each}{/if}
			</div>
			<div class="tensor-details">
				<div class="tensor-matrix">
					<div class="matrix-heading">
						<strong>{tensor.name}</strong><span>Shape [{tensor.shape.join(' × ')}]</span>
					</div>
					<div class="matrix-caption">
						{tensor.slice ?? 'Complete recorded tensor'} · {selectionMissing
							? 'The selected input position is not a row in this tensor. Choose a recorded row below.'
							: 'Click a cell to inspect its value'}
					</div>
					<div class="matrix-window" style={`--columns:${Math.min(12, maxWidth - columnStart)}`}>
						<div class="corner"></div>
						{#each Array.from({ length: Math.min(12, maxWidth - columnStart) }, (_, offset) => columnStart + offset) as index}
							<div class="axis column" title={tensor.columnLabels?.[index] ?? String(index)}>
								{tensor.columnLabels?.[index] ?? index}
							</div>
						{/each}
						{#each displayedRows as values, rowOffset}
							<div
								class="axis row"
								title={tensor.rowLabels?.[rowStart + rowOffset] ?? String(rowStart + rowOffset)}
							>
								{tensor.rowLabels?.[rowStart + rowOffset] ?? rowStart + rowOffset}
							</div>
							{#each values.slice(columnStart, columnStart + 12) as cell, columnOffset}
								<button
									class="cell"
									class:selected={row === rowStart + rowOffset &&
										column === columnStart + columnOffset}
									class:row-selected={row === rowStart + rowOffset}
									style={`background:${tensorColor(cell, scale, probability)}`}
									title={`[${rowStart + rowOffset}, ${columnStart + columnOffset}] = ${numeric(cell)}`}
									aria-label={`Row ${rowStart + rowOffset}, column ${columnStart + columnOffset}: ${numeric(cell)}`}
									on:click={() => {
										row = rowStart + rowOffset;
										column = columnStart + columnOffset;
									}}
								></button>
							{/each}
						{/each}
					</div>
					<div class="window-controls">
						<div>
							<button
								aria-label="Previous tensor rows"
								disabled={rowStart === 0}
								on:click={() => (row = Math.max(0, rowStart - 12))}>‹</button
							><span>Rows {rowStart}–{Math.min(rows.length - 1, rowStart + 11)}</span><button
								aria-label="Next tensor rows"
								disabled={rowStart + 12 >= rows.length}
								on:click={() => (row = rowStart + 12)}>›</button
							>
						</div>
						<div>
							<button
								aria-label="Previous tensor columns"
								disabled={columnStart === 0}
								on:click={() => (column = Math.max(0, columnStart - 12))}>‹</button
							><span>Columns {columnStart}–{Math.min(maxWidth - 1, columnStart + 11)}</span><button
								aria-label="Next tensor columns"
								disabled={columnStart + 12 >= maxWidth}
								on:click={() => (column = columnStart + 12)}>›</button
							>
						</div>
					</div>
				</div>
				<aside>
					<span class="eyebrow">RECORDED VALUE</span>
					<div class="value">{value === undefined ? '—' : numeric(value)}</div>
					<dl>
						<dt>{probability ? 'Query / output row' : 'Row'}</dt>
						<dd>{rowLabel}</dd>
						<dt>{probability ? 'Key / input column' : 'Column'}</dt>
						<dd>{columnLabel}</dd>
					</dl>
					{#if probability && value !== undefined}<p class="probability">
							{(value * 100).toFixed(3)}%
						</p>{/if}
					{#if operation.metrics?.length}<div class="metrics">
							{#each operation.metrics as metric}<div>
									<span>{metric.label}</span><strong
										>{typeof metric.value === 'number' && !Number.isInteger(metric.value)
											? numeric(metric.value)
											: metric.value}{metric.unit ? ` ${metric.unit}` : ''}</strong
									>
								</div>{/each}
						</div>{/if}
				</aside>
			</div>
		{:else}
			<p class="empty">
				This step is shown for architectural context. This recording does not contain its
				intermediate tensor.
			</p>
		{/if}
		<footer>
			Values come from the recorded forward pass. Shapes describe the original tensors; any saved
			coordinate slice is identified above.
		</footer>
	</div>
</div>

<style>
	.tensor-inspector {
		position: fixed;
		inset: 0;
		margin: auto;
		width: min(1120px, calc(100vw - 40px));
		height: min(690px, calc(100dvh - 40px));
		z-index: 1200;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		border: 1px solid #ded0ee;
		border-radius: 14px;
		background: white;
		box-shadow: 0 18px 55px #35214529;
		color: #6e5685;
		font-family: Arial, sans-serif;
		outline: none;
	}
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex: none;
		padding: 20px 25px;
		background: #faf8fd;
		border-bottom: 1px solid #e8def1;
	}
	h2 {
		font-size: 24px;
		font-weight: 700;
		margin: 0;
		color: #644784;
	}
	header p {
		font-size: 13px;
		margin: 7px 0 0;
		color: #9485a2;
	}
	.close {
		width: 38px;
		height: 38px;
		border: 1px solid #ded1ec;
		border-radius: 8px;
		background: #fff;
		color: #826a98;
		font-size: 28px;
		line-height: 1;
	}
	.close:hover {
		background: #f1e9f8;
	}
	.body {
		padding: 20px 25px;
		display: flex;
		flex-direction: column;
		gap: 14px;
		min-height: 0;
		flex: 1;
		overflow: auto;
	}
	.description {
		font-size: 14px;
		line-height: 1.5;
		margin: 0;
		min-height: 21px;
	}
	.formula {
		padding: 12px 16px;
		border-radius: 8px;
		background: #faf7fd;
		color: #8056a5;
		font:
			17px/1.5 Georgia,
			serif;
		white-space: pre-wrap;
	}
	.tensor-tabs {
		display: flex;
		gap: 3px;
		padding: 4px;
		border: 1px solid #e8dcf3;
		border-radius: 7px;
		background: #fbf9fe;
		flex: none;
		flex-wrap: wrap;
	}
	.tensor-tabs label {
		font-size: 12px;
		align-self: center;
		padding: 0 10px;
		color: #9b83ad;
	}
	.tensor-tabs select {
		flex: 1;
		padding: 8px 12px;
		color: #8054a3;
		border: 1px solid #dfcdec;
		border-radius: 5px;
		background: white;
		font-size: 13px;
	}
	.tensor-tabs button {
		flex: 1;
		padding: 9px 12px;
		white-space: nowrap;
		color: #8d74a4;
		font-size: 13px;
		border-radius: 4px;
		border: 1px solid transparent;
	}
	.tensor-tabs button.active {
		background: #eee2f8;
		border-color: #d7bce9;
		color: #794ba0;
	}
	.tensor-details {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 255px;
		gap: 25px;
		min-height: 250px;
		flex: 1;
	}
	.tensor-matrix {
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 9px;
	}
	.matrix-heading {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		font-size: 13px;
	}
	.matrix-heading span {
		color: #9a88ac;
	}
	.matrix-caption {
		font-size: 11px;
		color: #a18daf;
	}
	.matrix-window {
		display: grid;
		grid-template-columns: minmax(38px, 80px) repeat(var(--columns), minmax(0, 1fr));
		grid-auto-rows: minmax(12px, 1fr);
		gap: 3px;
		min-height: 140px;
		max-height: 235px;
		flex: 1;
	}
	.axis {
		font-size: 10px;
		color: #9c8aa9;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		align-self: center;
	}
	.axis.column {
		text-align: center;
	}
	.axis.row {
		text-align: right;
		padding-right: 6px;
	}
	.cell {
		min-width: 0;
		border-radius: 3px;
		border: 1px solid transparent;
		transition: filter 0.1s;
	}
	.cell:hover {
		filter: brightness(0.88);
	}
	.cell.row-selected {
		border-color: #bda0d399;
	}
	.cell.selected {
		outline: 2px solid #814eb5;
		outline-offset: 1px;
		z-index: 1;
	}
	.window-controls {
		display: flex;
		justify-content: space-between;
		gap: 8px;
		font-size: 11px;
		color: #9b87aa;
	}
	.window-controls div {
		display: flex;
		align-items: center;
		gap: 7px;
	}
	.window-controls button {
		border: 1px solid #ded0eb;
		border-radius: 4px;
		width: 24px;
		height: 23px;
		font-size: 19px;
		line-height: 1;
		color: #8d67ac;
	}
	.window-controls button:disabled {
		opacity: 0.3;
	}
	aside {
		padding: 16px;
		background: #fbf9fd;
		border: 1px solid #eee6f5;
		border-radius: 8px;
		min-width: 0;
	}
	.eyebrow {
		font-size: 10px;
		letter-spacing: 1px;
		color: #a08caf;
	}
	.value {
		font: 600 24px/1.4 monospace;
		color: #8055ab;
		margin: 13px 0 16px;
	}
	dl {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 10px;
		font-size: 12px;
		margin: 0;
	}
	dt {
		color: #a18bab;
	}
	dd {
		text-align: right;
		margin: 0;
		word-break: break-word;
	}
	.probability {
		font-size: 15px;
		color: #9564be;
		margin-top: 12px;
	}
	.metrics {
		border-top: 1px solid #eadff2;
		margin-top: 15px;
		padding-top: 12px;
		font-size: 11px;
		display: grid;
		gap: 9px;
	}
	.metrics div {
		display: flex;
		justify-content: space-between;
		gap: 12px;
	}
	.metrics strong {
		font-family: monospace;
		text-align: right;
	}
	footer {
		font-size: 11px;
		line-height: 1.5;
		color: #a08aaf;
		flex: none;
	}
	.empty {
		padding: 30px;
		background: #fbf9fe;
		border-radius: 8px;
		font-size: 14px;
	}
	button:focus-visible {
		outline: 2px solid #8d5bcc;
		outline-offset: 3px;
	}
	@media (max-height: 730px) {
		.body {
			padding: 14px 20px;
			gap: 10px;
		}
		header {
			padding: 15px 20px;
		}
		.tensor-details {
			min-height: 210px;
		}
		.matrix-window {
			max-height: 190px;
		}
		.formula {
			padding: 8px 12px;
		}
	}
	@media (max-width: 760px) {
		.tensor-inspector {
			width: calc(100vw - 16px);
			height: calc(100dvh - 16px);
		}
		.tensor-details {
			grid-template-columns: 1fr;
		}
		.tensor-matrix {
			min-height: 260px;
		}
		aside {
			display: grid;
			grid-template-columns: 1fr 1fr;
		}
		.value {
			margin: 0;
		}
		.body {
			overflow: auto;
		}
		header {
			padding: 14px;
		}
		h2 {
			font-size: 20px;
		}
	}
</style>
