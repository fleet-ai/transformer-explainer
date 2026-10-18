<script lang="ts">
	import type { ExpertOperation } from '$lib/moe/operations';
	import FeedForwardFlow from '../common/FeedForwardFlow.svelte';
	import MatrixProduct from '../common/MatrixProduct.svelte';
	export let operation: ExpertOperation | null = null;
	export let inputPreview: number[] = [];
	export let hiddenSize = 1024;
	export let intermediateSize = 512;
	export let expert: number;
	export let label: string | undefined = undefined;
	export let token = 'x';
	let step = 'up';
	$: projection = operation?.[step === 'down' ? 'down' : step === 'gate' ? 'gate' : 'up'];
	$: input = step === 'down' ? (operation?.product ?? []) : inputPreview;
	$: inDimensions = step === 'down' ? intermediateSize : hiddenSize;
	$: outDimensions = step === 'down' ? hiddenSize : intermediateSize;
	$: matrixName = step === 'down' ? 'Wdownᵀ' : step === 'gate' ? 'Wgateᵀ' : 'Wupᵀ';
	$: title = {
		up: 'Up projection',
		gate: 'Gate projection',
		activation: 'SiLU activation',
		product: 'SiLU × up',
		down: 'Down projection'
	}[step];
	const f = (v: number | undefined) => (v === undefined ? '—' : v.toFixed(6));
</script>

<div class="expert-operations">
	<div class="heading">
		<strong>{label ?? `Expert ${expert}`}</strong><span
			>SwiGLU · {hiddenSize} → {intermediateSize} → {hiddenSize}</span
		>
	</div>
	<FeedForwardFlow
		{hiddenSize}
		{intermediateSize}
		{token}
		selected={step}
		on:select={(e) => (step = e.detail)}
	/>
	<div class="operation-detail" aria-label="Expert matrix product" aria-live="polite">
		<h3>{title}</h3>
		{#if !operation}<p role="status">Recorded operations are not available.</p>
		{:else if step === 'activation'}
			<MatrixProduct
				input="Gate G"
				output="SiLU(G)"
				activation="SiLU"
				inputSize={intermediateSize}
				outputSize={intermediateSize}
				{token}
				inputValues={operation.gate.output}
				outputValues={operation.siluGate}
			/>
		{:else if step === 'product'}
			<MatrixProduct
				input="SiLU(G)"
				weight="Up U"
				output="Hidden H"
				inputSize={intermediateSize}
				outputSize={intermediateSize}
				elementwise
				{token}
				inputValues={operation.siluGate}
				outputValues={operation.product}
			/>
		{:else if projection}
			<MatrixProduct
				input={step === 'down' ? 'Hidden H' : 'Input X'}
				weight={matrixName}
				output={step === 'down' ? 'Output Y' : step === 'gate' ? 'Gate G' : 'Up U'}
				inputSize={inDimensions}
				outputSize={outDimensions}
				{token}
				inputValues={input}
				outputValues={projection.output}
			/>
		{/if}
		{#if operation}<details>
				<summary>Recorded values and products</summary>
				{#if step === 'activation' || step === 'product'}
					<table>
						<thead
							><tr><th>Coordinate</th><th>Gate</th><th>SiLU(g)</th><th>Up</th><th>Product</th></tr
							></thead
						><tbody
							>{#each operation.product.slice(0, 4) as value, i}<tr
									><td>{i}</td><td>{f(operation.gate.output[i])}</td><td
										>{f(operation.siluGate[i])}</td
									><td>{f(operation.up.output[i])}</td><td>{f(value)}</td></tr
								>{/each}</tbody
						>
					</table>
				{:else if projection}
					{#each projection.output.slice(0, 4) as value, i}<p class="recorded">
							[{i}] = {#each projection.weights[i].slice(0, 4) as weight, j}({f(input[j])} × {f(
									weight
								)}) +
							{/each}{f(projection.omittedContribution[i])}
							<small>remaining {inDimensions - 4} inputs</small>
							≈ <b>{f(value)}</b>
						</p>{/each}
				{/if}
				<p>
					Recorded forward-pass values; displayed numbers are rounded. Matrix cells above illustrate
					dimensions, not individual weights.
				</p>
			</details>{/if}
	</div>
</div>

<style>
	.expert-operations {
		display: flex;
		flex-direction: column;
		flex: 1;
		min-height: 0;
		min-width: 0;
		font:
			12px/1.5 Arial,
			sans-serif;
		color: #735b85;
	}
	.heading {
		display: flex;
		justify-content: space-between;
		gap: 12px;
	}
	.heading span {
		color: #927b9f;
		font-size: 13px;
	}
	.operation-detail {
		min-height: 0;
		overflow: auto;
		padding: 0 2px 2px;
	}
	h3 {
		font:
			19px Georgia,
			serif;
		margin: 0 0 10px;
	}
	details {
		border-top: 1px solid #ede5f4;
		margin-top: 8px;
		padding-top: 10px;
		font-size: 13px;
	}
	summary {
		cursor: pointer;
		color: #895eb0;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		margin-top: 8px;
	}
	td,
	th {
		text-align: right;
		padding: 5px;
		border-bottom: 1px solid #eee7f4;
	}
	th {
		font-weight: 500;
	}
	.recorded {
		font-family: monospace;
		line-height: 1.8;
		overflow-wrap: anywhere;
	}
	small {
		color: #9984a7;
	}
</style>
