<script lang="ts">
	export let input = 'X';
	export let weight = 'Wᵀ';
	export let output = 'Y';
	export let inputSize: number | string;
	export let outputSize: number | string;
	export let rows: number | string = 1;
	export let token = '';
	export let elementwise = false;
	export let activation = '';
	export let bias = false;
	export let inputValues: number[] = [];
	export let outputValues: number[] = [];
	const preview = (name: string, values: number[]) =>
		values.length
			? `${name} · recorded: ${values
					.slice(0, 4)
					.map((v) => v.toFixed(6))
					.join(', ')}`
			: `${name} · schematic dimensions`;
</script>

<div
	class="matrix-product"
	aria-label={`${input} ${activation ? 'through ' + activation : elementwise ? 'elementwise times ' + weight : 'times ' + weight} ${bias ? 'plus bias ' : ''}equals ${output}`}
	aria-live="polite"
>
	<div class="term">
		<h4>{input}</h4>
		<div class="tensor vectors" title={preview(input, inputValues)}>
			<span>{token}</span><i></i>{#if rows !== 1}<span>⋮</span><i></i>{/if}
		</div>
		<strong>{rows} × {inputSize}</strong>
	</div>
	<span class="symbol">{activation ? '→' : elementwise ? '⊙' : '×'}</span>
	<div class="term">
		<h4>{activation || weight}</h4>
		<div
			class="tensor"
			class:matrix={!elementwise && !activation}
			class:activation={!!activation}
			class:vectors={elementwise}
			aria-label="Schematic tensor shape"
		>
			{#if activation}{activation}{:else if elementwise}<i></i>{#if rows !== 1}<i></i>{/if}{/if}
		</div>
		<strong
			>{activation
				? 'elementwise'
				: elementwise
					? `${rows} × ${inputSize}`
					: `${inputSize} × ${outputSize}`}</strong
		>
	</div>
	<span class="symbol"
		>{#if bias}<small>+ b</small>{/if}{activation ? '→' : '='}</span
	>
	<div class="term">
		<h4>{output}</h4>
		<div class="tensor vectors result" title={preview(output, outputValues)}>
			<span>{token}</span><i></i>{#if rows !== 1}<span>⋮</span><i></i>{/if}
		</div>
		<strong>{rows} × {outputSize}</strong>
	</div>
</div>
<div class="shape-note">
	{#if bias}+ bias ({outputSize}), broadcast over token rows ·
	{/if}Schematic cells · full tensor dimensions{#if inputValues.length || outputValues.length}
		· hover vectors for recorded values{/if}
</div>

<style>
	.matrix-product {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 22px minmax(0, 1fr) 22px minmax(0, 1fr);
		gap: 9px;
		align-items: center;
		margin: 12px 0;
		color: #715883;
		font:
			13px/1.4 Arial,
			sans-serif;
	}
	.term {
		min-width: 0;
		text-align: center;
	}
	h4 {
		font:
			16px Georgia,
			serif;
		margin: 8px 0 12px;
		overflow-wrap: anywhere;
	}
	.tensor {
		height: 84px;
	}
	.vectors {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		align-content: center;
		align-items: center;
		gap: 8px;
	}
	.vectors span {
		font:
			11px Arial,
			sans-serif;
		max-width: 65px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.vectors i {
		display: block;
		height: 24px;
		background: repeating-linear-gradient(90deg, transparent 0 5px, #fff 5px 6px), #bed9ff;
	}
	.result i {
		background-color: #d9c4ed;
	}
	.matrix {
		background:
			repeating-linear-gradient(90deg, transparent 0 8px, #fff 8px 9px),
			repeating-linear-gradient(0deg, transparent 0 8px, #fff 8px 9px), #d9c4ed;
	}
	.term:nth-child(3) .vectors {
		grid-template-columns: 1fr;
	}
	.term:nth-child(3) i {
		background-color: #d9c4ed;
	}
	.activation {
		display: flex;
		align-items: center;
		justify-content: center;
		border: 1px solid #e2d5ed;
		background: #faf7fd;
		border-radius: 5px;
		font-size: 18px;
	}
	strong {
		display: block;
		font:
			500 16px/1.4 Arial,
			sans-serif;
		margin-top: 10px;
		font-variant-numeric: tabular-nums;
	}
	.symbol {
		font-size: 23px;
		text-align: center;
	}
	.symbol small {
		display: block;
		font-size: 12px;
		white-space: nowrap;
	}
	.shape-note {
		font:
			11px/1.5 Arial,
			sans-serif;
		color: #90799f;
		margin: 6px 0 12px;
	}
	@media (max-width: 600px) {
		.matrix-product {
			gap: 3px;
			grid-template-columns: minmax(0, 1fr) 12px minmax(0, 1fr) 12px minmax(0, 1fr);
		}
		h4 {
			font-size: 13px;
		}
		strong {
			font-size: 12px;
		}
		.vectors {
			grid-template-columns: 1fr;
		}
		.tensor {
			height: 65px;
		}
		.symbol {
			font-size: 18px;
		}
	}
</style>
