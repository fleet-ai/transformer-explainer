<script lang="ts">
	import MatrixProduct from './MatrixProduct.svelte';
	export let rows: {
		label: string;
		input: string;
		weight: string;
		output: string;
		color?: string;
	}[];
</script>

{#each rows as row}
	{@const dims = row.weight.replaceAll(',', '').match(/\d+/g)?.map(Number) ?? []}
	<MatrixProduct
		input={row.input}
		weight={row.label + 'ᵀ'}
		output={row.output}
		inputSize={dims[1] ?? '?'}
		outputSize={dims[0] ?? '?'}
	/>
{/each}
