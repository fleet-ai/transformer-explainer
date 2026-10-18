<script lang="ts">
	import TraceSelector from './TraceSelector.svelte';
	export let label = 'Token';
	export let heading = 'Token';
	export let options: { value: number; label: string }[];
	export let value: number;
</script>

<div class="token-picker">
	<span>{heading}</span>
	<TraceSelector {label} {options} bind:value vertical />
</div>

<style>
	.token-picker {
		display: flex;
		flex-direction: column;
		flex: 1 1 auto;
		gap: 8px;
		min-height: 0;
		font:
			12px/1.5 Arial,
			sans-serif;
	}
	span {
		color: #91849e;
		padding: 5px 10px 0;
	}
	.token-picker :global(.vertical .segments button) {
		white-space: normal;
		overflow-wrap: anywhere;
	}
</style>
