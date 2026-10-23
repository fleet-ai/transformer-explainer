<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import OperationFlow from './OperationFlow.svelte';
	export let items: { id: string; label: string; disabled?: boolean }[] = [];
	export let selected = '';
	let open = false;
	const dispatch = createEventDispatcher<{ select: string }>();
</script>

<details class="block-chooser" bind:open>
	<summary>Browse other blocks</summary><OperationFlow
		{items}
		{selected}
		on:select={(e) => {
			dispatch('select', e.detail);
			open = false;
		}}
	/>
</details>

<style>
	.block-chooser {
		margin: 2px 0 8px;
		color: #8961a6;
		font:
			13px/1.4 Arial,
			sans-serif;
	}
	summary {
		cursor: pointer;
		padding: 5px 0;
	}
	summary:focus-visible {
		outline: 2px solid #976bc4;
	}
	.block-chooser[open] {
		padding-bottom: 6px;
		border-bottom: 1px solid #eee4f5;
	}
</style>
