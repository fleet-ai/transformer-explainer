<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	export let items: { id: string; label: string; disabled?: boolean }[] = [];
	export let selected = '';
	export let label = 'Operations';
	const dispatch = createEventDispatcher<{ select: string }>();
</script>

<div class="operation-map" role="group" aria-label={label}>
	{#each items as item}<div class="operation-branch" class:active={selected === item.id}>
			<button
				class="band"
				disabled={item.disabled}
				aria-label={`Inspect ${item.label} band`}
				on:click={() => dispatch('select', item.id)}><i></i></button
			>
			<button
				class="box"
				disabled={item.disabled}
				aria-pressed={selected === item.id}
				on:click={() => dispatch('select', item.id)}>{item.label}</button
			>
		</div>{/each}
</div>

<style>
	.operation-map {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
		gap: 8px 14px;
		margin: 4px 0 8px;
	}
	.operation-branch {
		display: flex;
		align-items: center;
		min-width: 0;
	}
	.band {
		border: 0;
		background: transparent;
		width: 30px;
		flex: 0 0 30px;
		padding: 7px 0;
		cursor: pointer;
	}
	.band i {
		display: block;
		background: #ded4ee;
		height: 20px;
		border-radius: 50% 0 0 50%;
	}
	.box {
		font:
			14px/1.4 Arial,
			sans-serif;
		border: 1px solid #ded2eb;
		background: #faf8fc;
		border-radius: 4px;
		color: #694684;
		padding: 8px;
		flex: 1;
		min-height: 44px;
		cursor: pointer;
	}
	.active .band i {
		background: #d7bee9;
	}
	.active .box {
		background: #eddef7;
		border-color: #b591ce;
	}
	.box:hover {
		border-color: #aa7cc6;
	}
	button:disabled {
		opacity: 0.4;
		cursor: default;
	}
	button:focus-visible {
		outline: 2px solid #976bc4;
		outline-offset: 2px;
	}
</style>
