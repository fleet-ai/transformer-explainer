<script lang="ts">
	import { tick } from 'svelte';

	export let label: string;
	export let options: { value: number; label: string }[];
	export let value: number;
	export let rows = 1;
	let rail: HTMLDivElement;
	let buttons: HTMLButtonElement[] = [];
	$: selectedIndex = options.findIndex((option) => option.value === value);
	$: columns = Math.ceil(options.length / rows);
	$: revealSelection(value, options, rows);

	async function revealSelection(_value: number, _options: typeof options, rowCount: number) {
		if (rowCount > 1) return;
		await tick();
		const selected = rail?.querySelector<HTMLButtonElement>('[aria-checked="true"]');
		if (!selected) return;
		const left = selected.offsetLeft;
		const right = left + selected.offsetWidth;
		if (left < rail.scrollLeft) rail.scrollLeft = left;
		else if (right > rail.scrollLeft + rail.clientWidth) rail.scrollLeft = right - rail.clientWidth;
	}

	async function choose(index: number, focus = false) {
		if (index < 0 || index >= options.length) return;
		value = options[index].value;
		if (focus) {
			await tick();
			buttons[index]?.focus({ preventScroll: true });
		}
	}

	function onKeyDown(event: KeyboardEvent, index: number) {
		let next: number;
		switch (event.key) {
			case 'ArrowRight':
				next = Math.min(index + 1, options.length - 1);
				break;
			case 'ArrowDown':
				next = Math.min(index + (rows > 1 ? columns : 1), options.length - 1);
				break;
			case 'ArrowLeft':
				next = Math.max(index - 1, 0);
				break;
			case 'ArrowUp':
				next = Math.max(index - (rows > 1 ? columns : 1), 0);
				break;
			case 'Home':
				next = 0;
				break;
			case 'End':
				next = options.length - 1;
				break;
			default:
				return;
		}
		event.preventDefault();
		event.stopPropagation();
		choose(next, true);
	}
</script>

<svelte:window on:resize={() => revealSelection(value, options, rows)} />

<div class="selector" class:multirow={rows > 1} style={`--columns:${columns};`}>
	{#if rows === 1}
		<button
			type="button"
			class="step"
			aria-label={`Previous ${label.toLowerCase()}`}
			disabled={selectedIndex <= 0}
			on:click={() => choose(selectedIndex - 1)}>‹</button
		>
	{/if}
	<div class="segments" role="radiogroup" aria-label={label} bind:this={rail}>
		{#each options as option, index (option.value)}
			<button
				type="button"
				role="radio"
				bind:this={buttons[index]}
				class:chosen={value === option.value}
				aria-checked={value === option.value}
				tabindex={value === option.value ? 0 : -1}
				title={option.label}
				on:click={() => choose(index)}
				on:keydown={(event) => onKeyDown(event, index)}>{option.label}</button
			>
		{/each}
	</div>
	{#if rows === 1}
		<button
			type="button"
			class="step"
			aria-label={`Next ${label.toLowerCase()}`}
			disabled={selectedIndex < 0 || selectedIndex >= options.length - 1}
			on:click={() => choose(selectedIndex + 1)}>›</button
		>
	{/if}
</div>

<style>
	.selector {
		display: flex;
		min-width: 0;
		width: 100%;
	}
	.segments {
		position: relative;
		display: flex;
		min-width: 0;
		overflow-x: auto;
		scrollbar-width: thin;
		scrollbar-color: #cbb3e1 #faf8fd;
	}
	button {
		flex: 0 0 auto;
		min-width: 34px;
		max-width: 160px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		background: #faf8fd;
		border: 1px solid #e1d7ed;
		padding: 6px 10px;
		font: inherit;
		color: #9377ad;
		cursor: pointer;
	}
	.segments button + button {
		border-left: 0;
	}
	.multirow .segments {
		display: grid;
		grid-template-columns: repeat(var(--columns), minmax(0, 1fr));
		width: 100%;
		overflow: hidden;
		border: 1px solid #e1d7ed;
		border-radius: 4px;
	}
	.multirow .segments button {
		min-width: 0;
		max-width: none;
		padding: 6px 0;
		border: 0;
		box-shadow: inset -1px -1px #e1d7ed;
	}
	button.chosen {
		background: #a379ce;
		border-color: #a379ce;
		color: white;
	}
	button:hover:not(:disabled):not(.chosen) {
		background: #f0e7f8;
	}
	.step {
		font-size: 18px;
		line-height: 16px;
		padding: 6px 8px;
	}
	.step:first-child {
		border-radius: 4px 0 0 4px;
		border-right: 0;
	}
	.step:last-child {
		border-radius: 0 4px 4px 0;
		border-left: 0;
	}
	button:disabled {
		color: #cbbdd8;
		cursor: default;
	}
	button:focus-visible {
		outline: 2px solid #76509b;
		outline-offset: -3px;
	}
</style>
