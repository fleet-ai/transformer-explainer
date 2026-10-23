<script lang="ts">
	import TraceSelector from './TraceSelector.svelte';
	export let label = 'Token';
	export let heading = 'Token';
	export let options: { value: number; label: string }[];
	export let value: number;
	export let pageSize = 0;
	$: selectedIndex = Math.max(
		0,
		options.findIndex((option) => option.value === value)
	);
	$: pageStart = pageSize ? Math.floor(selectedIndex / pageSize) * pageSize : 0;
	$: visibleOptions = pageSize ? options.slice(pageStart, pageStart + pageSize) : options;
</script>

<div class="token-picker">
	<span>{heading}</span>
	<TraceSelector {label} options={visibleOptions} bind:value vertical />
	{#if pageSize && options.length > pageSize}
		<div class="pages">
			<button
				aria-label={`Previous ${label.toLowerCase()} page`}
				disabled={pageStart === 0}
				on:click={() => (value = options[Math.max(0, pageStart - pageSize)].value)}>‹</button
			>
			<span
				>{pageStart + 1}–{Math.min(pageStart + pageSize, options.length)} / {options.length}</span
			>
			<button
				aria-label={`Next ${label.toLowerCase()} page`}
				disabled={pageStart + pageSize >= options.length}
				on:click={() => (value = options[pageStart + pageSize].value)}>›</button
			>
		</div>
	{/if}
</div>

<style>
	.pages {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 4px;
	}
	.pages span {
		padding: 0;
		font-size: 11px;
	}
	.pages button {
		background: #faf8fd;
		border: 1px solid #e1d7ed;
		border-radius: 4px;
		color: #9377ad;
		font: 18px Arial;
		cursor: pointer;
		padding: 2px 8px;
	}
	.pages button:disabled {
		opacity: 0.4;
		cursor: default;
	}

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
