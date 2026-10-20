<script lang="ts">
	import type { ExpertOperation } from '$lib/moe/operations';
	import FeedForwardFlow from '../common/FeedForwardFlow.svelte';
	import ExpertProjectionPopup from './ExpertProjectionPopup.svelte';
	export let operation: ExpertOperation | null = null;
	export let inputPreview: number[] = [];
	export let hiddenSize = 1024;
	export let intermediateSize = 512;
	export let expert: number;
	export let label: string | undefined = undefined;
	export let token = 'x';
	let step: 'up' | 'gate' | 'down' | null = null;
	function select(value: string) {
		if (value === 'up' || value === 'gate' || value === 'down') step = value;
	}
</script>

<div class="expert-operations">
	<div class="heading">
		<strong>{label ?? `Expert ${expert}`}</strong>
		<span>SwiGLU · {hiddenSize} → {intermediateSize} → {hiddenSize}</span>
	</div>
	<FeedForwardFlow
		{hiddenSize}
		{intermediateSize}
		{token}
		selected={step ?? ''}
		projectionsOnly
		denseStyle
		on:select={(event) => select(event.detail)}
	/>
	{#if !operation}<p role="status">Recorded operations are not available.</p>{/if}
</div>
{#if step && operation}
	{#key step}
		<ExpertProjectionPopup
			{operation}
			{step}
			{inputPreview}
			{hiddenSize}
			{intermediateSize}
			{token}
			label={label ?? `Expert ${expert}`}
			on:close={() => (step = null)}
		/>
	{/key}
{/if}

<style>
	.expert-operations {
		min-width: 0;
		font:
			13px/1.5 ui-sans-serif,
			system-ui,
			sans-serif;
		color: #6b7280;
	}
	.heading {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 12px;
		flex-wrap: wrap;
	}
	.heading strong {
		color: #4b5563;
		font-weight: 500;
	}
	.heading span {
		color: #9ca3af;
	}
</style>
