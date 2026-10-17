<script lang="ts">
	import type { PopoverProps } from 'flowbite-svelte/Popover.svelte';
	import CommonPopover from './CommonPopover.svelte';
	import { moeModel } from '~/store';

	export let offset: PopoverProps['offset'] = undefined;
	export let className: PopoverProps['class'] = undefined;
	export let triggeredBy: PopoverProps['triggeredBy'] = undefined;
	export let trigger: PopoverProps['trigger'] = undefined;
	export let placement: PopoverProps['placement'] = undefined;
	export let reference: PopoverProps['reference'];
</script>

<CommonPopover
	className="residual-popover"
	title="Residual Connection"
	{offset}
	{triggeredBy}
	{trigger}
	{placement}
	{reference}
	goTo="article-residual"
>
	<div class="residual-content">
		{#if ($moeModel?.residualMultiplier ?? 0.22) === 1}
			Adds the branch output directly to its input residual stream: output = input + branch.
		{:else}
			Adds the residual stream to the branch output multiplied by {$moeModel?.residualMultiplier ??
				0.22}, the scale used by this checkpoint.
		{/if}
	</div></CommonPopover
>

<style lang="scss">
	.residual-content {
		width: 12rem;
	}
</style>
