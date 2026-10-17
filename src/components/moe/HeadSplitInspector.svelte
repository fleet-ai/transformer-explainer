<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { attentionHeadIdx, blockIdx, moeModel } from '~/store';
	import PlainHeadFlow from '../vit/PlainHeadFlow.svelte';
	export let plain = false;
	import HeadFlow from './HeadFlow.svelte';
	const dispatch = createEventDispatcher<{ close: void; rope: void; projections: void }>();
	let panel: HTMLDivElement;
	$: queryHeads = $moeModel?.attentionHeads ?? 16;
	$: keyValueHeads = $moeModel?.keyValueHeads ?? 8;
	$: headSize = $moeModel?.attentionHeadSize ?? 64;
	function portal(element: HTMLElement) {
		document.body.appendChild(element);
		return { destroy: () => element.remove() };
	}
	onMount(() => {
		const previous = document.activeElement;
		panel.focus({ preventScroll: true });
		return () => {
			if (previous instanceof HTMLElement && previous.isConnected)
				previous.focus({ preventScroll: true });
		};
	});
</script>

<svelte:window
	on:keydown={(event) => {
		if (event.key === 'Escape') dispatch('close');
	}}
/>
<div
	use:portal
	bind:this={panel}
	class="head-split-inspector"
	role="dialog"
	aria-modal="false"
	aria-label="Head split"
	tabindex="-1"
>
	<header>
		<div>
			<h2>Head split</h2>
			<p>Block {$blockIdx + 1} · Query head {$attentionHeadIdx + 1}</p>
		</div>
		<button class="close" aria-label="Close head split" on:click={() => dispatch('close')}>×</button
		>
	</header>
	<div class="body">
		{#if plain}<PlainHeadFlow on:inspect={() => dispatch('projections')} />{:else}<HeadFlow
				{queryHeads}
				{keyValueHeads}
				{headSize}
				on:inspect={() => dispatch('rope')}
			/>{/if}
		<p>
			Each token’s Q projection splits into {queryHeads} heads of {headSize} coordinates. K and V each
			split into {keyValueHeads} heads. Splitting reshapes the projected values; it does not change them.
		</p>
		{#if !plain}<p>
				RoPE then rotates Q and K into <strong>Q′ and K′</strong>. These rotated values enter the
				attention dot product. <strong>V stays unchanged.</strong>
			</p>{/if}
	</div>
</div>

<style>
	.head-split-inspector {
		position: fixed;
		right: 16px;
		bottom: 16px;
		width: min(820px, calc(100vw - 32px));
		max-height: calc(100dvh - 32px);
		z-index: 1200;
		border: 1px solid #ded5eb;
		border-radius: 8px;
		background: white;
		color: #665775;
		box-shadow: 0 8px 34px #33224b30;
		font:
			14px/1.5 Arial,
			sans-serif;
		outline: none;
		overflow: auto;
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 16px 22px;
		background: #faf8fc;
		border-bottom: 1px solid #e9e2ef;
	}
	h2 {
		margin: 0;
		color: #624685;
		font-size: 22px;
		font-weight: 600;
	}
	header p {
		margin: 3px 0 0;
		color: #91849e;
		font-size: 12px;
	}
	.close {
		width: 34px;
		height: 34px;
		border: 1px solid #ddd5e5;
		border-radius: 5px;
		background: white;
		color: #79698a;
		font-size: 24px;
		cursor: pointer;
	}
	.close:focus-visible {
		outline: 2px solid #b28bcb;
		outline-offset: 2px;
	}
	.body {
		padding: 20px 24px;
	}
	.body p {
		margin: 18px 0 0;
	}
	.body :global(.head-flow) {
		min-width: 0;
	}
	@media (max-width: 600px) {
		.body {
			padding: 14px;
		}
	}
</style>
