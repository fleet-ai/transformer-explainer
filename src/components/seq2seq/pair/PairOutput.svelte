<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	/** The Dense explainer's probability list for the next decoder token; it follows temperature and sampling. */
	export let id: string;
	export let candidates: { token: string; probability: number }[] = [];
	export let height = 300;
	const dispatch = createEventDispatcher<{ open: void }>();
	$: pitch = Math.min(36, height / Math.max(candidates.length, 1));
	$: top = candidates[0]?.probability || 1;
	$: font = Math.max(10, Math.min(16, pitch * 0.45));
</script>

<div class="probs" style={`height:${height}px`}>
	<div data-b={`${id}:list`} class="cand-list">
		{#each candidates as item, index}<button
				class="cand"
				class:first={index === 0}
				style={`height:${pitch}px;font-size:${font}px`}
				on:click={() => dispatch('open')}
				aria-label={`${item.token}, ${(item.probability * 100).toFixed(2)} percent`}
			>
				<span class="ctoken">{item.token.trim() ? item.token : '␣'}</span>
				<span class="pbar"
					><i style={`width:${Math.max(3, (item.probability / top) * 70)}px`}></i></span
				>
				<span class="ppercent"
					>{(item.probability * 100).toFixed(item.probability < 0.1 ? 2 : 1)}%</span
				>
			</button>{/each}
	</div>
</div>

<style>
	.probs {
		flex: none;
		width: 230px;
		font-family: Arial, sans-serif;
	}
	.cand-list {
		display: grid;
	}
	.cand {
		display: grid;
		grid-template-columns: 82px 76px minmax(0, 1fr);
		align-items: center;
		gap: 8px;
		border: 0;
		background: none;
		padding: 0;
		cursor: pointer;
		text-align: left;
		font: inherit;
		color: #9ca3af;
	}
	.ctoken {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		text-align: right;
		color: #6b7280;
	}
	.pbar {
		display: block;
	}
	.pbar i {
		display: block;
		height: 4px;
		background: #d1d5db;
	}
	.cand.first .pbar i {
		background: #7c3aed;
	}
	.cand.first .ctoken,
	.cand.first .ppercent {
		color: #7c3aed;
		font-weight: 600;
	}
</style>
