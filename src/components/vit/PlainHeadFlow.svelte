<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	const dispatch = createEventDispatcher<{ inspect: void }>();
	import { attentionHeadIdx, attentionHeadIdxTemp, moeModel } from '~/store';
	$: heads = $moeModel?.attentionHeads ?? 3;
	$: size = $moeModel?.attentionHeadSize ?? 64;
	function selectHead(head: number) {
		attentionHeadIdx.set(head);
		attentionHeadIdxTemp.set(head);
	}
</script>

<div class="plain-head-flow">
	<div class="heading">
		<span>Split into {heads} heads · {size}D each</span><button
			on:click|stopPropagation={() => dispatch('inspect')}>Inspect projections ↗</button
		>
	</div>
	{#each ['Q', 'K', 'V'] as label}<div
			class="row"
			class:query={label === 'Q'}
			class:key={label === 'K'}
			class:value={label === 'V'}
		>
			<strong>{label}</strong><span>{heads * size}D →</span>
			<div class="heads">
				{#each Array(heads) as _, head}<button
						class:selected={$attentionHeadIdx === head}
						aria-label={`${label} head ${head + 1}`}
						aria-pressed={$attentionHeadIdx === head}
						on:click={() => selectHead(head)}>{head + 1}</button
					>{/each}
			</div>
			<span>→ {label} head {$attentionHeadIdx + 1} · {size}D</span>
		</div>{/each}
	<small
		>{$moeModel?.normalization === 'layernorm'
			? 'Learned position embeddings were added before these projections.'
			: 'Relative position bias is added to the attention scores.'}</small
	>
</div>

<style>
	.plain-head-flow {
		font-size: 10px;
		color: #8c72a1;
	}
	.heading {
		display: flex;
		justify-content: space-between;
		gap: 10px;
		margin-bottom: 10px;
	}
	.heading button {
		border: 1px solid #d9c8eb;
		border-radius: 3px;
		padding: 3px 7px;
		background: #faf7fd;
		color: #8d61ad;
	}
	.row {
		display: flex;
		align-items: center;
		gap: 8px;
		margin: 8px 0;
	}
	.row > strong {
		font-size: 14px;
		width: 12px;
	}
	.query {
		color: #6d9dce;
	}
	.key {
		color: #ca8c94;
	}
	.value {
		color: #75aa90;
	}
	.heads {
		display: flex;
		flex: 1;
		gap: 3px;
	}
	.heads button {
		border: 1px solid currentColor;
		opacity: 0.4;
		min-width: 12px;
		flex: 1;
		border-radius: 2px;
		background: #f5effa;
		font-size: 9px;
		position: relative;
		color: inherit;
	}
	.heads button.selected {
		opacity: 1;
		outline: 1px solid #9565b7;
		outline-offset: 1px;
	}
	.heads button:hover {
		opacity: 1;
	}
	small {
		font-size: 9px;
		display: block;
	}
</style>
