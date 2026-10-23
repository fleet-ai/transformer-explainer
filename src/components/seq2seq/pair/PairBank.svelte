<script lang="ts">
	/** The Q · K · V bank of the Dense explainer: each token's three projections stacked, with their letters. */
	export let id: string;
	export let rows = 6;
	export let height = 300;
	export let selected = -1;
	$: pitch = height / Math.max(rows, 1);
	$: block = Math.max(9, pitch - 4);
	$: sub = block / 3;
	$: font = Math.max(7, Math.min(13, sub * 1.05));
</script>

<div class="bank" style={`height:${height}px`}>
	{#each Array(rows) as _, row}
		<div
			class="bank-token"
			class:selected={row === selected}
			data-b={`${id}:tok`}
			data-i={row}
			style={`top:${row * pitch + (pitch - block) / 2}px;height:${block}px`}
		>
			{#each ['q', 'k', 'v'] as part}<span
					class={`part ${part}`}
					data-b={`${id}:${part}`}
					data-i={row}
					style={`height:${sub}px;font-size:${font}px;line-height:${sub}px`}
					>{part.toUpperCase()}</span
				>{/each}
		</div>
	{/each}
</div>

<style>
	.bank {
		position: relative;
		width: 16px;
		flex: none;
	}
	.bank-token {
		position: absolute;
		left: 0;
		width: 16px;
		display: flex;
		flex-direction: column;
		z-index: 3;
	}
	.part {
		display: block;
		text-align: center;
		font-family: Arial, sans-serif;
		font-weight: 700;
	}
	.part.q {
		background: #bfdbfe;
		color: #60a5fa;
	}
	.part.k {
		background: #fecaca;
		color: #f87171;
	}
	.part.v {
		background: #bbf7d0;
		color: #34d399;
	}
	.bank-token.selected {
		outline: 1px solid #a78bfa;
	}
</style>
