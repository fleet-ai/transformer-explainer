<script lang="ts">
	/** "N more identical blocks", drawn as the Dense explainer draws them: thin stacked columns per token. */
	export let id: string;
	export let rows = 6;
	export let height = 300;
	export let count = 5;
	$: pitch = height / Math.max(rows, 1);
	$: bar = Math.max(3, Math.min(pitch - 3, (pitch - 3) / 2));
</script>

<div class="blocks" style={`height:${height}px;width:${count * 10 + 4}px`}>
	{#each Array(count) as _, column}
		{#each Array(rows) as _, row}<span
				class="seg"
				data-b={column === 0 ? `${id}:in` : column === count - 1 ? `${id}:out` : undefined}
				data-i={row}
				style={`left:${column * 10}px;top:${row * pitch + pitch / 2 - bar / 2}px;height:${bar}px;opacity:${0.45 + (column / count) * 0.5}`}
			></span>{/each}
	{/each}
	<span class="column-group" data-b={`${id}:outcol`} style={`left:${(count - 1) * 10}px`}></span>
</div>

<style>
	.blocks {
		position: relative;
		flex: none;
	}
	.seg {
		position: absolute;
		width: 6px;
		background: #bfdbfe;
		z-index: 3;
	}
	.column-group {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 6px;
		pointer-events: none;
	}
</style>
