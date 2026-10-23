<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Vec from './Vec.svelte';
	import { STRIP_W } from './metrics';

	/** A Dense-style token column: large token labels beside thin vectors, spread over the row height. */
	export let id: string;
	export let labels: string[] = [];
	export let vectors: number[][] = [];
	export let color = 'gray';
	export let height = 300;
	export let maxPitch = 40;
	export let selected = -1;
	export let labelSide: 'left' | 'right' = 'left';
	export let interactive = true;
	const dispatch = createEventDispatcher<{ select: number }>();
	$: rows = Math.max(labels.length, 1);
	$: pitch = Math.min(maxPitch, height / rows);
	$: stripH = Math.max(4, Math.min(22, pitch * 0.6));
	$: font = Math.max(9, Math.min(16, pitch * 0.55));
</script>

<div class="strips" class:right={labelSide === 'right'} style={`height:${height}px`}>
	<div
		class="lane"
		data-b={`${id}:lane`}
		style={`top:0;height:${rows * pitch}px;${labelSide === 'right' ? 'left:0' : 'right:0'}`}
	></div>
	<div data-b={`${id}:col`} class="tokcol" style={`height:${rows * pitch}px`}>
		{#each labels as label, row}<button
				class="tokrow"
				class:selected={row === selected}
				style={`height:${pitch}px`}
				disabled={!interactive}
				aria-label={`Select token ${label}`}
				aria-pressed={row === selected}
				on:click={() => dispatch('select', row)}
			>
				<span class="tokname" style={`font-size:${font}px`}>{label}</span>
				<span data-b={id} data-i={row} class="slot"
					><Vec
						data={vectors[row]}
						{color}
						w={STRIP_W}
						h={stripH}
						hot={row === selected}
						title={label}
					/></span
				>
			</button>{/each}
	</div>
</div>

<style>
	.strips {
		position: relative;
		flex: none;
		min-width: 110px;
		font-family: Arial, sans-serif;
	}
	.lane {
		position: absolute;
		width: 12px;
		pointer-events: none;
	}
	.tokcol {
		position: relative;
	}
	.tokrow {
		display: flex;
		align-items: center;
		justify-content: flex-end;
		gap: 8px;
		width: 100%;
		padding: 0;
		border: 0;
		background: none;
		color: #374151;
		font: inherit;
		cursor: pointer;
	}
	.right .tokrow {
		flex-direction: row-reverse;
		justify-content: flex-end;
	}
	.tokrow:disabled {
		cursor: default;
	}
	.tokname {
		white-space: nowrap;
		line-height: 1;
	}
	.tokrow.selected .tokname {
		color: #7c3aed;
		font-weight: 600;
	}
	.slot {
		display: block;
		line-height: 0;
		position: relative;
		z-index: 3;
	}
	.tokrow:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: 1px;
	}
</style>
