<script lang="ts">
	import { afterUpdate, createEventDispatcher, onMount } from 'svelte';
	export let selected = 'compression';
	export let hiddenSize = 2048;
	export let heads = 16;
	export let contentSize = 128;
	export let rotarySize = 64;
	export let latentSize = 512;
	export let valueSize = 128;
	const dispatch = createEventDispatcher<{ select: string; rope: void; attention: void }>();
	let map: HTMLDivElement;
	let paths: { id: string; label: string; d: string }[] = [];
	const edges = [
		{ from: 'hidden', to: 'query', id: 'query', label: 'Query projection' },
		{ from: 'hidden', to: 'compression', id: 'compression', label: 'KV compression' },
		{ from: 'query', to: 'rope', id: 'rope', label: 'Query rotary coordinates to RoPE' },
		{ from: 'compression', to: 'rope', id: 'rope', label: 'Shared key rotary coordinates to RoPE' },
		{
			from: 'compression',
			to: 'expansion',
			id: 'expansion',
			label: 'Normalized latent to K / V expansion'
		},
		{
			from: 'query',
			to: 'scores',
			id: 'scores',
			label: 'Query content bypasses RoPE',
			bypass: true
		},
		{ from: 'rope', to: 'scores', id: 'scores', label: 'Rotated Q′ and K′ to dot product' },
		{
			from: 'expansion',
			to: 'scores',
			id: 'scores',
			label: 'Content K and unchanged V to attention'
		},
		{
			from: 'scores',
			to: 'attention',
			id: 'attention',
			label: 'Concatenate head contexts and project output'
		}
	];
	function activate(id: string) {
		if (id === 'rope') dispatch('rope');
		else if (id === 'scores') dispatch('attention');
		else dispatch('select', id);
	}
	function measure() {
		if (!map) return;
		const parent = map.getBoundingClientRect();
		if (!parent.width) return;
		const next = edges.map((edge) => {
			const a = map
				.querySelector<HTMLElement>(`[data-node="${edge.from}"]`)!
				.getBoundingClientRect();
			const b = map.querySelector<HTMLElement>(`[data-node="${edge.to}"]`)!.getBoundingClientRect();
			let x1 = a.right - parent.left,
				y1 = a.top - parent.top + a.height / 2;
			let x2 = b.left - parent.left,
				y2 = b.top - parent.top + b.height / 2;
			let d: string;
			if (edge.bypass) {
				x1 = a.left - parent.left + a.width / 2;
				y1 = a.top - parent.top;
				x2 = b.left - parent.left + b.width / 2;
				y2 = b.top - parent.top;
				d = `M${x1},${y1} V8 H${x2} V${y2}`;
			} else if (edge.to === 'attention' || Math.abs(a.left - b.left) < 5 || b.top >= a.bottom) {
				x1 = a.left - parent.left + a.width / 2;
				y1 = a.bottom - parent.top;
				x2 = b.left - parent.left + b.width / 2;
				y2 = b.top - parent.top;
				const mid = y2 - 12;
				d = `M${x1},${y1} V${mid} H${x2} V${y2}`;
			} else {
				const mid = (x1 + x2) / 2;
				d = `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`;
			}
			return { id: edge.id, label: edge.label, d };
		});
		if (JSON.stringify(next) !== JSON.stringify(paths)) paths = next;
	}
	onMount(() => {
		const observer = new ResizeObserver(measure);
		observer.observe(map);
		measure();
		return () => observer.disconnect();
	});
	afterUpdate(measure);
</script>

<div class="latent-flow" bind:this={map} role="group" aria-label="MLA parallel projection branches">
	<svg aria-label="Projection connections">
		{#each paths as path}
			<path
				d={path.d}
				class:chosen={selected === path.id}
				role="button"
				tabindex="0"
				aria-label={`Inspect ${path.label} band`}
				on:click={() => activate(path.id)}
				on:keydown={(event) => {
					if (event.key === 'Enter' || event.key === ' ') {
						event.preventDefault();
						activate(path.id);
					}
				}}
			/>
		{/each}
	</svg>
	<div class="hidden-state" data-node="hidden">
		<b>x</b><span>RMSNorm</span><small>{hiddenSize}D</small>
	</div>
	<button
		class="query"
		data-node="query"
		aria-pressed={selected === 'query'}
		on:click={() => activate('query')}
	>
		Query projection <small>{heads} × ({contentSize} + {rotarySize})</small>
	</button>
	<button
		class="compression"
		data-node="compression"
		aria-pressed={selected === 'compression'}
		on:click={() => activate('compression')}
	>
		KV compression <small>Latent {latentSize} + rotary K {rotarySize}</small>
	</button>
	<button class="rope" data-node="rope" on:click={() => activate('rope')}>
		RoPE · Rotate Q and K <small>Rotary {rotarySize}D → Q′ / K′</small>
	</button>
	<button
		class="expansion"
		data-node="expansion"
		aria-pressed={selected === 'expansion'}
		on:click={() => activate('expansion')}
	>
		K / V expansion <small>K content {contentSize} · V {valueSize}</small>
	</button>
	<button class="scores" data-node="scores" on:click={() => activate('scores')}>
		Attention <small>Dot product → softmax → × V</small>
	</button>
	<button
		class="output"
		data-node="attention"
		aria-pressed={selected === 'attention'}
		on:click={() => activate('attention')}
	>
		Concat → Output projection <small>{heads} × {valueSize} → {hiddenSize}D</small>
	</button>
	<span class="bypass-label">Q content bypasses RoPE</span>
</div>

<style>
	.latent-flow {
		position: relative;
		display: grid;
		grid-template-columns: 76px 1fr 1.25fr 1fr;
		grid-template-rows: 64px 64px 54px;
		gap: 20px 24px;
		padding: 27px 0 0;
		min-width: 0;
	}
	svg {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		overflow: visible;
	}
	path {
		fill: none;
		stroke: #ded4ee;
		stroke-width: 13;
		stroke-linejoin: round;
		cursor: pointer;
		pointer-events: stroke;
	}
	path:hover,
	path.chosen {
		stroke: #bf99da;
	}
	path:focus-visible {
		outline: 2px solid #9362c1;
	}
	button {
		position: relative;
		z-index: 1;
		align-self: center;
		min-height: 58px;
		width: 100%;
		border: 1.5px solid #cbb4df;
		border-radius: 5px;
		padding: 8px 6px;
		background: #faf7fd;
		color: #694684;
		font:
			14px/1.25 Arial,
			sans-serif;
		cursor: pointer;
	}
	button small {
		display: block;
		margin-top: 5px;
		color: #856795;
		font-size: 11px;
	}
	button:hover,
	button[aria-pressed='true'] {
		background: #e0c8ef;
		border-color: #b28bce;
	}
	button:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: 2px;
	}
	.hidden-state {
		grid-column: 1;
		grid-row: 1/3;
		align-self: center;
		position: relative;
		z-index: 1;
		background: white;
		text-align: center;
		padding: 8px 0;
		color: #725687;
	}
	.hidden-state b {
		font:
			italic 24px Georgia,
			serif;
	}
	.hidden-state span,
	.hidden-state small {
		display: block;
		font:
			11px/1.5 Arial,
			sans-serif;
	}
	.query {
		grid-column: 2;
		grid-row: 1;
	}
	.compression {
		grid-column: 2;
		grid-row: 2;
	}
	.rope {
		grid-column: 3;
		grid-row: 1;
	}
	.expansion {
		grid-column: 3;
		grid-row: 2;
	}
	.scores {
		grid-column: 4;
		grid-row: 1/3;
	}
	.output {
		grid-column: 3/5;
		grid-row: 3;
		justify-self: end;
		width: 88%;
		min-height: 50px;
	}
	.bypass-label {
		position: absolute;
		top: 0;
		left: 39%;
		font:
			11px Arial,
			sans-serif;
		color: #856795;
		padding: 0 6px;
		background: white;
	}
	@media (max-width: 850px) {
		.latent-flow {
			grid-template-columns: 55px 1fr 1.2fr 1fr;
			gap: 22px 12px;
		}
		button {
			font-size: 12px;
			padding: 6px 3px;
		}
	}
	@media (max-width: 600px) {
		.latent-flow {
			grid-template-columns: 1fr 1fr;
			grid-template-rows: 40px 72px 72px 72px 52px;
			gap: 20px 18px;
		}
		.hidden-state {
			grid-column: 1/3;
			grid-row: 1;
			justify-self: start;
			width: 75px;
		}
		.query {
			grid-column: 1;
			grid-row: 2;
		}
		.compression {
			grid-column: 1;
			grid-row: 3;
		}
		.rope {
			grid-column: 2;
			grid-row: 2;
		}
		.expansion {
			grid-column: 2;
			grid-row: 3;
		}
		.scores {
			grid-column: 2;
			grid-row: 4;
		}
		.output {
			grid-column: 1/3;
			grid-row: 5;
			width: 100%;
		}
		.bypass-label {
			left: 25%;
			font-size: 11px;
		}
	}
</style>
