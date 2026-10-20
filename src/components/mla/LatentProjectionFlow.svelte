<script lang="ts">
	import { afterUpdate, createEventDispatcher, onMount } from 'svelte';
	export let selected = 'query';
	export let hiddenSize = 2048;
	export let heads = 16;
	export let contentSize = 128;
	export let rotarySize = 64;
	export let latentSize = 512;
	export let valueSize = 128;
	const dispatch = createEventDispatcher<{
		select: string;
		attention: void;
	}>();
	let map: HTMLDivElement;
	let paths: { id: string; label: string; d: string; bridge: boolean }[] = [];
	const edges = [
		['hidden', 'query', 'query', 'Query projection'],
		['query', 'query-output', 'query', 'Unrotated query projection'],
		['hidden', 'compression', 'compression', 'KV compression'],
		['compression', 'norm', 'compression', 'Raw KV latent'],
		['norm', 'key', 'key', 'Key expansion'],
		['key', 'key-output', 'key', 'Unrotated key content'],
		['compression', 'position', 'compression', 'Shared unrotated key position'],
		['position', 'key-output', 'compression', 'Shared key position'],
		['norm', 'value', 'value', 'Value expansion'],
		['value', 'value-output', 'value', 'Expanded value']
	];
	function activate(id: string) {
		dispatch('select', id);
	}
	function keyboard(event: KeyboardEvent, id: string) {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			activate(id);
		}
	}
	function measure() {
		if (!map) return;
		const parent = map.getBoundingClientRect();
		if (!parent.width) return;
		const next = edges.map(([from, to, id, label]) => {
			const a = map.querySelector<HTMLElement>(`[data-node="${from}"]`)!.getBoundingClientRect();
			const b = map.querySelector<HTMLElement>(`[data-node="${to}"]`)!.getBoundingClientRect();
			const x = a.right - parent.left,
				y = a.top - parent.top + a.height / 2;
			const xx = b.left - parent.left,
				yy = b.top - parent.top + b.height / 2;
			const mid = x + (xx - x) / 2;
			return {
				id,
				label,
				d: `M${x} ${y} C${mid} ${y} ${mid} ${yy} ${xx} ${yy}`,
				bridge: from === 'norm' && to === 'value'
			};
		});
		if (JSON.stringify(paths) !== JSON.stringify(next)) paths = next;
	}
	onMount(() => {
		const observer = new ResizeObserver(measure);
		observer.observe(map);
		measure();
		return () => observer.disconnect();
	});
	afterUpdate(measure);
</script>

<div class="hint">Click a band or box to inspect its operation</div>
<div class="latent-flow" bind:this={map} role="group" aria-label="MLA parallel projection branches">
	<svg aria-label="Projection bands">
		{#each paths as path}
			{#if path.bridge}<path class="bridge" d={path.d} />{/if}
			<path
				d={path.d}
				class:chosen={selected === path.id}
				role="button"
				tabindex="0"
				aria-label={`Inspect ${path.label} band`}
				on:click={() => activate(path.id)}
				on:keydown={(e) => keyboard(e, path.id)}
			/>
		{/each}
	</svg>
	<div class="node hidden-state" data-node="hidden">
		<b>x</b><small>RMSNorm</small><small>{hiddenSize}D</small>
	</div>
	<button
		class="node query"
		data-node="query"
		aria-pressed={selected === 'query'}
		on:click={() => activate('query')}
	>
		Wq<small>Query projection</small>
	</button>
	<div class="node result query-result" data-node="query-output">
		<b>Q</b><small>{heads * (contentSize + rotarySize)}D</small><small>All heads</small>
	</div>
	<button
		class="node compression"
		data-node="compression"
		aria-pressed={selected === 'compression'}
		on:click={() => activate('compression')}
	>
		W↓KV<small>Latent {latentSize}</small><small>+ rotary K {rotarySize}</small>
	</button>
	<button
		class="node norm"
		data-node="norm"
		aria-pressed={selected === 'norm'}
		on:click={() => activate('norm')}>RMSNorm<small>Latent · {latentSize}D</small></button
	>
	<button
		class="node key"
		data-node="key"
		aria-pressed={selected === 'key' || selected === 'expansion'}
		on:click={() => activate('key')}>W↑K<small>K content · {heads * contentSize}D</small></button
	>
	<div class="node position" data-node="position">
		Shared position<small>{rotarySize}D · before RoPE</small>
	</div>
	<div class="node result key-result" data-node="key-output">
		<b>K</b><small>{heads * contentSize}D content</small><small>+ shared {rotarySize}D</small>
	</div>
	<button
		class="node value"
		data-node="value"
		aria-pressed={selected === 'value'}
		on:click={() => activate('value')}>W↑V<small>V · {heads * valueSize}D</small></button
	>
	<div class="node result value-result" data-node="value-output">
		<b>V</b><small>{heads * valueSize}D</small><small>All heads</small>
	</div>
</div>
<div class="next-operations">
	<button on:click={() => dispatch('attention')}>Head split → Attention ↗</button>
</div>

<style>
	.hint {
		color: #8b709d;
		font:
			12px/1.4 Arial,
			sans-serif;
	}
	.latent-flow {
		position: relative;
		height: 300px;
		min-width: 0;
		container-type: inline-size;
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
		stroke-width: 19;
		cursor: pointer;
		pointer-events: stroke;
	}
	path.chosen,
	path:hover {
		stroke: #bb91d2;
		stroke-width: 23;
	}
	path.bridge {
		stroke: white;
		stroke-width: 32;
		pointer-events: none;
	}
	path:focus-visible {
		outline: 2px solid #9362c1;
	}
	.node {
		position: absolute;
		transform: translate(-50%, -50%);
		width: clamp(90px, 14cqw, 138px);
		min-height: 46px;
		padding: 7px 4px;
		background: #faf7fd;
		color: #75508f;
		border: 1.5px solid #cbb4df;
		border-radius: 5px;
		text-align: center;
		font:
			15px/1.2 Arial,
			sans-serif;
	}
	.node small {
		display: block;
		margin-top: 4px;
		font-size: 11px;
		color: #8e779e;
	}
	.node b {
		font-weight: 600;
	}
	button {
		cursor: pointer;
	}
	button:hover,
	button[aria-pressed='true'] {
		background: #e1c9ee;
		border-color: #b28bce;
	}
	button:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: 3px;
	}
	.hidden-state {
		left: 3.5%;
		top: 158px;
		width: 52px;
		background: white;
		border: 0;
		border-left: 7px solid #b6d1fa;
		border-radius: 0;
	}
	.hidden-state b {
		font:
			italic 23px Georgia,
			serif;
	}
	.query {
		left: 25%;
		top: 48px;
	}
	.compression {
		left: 25%;
		top: 175px;
	}
	.norm {
		left: 46%;
		top: 151px;
	}
	.key {
		left: 68%;
		top: 151px;
	}
	.position {
		left: 68%;
		top: 215px;
	}
	.value {
		left: 68%;
		top: 265px;
	}
	.result {
		left: 93%;
		width: 90px;
		border: 0;
		border-left: 5px solid currentColor;
		border-radius: 0;
		background: white;
	}
	.result b {
		font-size: 21px;
	}
	.query-result {
		top: 48px;
		color: #75a0e1;
	}
	.key-result {
		top: 175px;
		color: #cf8793;
	}
	.value-result {
		top: 265px;
		color: #66af91;
	}
	.next-operations {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 8px;
	}
	.next-operations button {
		color: #9066ad;
		background: none;
		border: 0;
		padding: 3px 0;
		font:
			12px/1.4 Arial,
			sans-serif;
	}
	@container (max-width: 610px) {
		.node {
			font-size: 12px;
			width: 80px;
		}
		.node small {
			font-size: 11px;
		}
		.result {
			width: 70px;
			left: 93%;
		}
		.hidden-state {
			width: 42px;
		}
	}
	@media (max-width: 600px) {
		.latent-flow {
			display: grid;
			grid-template-columns: 1fr 1fr;
			gap: 14px;
			height: auto;
			padding: 8px 0;
		}
		svg {
			display: none;
		}
		.node {
			position: relative;
			left: auto;
			top: auto;
			transform: none;
			width: 100%;
			min-height: 58px;
			font-size: 13px;
		}
		.hidden-state,
		.compression {
			grid-column: 1/3;
		}
	}
</style>
