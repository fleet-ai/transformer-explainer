<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { moeModel } from '~/store';
	const dispatch = createEventDispatcher<{ inspect: string; rope: 'query' | 'key' }>();
	$: contentSize = $moeModel?.contentKeySize ?? 128;
	$: rotarySize = $moeModel?.rotarySize ?? 64;
	$: latentSize = $moeModel?.kvLatentSize ?? 512;
	$: valueSize = $moeModel?.valueHeadSize ?? 128;
	function inspect(stage: string) {
		if (stage.startsWith('rope-')) dispatch('rope', stage === 'rope-key' ? 'key' : 'query');
		else dispatch('inspect', stage);
	}
	function keyboard(event: KeyboardEvent, stage: string) {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			event.stopPropagation();
			inspect(stage);
		}
	}
</script>

<div class="latent-flow">
	<div class="flow-heading">
		<span>MLA projections</span><button on:click|stopPropagation={() => inspect('compression')}
			>Inspect MLA ↗</button
		>
	</div>
	<svg
		viewBox="0 0 400 475"
		role="img"
		aria-label="One normalized hidden-state input. Query projection and KV compression produce the head vectors. Only positional Q and K receive RoPE, before Q prime and K prime enter the attention dot product. V bypasses RoPE."
	>
		<defs
			><marker
				id="mla-flow-arrow"
				markerWidth="5"
				markerHeight="5"
				refX="4"
				refY="2.5"
				orient="auto"><path d="M0 0 L5 2.5 L0 5" fill="#a48bb9" /></marker
			></defs
		>
		<g class="connections" marker-end="url(#mla-flow-arrow)">
			<path d="M34 230 H48 V99 H65" /><path d="M48 230 V304 H65" />
			<path d="M153 99 H165 V57 H180" /><path d="M165 99 V170 H285" />
			<path d="M275 57 H285" />
			<path
				d="M278 57 V160 Q286 170 278 180 V260 Q286 270 278 280 V335 Q286 345 278 355 V430 H385"
			/>
			<path d="M373 65 H378 V110 H385" /><path d="M373 170 H378 V110" />
			<path d="M153 304 H165 V270 H180" /><path d="M165 304 V345 H285" />
			<path d="M270 270 H378 V285 H385" /><path d="M373 345 H378 V285" />
		</g>
		<rect class="mla-input input" x="6" y="210" width="28" height="40" rx="4" /><text
			x="20"
			y="236"
			class="input-label">x</text
		>
		<g
			class="clickable"
			role="button"
			tabindex="0"
			aria-label="Inspect shared KV latent compression"
			on:click|stopPropagation={() => inspect('compression')}
			on:keydown={(e) => keyboard(e, 'compression')}
		>
			<rect x="65" y="75" width="88" height="48" rx="5" /><text x="109" y="104">W↓KV · x</text>
			<text x="226" y="151" class="key-text small">K position</text><text
				x="225"
				y="193"
				class="small">shared · {rotarySize}D</text
			>
		</g>
		<g
			class="clickable"
			role="button"
			tabindex="0"
			aria-label="Inspect expansion of the compressed KV latent"
			on:click|stopPropagation={() => inspect('expansion')}
			on:keydown={(e) => keyboard(e, 'expansion')}
		>
			<rect x="180" y="30" width="95" height="54" rx="5" /><text x="227" y="52">RMSNorm</text><text
				x="227"
				y="73">W↑KV</text
			>
			<text x="227" y="18" class="small">Latent · {latentSize}D</text>
			<rect class="key" x="285" y="40" width="88" height="50" rx="5" /><text
				x="329"
				y="61"
				class="key-text">K content</text
			><text x="329" y="80" class="key-text small">{contentSize}D</text>
		</g>
		<g
			class="clickable"
			role="button"
			tabindex="0"
			aria-label="Inspect MLA query projection"
			on:click|stopPropagation={() => inspect('query')}
			on:keydown={(e) => keyboard(e, 'query')}
		>
			<rect class="query" x="65" y="280" width="88" height="48" rx="5" /><text
				x="109"
				y="309"
				class="query-text">WQ · x</text
			>
			<rect class="query" x="180" y="245" width="90" height="50" rx="5" /><text
				x="225"
				y="266"
				class="query-text">Q content</text
			><text x="225" y="285" class="query-text small">{contentSize}D</text>
			<text x="225" y="328" class="query-text small">Q position</text><text
				x="225"
				y="368"
				class="small">{rotarySize}D</text
			>
		</g>
		{#each [{ y: 145, label: 'rotate K', kind: 'key' }, { y: 320, label: 'rotate Q', kind: 'query' }] as gate}
			<g
				class="clickable rope"
				role="button"
				tabindex="0"
				aria-label={`Inspect MLA RoPE: ${gate.label}`}
				on:click|stopPropagation={() => inspect(`rope-${gate.kind}`)}
				on:keydown={(e) => keyboard(e, `rope-${gate.kind}`)}
			>
				<rect x="285" y={gate.y} width="88" height="50" rx="5" /><text
					x="329"
					y={gate.y + 23}
					class="rope-title">RoPE</text
				><text x="329" y={gate.y + 42} class="small">{gate.label}</text>
			</g>
		{/each}
		<rect class="mla-output-key port key" x="385" y="104" width="3" height="12" /><text
			x="368"
			y="130"
			class="key-text output">K′</text
		>
		<rect class="mla-output-query port query" x="385" y="279" width="3" height="12" /><text
			x="368"
			y="309"
			class="query-text output">Q′</text
		>
		<rect class="mla-output-value port value" x="385" y="424" width="3" height="12" /><text
			x="366"
			y="420"
			class="value-text output">V</text
		>
		<text x="225" y="420" class="value-text small">V bypasses RoPE</text>
		<text x="200" y="465" class="caption"
			>Q′ / K′ · {contentSize + rotarySize}D V · {valueSize}D</text
		>
	</svg>
	<div class="flow-caption">
		Applied to each token. Q′ and K′ enter the dot product; V enters the weighted sum.
	</div>
</div>

<style>
	.latent-flow {
		width: 100%;
		color: #695078;
		font:
			14px/1.4 Arial,
			Helvetica,
			sans-serif;
	}
	.flow-heading {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
	}
	button {
		font: inherit;
		color: #87669e;
		border: 1px solid #e5d8ee;
		border-radius: 3px;
		background: #fcf9fe;
		padding: 4px 6px;
		cursor: pointer;
	}
	svg {
		width: 100%;
		display: block;
		overflow: visible;
	}
	.connections {
		fill: none;
		stroke: #bda8ce;
		stroke-width: 1.7;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	text {
		font:
			18px Arial,
			Helvetica,
			sans-serif;
		fill: #765887;
		text-anchor: middle;
	}
	rect {
		stroke: #d5c3e4;
		fill: #faf6fe;
	}
	.input {
		fill: #f0f1f4;
		stroke: #a9abb4;
	}
	.input-label {
		font:
			italic 21px Georgia,
			serif;
		fill: #737782;
	}
	.query {
		stroke: #a4c4ec;
		fill: #f1f7fe;
	}
	.query-text {
		fill: #4878b7;
	}
	.key {
		fill: #fdf0f1;
		stroke: #edc2c6;
	}
	.key-text {
		fill: #a15c6a;
	}
	.value {
		fill: #eff8f3;
		stroke: #b2d8c3;
	}
	.value-text {
		fill: #438769;
	}
	.rope rect {
		fill: #f3e9fa;
		stroke: #b48acb;
		stroke-width: 1.5;
	}
	.rope-title {
		font-size: 21px;
		font-weight: 600;
		fill: #755092;
	}
	.small {
		font-size: 14px;
	}
	.output {
		font-size: 20px;
		font-weight: 600;
	}
	.caption,
	.flow-caption {
		font-size: 14px;
		color: #80628e;
	}
	.clickable {
		cursor: pointer;
	}
	.clickable:hover rect {
		stroke: #9c71be;
		stroke-width: 2;
	}
	.clickable:focus-visible {
		outline: none;
	}
	.clickable:focus-visible rect {
		stroke: #8b5cc0;
		stroke-width: 2;
	}
	button:focus-visible {
		outline: 2px solid #b28bcb;
		outline-offset: 2px;
	}
</style>
