<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { attentionHeadIdx, moeModel } from '~/store';
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
		<span>MLA projections · Head {$attentionHeadIdx + 1}</span><button
			on:click|stopPropagation={() => inspect('compression')}>Inspect MLA ↗</button
		>
	</div>
	<svg
		viewBox="0 0 520 460"
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
				orient="auto"><path d="M0 0 L5 2.5 L0 5" fill="#b7a3c8" /></marker
			></defs
		>
		<g class="connections" marker-end="url(#mla-flow-arrow)">
			<!-- One input branches to the two learned projections. -->
			<path d="M40 220 H58 V80 H78" /><path d="M58 220 V280 H78" />
			<!-- KV down projection splits latent and shared positional key. -->
			<path d="M163 80 H181 V42 H202" /><path d="M181 80 V140 H325" />
			<path d="M277 42 H294 V70 H325" />
			<path d="M294 42 H310 V130 Q320 140 310 150 V240 Q320 250 310 260 V405 H452" />
			<path d="M415 70 H460 V90 H505" />
			<path d="M415 140 H460 V90" />
			<!-- Q content bypasses RoPE; its positional branch rotates once. -->
			<path d="M163 280 H185 V250 H210" /><path d="M185 280 V320 H325" />
			<path d="M300 250 H460 V275 H505" />
			<path d="M415 320 H460 V275" />
		</g>
		<rect class="mla-input input" x="8" y="200" width="32" height="40" rx="4" /><text
			x="24"
			y="225"
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
			<rect x="78" y="62" width="85" height="36" rx="4" /><text x="120" y="84">W↓KV · x</text>
			<text x="242" y="126" class="key-text">Shared K position · {rotarySize}</text>
		</g>
		<g
			class="clickable"
			role="button"
			tabindex="0"
			aria-label="Inspect expansion of the compressed KV latent"
			on:click|stopPropagation={() => inspect('expansion')}
			on:keydown={(e) => keyboard(e, 'expansion')}
		>
			<rect x="202" y="20" width="75" height="44" rx="4" /><text x="239" y="37">RMSNorm</text><text
				x="239"
				y="54">W↑KV</text
			>
			<text x="239" y="12" class="small">Latent · {latentSize}</text>
			<rect class="key" x="325" y="52" width="90" height="36" rx="4" /><text
				x="370"
				y="74"
				class="key-text">K content · {contentSize}</text
			>
			<text x="349" y="395" class="value-text">V · {valueSize}D · no RoPE</text>
		</g>
		<g
			class="clickable"
			role="button"
			tabindex="0"
			aria-label="Inspect MLA query projection"
			on:click|stopPropagation={() => inspect('query')}
			on:keydown={(e) => keyboard(e, 'query')}
		>
			<rect class="query" x="78" y="262" width="85" height="36" rx="4" /><text
				x="120"
				y="284"
				class="query-text">WQ · x</text
			>
			<rect class="query" x="210" y="232" width="90" height="36" rx="4" /><text
				x="255"
				y="254"
				class="query-text">Q content · {contentSize}</text
			>
			<text x="244" y="306" class="query-text">Q position · {rotarySize}</text>
		</g>
		{#each [{ y: 119, label: 'rotate K', kind: 'key' }, { y: 299, label: 'rotate Q', kind: 'query' }] as gate}
			<g
				class="clickable rope"
				role="button"
				tabindex="0"
				aria-label={`Inspect MLA RoPE: ${gate.label}`}
				on:click|stopPropagation={() => inspect(`rope-${gate.kind}`)}
				on:keydown={(e) => keyboard(e, `rope-${gate.kind}`)}
			>
				<rect x="325" y={gate.y} width="90" height="42" rx="5" /><text
					x="370"
					y={gate.y + 19}
					class="rope-title">RoPE</text
				><text x="370" y={gate.y + 34} class="small">{gate.label}</text>
			</g>
		{/each}
		<rect class="mla-output-key port key" x="505" y="84" width="3" height="12" /><text
			x="485"
			y="76"
			class="key-text output">K′</text
		>
		<rect class="mla-output-query port query" x="505" y="269" width="3" height="12" /><text
			x="485"
			y="261"
			class="query-text output">Q′</text
		>
		<path class="connections" d="M452 405 H505" marker-end="url(#mla-flow-arrow)" />
		<rect class="mla-output-value port value" x="505" y="399" width="3" height="12" /><text
			x="485"
			y="391"
			class="value-text output">V</text
		>
		<text x="120" y="326" class="small">Split into heads</text>
		<text x="260" y="443" class="caption"
			>Content + rotated position → Q′ / K′ · {contentSize + rotarySize}D</text
		>
	</svg>
	<div class="flow-caption">
		Applied to each token. Q′ and K′ enter the dot product; V enters the weighted sum.
	</div>
</div>

<style>
	.latent-flow {
		width: 100%;
		color: #7e6a8b;
		font:
			11px/1.4 Arial,
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
			12px Arial,
			Helvetica,
			sans-serif;
		fill: #897599;
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
		fill: #6793cc;
	}
	.key {
		fill: #fdf0f1;
		stroke: #edc2c6;
	}
	.key-text {
		fill: #b77b86;
	}
	.value {
		fill: #eff8f3;
		stroke: #b2d8c3;
	}
	.value-text {
		fill: #6da488;
	}
	.rope rect {
		fill: #f3e9fa;
		stroke: #b48acb;
		stroke-width: 1.5;
	}
	.rope-title {
		font-size: 16px;
		font-weight: 600;
		fill: #755092;
	}
	.small {
		font-size: 10px;
	}
	.output {
		font-size: 17px;
		font-weight: 600;
	}
	.caption,
	.flow-caption {
		font-size: 10px;
		color: #91769f;
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
