<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { attentionHeadIdx, moeModel } from '~/store';
	const dispatch = createEventDispatcher<{ inspect: string; rope: void }>();
	$: contentSize = $moeModel?.contentKeySize ?? 128;
	$: rotarySize = $moeModel?.rotarySize ?? 64;
	$: latentSize = $moeModel?.kvLatentSize ?? 512;
	$: valueSize = $moeModel?.valueHeadSize ?? 128;
	function keyboard(event: KeyboardEvent, stage: string) {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			dispatch('inspect', stage);
		}
	}
</script>

<div class="latent-flow">
	<div class="flow-heading">
		<span>Multi-head Latent Attention <small>· Head {$attentionHeadIdx + 1}</small></span><button
			type="button"
			on:click={() => dispatch('inspect', 'compression')}>Inspect MLA ↗</button
		>
	</div>
	<svg
		viewBox="0 0 500 160"
		aria-label="MLA: the query splits into content and rotary coordinates. A shared compressed KV latent expands to head-specific content keys and values. A separate shared rotary key receives RoPE."
		role="img"
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
			<path d="M43 35 H67" /><path d="M143 35 H164 V20 H192" /><path d="M164 35 V59 H192" />
			<path d="M287 20 H355" /><path d="M287 59 H312" /><path d="M348 59 H368 V32" />
			<path d="M43 116 H67" /><path d="M143 116 H175 V100 H192" /><path d="M175 116 V142 H192" />
			<path d="M287 100 H355" /><path d="M287 142 H449" />
			<path d="M43 116 V77 H312" /><path d="M348 77 H368 V88" />
		</g>
		<text x="3" y="40" class="input">x</text><text x="3" y="121" class="input">x</text>
		<g
			class="clickable query"
			role="button"
			tabindex="0"
			aria-label="Inspect MLA query projection"
			on:click={() => dispatch('inspect', 'query')}
			on:keydown={(e) => keyboard(e, 'query')}
		>
			<rect x="67" y="23" width="76" height="24" rx="3" /><text x="105" y="39">Wq · x</text>
			<rect x="192" y="8" width="95" height="24" rx="3" /><text x="239.5" y="24"
				>Q content · {contentSize}</text
			>
			<rect x="192" y="47" width="95" height="24" rx="3" /><text x="239.5" y="63"
				>Q rotary · {rotarySize}</text
			>
		</g>
		<g
			class="clickable compression"
			role="button"
			tabindex="0"
			aria-label="Inspect shared KV latent compression"
			on:click={() => dispatch('inspect', 'compression')}
			on:keydown={(e) => keyboard(e, 'compression')}
		>
			<rect x="67" y="104" width="76" height="24" rx="3" /><text x="105" y="120"
				>KV latent · {latentSize}</text
			>
		</g>
		<g
			class="clickable key"
			role="button"
			tabindex="0"
			aria-label="Inspect expansion of the compressed KV latent"
			on:click={() => dispatch('inspect', 'expansion')}
			on:keydown={(e) => keyboard(e, 'expansion')}
		>
			<rect x="192" y="88" width="95" height="24" rx="3" /><text x="239.5" y="104"
				>K content · {contentSize}</text
			>
			<rect x="192" y="130" width="95" height="24" rx="3" class="value" /><text
				x="239.5"
				y="146"
				class="value-text">V · {valueSize}</text
			>
		</g>
		<g
			class="clickable rope"
			role="button"
			tabindex="0"
			aria-label="Inspect MLA rotary query and shared rotary key"
			on:click={() => dispatch('rope')}
			on:keydown={(e) => {
				if (e.key === 'Enter' || e.key === ' ') {
					e.preventDefault();
					dispatch('rope');
				}
			}}
		>
			<rect x="312" y="47" width="36" height="42" rx="3" /><text x="330" y="72">RoPE</text>
		</g>
		<text x="182" y="81" class="shared-label">shared K rotary · {rotarySize}</text>
		<text x="376" y="24" class="query-output">Q · {contentSize + rotarySize}D</text>
		<text x="376" y="104" class="key-output">K · {contentSize + rotarySize}D</text>
		<text x="457" y="146" class="value-output">{valueSize}D</text>
		<text x="4" y="158" class="cache-label"
			>cache: {latentSize} + {rotarySize} = {latentSize + rotarySize}D / token</text
		>
	</svg>
	<div class="flow-caption">
		Content Q/K bypass RoPE · all heads share the compressed cache and rotary key
	</div>
</div>

<style>
	.latent-flow {
		width: 100%;
		min-width: 380px;
		color: #7e6a8b;
		font:
			10px/1.4 Arial,
			Helvetica,
			sans-serif;
		padding: 3px 0 5px;
		box-sizing: border-box;
	}
	.flow-heading {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
	}
	.flow-heading small {
		color: #96859f;
		font: inherit;
	}
	button {
		font: inherit;
		color: #87669e;
		border: 1px solid #e5d8ee;
		border-radius: 3px;
		background: #fcf9fe;
		padding: 3px 6px;
		cursor: pointer;
	}
	svg {
		width: 100%;
		display: block;
		overflow: visible;
	}
	.connections {
		fill: none;
		stroke: #c8b8d7;
		stroke-width: 1.2;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	text {
		font:
			10px Arial,
			Helvetica,
			sans-serif;
		fill: #897599;
	}
	g text {
		text-anchor: middle;
	}
	.input {
		font:
			italic 18px Georgia,
			serif;
		fill: #9985ac;
	}
	rect {
		stroke: #d5c3e4;
		fill: #faf6fe;
	}
	.query rect {
		stroke: #a4c4ec;
		fill: #f1f7fe;
	}
	.query text,
	.query-output {
		fill: #729bd0;
	}
	.key rect {
		fill: #fdf0f1;
		stroke: #edc2c6;
	}
	.key text,
	.key-output {
		fill: #be8b94;
	}
	.key rect.value {
		fill: #eff8f3;
		stroke: #b2d8c3;
	}
	.key text.value-text,
	.value-output {
		fill: #6da488;
	}
	.shared-label {
		font-size: 8px;
		fill: #b18490;
		text-anchor: middle;
	}
	.cache-label {
		font-size: 8px;
		fill: #9a85ac;
	}
	.flow-caption {
		font-size: 9px;
		color: #91769f;
	}
	.clickable {
		cursor: pointer;
	}
	.clickable:hover rect {
		stroke: #9c71be;
		stroke-width: 1.5;
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
