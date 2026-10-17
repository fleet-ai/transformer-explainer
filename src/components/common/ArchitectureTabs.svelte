<script lang="ts">
	import { base } from '$app/paths';
	export let active: 'dense' | 'moe';
	$: root = active === 'dense' ? base.replace(/\/dense$/, '') : base;
</script>

<nav class="architecture-tabs" aria-label="Transformer architecture">
	<a
		href={`${root}/dense/`}
		data-sveltekit-reload
		rel="external"
		class:active={active === 'dense'}
		aria-current={active === 'dense' ? 'page' : undefined}
		title="Dense transformer · GPT-2">Dense</a
	>
	<a
		href={`${root}/`}
		data-sveltekit-reload
		rel="external"
		class:active={active === 'moe'}
		aria-current={active === 'moe' ? 'page' : undefined}
		title="Mixture of Experts · Granite">MoE</a
	>
</nav>

<style>
	.architecture-tabs {
		display: inline-flex;
		align-self: flex-start;
		gap: 3px;
		padding: 3px;
		border: 1px solid #e6dcf5;
		border-radius: 7px;
		background: #faf8fe;
		font:
			500 12px/1.2 Arial,
			sans-serif;
	}
	a {
		min-width: 64px;
		padding: 5px 14px;
		border-radius: 4px;
		color: #8b76a3;
		text-align: center;
		text-decoration: none;
		transition:
			background 120ms,
			color 120ms;
	}
	a:hover {
		background: #eee5fa;
		color: #7449b3;
	}
	a.active {
		background: #945fdd;
		color: white;
	}
	a:focus-visible {
		outline: 2px solid #8252d9;
		outline-offset: 3px;
	}
</style>
