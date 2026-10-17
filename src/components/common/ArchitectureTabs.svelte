<script lang="ts">
	import { base } from '$app/paths';
	import { architectures, type ArchitectureId } from '../../lib/architecture-navigation';
	export let active: ArchitectureId;
	$: root = active === 'dense' ? base.replace(/\/dense$/, '') : base;
</script>

<nav class="architecture-tabs" aria-label="Transformer architecture">
	{#each architectures as architecture}
		<a
			href={`${root}${architecture.path}`}
			data-sveltekit-reload
			rel="external"
			class:active={active === architecture.id}
			aria-current={active === architecture.id ? 'page' : undefined}
			title={architecture.title}>{architecture.label}</a
		>
	{/each}
</nav>

<style>
	.architecture-tabs {
		display: inline-flex;
		flex-wrap: wrap;
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
		min-width: 54px;
		padding: 6px 13px;
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
