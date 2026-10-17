<script lang="ts">
	import { base } from '$app/paths';
	import ArchitectureTabs from './ArchitectureTabs.svelte';
	import type { ArchitectureId } from '../../lib/architecture-navigation';
	export let active: ArchitectureId;
</script>

<header class="architecture-header">
	<div class="header-line">
		<a class="logo" href={`${base}/`}>Transformer Explainer</a><span
			>Recorded model computations</span
		><a
			class="source"
			href="https://github.com/fleet-ai/transformer-explainer"
			target="_blank"
			rel="noreferrer">Source ↗</a
		>
	</div>
	<ArchitectureTabs {active} />
</header>

<style>
	.architecture-header {
		padding: 14px 32px 12px;
		background: #fff;
		border-bottom: 1px solid #f0ebf7;
	}
	.header-line {
		display: flex;
		align-items: center;
		gap: 24px;
		margin-bottom: 9px;
	}
	.logo {
		font-family: 'Jersey 10', sans-serif;
		text-transform: uppercase;
		font-size: 34px;
		line-height: 1;
		background: linear-gradient(90deg, #4285ff, #985bff);
		background-clip: text;
		-webkit-background-clip: text;
		color: transparent;
		text-decoration: none;
	}
	span {
		font-size: 12px;
		color: #9a8aaa;
	}
	.source {
		margin-left: auto;
		font-size: 13px;
		color: #8d65ad;
		text-decoration: none;
	}
	a:focus-visible {
		outline: 2px solid #945fdd;
		outline-offset: 4px;
	}
	@media (max-width: 700px) {
		.architecture-header {
			padding: 12px 16px;
		}
		.logo {
			font-size: 28px;
		}
		span {
			display: none;
		}
	}
</style>
