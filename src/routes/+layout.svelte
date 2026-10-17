<script lang="ts">
	import '~/styles/app.css';
	import '~/styles/global.scss';
	import 'katex/dist/katex.min.css';
	import { page } from '$app/stores';
	import ArchitectureHeader from '~/components/common/ArchitectureHeader.svelte';
	import type { RecordedArchitectureId } from '~/lib/architecture-navigation';
	$: architecture = $page.params.architecture as RecordedArchitectureId | undefined;
	import Topbar from '~/components/Topbar.svelte';
	import { predictedColor, rootRem } from '~/store';
	let topBarHeight = 0;
	let minScreenWidth = 1600;
	let minColumnWidth = Math.floor(minScreenWidth / 24) - rootRem * 2;
</script>

{#if architecture}
	<div class="architecture-app">
		<ArchitectureHeader active={architecture} />
		<main class="architecture-main"><slot /></main>
	</div>
{:else}
	<div
		id="app"
		style={`--min-screen-width:${minScreenWidth}px;--min-column-width:${minColumnWidth}px;--predicted-color:${predictedColor};`}
	>
		<div id="landing">
			<header bind:offsetHeight={topBarHeight}><Topbar isActive={true} /></header>
			<main id="main" style={`padding-top:${topBarHeight}px`}><slot /></main>
		</div>
	</div>
{/if}

<style lang="scss">
	.architecture-app {
		min-height: 100vh;
		background: white;
		color: #4b5563;
	}
	main.architecture-main {
		display: block;
		height: auto;
		overflow: visible;
	}
	#app {
		height: 100vh;
		min-width: 1600px;
	}
	#landing {
		height: 100%;
		width: 100%;
		min-width: var(--min-screen-width);
	}
	header {
		min-width: var(--min-screen-width);
		width: 100%;
		position: absolute;
		z-index: $TOP_BAR_INDEX;
	}
	main {
		position: relative;
		height: 100%;
		width: 100%;
		display: flex;
		justify-content: start;
		overflow-x: clip;
		overflow-y: visible;
	}
</style>
