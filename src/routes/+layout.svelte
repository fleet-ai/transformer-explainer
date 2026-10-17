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
	$: originalArchitecture =
		!architecture ||
		['mla', 'vit', 'encoder-decoder', 'vision-language', 'hybrid'].includes(architecture);
	$: minScreenWidth = architecture === 'mla' ? 0 : architecture === 'encoder-decoder' ? 1900 : 1600;
	$: minColumnWidth = Math.max(16, Math.floor(minScreenWidth / 24) - rootRem * 2);
</script>

{#if !originalArchitecture}
	<div class="architecture-app">
		<ArchitectureHeader active={architecture} />
		<main class="architecture-main"><slot /></main>
	</div>
{:else}
	<div
		id="app"
		class:mla-page={architecture === 'mla'}
		style={`--min-screen-width:${minScreenWidth}px;--min-column-width:${minColumnWidth}px;--predicted-color:${predictedColor};`}
	>
		<div id="landing">
			<header bind:offsetHeight={topBarHeight}>
				<Topbar isActive={true} architecture={architecture ?? 'moe'} />
			</header>
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
	#app.mla-page {
		min-width: 0;
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
