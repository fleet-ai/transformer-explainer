<script lang="ts">
	import Attention from '../Attention.svelte';
	import MambaCore from './MambaCore.svelte';
	import { blockIdx } from '~/store';
	import { hybridRecording } from '$lib/hybrid/store';
	export let className: string | undefined = undefined;
	$: hasAttention = $hybridRecording?.document.layers[$blockIdx].type === 'hybrid';
</script>

{#if hasAttention}<Attention {className} />{:else}<div class={`mamba-step ${className ?? ''}`}>
		<div class="title">Selective State-space Model</div>
		<div class="content"><MambaCore /></div>
	</div>{/if}

<style>
	.mamba-step .content {
		padding: 0 45px 0 65px;
	}
	.mamba-step > .title {
		padding-left: 40px;
	}
</style>
