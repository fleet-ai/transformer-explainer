<script lang="ts">
	import Attention from '../Attention.svelte';
	import MambaHeads from './MambaHeads.svelte';
	import { hybridMambaView } from '$lib/hybrid/store';
	export let className: string | undefined = undefined;
</script>

{#if $hybridMambaView}<MambaHeads {className} />{:else}<Attention {className} />{/if}
