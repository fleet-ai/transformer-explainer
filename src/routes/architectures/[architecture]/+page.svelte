<script lang="ts">
	import { base } from '$app/paths';
	import ArchitectureExplorer from '~/components/architectures/ArchitectureExplorer.svelte';
	import type { RecordedArchitectureId } from '~/lib/architecture-navigation';
	export let data: { architecture: RecordedArchitectureId };
</script>

<svelte:head>
 <title>Transformer Explainer · {data.architecture}</title>
 <meta name="description" content="Explore real pretrained transformer computations with interactive architecture diagrams and recorded tensors." />
 <link rel="stylesheet" href={`${base}/fonts/font.css`} />
</svelte:head>

{#key data.architecture}<ArchitectureExplorer architecture={data.architecture} />{/key}
