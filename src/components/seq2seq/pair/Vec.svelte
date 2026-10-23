<script lang="ts">
	import VectorCanvas from '../../common/VectorCanvas.svelte';
	import { FILL } from './metrics';
	/** One recorded token vector as a thin strip; the captured values show on hover. */
	export let data: number[] | undefined = undefined;
	export let color = 'purple';
	export let w = 12;
	export let h = 10;
	export let hot = false;
	export let title: string | undefined = undefined;
</script>

<div
	class="vec"
	class:hot
	style={`width:${w}px;height:${h}px;background:${FILL[color] ?? FILL.gray}`}
	{title}
>
	<VectorCanvas colorScale={color} {data} />
</div>

<style>
	.vec {
		position: relative;
		overflow: hidden;
		flex: none;
	}
	.vec.hot {
		outline: 1px solid #9b7be0;
		outline-offset: 0;
	}
</style>
