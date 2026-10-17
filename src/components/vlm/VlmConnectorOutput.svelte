<script lang="ts">
	import { vlmInspectorStage } from '$lib/vlm/store';
	import { selectVlmPhase } from '$lib/vlm/adapter';
	export let className: string | undefined = undefined;
</script>

<div class={`softmax ${className ?? ''}`}>
	<div class="title">Vision → language</div>
	<div class="content">
		<button on:click={() => vlmInspectorStage.set('connector')}
			><strong>256 image patches</strong><span>4 × 4 pixel shuffle</span><strong
				>16 × 12,288D</strong
			><span>Learned projection</span><strong>16 × 576D visual tokens</strong></button
		><button class="next" on:click={() => selectVlmPhase('language')}
			>Open all 30 language blocks →</button
		><small>The visual embeddings join the text sequence before language attention.</small>
	</div>
</div>

<style>
	.content {
		width: 230px;
		display: flex;
		flex-direction: column;
		gap: 16px;
	}
	.content > button:first-child {
		display: flex;
		flex-direction: column;
		gap: 15px;
		padding: 20px;
		border: 1px solid #dac8e9;
		border-radius: 5px;
		background: #fcf9fe;
		color: #8b64a7;
		font-size: 12px;
	}
	.content span {
		font-size: 10px;
		color: #ad96bd;
	}
	.next {
		color: #9861be;
		font-size: 12px;
	}
	small {
		font-size: 10px;
		line-height: 1.6;
		color: #a08daf;
	}
</style>
