<script lang="ts">
	import { base } from '$app/paths';
	import { tokens, moeExample, moeTokenIndex } from '~/store';
	import {
		vlmRecording,
		vlmPhase,
		vlmVisiblePositions,
		vlmSelectedPosition,
		vlmInspectorStage
	} from '$lib/vlm/store';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	export let className: string | undefined = undefined;
	function inspect(index: number) {
		moeTokenIndex.set(index);
		vlmSelectedPosition.set($vlmVisiblePositions[index]);
		vlmInspectorStage.set('embedding');
	}
</script>

<div class={`embedding ${className ?? ''}`}>
	<div class="title">
		<button on:click={() => vlmInspectorStage.set('embedding')}
			>{$vlmPhase === 'vision' ? 'Image patches ⊕ position' : 'Image + text embeddings'}</button
		>
	</div>
	<div class="content relative">
		<div class="token-column resize-watch">
			<div class="column token-string">
				{#each $tokens as token, index}<div class="cell">
						<button
							class="label"
							class:image-token={$vlmPhase === 'language' &&
								$vlmRecording?.document.imageTokenPositions.includes($vlmVisiblePositions[index])}
							on:click={() => inspect(index)}>{token}</button
						>
					</div>{/each}
			</div>
		</div>
		<div class="vector-column block-start-column">
			<div class="column vectors embedding-column">
				{#each $tokens as token, index}<button
						class="vector bg-gray-300"
						aria-label={`Inspect embedding for ${token}`}
						on:click={() => inspect(index)}
						><VectorCanvas data={$moeExample?.embeddingPreview[index]} /></button
					>{/each}
			</div>
		</div>
	</div>
</div>

<style>
	.content {
		display: flex;
		gap: 12px;
	}
	.title button {
		font-size: 12px;
		white-space: nowrap;
	}
	.image-token {
		color: #9064b0;
		background: #f3e9fb;
	}
	.label {
		font-size: 10px;
		max-width: 85px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.vector {
		border: 0;
		padding: 0;
	}
</style>
