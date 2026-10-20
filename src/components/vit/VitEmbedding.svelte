<script lang="ts">
	import { base } from '$app/paths';
	import { tokens, moeExample, moeTokenIndex } from '~/store';
	import {
		vitRecording,
		vitVisiblePositions,
		vitSelectedPosition,
		vitInspectorStage
	} from '$lib/vit/store';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	import VitPatchControls from './VitPatchControls.svelte';
	export let className: string | undefined = undefined;
	function inspect(index: number) {
		moeTokenIndex.set(index);
		vitSelectedPosition.set($vitVisiblePositions[index]);
		vitInspectorStage.set('embedding');
	}
</script>

<div class={`embedding ${className ?? ''}`} data-click="embedding-step">
	<div class="title">
		<button on:click={() => vitInspectorStage.set('embedding')}>Image patches ⊕ position</button>
	</div>
	<div class="content relative">
		<div class="patch-inputs">
			<div class="token-column resize-watch">
				<div class="column token-string">
					{#each $tokens as token, index}<div class="cell">
							<button class="patch-label" on:click={() => inspect(index)}
								>{#if $vitVisiblePositions[index] === 0}<span class="cls">CLS</span
									>{:else if $vitRecording}<span
										class="patch"
										style={`background-image:url('${base}/${$vitRecording.document.image.src}');background-position:${((($vitVisiblePositions[index] - 1) % 14) * 100) / 13}% ${(Math.floor(($vitVisiblePositions[index] - 1) / 14) * 100) / 13}%;`}
									></span>{/if}<span>{token}</span></button
							>
						</div>{/each}
				</div>
			</div>
			<div class="vector-column block-start-column">
				<div class="column vectors embedding-column">
					{#each $tokens as _, index}<button
							class="vector bg-gray-300"
							aria-label={`Inspect embedding for ${$tokens[index]}`}
							on:click={() => inspect(index)}
							><VectorCanvas data={$moeExample?.embeddingPreview[index]} /></button
						>{/each}
				</div>
			</div>
		</div>
		<VitPatchControls />
	</div>
</div>

<style>
	.content {
		display: flex;
		flex-direction: column;
	}
	.patch-inputs {
		display: flex;
		gap: 12px;
	}
	.token-column {
		display: flex;
	}
	.patch-label {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 11px;
		justify-content: flex-end;
		height: var(--vector-height);
		color: #4b5563;
		white-space: nowrap;
	}
	.patch,
	.cls {
		display: inline-block;
		width: 20px;
		height: 20px;
		border: 1px solid #dfd6e9;
		border-radius: 2px;
		background-size: 1400%;
		flex-shrink: 0;
	}
	.cls {
		font-size: 8px;
		background: #f2eaf9;
		color: #8f65ad;
		line-height: 18px;
		text-align: center;
	}
	.title button {
		white-space: nowrap;
		font-size: 12px;
	}
</style>
