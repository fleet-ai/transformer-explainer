<script lang="ts">
	export let showTokens = true;
	export let gap = '12px';
</script>

<div class="token-layout" class:without-tokens={!showTokens} style={`--content-gap:${gap}`}>
	{#if showTokens}<aside class="token-sidebar" aria-label="Token selection">
			<slot name="tokens" />
		</aside>{/if}
	<div class="token-content"><slot /></div>
</div>

<style>
	.token-layout {
		display: grid;
		grid-template-columns: 112px minmax(0, 1fr);
		gap: 14px;
		min-width: 0;
		align-items: start;
	}
	.without-tokens {
		grid-template-columns: minmax(0, 1fr);
	}
	.token-sidebar {
		position: sticky;
		top: 0;
		display: flex;
		flex-direction: column;
		gap: 12px;
		min-height: 0;
		max-height: min(580px, calc(100dvh - 210px));
	}
	.token-content {
		display: flex;
		flex-direction: column;
		gap: var(--content-gap);
		min-width: 0;
	}
	@media (max-width: 560px) {
		.token-layout {
			grid-template-columns: 88px minmax(0, 1fr);
			gap: 8px;
		}
		.without-tokens {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
