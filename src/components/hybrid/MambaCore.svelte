<script lang="ts">
	import VectorCanvas from '../common/VectorCanvas.svelte';
	import { tokens, moeTokenIndex, blockIdx } from '~/store';
	import { hybridLayer, hybridInspector } from '$lib/hybrid/store';
	import { row, scalar } from '$lib/vit/types';
	export let compact = false;
	let scanHead = 0;
	function inspect(stage: string, token: number) {
		moeTokenIndex.set(token);
		hybridInspector.set(stage);
	}
	$: n = $tokens.length;
	$: headOptions = Array.from({ length: 64 }, (_, i) => i);
</script>

<div class="mamba-core" class:compact>
	<div class="mamba-heading">
		<b>Mamba state-space mixer</b><button on:click={() => hybridInspector.set('scan')}
			>Inspect state scan ↗</button
		>
	</div>
	<div class="mamba-caption">
		RMSNorm → input projection → causal convolution + SiLU → selective scan → gated RMSNorm → output
		projection
	</div>
	<div class="scan-head">
		<label
			>State head <select bind:value={scanHead}
				>{#each headOptions as head}<option value={head}>{head + 1} of 64</option>{/each}</select
			></label
		><span>64 channels × 128 recurrent states per head</span>
	</div>
	<div class="operation-labels">
		<span>Input<br />2048D</span><span>Conv + SiLU<br />4096D</span><span
			>State sₜ<br />128D / channel</span
		><span>Readout<br />4096D</span><span>Gated norm<br />4096D</span>
	</div>
	<div class="token-pipeline">
		{#each $tokens as token, index}<div class="token-row">
				<span class="token-label">{token}</span>
				<button
					class="vector state-input"
					aria-label={`Input projection ${token}`}
					on:click={() => inspect('projection', index)}
					><VectorCanvas
						active
						colorScale="blue"
						data={$hybridLayer ? row($hybridLayer, 'norm1', index, 8) : undefined}
					/></button
				><span class="arrow">→</span>
				<button
					class="vector"
					aria-label={`Convolution ${token}`}
					on:click={() => inspect('convolution', index)}
					><VectorCanvas
						active
						colorScale="green"
						data={$hybridLayer ? row($hybridLayer, 'x', index * 64 + scanHead, 8) : undefined}
					/></button
				><span class="arrow">→</span>
				<button
					class="state-vector"
					aria-label={`Reconstructed recurrent state ${token}`}
					on:click={() => inspect('scan', index)}
					>{#if $hybridLayer}<VectorCanvas
							active
							colorScale="purple"
							data={row($hybridLayer, 'state', (index * 64 + scanHead) * 4, 8)}
						/>{/if}<small>s{index}</small></button
				><span class="arrow">→</span>
				<button
					class="vector readout"
					aria-label={`Recorded readout ${token}`}
					on:click={() => inspect('scan', index)}
					><VectorCanvas
						active
						colorScale="purple"
						data={$hybridLayer
							? row($hybridLayer, 'scanOutput', index).slice(scanHead * 64, scanHead * 64 + 8)
							: undefined}
					/></button
				><span class="arrow">→</span>
				<button
					class="vector gated"
					aria-label={`Gated normalization ${token}`}
					on:click={() => inspect('gate', index)}
					><VectorCanvas
						active
						colorScale="blue"
						data={$hybridLayer
							? row($hybridLayer, 'gated', index).slice(scanHead * 64, scanHead * 64 + 8)
							: undefined}
					/></button
				>
			</div>{/each}
	</div>
	<div class="state-formula">
		<span>sₜ = exp(ΔₜA) ⊙ sₜ₋₁ + ΔₜBₜxₜ</span><span>yₜ = Cₜsₜ + D xₜ</span>
	</div>
	<p>
		State passes from one token to the next. The selected head’s first 8 values are shown; click any
		operation for its recorded tensors and arithmetic.
	</p>
</div>

<style>
	.mamba-core {
		background: white;
		border: 1px solid #f0e9f5;
		box-shadow: 0 3px 6px #21162d18;
		min-width: 300px;
		padding: 18px 24px;
		position: relative;
		z-index: 30;
		color: #9274a3;
	}
	.mamba-heading {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
		font-size: 13px;
	}
	.mamba-heading button {
		font-size: 10px;
		background: #fcf9fe;
		color: #9369af;
		border: 1px solid #e2d2ed;
		border-radius: 4px;
		padding: 5px 8px;
		cursor: pointer;
	}
	.mamba-caption {
		font-size: 10px;
		color: #af9cba;
		margin: 10px 0 20px;
		line-height: 1.5;
	}
	.scan-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 28px;
		font-size: 10px;
	}
	.scan-head select {
		border: 1px solid #deceed;
		border-radius: 3px;
		padding: 5px 6px;
		background: #fcf9ff;
		color: #9369ad;
		font-size: 11px;
	}
	.scan-head span {
		font-size: 9px;
		color: #b09cbb;
	}
	.operation-labels {
		display: grid;
		grid-template-columns: repeat(5, 1fr);
		font-size: 10px;
		text-align: center;
		margin-bottom: 12px;
		color: #9e87b0;
	}
	.token-row {
		display: grid;
		grid-template-columns: 1fr 20px 1fr 20px 1.2fr 20px 1fr 20px 1fr;
		align-items: center;
		position: relative;
		min-height: 32px;
		height: var(--vector-height);
		margin-bottom: 0.5rem;
		text-align: center;
	}
	.token-row button {
		margin: auto;
	}
	.token-label {
		position: absolute;
		right: calc(100% + 8px);
		font-size: 10px;
		color: #817386;
		max-width: 80px;
		overflow: hidden;
	}
	.token-row .vector {
		width: 12px;
		height: 100%;
		min-height: 20px;
		border: 0;
		padding: 0;
		background: #e9e1f4;
		cursor: pointer;
	}
	.state-vector {
		height: 100%;
		min-height: 20px;
		width: 36px;
		position: relative;
		cursor: pointer;
		border: 1px solid #cfb5e2;
		background: #f4ebfa;
		padding: 0;
	}
	.state-vector:before {
		content: '';
		position: absolute;
		left: 50%;
		bottom: 100%;
		height: 0.5rem;
		border-left: 1px solid #bd94d6;
	}
	.token-row:first-child .state-vector:before {
		display: none;
	}
	.state-vector small {
		position: absolute;
		left: calc(100% + 3px);
		top: 40%;
		font-size: 8px;
		color: #b090c7;
	}
	.arrow {
		font-size: 14px;
		color: #ceb8de;
	}
	.state-formula {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		margin-top: 25px;
		padding: 12px;
		background: #fbf8fe;
		border-radius: 5px;
		font:
			11px/1.6 Georgia,
			serif;
	}
	.mamba-core p {
		font-size: 10px;
		color: #ad9abb;
		margin-top: 12px;
		line-height: 1.5;
	}
	.compact {
		padding: 14px 18px;
		min-width: 0;
	}
	.compact .mamba-heading {
		font-size: 11px;
	}
	.compact .mamba-caption {
		font-size: 9px;
		margin-bottom: 12px;
	}
	.compact .scan-head {
		margin-bottom: 14px;
		flex-wrap: wrap;
		gap: 5px;
	}
	.compact .operation-labels {
		font-size: 8px;
	}
	.compact .token-row {
		grid-template-columns: 1fr 10px 1fr 10px 1.2fr 10px 1fr 10px 1fr;
		min-height: 22px;
	}
	.compact .state-formula {
		font-size: 9px;
		display: block;
		margin-top: 12px;
		padding: 8px;
	}
	.compact .state-formula span {
		display: block;
	}
	.compact p {
		font-size: 9px;
	}
	.compact .token-label {
		display: none;
	}
</style>
