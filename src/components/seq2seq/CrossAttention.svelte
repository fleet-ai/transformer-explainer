<script lang="ts">
	import HeadStack from '../HeadStack.svelte';
	import Matrix from '../common/Matrix.svelte';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	import OperationGroup from '../OperationGroup.svelte';
	import {
		attentionHeadIdx,
		blockIdx,
		moeTokenIndex,
		tokens,
		headContentHeight,
		rootRem
	} from '~/store';
	import { t5Example, t5Inspector } from '$lib/seq2seq/store';
	export let className: string | undefined = undefined;
	$: layer = $t5Example?.crossAttention[$blockIdx];
	$: head = $attentionHeadIdx;
	$: sourceTokens = $t5Example?.encoder.example.tokens ?? [];
	$: probabilities = layer?.probabilities[head] ?? [];
	$: cellSize = Math.max(8, Math.min(16, 112 / Math.max(sourceTokens.length, $tokens.length)));
	function inspect() {
		t5Inspector.set('cross');
	}
</script>

<div class={`cross-attention ${className ?? ''}`}>
	<div class="title">Encoder → decoder Attention</div>
	<div class="content relative">
		<div class="cross-card-wrap">
			<HeadStack>
				<div class="cross-flow">
					<span>Q: decoder state · K/V: final encoder memory</span><button on:click={inspect}
						>Inspect cross-attention ↗</button
					>
				</div>
				<div
					class="cross-head-block"
					style={`height:${Math.max($headContentHeight - 110, 320)}px;`}
				>
					<div class="cross-input column">
						<span class="port-label">Self-attn out</span>{#each $tokens as token, index}<div
								class="cell x1-12"
							>
								<div class="vector x1-12" title={`Recorded self-attention output token ${index}`}>
									<VectorCanvas
										colorScale="purple"
										data={$t5Example?.decoder.example.layers[$blockIdx].attentionOutputPreview[
											index
										]}
									/>
								</div>
							</div>{/each}
					</div>
					<div class="cross-residual">
						<OperationGroup type="residual-end" id="embedding-residual" /><OperationGroup
							type="residual-start"
							id="cross-residual"
						/>
					</div>
					<div class="cross-qkv">
						<div class="column cross-key">
							<div class="mini-title key">Key {head + 1}<small>encoder · 64D</small></div>
							{#each sourceTokens as token, index}<div class="cell x1-12">
									<span class="label float">{token.text}</span>
									<div class="vector x1-12">
										<VectorCanvas colorScale="red" data={layer?.keyAfter[head][index]} />
									</div>
								</div>{/each}
						</div>
						<div class="column cross-query">
							<div class="mini-title query">Query {head + 1}<small>decoder · 64D</small></div>
							{#each $tokens as token, index}<div class="cell x1-12">
									<span class="label float">{token}</span>
									<div class="vector x1-12">
										<VectorCanvas colorScale="blue" data={layer?.queryAfter[head][index]} />
									</div>
								</div>{/each}
						</div>
						<div class="column cross-value">
							<div class="mini-title value">Value {head + 1}<small>encoder · 64D</small></div>
							{#each sourceTokens as token, index}<div class="cell x1-12">
									<span class="label float">{token.text}</span>
									<div class="vector x1-12">
										<VectorCanvas colorScale="green" data={layer?.value[head][index]} />
									</div>
								</div>{/each}
						</div>
					</div>
					<button
						class="cross-matrix"
						on:click={inspect}
						aria-label="Inspect rectangular encoder-decoder attention matrix"
						><Matrix
							data={probabilities}
							shape="circle"
							colorScale="purple"
							cellWidth={cellSize}
							cellHeight={cellSize}
							cellGap={2}
							showSize={false}
							onMouseOverCell={() => undefined}
							onMouseOutCell={() => undefined}
							onMouseOutSvg={() => undefined}
							showTooltip={() => undefined}
							highlightRow={undefined}
							highlightCol={undefined}
						/><span>Cross-attention ⊕</span></button
					>
					<div class="column cross-head-out">
						<div class="mini-title out">Out<small>64D / head</small></div>
						{#each $tokens as token, index}<div class="cell x1-12">
								<div class="vector x1-12">
									<VectorCanvas
										colorScale="purple"
										data={layer?.attentionContext?.[head]?.[index]}
									/>
								</div>
							</div>{/each}
					</div>
				</div>
			</HeadStack>
		</div>
	</div>
</div>

<style>
	.cross-attention > .title {
		white-space: nowrap;
	}
	.cross-card-wrap {
		padding: 0 16px;
	}
	.cross-flow {
		position: relative;
		z-index: 30;
		height: 60px;
		padding: 10px 12px;
		font:
			10px/1.4 Arial,
			sans-serif;
		color: #9075a3;
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: center;
		gap: 6px;
	}
	.cross-flow small {
		display: block;
		width: 100%;
		font-size: 9px;
		color: #a591b3;
	}
	.cross-flow button {
		background: #fcf9fe;
		border: 1px solid #e5d8ee;
		border-radius: 3px;
		color: #87669e;
		padding: 3px 6px;
		cursor: pointer;
		font: inherit;
	}
	.cross-head-block {
		display: flex;
		align-items: center;
		justify-content: space-between;
		position: relative;
		min-height: 0;
		height: calc(var(--t5-card-height) - 60px) !important;
		padding: 30px 10px 40px 18px;
	}
	.cross-input {
		position: absolute;
		left: -24px;
		top: 47%;
		z-index: 31;
	}
	.port-label {
		position: absolute;
		bottom: 100%;
		font:
			8px Arial,
			sans-serif;
		color: #ae95bc;
		white-space: nowrap;
		transform: translateX(-30%);
		padding-bottom: 8px;
	}
	.cross-input .vector {
		background: #c4b5fd;
	}
	.cross-residual {
		position: absolute;
		left: -14px;
		top: 47%;
		display: flex;
		z-index: 31;
	}
	.cross-residual :global(.operation-group) {
		width: 5px;
	}
	.cross-qkv {
		position: relative;
		top: 0;
		height: 100%;
		justify-content: space-between;
		display: flex;
		flex-direction: column;
		gap: 20px;
		margin-left: 32px;
	}
	.mini-title {
		position: absolute;
		bottom: calc(100% + 6px);
		left: 50%;
		transform: translateX(-50%);
		white-space: nowrap;
		font-size: 12px;
		text-align: center;
		z-index: 35;
	}
	.mini-title small {
		display: inline;
		margin-left: 5px;
		color: #a6a1ad;
		font:
			8px Arial,
			sans-serif;
	}
	.key {
		color: #e896a1;
	}
	.query {
		color: #81a8dc;
	}
	.value {
		color: #75bc9b;
	}
	.out {
		color: #aa88d5;
	}
	.cross-qkv .label {
		font-size: 10px;
		line-height: 1 !important;
		height: auto !important;
		max-width: 70px;
		color: #83818c;
	}
	.cross-qkv .cell {
		font-size: 8px;
	}
	.cross-matrix {
		border: 0;
		padding: 0;
		margin-left: 10px;
		background: none;
		position: relative;
		z-index: 35;
		cursor: pointer;
		align-self: center;
	}
	.cross-matrix > span {
		position: absolute;
		top: calc(100% + 8px);
		left: 50%;
		transform: translateX(-50%);
		white-space: nowrap;
		font-size: 11px;
		color: #a099aa;
	}
	.cross-head-out {
		margin-left: 18px;
		z-index: 35;
	}
	.cross-head-out .vector {
		background: #c4b5fd;
	}
	.cross-qkv .column {
		gap: 7px;
	}
	.cross-qkv .cell {
		height: 2px;
	}
	.cross-qkv .label {
		line-height: 10px !important;
		top: -4px;
	}
	.cross-card-wrap :global(.head-title) {
		font-size: 13px;
	}
</style>
