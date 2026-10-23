<script lang="ts">
	import QKV from '../QKV.svelte';
	import OperationGroup from '../OperationGroup.svelte';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	import { Tooltip } from 'flowbite-svelte';
	import {
		tokens,
		blockIdx,
		moeExample,
		moeTokenIndex,
		vectorHeight,
		attentionHeadIdx
	} from '~/store';
	import {
		hybridRecording,
		hybridInspector,
		hybridMambaView,
		hybridMambaOperations,
		hybridStage
	} from '$lib/hybrid/store';
	export let className: string | undefined = undefined;

	/** The Mamba in_proj bank, drawn like the Dense QKV bank: one stack per token. */
	$: model = $hybridRecording?.document.model;
	$: inner = (model?.mambaHeads ?? 64) * (model?.mambaHeadSize ?? 64);
	$: state = model?.stateSize ?? 128;
	$: heads = model?.mambaHeads ?? 64;
	$: ops = $hybridMambaOperations;
	$: isHybridBlock = $hybridRecording?.document.layers[$blockIdx]?.type === 'hybrid';
	// Same total height as the QKV bank (3 vectors); B, C and dt are too thin to see at scale,
	// so they keep a floor and the labels carry the sizes.
	const THIN = 3;
	$: bank = 3 * $vectorHeight;
	$: wide = (bank - 3 * THIN) / 2;
	$: parts = [
		{
			id: 'mamba-z',
			letter: 'z',
			size: inner,
			height: wide,
			color: 'bg-purple-200',
			scale: 'purple'
		},
		{ id: 'value', letter: 'x', size: inner, height: wide, color: 'bg-green-300', scale: 'green' },
		{ id: 'key', letter: 'B', size: state, height: THIN, color: 'bg-red-300', scale: 'red' },
		{ id: 'query', letter: 'C', size: state, height: THIN, color: 'bg-blue-300', scale: 'blue' },
		{ id: 'mamba-dt', letter: 'Δ', size: heads, height: THIN, color: 'bg-amber-300', scale: 'gray' }
	];
	/** Where the selected head reads each part: B and C are shared by every head. */
	$: slice = (id: string) =>
		id === 'key' || id === 'query'
			? { top: 0, height: 100 }
			: id === 'mamba-z'
				? null
				: { top: (100 * $attentionHeadIdx) / heads, height: 100 / heads };
	function data(id: string, index: number) {
		if (!ops) return undefined;
		const xbc = ops.xbc[index];
		if (id === 'mamba-z') return ops.z[index];
		if (id === 'value') return xbc.slice(0, inner);
		if (id === 'key') return xbc.slice(inner, inner + state);
		if (id === 'query') return xbc.slice(inner + state);
		return ops.dtRaw[index];
	}
	function inspect(index: number) {
		moeTokenIndex.set(index);
		hybridInspector.set('in-proj');
	}
</script>

{#if !$hybridMambaView}<div class="hybrid-qkv">
		<QKV {className} />
		<span class="concat-note">[x ; e₀] → RMSNorm · 4096</span>
	</div>{:else}<div
		class={`qkv mamba-bank expert-projections ${className ?? ''}`}
		data-click="qkv-step"
	>
		<div class="content relative">
			<div
				class="vector-column block-start-column relative flex"
				class:initial-column={$blockIdx === 0}
			>
				<div class="column vectors embedding-column">
					{#each $tokens as token, index}<div
							class={`vector ${$blockIdx !== 0 ? 'bg-blue-200' : 'bg-gray-300'}`}
							class:last={index === $tokens.length - 1}
						>
							<VectorCanvas
								colorScale={$blockIdx !== 0 ? 'blue' : 'gray'}
								data={$moeExample?.layers[$blockIdx]?.residualInputPreview?.[index]}
							/>
						</div>{/each}
				</div>
				<div class="operations flex">
					<OperationGroup type="residual-start" id="embedding-residual" />
					<OperationGroup type="ln" id="embedding-ln" />
				</div>
				{#if isHybridBlock}<button
						class="plus-t"
						title="The shared transformer output t is added to x before this norm"
						on:click={() => hybridStage.set('transformer')}>+ t</button
					>{/if}
			</div>
			<div class="column qkv-column">
				{#each $tokens as token, index}<div
						class="qkv-weighted vector flex flex-col"
						class:last={index === $tokens.length - 1}
						style={`height:${bank}px`}
					>
						{#each parts as part}{@const head = slice(part.id)}
							<div
								class={`sub-vector ${part.id} relative flex flex-col ${part.color}`}
								style={`height:${part.height}px`}
							>
								<VectorCanvas colorScale={part.scale} data={data(part.id, index)} />
								{#if head}<div
										class="sub-vector head1 absolute"
										style={`top:${head.top}%;height:${head.height}%;`}
									></div>{/if}
								{#if part.height > 8}<div class="sub-vector head-rest">
										<span>{part.letter}</span>
									</div>{/if}
							</div>{/each}
						<button
							type="button"
							class="projection-bank-target"
							aria-label={`Inspect the Mamba input projection for token ${index}: ${token}`}
							on:click|stopPropagation={() => inspect(index)}
						></button>
					</div>{/each}
				<div class="bank-label">Win<br />2048 → {2 * inner + 2 * state + heads}</div>
				<div class="thin-letters" aria-hidden="true">
					<span class="b">B</span><span class="c">C</span><span class="d">Δ</span>
				</div>
				<Tooltip class="popover" triggeredBy={'.mamba-bank .qkv-column .vector'} placement="right"
					>Mamba in_proj, in its real order: z {inner}, x {inner}, B {state}, C {state}, dt {heads}.
					Click to multiply.</Tooltip
				>
			</div>
		</div>
	</div>{/if}

<style lang="scss">
	.hybrid-qkv {
		position: relative;
		height: 100%;
		min-width: 0;
	}
	.hybrid-qkv :global(> .qkv) {
		height: 100%;
	}
	.concat-note {
		position: absolute;
		top: calc(var(--title-height) + var(--vector-height) * 18 + 2.5rem + 6px);
		right: -12px;
		font-size: 8.5px;
		color: #a898ba;
		white-space: nowrap;
		z-index: 40;
	}
	.mamba-bank {
		position: relative;
	}
	.projection-bank-target {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		border: 0;
		background: transparent;
		padding: 0;
		z-index: 40;
		cursor: pointer;
	}
	.projection-bank-target:focus-visible {
		outline: 2px solid #b28bcb;
		outline-offset: 3px;
	}
	.qkv-weighted :global(.sub-vector),
	.qkv-weighted :global(.sub-vector *) {
		pointer-events: none;
	}
	.qkv-weighted {
		cursor: pointer;
	}
	.content {
		display: grid;
		grid-template-columns: 1fr 1fr;
	}
	.vector-column {
		position: relative;
		left: -12px;
	}
	.qkv-column {
		position: relative;
		left: 12px;
		display: flex;
		flex-direction: column;
		align-items: end;
	}
	.sub-vector {
		user-select: none;
		font-size: 0.8rem;
		opacity: 0.85;
		flex-shrink: 0;
		&.mamba-z {
			color: theme('colors.purple.600');
		}
		&.value {
			color: theme('colors.green.600');
		}
	}
	.head1 {
		left: 0;
		width: 100%;
		opacity: 0.9;
	}
	.value .head1 {
		background: theme('colors.green.500');
	}
	.key .head1 {
		background: theme('colors.red.500');
	}
	.query .head1 {
		background: theme('colors.blue.500');
	}
	.mamba-dt .head1 {
		background: theme('colors.amber.500');
	}
	.head-rest {
		height: 100%;
		display: flex;
		justify-content: center;
		align-items: center;
		font-weight: 700;
		text-shadow:
			-1px -1px 0 white,
			1px -1px 0 white,
			-1px 1px 0 white,
			1px 1px 0 white;
		span {
			opacity: 0.55;
		}
	}
	.thin-letters {
		position: absolute;
		top: calc(var(--vector-height) * 3 - 9px);
		right: 14px;
		display: flex;
		flex-direction: column;
		font-size: 7px;
		line-height: 3px;
		font-weight: 700;
		.b {
			color: theme('colors.red.500');
		}
		.c {
			color: theme('colors.blue.500');
		}
		.d {
			color: theme('colors.amber.600');
		}
	}
	.bank-label {
		position: absolute;
		top: calc(100% + 6px);
		right: 0;
		text-align: right;
		line-height: 1.2;
		font-size: 8px;
		color: #a898ba;
		white-space: nowrap;
	}
	.plus-t {
		position: absolute;
		left: -4px;
		top: calc(100% + 8px);
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font-size: 10px;
		padding: 1px 6px;
		cursor: pointer;
		z-index: 40;
		white-space: nowrap;
	}
	.plus-t:hover {
		background: #f1e9f9;
	}
</style>
