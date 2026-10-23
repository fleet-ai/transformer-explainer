<script lang="ts">
	import HeadStack from '~/components/HeadStack.svelte';
	import VectorCanvas from '../common/VectorCanvas.svelte';
	import MambaMixingMatrix from './MambaMixingMatrix.svelte';
	import { Tooltip } from 'flowbite-svelte';
	import classNames from 'classnames';
	import { setContext } from 'svelte';
	import {
		tokens,
		headContentHeight,
		expandedBlock,
		hoveredMatrixCell,
		attentionHeadIdx,
		moeTokenIndex
	} from '~/store';
	import {
		hybridRecording,
		hybridInspector,
		hybridMambaOperations,
		hybridMambaView
	} from '$lib/hybrid/store';
	export let className: string | undefined = undefined;

	/**
	 * One Mamba-2 SSM head in the Dense head card. Its scan is a causal token-mixing matrix,
	 * so B plays the key's part, C the query's and x the value's (the "SSD" form).
	 */
	setContext('block-id', 'attention');
	$: expanded = $expandedBlock.id === 'attention';
	$: model = $hybridRecording?.document.model;
	$: state = model?.stateSize ?? 128;
	$: headSize = model?.mambaHeadSize ?? 64;
	$: ops = $hybridMambaOperations;
	$: columns = [
		{
			id: 'key',
			name: 'B',
			title: 'input',
			sub: `${state}D · all heads`,
			color: 'bg-red-400',
			scale: 'red',
			data: ops?.B
		},
		{
			id: 'query',
			name: 'C',
			title: 'output',
			sub: `${state}D · all heads`,
			color: 'bg-blue-400',
			scale: 'blue',
			data: ops?.C
		},
		{
			id: 'value',
			name: 'x',
			title: `head ${$attentionHeadIdx + 1}`,
			sub: `${headSize}D · + D·x to y`,
			color: 'bg-green-400',
			scale: 'green',
			data: ops?.x
		}
	];
	function open(stage: string) {
		hybridInspector.set(stage);
	}
</script>

<div
	class={classNames('attention mamba-heads rotary', className, { expanded })}
	data-click="attention-step"
>
	<div class="title">
		<div class="w-max">Mamba2 mixer · selective state space</div>
	</div>
	<div class="content relative">
		<div class="heads">
			{#if $hybridMambaView}<HeadStack>
					<div
						class="head-block flex w-full items-center justify-between px-2"
						style={`height:${Math.max($headContentHeight - 110, 320)}px;min-height:calc(var(--vector-height) * ${($tokens.length * 3) / 16} + ${($tokens.length - 1) * 1.5 + 14}rem);`}
					>
						<div class="qkv flex h-full flex-col justify-center gap-[5rem] pl-[6rem]">
							{#each columns as column}
								<div class={`column ${column.id}`}>
									<div class="head1 title">
										<strong class="role">{column.name}</strong> · {column.title}<small
											>{column.sub}</small
										>
									</div>
									<button
										class="rotary-gate"
										on:click={() => open(`conv:${column.id}`)}
										aria-label={`Causal convolution and SiLU on ${column.name}`}
										>conv1d<small>k = 4 · SiLU</small></button
									>
									{#each $tokens as token, index}
										<div
											class={classNames('head1 cell x1-12 text-xs', {
												key: column.id === 'key',
												query: column.id === 'query'
											})}
											class:last={index === $tokens.length - 1}
											class:active={(column.id === 'key' && $hoveredMatrixCell.col === index) ||
												(column.id === 'query' && $hoveredMatrixCell.row === index)}
										>
											<span class="label float">{token}</span>
											<div class="rotary-token-input">
												<span class="rotary-port"></span>
												<span class="rotary-wire"></span>
												<span class="rotary-gate-space"></span>
												<span class="rotary-wire outgoing">→</span>
											</div>
											<div class={`vector x1-12 ${column.color}`}>
												<VectorCanvas colorScale={column.scale} data={column.data?.[index]} />
											</div>
										</div>
									{/each}
								</div>
							{/each}
							<Tooltip
								class="popover"
								triggeredBy={'.mamba-heads .column.key .cell'}
								placement="right">B after the convolution, {state}D, shared by all heads</Tooltip
							>
							<Tooltip
								class="popover"
								triggeredBy={'.mamba-heads .column.query .cell'}
								placement="right">C after the convolution, {state}D, shared by all heads</Tooltip
							>
						</div>
						<div class="resize-watch attention-matrix flex"><MambaMixingMatrix /></div>
						<div class="head-out mx-[2rem]">
							<div class="column out">
								<div class="head1 title">
									<strong class="role">y</strong><small
										>head {$attentionHeadIdx + 1} · {headSize}D</small
									>
								</div>
								{#each $tokens as token, index}
									<div class="head1 cell x1-12" class:last={index === $tokens.length - 1}>
										<button
											class="vector x1-12 bg-purple-500"
											class:chosen={index === $moeTokenIndex}
											aria-label={`Scan output of head ${$attentionHeadIdx + 1} for ${token}`}
											on:click={() => {
												moeTokenIndex.set(index);
												open('scan');
											}}
										>
											<VectorCanvas colorScale="purple" data={ops?.y[index]} />
										</button>
									</div>
								{/each}
							</div>
						</div>
					</div>
				</HeadStack>{/if}
		</div>
	</div>
</div>

<style lang="scss">
	.role {
		font-size: 1.2rem;
		font-weight: 700;
	}
	.mamba-heads {
		user-select: none;
		-webkit-user-select: none;
	}
	.rotary .head-block > .qkv {
		padding-left: 13rem;
	}
	.rotary .column .label.float {
		left: -138px;
	}
	.rotary-token-input {
		position: absolute;
		left: -128px;
		top: 0;
		height: 100%;
		width: 128px;
		display: flex;
		align-items: center;
		color: #80afe9;
		z-index: 35;
	}
	.key .rotary-token-input {
		color: #e9a0a5;
	}
	.value .rotary-token-input {
		color: #86d6a0;
	}
	.rotary-port {
		width: 2px;
		height: 100%;
		flex: 0 0 2px;
		background: currentColor;
	}
	.rotary-gate {
		position: absolute;
		left: -114px;
		top: 50%;
		transform: translateY(-50%);
		height: calc(100% + 16px);
		min-height: 36px;
		width: 76px;
		border: 1.5px solid #b48acb;
		border-radius: 3px;
		background: #f3e9fa;
		color: #755092;
		font-weight: 600;
		z-index: 36;
		padding: 0;
		font-size: 14px;
		cursor: pointer;
	}
	.rotary-gate:hover {
		background: #eee0f7;
		border-color: #b28bcb;
	}
	.rotary-gate:focus-visible {
		outline: 2px solid #b28bcb;
		outline-offset: 2px;
	}
	.rotary-gate small {
		display: block;
		font-size: 9px;
		font-weight: 400;
		color: #87669e;
	}
	.rotary-gate-space {
		width: 76px;
		flex-shrink: 0;
	}
	.rotary-wire {
		border-top: 1px solid currentColor;
		width: 12px;
	}
	.rotary-wire.outgoing {
		border: 0;
		flex: 1;
		font-size: 18px;
		line-height: 1;
		text-align: center;
	}
	.head-out .vector {
		padding: 0;
		border: 0;
		cursor: pointer;
		display: block;
	}
	.column .title small {
		display: block;
		font-size: 9px;
		line-height: 1.2;
		white-space: nowrap;
		color: #9ca3af;
	}
	.attention-matrix {
		z-index: $COLUMN_TITLE_INDEX;
	}
	.attention {
		&.expanded {
			.title,
			:global(.head-content) {
				z-index: $EXPANDED_CONTENT_INDEX;
			}
			:global(.multi-head .head-card:first-child) {
				z-index: $EXPANDED_CONTENT_INDEX !important;
			}
		}
		.column {
			.label {
				font-size: 0.7rem;
				white-space: nowrap;
				max-width: 6rem;
				overflow: hidden;
				text-overflow: ellipsis;
				color: theme('colors.gray.600');
			}
			.title {
				z-index: $COLUMN_TITLE_INDEX;
				position: absolute;
				top: -2.65rem;
				left: 50%;
				transform: translateX(-50%);
				font-size: 0.9rem;
				white-space: nowrap;
				transition: none;
			}
			&.query .title {
				color: theme('colors.blue.400');
			}
			&.key .title {
				color: theme('colors.red.400');
			}
			&.value .title {
				color: theme('colors.green.500');
			}
			&.out .title {
				color: theme('colors.purple.400');
			}
		}
		.content {
			display: grid;
			grid-template-columns: auto 0;
		}
		.heads {
			padding: 0 7rem 0 8rem;

			.head1.cell {
				.label {
					height: auto;
					line-height: 1;
				}
				&.active {
					&.query .label {
						background-color: theme('colors.blue.100');
						color: theme('colors.blue.700');
						font-size: 1rem;
						z-index: 100;
						padding: 0.2rem;
					}
					&.key .label {
						background-color: theme('colors.red.100');
						color: theme('colors.red.700');
						font-size: 1rem;
						z-index: 100;
						padding: 0.2rem;
					}
				}
			}
		}
	}
</style>
