<script lang="ts">
	import { onMount } from 'svelte';
	import QkvWeightPopover from './Popovers/QKVWeightPopover.svelte';
	import { vlmVisiblePositions, vlmSelectedPosition, vlmInspectorStage } from '$lib/vlm/store';
	import { hybridInspector } from '$lib/hybrid/store';
	import { vitVisiblePositions, vitSelectedPosition, vitInspectorStage } from '$lib/vit/store';
	import { t5Inspector } from '$lib/seq2seq/store';
	import { attentionInspector } from '$lib/inspectors';
	import RecordedLMHead from './moe/RecordedLMHead.svelte';
	import { weightPopover, tooltip, moeModel, moeTokenIndex } from '~/store';
	import { fade } from 'svelte/transition';
	import AttentionWeightPopover from './Popovers/AttentionWeightPopover.svelte';
	import MLPWeightPopover from './Popovers/MLPWeightPopover.svelte';
	import MLPDownWeightPopover from './Popovers/MLPDownWeightPopover.svelte';

	import LogitWeightPopover from './Popovers/LogitWeightPopover.svelte';
	import { ArrowUpRightDownLeftOutline } from 'flowbite-svelte-icons';
	import {
		highlightAttentionPath,
		highlightLogitPath,
		highlightPath,
		removeAttentionPathHighlight,
		removePathHighlight
	} from '~/utils/textbook';

	$: if ($moeModel?.architecture === 'mla' && $weightPopover === 'qkv') {
		attentionInspector.set({ kind: 'mla', stage: 'query' });
		$weightPopover = null;
	}

	$: if ($moeModel?.architecture === 'vit' && $weightPopover) {
		const stages: Record<string, string> = {
			qkv: 'qkv',
			attention: 'attention',
			mlpUp: 'up',
			mlpDown: 'down',
			softmax: 'classifier'
		};
		$vitSelectedPosition = $vitVisiblePositions[$moeTokenIndex] ?? 0;
		vitInspectorStage.set(stages[String($weightPopover)] ?? 'attention');
		$weightPopover = null;
	}
	$: if ($moeModel?.architecture === 'encoder-decoder' && $weightPopover) {
		t5Inspector.set(
			$weightPopover === 'qkv'
				? 'qkv'
				: $weightPopover === 'softmax'
					? 'output'
					: $weightPopover === 'mlpUp'
						? 'ffn-up'
						: $weightPopover === 'mlpDown'
							? 'ffn-down'
							: 'self'
		);
		$weightPopover = null;
	}
	$: if ($moeModel?.architecture === 'vision-language' && $weightPopover) {
		const stages: Record<string, string> = {
			qkv: 'qkv',
			attention: 'attention',
			mlpUp: 'up',
			mlpDown: 'down',
			softmax: 'output'
		};
		$vlmSelectedPosition = $vlmVisiblePositions[$moeTokenIndex] ?? 0;
		vlmInspectorStage.set(stages[String($weightPopover)] ?? 'attention');
		$weightPopover = null;
	}
	$: if ($moeModel?.architecture === 'hybrid' && $weightPopover) {
		hybridInspector.set(
			$weightPopover === 'qkv'
				? 'qkv'
				: $weightPopover === 'softmax'
					? 'classifier'
					: ['mlpUp', 'mlpDown'].includes(String($weightPopover))
						? $weightPopover === 'mlpUp'
							? 'ffn-up'
							: 'ffn-down'
						: 'attention'
		);
		$weightPopover = null;
	}
	let resizeObserver: ResizeObserver;
	let qkvPos = { left: 0, top: 0 };
	let mlpPos = { left: 0, top: 0 };
	let mlpDownPos = { left: 0, top: 0 };
	let attentionPos = { left: 0, top: 0 };
	let softmaxPos = { left: 0, top: 0 };

	let popoverEl: HTMLElement = null;

	function handleOutsideClick(e) {
		if (
			!!$weightPopover &&
			!popoverEl?.contains(e.target) &&
			!(e.target instanceof Element && e.target.closest('.latent-inspector'))
		) {
			weightPopover.set(null);
		}
	}
	// add global event
	onMount(() => {
		document.querySelector('.main-section').addEventListener('click', handleOutsideClick);

		return () => {
			document.querySelector('.main-section')?.removeEventListener('click', handleOutsideClick);
		};
	});

	//set popover pos
	onMount(() => {
		const setPosition = () => {
			const scrollLeft = window.scrollX;
			const topbarHeight = document.querySelector('.top-bar')?.offsetHeight;

			const embedding = document.querySelector('.step.qkv .content');
			const mlp = document.querySelector('.step.mlp .content');
			const mlpDown = document.querySelector('.step.mlp .second-layer');
			const attention = document.querySelector('.step.attention .content .multi-head');
			const softmax = document.querySelector('.step.transformer-blocks .content');

			const embeddingRect = embedding?.getBoundingClientRect();
			const mlpRect = mlp?.getBoundingClientRect();
			const mlpDownRect = mlpDown?.getBoundingClientRect();
			const attentionRect = attention?.getBoundingClientRect();
			const softmaxRect = softmax?.getBoundingClientRect();

			qkvPos = { left: embeddingRect?.right + scrollLeft, top: embeddingRect?.top - topbarHeight };
			mlpPos = { left: mlpRect?.left + scrollLeft, top: mlpRect?.top - topbarHeight };
			mlpDownPos = { left: mlpDownRect?.left + scrollLeft, top: mlpDownRect?.top - topbarHeight };
			attentionPos = {
				left: attentionRect?.right + scrollLeft,
				top: attentionRect?.top + attentionRect?.height / 2 - topbarHeight
			};
			softmaxPos = {
				left: softmaxRect?.left + scrollLeft,
				top: softmaxRect?.top - topbarHeight
			};
		};

		setPosition();

		resizeObserver = new ResizeObserver((entries) => {
			setPosition();
		});
		const elements = document?.querySelectorAll('.resize-watch');
		elements.forEach((el) => resizeObserver.observe(el));

		return () => {
			document.querySelector('.main-section')?.removeEventListener('click', handleOutsideClick);
		};
	});

	//tooltip
	$: isVisible = !!$tooltip && !$weightPopover;

	let x = 0;
	let y = 0;

	function handleMouseMove(e) {
		x = e.clientX + 10;
		y = e.clientY + 10;
	}

	onMount(() => {
		window.addEventListener('mousemove', handleMouseMove);

		const unsubscribe = weightPopover.subscribe((value) => {
			if (!value) {
				removePathHighlight();
				removeAttentionPathHighlight();
				return;
			}

			if (value === 'attention') {
				highlightAttentionPath();
			} else if (value === 'softmax') {
				highlightLogitPath();
			} else {
				highlightPath(value);
			}
		});
		return () => {
			window.removeEventListener('mousemove', handleMouseMove);
			unsubscribe();
		};
	});
</script>

{#if isVisible}
	<div
		class="tooltip-box rounded shadow-lg"
		style="left: {x}px; top: {y}px;"
		in:fade={{ duration: 100 }}
	>
		{$tooltip}
		<ArrowUpRightDownLeftOutline size="sm" />
	</div>
{/if}

{#if $weightPopover === 'qkv' && (!$moeModel?.architecture || $moeModel.architecture === 'moe')}
	<QkvWeightPopover />
{/if}
{#if $weightPopover === 'attention'}
	<div
		bind:this={popoverEl}
		class="attention-weight-popover weight-popover"
		style={`left:${attentionPos.left}px;top:${attentionPos.top}px;`}
		in:fade={{ duration: 300 }}
		role="group"
	>
		<AttentionWeightPopover></AttentionWeightPopover>
	</div>
{/if}
{#if $weightPopover === 'mlpUp'}
	<div
		bind:this={popoverEl}
		class="mlp-weight-popover weight-popover"
		style={`left:${mlpPos.left}px;top:${mlpPos.top}px;`}
		in:fade={{ duration: 300 }}
		role="group"
	>
		<MLPWeightPopover></MLPWeightPopover>
	</div>
{/if}
{#if $weightPopover === 'mlpDown'}
	<div
		bind:this={popoverEl}
		class="mlp-weight-popover weight-popover"
		style={`left:${mlpDownPos.left}px;top:${mlpDownPos.top}px;`}
		in:fade={{ duration: 300 }}
		role="group"
	>
		<MLPDownWeightPopover></MLPDownWeightPopover>
	</div>
{/if}
{#if $weightPopover === 'softmax'}
	<div
		bind:this={popoverEl}
		class="softmax-weight-popover weight-popover"
		style={`left:${softmaxPos.left}px;top:${softmaxPos.top}px;`}
		in:fade={{ duration: 300 }}
		role="group"
	>
		{#if $moeModel?.architecture === 'mla'}<RecordedLMHead />{:else}<LogitWeightPopover />{/if}
	</div>
{/if}

<style lang="scss">
	.weight-popover {
		z-index: $POPOVER_INDEX;
		position: absolute;
		width: max-content !important;
	}
	.qkv-weight-popover {
		transform: translateX(1rem);
	}
	.attention-weight-popover {
		transform: translate(1rem, -50%);
	}
	.mlp-weight-popover {
		transform: translateX(calc(-100% - 0.5rem));
	}
	.softmax-weight-popover {
		transform: translateX(calc(-100% + 1rem));
	}
	.tooltip-box {
		z-index: $POPOVER_INDEX;

		background-color: white;
		position: fixed;
		border: 1px solid theme('colors.gray.200');
		color: theme('colors.gray.600');
		padding: 0.2rem 0.4rem;
		border-radius: 0.4rem;
		pointer-events: none;
		white-space: nowrap;
		font-size: 1rem;
		font-weight: 300;

		display: flex;
		align-items: center;
		gap: 2px;
	}
	:global(.formula) {
		height: 4rem;
		position: relative;
	}
	:global(.formula > div) {
		position: absolute;
		width: 100%;
		top: 0;
	}
	:global(.first-row) {
		transform: translateY(100%);
	}
	:global(.katex-display) {
		margin: 0 !important;
	}
</style>
