<script lang="ts">
	import { AngleLeftOutline, AngleRightOutline } from 'flowbite-svelte-icons';
	import { onMount, tick } from 'svelte';
	import {
		blockIdx,
		rootRem,
		modelMeta,
		isBoundingBoxActive,
		expandedBlock,
		weightPopover,
		isOnAnimation,
		userId
	} from '~/store';
	import { textPages } from '~/utils/textbookPages';
	import TextbookTooltip from './common/TextbookTooltip.svelte';

	let resizeObserver: ResizeObserver;
	let boundingPos = { left: 0, top: 0, width: 0 };

	let mounted = false;

	const setPosition = async () => {
		await tick();
		if (!mounted) return;
		const container = document.querySelector('.nodes');
		const embedding = document.querySelector(
			'.block-steps.main .qkv .content .block-start-column'
		);
		const block =
			$blockIdx === $modelMeta.layer_num - 1
				? document.querySelector('.step.transformer-blocks .content .column.final')
				: document.querySelector('.step.transformer-blocks .content');

		if (!container || !embedding || !block) return;
		const containerRect = container.getBoundingClientRect();
		const embeddingRect = embedding.getBoundingClientRect();
		const blockRect = block.getBoundingClientRect();

		boundingPos = {
			left: embeddingRect.left - containerRect.left + rootRem,
			top: embeddingRect.top - containerRect.top,
			width: blockRect.left - embeddingRect.left - rootRem
		};
	};

	$: if (mounted && $blockIdx >= 0) setPosition();

	onMount(() => {
		mounted = true;
		setPosition();

		resizeObserver = new ResizeObserver(() => {
			setPosition();
		});
		const elements = document?.querySelectorAll('.resize-watch');
		elements.forEach((el) => resizeObserver.observe(el));
		return () => {
			mounted = false;
			resizeObserver.disconnect();
		};
	});

	const onClickNext = (e) => {
		e.stopPropagation();
		textPages.find((page) => page.id === 'blocks')?.complete();

		$blockIdx = Math.min($blockIdx + 1, $modelMeta.layer_num - 1);

		window.dataLayer?.push({
			event: `pagination-transformer-block-next`,
			page_num: $blockIdx,
			pagination_name: 'transformer-block',
			user_id: $userId
		});
	};
	const onClickPrev = (e) => {
		e.stopPropagation();
		textPages.find((page) => page.id === 'blocks')?.complete();

		$blockIdx = Math.max($blockIdx - 1, 0);

		window.dataLayer?.push({
			event: `pagination-transformer-block-prev`,
			page_num: $blockIdx,
			pagination_name: 'transformer-block',
			user_id: $userId
		});
	};
</script>

<div
	class="transformer-bounding"
	class:active={$isBoundingBoxActive}
	style={`left:${boundingPos.left}px;top:${boundingPos.top - rootRem * 3.5}px;width:${boundingPos.width}px;`}
></div>
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div
	class="transformer-bounding-title"
	data-click="transformer-bounding-title"
	class:deactive={!!$weightPopover}
	class:hide={!!$expandedBlock.id}
	on:mouseenter={() => {
		isBoundingBoxActive.set(true);
	}}
	on:mouseleave={() => {
		isBoundingBoxActive.set(false);
	}}
	class:active={$isBoundingBoxActive}
>
	<TextbookTooltip id="blocks">
		<span class="title-text">Transformer Block {$blockIdx + 1}</span>
	</TextbookTooltip>
	<button
		aria-label="Previous transformer block"
		data-click="transformer-block-prev-btn"
		on:click={onClickPrev}
		disabled={$isOnAnimation || $blockIdx === 0}
		><AngleLeftOutline size="sm" /></button
	>
	<button
		aria-label="Next transformer block"
		data-click="transformer-block-next-btn"
		on:click={onClickNext}
		disabled={$isOnAnimation || $blockIdx === $modelMeta.layer_num - 1}
		><AngleRightOutline size="sm" /></button
	>
</div>

<style lang="scss">
	.transformer-bounding-title {
		&.deactive {
			pointer-events: none;
		}
		position: absolute;
		left: 13rem;
		top: 0.4rem;
		z-index: $BOUNDING_BOX_INDEX;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.4rem;
		color: theme('colors.gray.700');
		background: white;
		transition: opacity 0.2s;

		&.hide {
			opacity: 0;
		}

		.title-text {
			width: 10rem;
			white-space: nowrap;
			text-align: center;
			font-weight: 600;
		}

		button {
			display: grid;
			place-items: center;
			width: 2rem;
			height: 2rem;
			border-radius: 0.3rem;
			padding: 0.1rem;
			border: 1px solid theme('colors.purple.300');
			color: theme('colors.purple.700');
			background: theme('colors.purple.50');
			box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
		}
		button:hover:not(:disabled) {
			background: theme('colors.purple.100');
		}
		button:focus-visible,
		:global(.textbook-tooltip:focus-visible) {
			outline: 2px solid theme('colors.purple.500');
			outline-offset: 3px;
		}
		button:disabled {
			color: theme('colors.gray.400');
			border: 1px solid theme('colors.gray.300');
			background: theme('colors.gray.50');
		}

		&.active {
			color: theme('colors.blue.600');

			button {
				border: 1px solid theme('colors.blue.500');
				color: theme('colors.blue.600');
			}
			button:disabled {
				border: 1px solid theme('colors.gray.300');
				color: theme('colors.gray.400');
			}
		}
	}
	.transformer-bounding {
		position: absolute;
		opacity: 1;
		z-index: $BOUNDING_BOX_INDEX;
		pointer-events: none;
		height: calc(95% - 3.5rem);
		border-radius: 1rem;
		transition: all 0.2s;

		&.active {
			border: 2px dashed theme('colors.blue.400');
		}
	}
</style>
