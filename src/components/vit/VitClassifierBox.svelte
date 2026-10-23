<script lang="ts">
	import { onMount, tick } from 'svelte';
	import * as d3 from 'd3';
	import { bandRedraw, blockIdx, expandedBlock, tokens, tooltip } from '~/store';
	import { LOGIT } from '~/constants/opacity';
	import { vitInspectorStage, vitRecording } from '$lib/vit/store';

	/** The classifier head sits on the one band that leaves [CLS]: Dense's Probabilities band. */
	let host: HTMLButtonElement;
	let left = 0;
	let top = 0;
	let width = 40;
	let visible = false;
	let hot = false;
	$: classes = $vitRecording?.document.classLabels.length ?? 1000;
	$: hidden = $vitRecording?.document.model.hiddenSize ?? 192;

	function measure() {
		const container = host?.offsetParent ?? document.body;
		const source = document.querySelector('.transformer-blocks .final .vector.last-token');
		const target = document.querySelector('.softmax .content');
		if (!host || !source || !target) return void (visible = false);
		const origin = container.getBoundingClientRect();
		const from = source.getBoundingClientRect();
		const to = target.getBoundingClientRect();
		const gap = to.left - from.right;
		width = Math.max(28, Math.min(44, gap - 8));
		left = from.right - origin.left + (gap - width) / 2;
		top = from.top - origin.top + from.height / 2;
		visible = gap > 24;
	}
	function bands(opacity: number) {
		d3.selectAll('.sankey-top g.softmax path').style('opacity', opacity);
	}
	function enter() {
		hot = true;
		bands(1);
		tooltip.set('Classifier head · click for the multiplication');
	}
	function leave() {
		hot = false;
		bands(LOGIT);
		tooltip.set(null);
	}
	$: if ($bandRedraw >= 0 && $tokens && $blockIdx >= 0 && $expandedBlock) tick().then(measure);
	onMount(() => {
		measure();
		const observer = new ResizeObserver(measure);
		observer.observe(document.body);
		// The band and the box are one operation, so hovering either lights up both.
		const over = (event: Event) => {
			if ((event.target as Element)?.closest?.('.sankey-top g.softmax')) hot = true;
		};
		const out = (event: Event) => {
			if ((event.target as Element)?.closest?.('.sankey-top g.softmax')) hot = false;
		};
		document.addEventListener('mouseover', over);
		document.addEventListener('mouseout', out);
		return () => {
			observer.disconnect();
			document.removeEventListener('mouseover', over);
			document.removeEventListener('mouseout', out);
		};
	});
</script>

<button
	bind:this={host}
	type="button"
	class="classifier-box"
	class:hot
	class:visible
	style={`left:${left}px;top:${top}px;width:${width}px;`}
	title={`Classifier head · LayerNorm, then Wcls ${hidden} → ${classes}, on [CLS] only`}
	aria-label={`Open the classifier head: Wcls, ${hidden} to ${classes}, applied to [CLS] only`}
	on:click|stopPropagation={() => vitInspectorStage.set('classifier')}
	on:mouseenter={enter}
	on:mouseleave={leave}
	on:focus={enter}
	on:blur={leave}>Wcls</button
>

<style>
	.classifier-box {
		position: absolute;
		z-index: 120;
		height: 22px;
		transform: translateY(-50%);
		padding: 0;
		border: 1.5px solid #cbb4df;
		border-radius: 4px;
		background: #faf7fd;
		color: #694684;
		font:
			10px Arial,
			sans-serif;
		cursor: pointer;
		visibility: hidden;
		user-select: none;
	}
	.classifier-box.visible {
		visibility: visible;
	}
	.classifier-box.hot {
		background: #f1e9f9;
		border-color: #b9a0d4;
	}
	.classifier-box:focus-visible {
		outline: 2px solid #9362c1;
		outline-offset: 1px;
	}
</style>
