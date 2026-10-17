<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { attentionHeadIdx } from '~/store';

	export let headIndex: number | undefined = undefined;
	export let queryHeads = 16;
	export let keyValueHeads = 8;
	export let headSize = 64;

	const dispatch = createEventDispatcher<{
		inspect: { queryHead: number; keyValueHead: number };
	}>();
	const bankX = 80;
	const bankWidth = 176;

	$: selectedHead = Math.max(0, Math.min(queryHeads - 1, headIndex ?? $attentionHeadIdx));
	$: queriesPerKeyValueHead = queryHeads / keyValueHeads;
	$: sharedHead = Math.floor(selectedHead / queriesPerKeyValueHead);
	$: firstSharedQuery = sharedHead * queriesPerKeyValueHead;
	$: sharedQueries = Array.from({ length: queriesPerKeyValueHead }, (_, i) => firstSharedQuery + i);
	$: sharedQueryLabel = sharedQueries.map((i) => i + 1).join(' & ');
	$: querySlots = Array.from({ length: queryHeads }, (_, index) => ({
		index,
		x: bankX + (index * bankWidth) / queryHeads,
		width: bankWidth / queryHeads - 2
	}));
	$: keyValueSlots = Array.from({ length: keyValueHeads }, (_, index) => ({
		index,
		x: bankX + (index * bankWidth) / queryHeads,
		width: bankWidth / queryHeads - 2
	}));
	$: keyValueBankEnd = bankX + (keyValueHeads * bankWidth) / queryHeads - 2;
	$: pairStart = bankX + (firstSharedQuery * bankWidth) / queryHeads;
	$: pairEnd = pairStart + bankWidth / keyValueHeads - 2;
</script>

<div class="head-flow">
	<div class="flow-heading">
		<span>Split into heads <span class="head-width">· {headSize}D each</span></span>
		<button
			type="button"
			on:click={() => dispatch('inspect', { queryHead: selectedHead, keyValueHead: sharedHead })}
			aria-label={`Inspect rotary position embedding for query head ${selectedHead + 1} and shared key head ${sharedHead + 1}`}
			>Inspect RoPE <span aria-hidden="true">↗</span></button
		>
	</div>
	<svg
		viewBox="0 0 480 116"
		role="img"
		aria-label={`Query splits into ${queryHeads} heads of ${headSize} dimensions; key and value each split into ${keyValueHeads} heads of ${headSize} dimensions. Query heads ${sharedQueryLabel} share key and value head ${sharedHead + 1}. RoPE rotates query and key, preserving their dimensions. Value bypasses RoPE.`}
	>
		<g class="query-row">
			<text x="2" y="31" class="projection-letter">Q</text>
			<text x="24" y="31" class="projection-size">{queryHeads * headSize}D</text>
			<path d="M61 27 H72 M69 24 L72 27 L69 30" class="trace" />
			<text x={bankX} y="13" class="split-label">{queryHeads} × {headSize}D</text>
			{#each querySlots as slot}
				<rect
					x={slot.x}
					y="20"
					width={slot.width}
					height="14"
					rx="1.5"
					class="head-slot"
					class:shared={sharedQueries.includes(slot.index)}
					class:selected={selectedHead === slot.index}
				>
					<title>Query head {slot.index + 1}: {headSize} coordinates</title>
				</rect>
			{/each}
			<path d={`M${pairStart} 36 V39 H${pairEnd} V36`} class="shared-bracket" />
			<path d="M261 27 H271 M268 24 L271 27 L268 30" class="trace" />
			<text x="279" y="31" class="selected-label">Q head {selectedHead + 1}</text>
			<path d="M333 27 H345 M342 24 L345 27 L342 30" class="trace" />
			<rect x="351" y="16" width="54" height="22" rx="3" class="rope-box" />
			<text x="378" y="31" text-anchor="middle" class="rope-label">RoPE</text>
			<path d="M410 27 H432 M429 24 L432 27 L429 30" class="trace" />
			<text x="440" y="31" class="output-size">{headSize}D</text>
		</g>
		<g class="key-row">
			<text x="2" y="67" class="projection-letter">K</text>
			<text x="24" y="67" class="projection-size">{keyValueHeads * headSize}D</text>
			<path d="M61 63 H72 M69 60 L72 63 L69 66" class="trace" />
			<text x={bankX} y="49" class="split-label">{keyValueHeads} × {headSize}D</text>
			{#each keyValueSlots as slot}
				<rect
					x={slot.x}
					y="56"
					width={slot.width}
					height="14"
					rx="1.5"
					class="head-slot"
					class:selected={sharedHead === slot.index}
				>
					<title
						>Key head {slot.index + 1}: {headSize} coordinates, shared by {queriesPerKeyValueHead} query
						heads</title
					>
				</rect>
			{/each}
			<path d={`M${keyValueBankEnd + 7} 63 H271 M268 60 L271 63 L268 66`} class="trace" />
			<text x="279" y="67" class="selected-label">K head {sharedHead + 1}</text>
			<path d="M333 63 H345 M342 60 L345 63 L342 66" class="trace" />
			<rect x="351" y="52" width="54" height="22" rx="3" class="rope-box" />
			<text x="378" y="67" text-anchor="middle" class="rope-label">RoPE</text>
			<path d="M410 63 H432 M429 60 L432 63 L429 66" class="trace" />
			<text x="440" y="67" class="output-size">{headSize}D</text>
		</g>
		<g class="value-row">
			<text x="2" y="103" class="projection-letter">V</text>
			<text x="24" y="103" class="projection-size">{keyValueHeads * headSize}D</text>
			<path d="M61 99 H72 M69 96 L72 99 L69 102" class="trace" />
			<text x={bankX} y="85" class="split-label">{keyValueHeads} × {headSize}D</text>
			{#each keyValueSlots as slot}
				<rect
					x={slot.x}
					y="92"
					width={slot.width}
					height="14"
					rx="1.5"
					class="head-slot"
					class:selected={sharedHead === slot.index}
				>
					<title
						>Value head {slot.index + 1}: {headSize} coordinates, shared by {queriesPerKeyValueHead}
						query heads</title
					>
				</rect>
			{/each}
			<path d={`M${keyValueBankEnd + 7} 99 H271 M268 96 L271 99 L268 102`} class="trace" />
			<text x="279" y="103" class="selected-label">V head {sharedHead + 1}</text>
			<path d="M333 99 H432 M429 96 L432 99 L429 102" class="trace value-bypass" />
			<rect x="351" y="90" width="54" height="18" fill="white" />
			<text x="378" y="103" text-anchor="middle" class="bypass-label">bypass</text>
			<text x="440" y="103" class="output-size">{headSize}D</text>
		</g>
	</svg>
	<div class="flow-caption">
		<span class="sharing"
			><span aria-hidden="true" class="shared-dot"></span>Q heads {sharedQueryLabel} share K/V head {sharedHead +
				1}</span
		>
		<span class="dimension-note">RoPE keeps {headSize}D</span>
	</div>
</div>

<style>
	.head-flow {
		width: 100%;
		min-width: 380px;
		padding: 5px 0 7px;
		font-family: Arial, Helvetica, sans-serif;
		color: #7e6a8b;
		box-sizing: border-box;
	}
	.flow-heading,
	.flow-caption {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		font-size: 10px;
		line-height: 1.4;
	}
	.flow-heading {
		margin-bottom: 3px;
	}
	.head-width {
		color: #96859f;
	}
	.flow-heading button {
		border: 1px solid #e5d8ee;
		border-radius: 3px;
		padding: 3px 6px;
		background: #fcf9fe;
		color: #87669e;
		font: inherit;
		cursor: pointer;
	}
	.flow-heading button:hover {
		background: #f4eaf9;
		border-color: #cbb0de;
	}
	.flow-heading button:focus-visible {
		outline: 2px solid #b28bcb;
		outline-offset: 2px;
	}
	.head-flow svg {
		display: block;
		width: 100%;
		height: auto;
		overflow: visible;
	}
	.query-row {
		--trace: #80afe9;
		--slot: #d8e8fa;
		--selected: #91bbee;
		--ink: #6c92c4;
	}
	.key-row {
		--trace: #e9a0a5;
		--slot: #f9dfe1;
		--selected: #eeabb3;
		--ink: #b98088;
	}
	.value-row {
		--trace: #85c5a0;
		--slot: #dcefe3;
		--selected: #9ad2ae;
		--ink: #729f84;
	}
	.projection-letter {
		font-size: 14px;
		font-weight: 600;
		fill: var(--ink);
	}
	.projection-size,
	.output-size,
	.selected-label {
		font-size: 11px;
		fill: var(--ink);
	}
	.split-label {
		font-size: 10px;
		fill: #8a7b94;
	}
	.trace {
		fill: none;
		stroke: var(--trace);
		stroke-width: 1.25;
		stroke-linejoin: round;
		stroke-linecap: round;
	}
	.head-slot {
		fill: var(--slot);
		stroke: white;
		stroke-width: 0.5;
	}
	.head-slot.shared {
		fill: #bed7f6;
	}
	.head-slot.selected {
		fill: var(--selected);
		stroke: #a47dbd;
		stroke-width: 1;
	}
	.query-row .head-slot.selected {
		stroke: #729bd0;
	}
	.shared-bracket {
		fill: none;
		stroke: #b08ac9;
		stroke-width: 1;
	}
	.rope-box {
		fill: #fbf7fd;
		stroke: #dfcee9;
		stroke-width: 1;
	}
	.rope-label {
		fill: #9472ab;
		font-size: 11px;
	}
	.bypass-label {
		fill: #8ca694;
		font-size: 10px;
	}
	.flow-caption {
		margin-top: 2px;
		font-size: 9px;
		color: #91769f;
	}
	.sharing {
		display: flex;
		align-items: center;
		gap: 5px;
	}
	.shared-dot {
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: #b591ca;
		flex-shrink: 0;
	}
	.dimension-note {
		color: #9a8da2;
	}
</style>
