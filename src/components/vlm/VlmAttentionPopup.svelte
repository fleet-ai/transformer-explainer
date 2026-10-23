<script lang="ts">
	import VlmFormulaPopup from './VlmFormulaPopup.svelte';
	/** One attention head for one query: scaled dot products, softmax, then a weighted sum of V. */
	export let label: string;
	export let token: string;
	export let sequenceLength: number;
	export let headSize = 64;
	export let head = 0;
	export let heads = 12;
	export let keyValueHeads = 12;
	export let causal = false;
	export let rotary = false;
	$: kvHead = Math.floor(head / (heads / keyValueHeads));
	$: grouped = keyValueHeads < heads;
	$: q = rotary ? 'q′' : 'q';
	$: k = rotary ? 'k′' : 'k';
	$: headNote = grouped
		? `Grouped-query attention: ${heads} query heads share ${keyValueHeads} key/value heads, so query head ${head + 1} reads shared head ${kvHead + 1}.`
		: 'Every head has its own keys and values.';
	$: maskNote = `${causal ? 'Text and image positions sit under the same causal mask.' : 'Every patch attends to all patches of the image.'} The weights always sum to 1 over all ${sequenceLength} positions, including the ones the page does not show.`;
</script>

<VlmFormulaPopup
	title={`${label} · Attention · head ${head + 1}`}
	notice={`Q·Kᵀ / √${headSize}${causal ? ' + causal mask' : ''} → softmax → × V · one query row of ${sequenceLength} keys · schematic animation`}
	size={sequenceLength}
	let:progress
	on:close
>
	<div class="equation">
		<div class="tensor">
			<div>Query {head + 1} <span class="token">“{token}”</span></div>
			<div class="cells input" aria-hidden="true">
				{#each Array(32) as _}<i></i>{/each}
			</div>
			<strong>(1, {headSize})</strong>
		</div>
		<span class="operator">×</span>
		<div class="tensor">
			<div>{grouped ? `Shared key ${kvHead + 1}` : `Key ${head + 1}`}ᵀ</div>
			<div class="key-matrix" aria-hidden="true">
				{#each Array(256) as _, i}<i
						class:active={i % 32 === progress.active}
						class:masked={causal && i % 32 > 23}
					></i>{/each}
			</div>
			<strong>({headSize}, {sequenceLength})</strong>
		</div>
		<span class="operator">=</span>
		<div class="tensor">
			<div>Weights a = softmax(·)</div>
			<div class="cells output" aria-hidden="true">
				{#each Array(32) as _, i}<i
						class:done={i < progress.completed}
						class:active={i === progress.active}
						class:masked={causal && i > 23}
					></i>{/each}
			</div>
			<strong>(1, {sequenceLength})</strong>
		</div>
	</div>
	<div class="formula-lines" aria-hidden="true">
		<div>
			a<sub>j</sub> = softmax<sub>j</sub>( {q} · {k}<sub>j</sub> / √{headSize}{causal
				? ' + m'
				: ''}{#if causal}<sub>j</sub>{/if} ), j = {progress.coordinate}
		</div>
		{#if causal}<div>m<sub>j</sub> = 0 for j ≤ t, −∞ for j &gt; t</div>{/if}
		<div>o<sub>d</sub> = ∑<sub>j=1…{sequenceLength}</sub> a<sub>j</sub> · v<sub>j,d</sub></div>
	</div>
	<p class="status" role="status">
		{progress.complete
			? `Complete · weights (1, ${sequenceLength}) · context (1, ${headSize})`
			: 'Scoring every key'} · schematic sweep
	</p>
	<p class="vlm-note">{headNote} {maskNote}</p>
</VlmFormulaPopup>

<style>
	.key-matrix {
		display: grid;
		grid-template-columns: repeat(32, minmax(0, 1fr));
		grid-template-rows: repeat(8, 1fr);
		gap: 1px;
		height: 88px;
	}
	.key-matrix i {
		background: #f8c6c6;
	}
	.key-matrix i.active {
		background: #e98a8a;
	}
	.key-matrix i.masked,
	.cells i.masked {
		background: #f3f4f6;
	}
</style>
