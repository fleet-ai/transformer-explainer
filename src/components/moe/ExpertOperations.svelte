<script lang="ts">
	import type { ExpertOperation } from '$lib/moe/operations';
	type Step = 'up' | 'gate' | 'activation' | 'down';
	export let operation: ExpertOperation | null = null;
	export let inputPreview: number[] = [];
	export let hiddenSize = 1024;
	export let intermediateSize = 512;
	export let expert: number;
	let step: Step = 'up';
	let coordinate = 0;
	const steps: Array<{ id: Step; title: string }> = [
		{ id: 'up', title: 'Up projection' },
		{ id: 'gate', title: 'Gate projection' },
		{ id: 'activation', title: 'SiLU × up' },
		{ id: 'down', title: 'Down projection' }
	];
	$: projection = operation?.[step === 'down' ? 'down' : step === 'gate' ? 'gate' : 'up'];
	$: input = (step === 'down' ? (operation?.product ?? []) : inputPreview).slice(0, 4);
	$: inDimensions = step === 'down' ? intermediateSize : hiddenSize;
	$: outDimensions = step === 'down' ? hiddenSize : intermediateSize;
	$: matrixName = step === 'down' ? 'Wdown' : step === 'gate' ? 'Wgate' : 'Wup';
	$: inputName = step === 'down' ? 'h' : 'x';
	$: outputName = step === 'down' ? 'y' : step === 'gate' ? 'g' : 'u';
	$: maxWeight = Math.max(...(projection?.weights.flat().map(Math.abs) ?? [1]), 1e-12);
	$: equationTitle = `${outputName}[${coordinate}]`;

	function number(value: number, digits = 4): string {
		if (!Number.isFinite(value)) return '—';
		const absolute = Math.abs(value);
		const text =
			absolute > 9999 || (absolute > 0 && absolute < 0.0001)
				? value.toExponential(2)
				: value.toFixed(digits);
		return text.replace('-', '−');
	}
	function weightShade(value: number, max: number): string {
		const strength = 0.06 + Math.min(1, Math.abs(value) / max) * 0.18;
		return value < 0 ? `rgba(109, 138, 195, ${strength})` : `rgba(155, 111, 195, ${strength})`;
	}
	function activateNode(event: KeyboardEvent, next: Step) {
		if (event.key !== 'Enter' && event.key !== ' ') return;
		event.preventDefault();
		event.stopPropagation();
		step = next;
	}
</script>

<div class="expert-operations">
	<div class="operation-heading">
		<strong>Inside expert {expert}</strong><span>SwiGLU feed-forward network</span>
	</div>
	<div class="architecture-chain">
		<svg
			viewBox="0 0 560 86"
			role="group"
			aria-label={`Expert computation: ${hiddenSize}-dimensional input branches into up and gate projections; SiLU of the gate multiplies the up vector coordinate by coordinate; down projection returns ${hiddenSize} output coordinates.`}
		>
			<path
				d="M44 43 H63 C78 43 78 20 93 20 H108 M63 43 C78 43 78 66 93 66 H108 M179 20 H318 C339 20 338 43 354 43 M179 66 H210 M277 66 H318 C339 66 338 43 354 43 M375 43 H401 M481 43 H518"
				class="chain-wire"
			/>
			<text x="25" y="49" class="chain-input" text-anchor="middle">x</text><text
				x="25"
				y="72"
				class="chain-dimension"
				text-anchor="middle">{hiddenSize}</text
			>
			<g
				class="chain-control"
				role="button"
				tabindex="0"
				aria-label="Inspect up projection"
				aria-pressed={step === 'up'}
				on:click={() => (step = 'up')}
				on:keydown={(event) => activateNode(event, 'up')}
			>
				<title>Up projection · Wup × x</title>
				<rect
					x="108"
					y="5"
					width="71"
					height="30"
					rx="4"
					class="chain-node"
					class:current={step === 'up'}
				/><text x="143.5" y="24" text-anchor="middle" class="chain-label">Wup · x</text>
			</g>
			<g
				class="chain-control"
				role="button"
				tabindex="0"
				aria-label="Inspect gate projection"
				aria-pressed={step === 'gate'}
				on:click={() => (step = 'gate')}
				on:keydown={(event) => activateNode(event, 'gate')}
			>
				<title>Gate projection · Wgate × x</title>
				<rect
					x="108"
					y="51"
					width="71"
					height="30"
					rx="4"
					class="chain-node"
					class:current={step === 'gate'}
				/><text x="143.5" y="70" text-anchor="middle" class="chain-label">Wgate · x</text>
			</g>
			<g
				class="chain-control"
				role="button"
				tabindex="0"
				aria-label="Inspect SiLU activation"
				aria-pressed={step === 'activation'}
				on:click={() => (step = 'activation')}
				on:keydown={(event) => activateNode(event, 'activation')}
			>
				<title>SiLU activation and multiplication by the up vector</title>
				<rect
					x="210"
					y="51"
					width="67"
					height="30"
					rx="4"
					class="chain-node"
					class:current={step === 'activation'}
				/><text x="243.5" y="70" text-anchor="middle" class="chain-label">SiLU</text>
			</g>
			<text x="241" y="15" class="chain-dimension" text-anchor="middle">u · {intermediateSize}</text
			><text x="309" y="82" class="chain-dimension" text-anchor="middle">{intermediateSize}</text>
			<g
				class="chain-control"
				role="button"
				tabindex="0"
				aria-label="Inspect elementwise multiplication"
				aria-pressed={step === 'activation'}
				on:click={() => (step = 'activation')}
				on:keydown={(event) => activateNode(event, 'activation')}
			>
				<title>Elementwise multiplication · SiLU(g) × u</title>
				<circle cx="364" cy="43" r="17" class="chain-hit-area" />
				<circle
					cx="364"
					cy="43"
					r="12"
					class="chain-node"
					class:current={step === 'activation'}
				/><text x="364" y="48" text-anchor="middle" class="multiply-symbol">⊙</text>
			</g>
			<g
				class="chain-control"
				role="button"
				tabindex="0"
				aria-label="Inspect down projection"
				aria-pressed={step === 'down'}
				on:click={() => (step = 'down')}
				on:keydown={(event) => activateNode(event, 'down')}
			>
				<title>Down projection · Wdown × h</title>
				<rect
					x="401"
					y="28"
					width="80"
					height="30"
					rx="4"
					class="chain-node"
					class:current={step === 'down'}
				/><text x="441" y="47" text-anchor="middle" class="chain-label">Wdown · h</text>
			</g>
			<text x="537" y="49" class="chain-input" text-anchor="middle">y</text><text
				x="537"
				y="72"
				class="chain-dimension"
				text-anchor="middle">{hiddenSize}</text
			>
		</svg>
		<p class="chain-hint">Select a block to inspect its recorded values.</p>
	</div>
	<div class="step-buttons" role="group" aria-label="Expert operation">
		{#each steps as item}<button
				type="button"
				class:active={step === item.id}
				aria-pressed={step === item.id}
				on:click={() => (step = item.id)}>{item.title}</button
			>{/each}
	</div>

	{#key step}
		<div class="operation-detail" role="region" aria-label="Recorded expert operation" tabindex="0">
			{#if !operation || inputPreview.length < 4}
				<div class="operation-empty" role="status">Expert operation data is not available yet.</div>
			{:else if step === 'activation'}
				<div class="step-description">
					<strong>SiLU(g) ⊙ u = h</strong><span
						>{intermediateSize} independent coordinate multiplications</span
					>
				</div>
				<div class="activation-formula">
					<span class="activation-expression">SiLU(g) = g / (1 + e<sup>−g</sup>)</span><span
						class="activation-hint">⊙ means element by element.</span
					>
				</div>
				<div class="operation-scroll">
					<table class="activation-table">
						<thead
							><tr
								><th scope="col">Coordinate</th><th scope="col">Gate g</th><th scope="col"
									>SiLU(g)</th
								><th class="operator" aria-label="multiplied by">×</th><th scope="col">Up u</th><th
									class="operator"
									aria-label="equals">=</th
								><th scope="col">Product h</th></tr
							></thead
						>
						<tbody
							>{#each operation.product.slice(0, 4) as value, index}<tr
									><th scope="row">{index}</th><td title={String(operation.gate.output[index])}
										>{number(operation.gate.output[index])}</td
									><td title={String(operation.siluGate[index])}
										>{number(operation.siluGate[index])}</td
									><td class="operator">×</td><td title={String(operation.up.output[index])}
										>{number(operation.up.output[index])}</td
									><td class="operator">=</td><td class="product-value" title={String(value)}
										>{number(value)}</td
									></tr
								>{/each}</tbody
						>
					</table>
				</div>
				<p class="slice-note">
					The first 4 of {intermediateSize} coordinates are shown. Each gate value controls the matching
					up-projection value.
				</p>
			{:else if projection && input.length === 4}
				<div class="step-description">
					<strong>{matrixName} · {inputName} = {outputName}</strong><span
						>{outDimensions.toLocaleString()} × {inDimensions.toLocaleString()} weight matrix</span
					>
				</div>
				<div class="projection-layout">
					<div class="operation-scroll">
						<div class="linear-operation">
							<div class="matrix-block">
								<div class="tensor-label">{matrixName}<small>rows 0–3 · columns 0–3</small></div>
								<div class="weight-matrix" aria-label={`Recorded 4 by 4 slice of ${matrixName}`}>
									{#each projection.weights.slice(0, 4) as row, rowIndex}<button
											type="button"
											class="matrix-row"
											class:selected={coordinate === rowIndex}
											aria-label={`Inspect output coordinate ${rowIndex}`}
											aria-pressed={coordinate === rowIndex}
											on:click={() => (coordinate = rowIndex)}
											>{#each row.slice(0, 4) as value, columnIndex}<span
													style={`background:${weightShade(value, maxWeight)}`}
													title={`${matrixName}[${rowIndex},${columnIndex}] = ${value}`}
													>{number(value)}</span
												>{/each}</button
										>{/each}
								</div>
							</div>
							<span class="math-operator" aria-hidden="true">×</span>
							<div class="vector-block input-block">
								<div class="tensor-label">{inputName}<small>first 4</small></div>
								<div class="number-vector">
									{#each input as value, index}<div title={`${inputName}[${index}] = ${value}`}>
											{number(value)}
										</div>{/each}
								</div>
							</div>
							<span class="math-operator" aria-hidden="true">+</span>
							<div class="vector-block tail-block">
								<div class="tensor-label">
									Remaining<small>{(inDimensions - 4).toLocaleString()} inputs</small>
								</div>
								<div class="number-vector">
									{#each projection.omittedContribution.slice(0, 4) as value, index}<div
											class:highlighted={coordinate === index}
											title={`Full dot-product contribution from input coordinates 4–${inDimensions - 1}: ${value}`}
										>
											{number(value)}
										</div>{/each}
								</div>
							</div>
							<span class="math-operator" aria-hidden="true">=</span>
							<div class="vector-block output-block">
								<div class="tensor-label">{outputName}<small>select a row</small></div>
								<div class="number-vector">
									{#each projection.output.slice(0, 4) as value, index}<button
											type="button"
											class:selected={coordinate === index}
											aria-label={`Inspect ${outputName} coordinate ${index}: ${value}`}
											aria-pressed={coordinate === index}
											on:click={() => (coordinate = index)}
											title={`${outputName}[${index}] = ${value}`}
											><span class="output-index">{index}</span>{number(value)}</button
										>{/each}
								</div>
							</div>
						</div>
					</div>
					<div class="dot-detail" aria-live="polite">
						<div class="dot-heading">
							<strong>One output coordinate</strong><span>{equationTitle}</span>
						</div>
						<div class="dot-equation">
							<span class="equation-label">{equationTitle} =</span>
							<div class="dot-terms">
								{#each projection.weights[coordinate].slice(0, 4) as weight, index}
									<span class="dot-addend">
										<span class="equation-operator">{index > 0 ? '+' : ''}</span>
										<span
											class="dot-term"
											title={`${matrixName}[${coordinate},${index}] × ${inputName}[${index}]`}
										>
											(<b>{number(weight)}</b> × <i>{number(input[index])}</i>)
										</span>
									</span>
								{/each}
							</div>
							<div class="dot-total">
								<span class="equation-operator">+</span>
								<span
									class="remaining-term"
									title={`Contribution of all other ${inDimensions - 4} dimensions`}
								>
									{number(projection.omittedContribution[coordinate])}
									<small>remaining {(inDimensions - 4).toLocaleString()} inputs</small>
								</span>
								<span class="equation-operator">=</span>
								<strong
									class="recorded-result"
									title={`Recorded projection output: ${projection.output[coordinate]}`}
								>
									{number(projection.output[coordinate])}
								</strong>
							</div>
						</div>
					</div>
				</div>
				<p class="slice-note">
					Weights are indexed by output row, input column. The remaining-input term is the recorded
					contribution of every omitted coordinate. Values are rounded for display; float32 model
					outputs may differ slightly from the reconstructed dot product.
				</p>
			{/if}
		</div>
	{/key}
</div>

<style>
	.expert-operations {
		display: flex;
		flex-direction: column;
		flex: 1;
		min-height: 0;
		width: 100%;
		min-width: 0;
		box-sizing: border-box;
		color: #6e597f;
		font-family: Arial, Helvetica, sans-serif;
		font-size: 12px;
		container-type: inline-size;
	}
	.expert-operations * {
		box-sizing: border-box;
	}
	.operation-detail {
		flex: 1;
		min-height: 0;
		overflow: auto;
		overscroll-behavior: contain;
		scrollbar-gutter: stable;
		scrollbar-width: thin;
		scrollbar-color: #d7c5e5 transparent;
		padding-right: 6px;
	}
	.operation-detail:focus-visible {
		outline: 2px solid #b799d1;
		outline-offset: -2px;
	}
	.operation-heading {
		flex: 0 0 auto;
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 10px;
		margin-bottom: 4px;
	}
	.operation-heading strong {
		font-size: 13px;
		font-weight: 600;
		color: #75548f;
	}
	.operation-heading > span {
		font-size: 10px;
		color: #7d698d;
	}
	.architecture-chain {
		flex: 0 0 auto;
		padding: 0 0 6px;
	}
	.architecture-chain svg {
		width: 100%;
		height: auto;
		display: block;
		max-height: 74px;
	}
	.chain-wire {
		fill: none;
		stroke: #d4c4e2;
		stroke-width: 1.4;
	}
	.chain-node {
		fill: #faf7fd;
		stroke: #dfd1eb;
		stroke-width: 1;
	}
	.chain-node.current {
		fill: #ecdef7;
		stroke: #b794ce;
	}
	.chain-control {
		cursor: pointer;
		user-select: none;
		outline: none;
	}
	.chain-control:hover .chain-node {
		fill: #f1e6f9;
		stroke: #a87bc6;
		stroke-width: 1.6;
	}
	.chain-control:focus-visible .chain-node {
		stroke: #76509b;
		stroke-width: 2;
		stroke-dasharray: 3 2;
	}
	.chain-control text {
		pointer-events: none;
	}
	.chain-hit-area {
		fill: transparent;
	}
	.chain-hint {
		margin: 3px 0 0;
		color: #9a83aa;
		font-size: 10px;
		text-align: center;
	}
	.chain-label {
		fill: #826597;
		font-size: 11px;
	}
	.chain-input {
		fill: #8c6ba4;
		font:
			italic 19px Georgia,
			serif;
	}
	.chain-dimension {
		fill: #82708f;
		font-size: 9px;
	}
	.multiply-symbol {
		fill: #9876af;
		font:
			19px Georgia,
			serif;
	}
	.step-buttons {
		flex: 0 0 auto;
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 4px;
		padding: 4px;
		border: 1px solid #eee6f4;
		background: #fcfafd;
		border-radius: 5px;
	}
	.step-buttons button {
		border: 1px solid transparent;
		border-radius: 3px;
		background: transparent;
		color: #806a91;
		font:
			12px/1.4 Arial,
			Helvetica,
			sans-serif;
		padding: 5px;
		cursor: pointer;
	}
	.step-buttons button.active {
		color: #7b5598;
		border-color: #dfcbea;
		background: #f1e6f9;
	}
	.step-buttons button:hover {
		color: #795197;
		background: #f5edf9;
	}
	.step-description {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 10px;
		margin: 10px 0 8px;
	}
	.step-description strong {
		font:
			13px Georgia,
			serif;
		color: #7c5b94;
	}
	.step-description > span {
		font-size: 10px;
		color: #826b93;
	}
	.projection-layout {
		display: grid;
		grid-template-columns: minmax(540px, 1.2fr) minmax(0, 1fr);
		align-items: start;
		gap: 12px;
	}
	.projection-layout > * {
		min-width: 0;
	}
	.operation-scroll {
		max-width: 100%;
		overflow-x: auto;
		padding-bottom: 3px;
		scrollbar-width: thin;
		scrollbar-color: #d7c5e5 transparent;
	}
	.linear-operation {
		display: flex;
		align-items: flex-end;
		gap: 6px;
		min-width: 540px;
		justify-content: center;
	}
	.matrix-block {
		width: 232px;
		flex-shrink: 0;
	}
	.tensor-label {
		text-align: center;
		font:
			12px Georgia,
			serif;
		color: #9273a7;
		margin-bottom: 5px;
	}
	.tensor-label small {
		display: block;
		font:
			9px/1.4 Arial,
			sans-serif;
		color: #8a759b;
		margin-top: 2px;
	}
	.weight-matrix {
		display: grid;
		gap: 3px;
		border-left: 1px solid #cdb8dc;
		border-right: 1px solid #cdb8dc;
		padding: 3px;
		position: relative;
	}
	.matrix-row {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 2px;
		padding: 0;
		background: transparent;
		border: 0;
		min-height: 27px;
		cursor: pointer;
		outline-offset: 1px;
	}
	.matrix-row span {
		display: flex;
		align-items: center;
		justify-content: center;
		color: #705c80;
		font:
			10px 'SFMono-Regular',
			Consolas,
			monospace;
		border-radius: 2px;
	}
	.matrix-row.selected {
		box-shadow: 0 0 0 1px #bc99d3;
		border-radius: 2px;
	}
	.matrix-row.selected span {
		color: #705083;
	}
	.math-operator {
		height: 123px;
		display: flex;
		align-items: center;
		font:
			17px Georgia,
			serif;
		color: #b3a0c0;
		flex-shrink: 0;
	}
	.vector-block {
		flex-shrink: 0;
	}
	.input-block {
		width: 72px;
	}
	.tail-block {
		width: 78px;
	}
	.output-block {
		width: 84px;
	}
	.number-vector {
		display: grid;
		gap: 3px;
		padding: 3px 0;
	}
	.number-vector > div,
	.number-vector > button {
		display: flex;
		align-items: center;
		justify-content: center;
		min-height: 27px;
		font:
			11px 'SFMono-Regular',
			Consolas,
			monospace;
		color: #7c658a;
		border: 1px solid #f0e9f5;
		border-radius: 3px;
		background: #fcfafc;
		position: relative;
	}
	.input-block .number-vector > div {
		background: #f2f5fa;
		border-color: #e7edf5;
		color: #62758f;
	}
	.number-vector > button {
		cursor: pointer;
		padding: 0 2px 0 9px;
	}
	.output-block button.selected {
		border-color: #b999cf;
		background: #efe1f8;
		color: #78508e;
	}
	.output-index {
		position: absolute;
		top: 3px;
		left: 4px;
		color: #b09abc;
		font:
			7px Arial,
			sans-serif;
	}
	.number-vector > div.highlighted {
		border-color: #dac7e7;
		color: #826394;
	}
	.dot-detail {
		padding: 9px 10px;
		border: 1px solid #e9dcef;
		border-radius: 4px;
		background: #fdfbfe;
	}
	.dot-heading {
		display: flex;
		justify-content: space-between;
		align-items: center;
		color: #806a92;
		margin-bottom: 6px;
	}
	.dot-heading strong {
		font-size: 11px;
		font-weight: 500;
	}
	.dot-heading > span {
		font:
			11px 'SFMono-Regular',
			Consolas,
			monospace;
		color: #805b98;
	}
	.dot-equation {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		align-items: start;
		column-gap: 8px;
		row-gap: 7px;
		font:
			11px/1.6 'SFMono-Regular',
			Consolas,
			monospace;
		color: #816b92;
	}
	.dot-terms {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 4px 12px;
	}
	.dot-addend {
		display: grid;
		grid-template-columns: 8px minmax(0, 1fr);
		gap: 4px;
	}
	.dot-total {
		grid-column: 2;
		display: grid;
		grid-template-columns: 8px minmax(100px, 1fr) 12px minmax(70px, 1fr);
		align-items: center;
		gap: 4px;
	}
	.equation-label {
		color: #88619f;
		margin-right: 3px;
	}
	.equation-operator {
		color: #b7a1c5;
	}
	.dot-term {
		white-space: nowrap;
	}
	.dot-term b {
		font-weight: 400;
		color: #805898;
	}
	.dot-term i {
		font-style: normal;
		color: #627591;
	}
	.remaining-term {
		position: relative;
		background: #f3ebf8;
		padding: 1px 5px;
		border-radius: 3px;
		color: #805f94;
	}
	.remaining-term small {
		display: block;
		font:
			10px/1.4 Arial,
			sans-serif;
		color: #887096;
		text-align: center;
	}
	.recorded-result {
		font-size: 11px;
		color: #805598;
		font-weight: 500;
	}
	.slice-note {
		font-size: 10px;
		line-height: 1.5;
		color: #826e91;
		margin: 8px 0 0;
	}
	.activation-formula {
		color: #9472a8;
		font:
			13px Georgia,
			serif;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		padding: 2px 0 7px;
	}
	.activation-formula .activation-expression {
		white-space: nowrap;
	}
	.activation-formula .activation-hint {
		font:
			9px Arial,
			sans-serif;
		color: #b09ac0;
	}
	.activation-table {
		border-collapse: separate;
		border-spacing: 0 4px;
		width: 100%;
		min-width: 440px;
		text-align: center;
	}
	.activation-table th {
		color: #9f85af;
		font-weight: 400;
		font-size: 10px;
	}
	.activation-table thead th {
		padding: 0 5px 4px;
	}
	.activation-table tbody th {
		width: 65px;
		background: #faf7fc;
		border-radius: 3px 0 0 3px;
		font:
			11px 'SFMono-Regular',
			Consolas,
			monospace;
	}
	.activation-table td {
		background: #faf7fc;
		padding: 7px 5px;
		font:
			11px 'SFMono-Regular',
			Consolas,
			monospace;
		color: #957eaa;
	}
	.activation-table .operator {
		width: 15px;
		padding-left: 0;
		padding-right: 0;
		color: #c2adcf;
	}
	.activation-table td.product-value {
		color: #825a99;
		background: #f1e6f8;
		border-radius: 0 3px 3px 0;
	}
	.operation-empty {
		padding: 30px 12px;
		text-align: center;
		color: #a58ab7;
		font-size: 11px;
	}
	button:focus-visible {
		outline: 2px solid #b799d1;
		outline-offset: 3px;
	}
	@container (max-width: 900px) {
		.projection-layout {
			grid-template-columns: minmax(0, 1fr);
			gap: 8px;
		}
		.dot-terms {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
		.dot-detail {
			padding: 8px 10px;
		}
	}
	@container (max-width:480px) {
		.dot-terms {
			grid-template-columns: minmax(0, 1fr);
		}
		.operation-heading {
			align-items: flex-start;
			flex-direction: column;
			gap: 4px;
		}
		.step-description {
			gap: 6px;
		}
		.step-description > span {
			font-size: 9px;
		}
		.step-buttons button {
			font-size: 10px;
		}
		.activation-formula {
			flex-wrap: wrap;
		}
		.architecture-chain svg {
			min-width: 330px;
		}
		.architecture-chain {
			overflow-x: auto;
		}
	}
	@media (max-height: 540px) {
		.expert-operations,
		.operation-detail {
			flex: 0 0 auto;
		}
		.operation-detail {
			overflow: visible;
			scrollbar-gutter: auto;
		}
	}
</style>
