<script lang="ts">
	import { base } from '$app/paths';
	import { onDestroy, onMount } from 'svelte';
	import {
		diagramFor,
		edgePath,
		operationFor,
		previewTensor,
		type Stage
	} from '$lib/architectures/diagrams';
	import {
		isArchitectureTrace,
		numeric,
		tensorColor,
		tensorRows,
		tensorRowForPosition,
		type ArchitectureId,
		type ArchitectureTrace,
		type RecordedOperation,
		type RecordedTensor
	} from '$lib/architectures/types';
	import TensorInspector from './TensorInspector.svelte';
	export let architecture: ArchitectureId;
	let trace: ArchitectureTrace | null = null;
	let error = '';
	let loading = true;
	let exampleIndex = 0;
	let layerIndex = 0;
	let tokenIndex = 0;
	let headIndex = 0;
	let selectedPatch = 0;
	let inspectorToken = 0;
	let inspectorTokenLabel = '';
	let selectedOperation: RecordedOperation | null = null;
	let replayStage = '';
	let replaying = false;
	let timer: ReturnType<typeof setTimeout>;
	let controller: AbortController;
	let requested: ArchitectureId;
	let mounted = false;
	$: diagram = diagramFor(architecture, layer);
	$: if (mounted && architecture !== requested) {
		requested = architecture;
		load();
	}
	$: example = trace?.examples[exampleIndex];
	$: layer = example?.layers[layerIndex];
	$: operations = layer?.operations ?? [];
	$: allHeads = [
		...new Set(
			operations.flatMap((operation) =>
				operation.tensors
					.filter((tensor) => tensor.role === 'attention' && tensor.head !== undefined)
					.map((tensor) => tensor.head!)
			)
		)
	].sort((a, b) => a - b);
	$: if (allHeads.length && !allHeads.includes(headIndex)) headIndex = allHeads[0];
	$: inputImage =
		typeof example?.input.image === 'string'
			? example.input.image
			: (example?.input.image?.src ?? example?.input.image?.url);
	$: imageUrl = inputImage?.startsWith('/') ? `${base}${inputImage}` : inputImage;
	$: isVision = architecture === 'vit' || architecture === 'vision-language';
	$: patchCount = Number(
		trace?.config.patchCount ?? (architecture === 'vit' ? (example?.tokens.length ?? 1) - 1 : 1024)
	);
	$: patchGrid = Math.max(1, Math.round(Math.sqrt(patchCount)));
	$: visibleTokens =
		example?.tokens.slice(Math.floor(tokenIndex / 8) * 8, Math.floor(tokenIndex / 8) * 8 + 8) ?? [];
	$: tokenOffset = Math.floor(tokenIndex / 8) * 8;
	$: dimensions = trace ? dimensionText(trace) : '';
	$: totalHeads = trace?.config.attentionHeads ?? trace?.config.numAttentionHeads;
	$: provenanceUrl = trace
		? `https://huggingface.co/${trace.provenance.modelId}/tree/${trace.provenance.revision}`
		: '';
	$: promptLines = (example?.input.text ?? '').match(/.{1,24}(?:\s|$)|.{1,24}/g)?.slice(0, 3) ?? [];
	async function load() {
		controller?.abort();
		controller = new AbortController();
		const signal = controller.signal;
		loading = true;
		error = '';
		trace = null;
		reset();
		try {
			const response = await fetch(`${base}/architectures/${architecture}.json`, { signal });
			if (!response.ok) throw new Error(`Recorded trace could not be loaded (${response.status}).`);
			const data: unknown = await response.json();
			if (!isArchitectureTrace(data, architecture))
				throw new Error('The recorded trace has an unsupported format.');
			if (!signal.aborted) {
				trace = data;
				if (architecture === 'hybrid')
					layerIndex = Math.max(
						0,
						data.examples[0].layers.findIndex((item) =>
							item.operations.some((operation) => operation.id === 'attention')
						)
					);
			}
		} catch (reason) {
			if (!signal.aborted)
				error =
					reason instanceof Error ? reason.message : 'Could not load this architecture recording.';
		} finally {
			if (!signal.aborted) loading = false;
		}
	}
	function stopReplay() {
		clearTimeout(timer);
		replaying = false;
		replayStage = '';
	}
	function reset() {
		stopReplay();
		exampleIndex = 0;
		layerIndex = 0;
		tokenIndex = 0;
		headIndex = 0;
		selectedPatch = 0;
		selectedOperation = null;
	}
	function chooseExample(index: number) {
		stopReplay();
		exampleIndex = index;
		layerIndex =
			architecture === 'hybrid'
				? Math.max(
						0,
						trace!.examples[index].layers.findIndex((item) =>
							item.operations.some((operation) => operation.id === 'attention')
						)
					)
				: 0;
		tokenIndex = 0;
		selectedOperation = null;
	}
	function chooseLayer(index: number) {
		stopReplay();
		layerIndex = index;
		selectedOperation = null;
	}
	function replay() {
		if (replaying) {
			stopReplay();
			return;
		}
		selectedOperation = null;
		const stages = diagram.stages.filter((stage) => operationFor(stage, layer));
		let index = 0;
		replaying = true;
		const step = () => {
			if (index >= stages.length) {
				stopReplay();
				return;
			}
			replayStage = stages[index++].id;
			timer = setTimeout(step, 1100);
		};
		step();
	}
	function inspect(stage: Stage) {
		const operation = operationFor(stage, layer);
		if (operation) inspectOperation(operation);
	}
	function inspectOperation(operation: RecordedOperation) {
		stopReplay();
		const decoder =
			architecture === 'encoder-decoder' &&
			['decoder-self', 'cross-attention', 'feed-forward', 'decoder-input', 'output'].includes(
				operation.id
			);
		inspectorToken = decoder ? 0 : tokenIndex;
		inspectorTokenLabel = decoder
			? (operation.tensors[0]?.rowLabels?.[0] ?? '')
			: (example?.tokens[tokenIndex] ?? '');
		selectedOperation = operation;
	}
	function choosePatch(patch: number) {
		if (architecture === 'vit') {
			tokenIndex = patch + 1;
			return;
		}
		selectedPatch = patch;
		const label = `patch ${Math.floor(patch / patchGrid)},${patch % patchGrid}`;
		const operation =
			operations.find(
				(item) =>
					['patches', 'vision-attention'].includes(item.id) &&
					item.tensors.some((tensor) => tensor.rowLabels?.includes(label))
			) ?? operations.find((item) => item.id === 'vision-attention');
		if (operation) {
			stopReplay();
			inspectorToken = patch;
			inspectorTokenLabel = label;
			selectedOperation = operation;
		}
	}
	function displayTensor(stage: Stage): RecordedTensor | undefined {
		const operation = operationFor(stage, layer);
		return (
			operation?.tensors.find(
				(tensor) => tensor.role === 'attention' && tensor.head === headIndex
			) ?? previewTensor(operation)
		);
	}
	function preview(stage: Stage) {
		const tensor = displayTensor(stage);
		const rows = tensorRows(tensor);
		const label = example?.tokens[tokenIndex];
		const decoder =
			architecture === 'encoder-decoder' &&
			['decoder-self', 'cross-attention', 'feed-forward', 'decoder-input', 'output'].includes(
				stage.id
			);
		const selectedRow = decoder ? -1 : tensorRowForPosition(tensor, tokenIndex, label);
		const start = Math.max(0, Math.min(selectedRow - 3, rows.length - 8));
		const values = rows.slice(start, start + 8).map((row) => row.slice(0, 8));
		const scale = Math.max(0.000001, ...values.flat().map((value) => Math.abs(value)));
		return {
			tensor,
			values,
			start,
			selectedRow,
			scale,
			probability: tensor?.role === 'attention' || tensor?.role === 'probabilities'
		};
	}
	function tokenLabel(token: string) {
		return token.replace(/▁|Ġ/g, '·').replace(/\n/g, '↵');
	}
	function dimensionText(data: ArchitectureTrace) {
		const config = data.config;
		const parts = [];
		if (config.hiddenSize) parts.push(`${config.hiddenSize}D hidden`);
		if (config.attentionHeads) parts.push(`${config.attentionHeads} attention heads`);
		if (config.kvLatentDim) parts.push(`${config.kvLatentDim}D K/V latent`);
		if (config.patchSize) parts.push(`${config.patchSize} × ${config.patchSize} px patches`);
		return parts.join(' · ');
	}
	function stageNote(stage: Stage): string {
		if (!trace) return stage.note;
		const config = trace.config;
		if (stage.id === 'compress' && config.kvLatentDim)
			return `${config.hiddenSize}D → ${config.kvLatentDim}D latent`;
		if (stage.id === 'expand' && config.contentKeyDim)
			return `${config.attentionHeads} heads · K ${config.contentKeyDim}D · V ${config.valueDim}D`;
		if (stage.id === 'rope' && config.rotaryDim)
			return `${config.rotaryDim}D position · shared key`;
		if (stage.id === 'output' && trace.provenance.partial)
			return 'Recorded block output · no prediction';
		return stage.note;
	}
	onDestroy(() => {
		controller?.abort();
		clearTimeout(timer);
	});
	onMount(() => {
		mounted = true;
	});
</script>

<section class="architecture-explorer" aria-label={diagram.title}>
	{#if loading}
		<div class="loading" role="status">
			<div class="loading-line"></div>
			Loading the recorded {diagram.title} forward pass…
		</div>
	{:else if error}
		<div class="error" role="alert">
			<strong>Recording unavailable</strong>
			<p>{error}</p>
			<button on:click={load}>Retry</button>
		</div>
	{:else if trace && example && layer}
		<div class="trace-toolbar">
			<div class="example-control">
				<span>Recorded examples</span>
				<div class="segments">
					{#each trace.examples as item, index}<button
							class:active={exampleIndex === index}
							aria-pressed={exampleIndex === index}
							on:click={() => chooseExample(index)}>{item.label}</button
						>{/each}
				</div>
			</div>
			<button class="replay" class:active={replaying} on:click={replay}
				>{replaying ? 'Stop replay' : 'Replay flow'}
				<span aria-hidden="true">{replaying ? '■' : '▷'}</span></button
			>
			<a
				class="trace-link"
				href={`${base}/architectures/${architecture}.json`}
				download={`${architecture}-recorded-trace.json`}>Trace JSON ↗</a
			>
		</div>
		<div class="title-row">
			<div>
				<h1>{diagram.title}</h1>
				<p>{diagram.caption}</p>
				<div class="scope-badge">
					<strong
						>{trace.provenance.partial
							? 'Pretrained blocks only · no generated prediction'
							: 'Full pretrained model · recorded forward pass'}</strong
					><a href={provenanceUrl} target="_blank" rel="noreferrer">{trace.provenance.modelId} ↗</a
					>
				</div>
			</div>
			<div class="layer-controls">
				<span>{layer.label ?? `Block ${layer.index + 1}`}</span><button
					aria-label="Previous recorded block"
					disabled={layerIndex === 0}
					on:click={() => chooseLayer(layerIndex - 1)}>‹</button
				><button
					aria-label="Next recorded block"
					disabled={layerIndex >= example.layers.length - 1}
					on:click={() => chooseLayer(layerIndex + 1)}>›</button
				><small>{example.layers.length} recorded</small>
			</div>
		</div>
		<div class="position-row">
			<div class="position-controls">
				<label for="recorded-position"
					>{isVision
						? 'Position'
						: architecture === 'encoder-decoder'
							? 'Source token'
							: 'Token'}</label
				><button
					aria-label="Previous position"
					disabled={tokenIndex === 0}
					on:click={() => tokenIndex--}>‹</button
				><select id="recorded-position" bind:value={tokenIndex}
					>{#each example.tokens as token, index}<option value={index}
							>{index}: {tokenLabel(token)}</option
						>{/each}</select
				><button
					aria-label="Next position"
					disabled={tokenIndex >= example.tokens.length - 1}
					on:click={() => tokenIndex++}>›</button
				>
			</div>
			{#if allHeads.length > 1}<div class="head-control">
					<label for="recorded-head">Attention head</label><select
						id="recorded-head"
						bind:value={headIndex}
						>{#each allHeads as head}<option value={head}>Head {head + 1}</option>{/each}</select
					>
				</div>{:else if operations.some( (operation) => operation.tensors.some((tensor) => tensor.role === 'attention') )}<span
					class="head-note"
					>Recorded {architecture === 'vision-language' ? 'language' : 'attention'} head {allHeads.length
						? allHeads[0] + 1
						: 1}{totalHeads ? ` of ${totalHeads}` : ''}{architecture === 'vision-language'
						? ' · vision head 1'
						: ''}</span
				>{/if}
			<span class="dimensions">{dimensions}</span>
		</div>
		<div class="diagram-scroll">
			<svg
				class="architecture-diagram"
				viewBox="0 100 1510 640"
				aria-label={`${diagram.title} recorded computation diagram`}
			>
				<defs
					><marker
						id="architecture-arrow"
						markerWidth="7"
						markerHeight="7"
						refX="6"
						refY="3.5"
						orient="auto"
						><path d="M0,0 L7,3.5 L0,7" fill="none" stroke="#bca5d7" stroke-width="1.4" /></marker
					></defs
				>
				{#each diagram.lanes ?? [] as lane}<text x="225" y={lane.y} class="lane-title"
						>{lane.title}</text
					>{/each}
				{#each diagram.edges as edge}
					{@const from = diagram.stages.find((stage) => stage.id === edge.from)!}
					{@const to = diagram.stages.find((stage) => stage.id === edge.to)!}
					<path
						d={edgePath(from, to)}
						class="flow-ribbon"
						class:illuminated={replayStage === edge.from || replayStage === edge.to}
					/>
					<path d={edgePath(from, to)} class="flow-line" marker-end="url(#architecture-arrow)" />
					{#if edge.label}<text
							x={from.x === to.x ? from.x + from.w / 2 + 16 : (from.x + from.w + to.x) / 2}
							y={from.x === to.x
								? (from.y + from.h + to.y) / 2
								: (from.y + from.h / 2 + to.y + to.h / 2) / 2 - 26}
							text-anchor={from.x === to.x ? 'start' : 'middle'}
							class="edge-label">{edge.label}</text
						>{/if}
				{/each}
				{#if architecture === 'hybrid'}{@const stateStage = diagram.stages.find(
						(stage) => stage.id === 'selective-state'
					)!}<path
						d={`M ${stateStage.x + 150} ${stateStage.y - 5} C ${stateStage.x + 200} ${stateStage.y - 155}, ${stateStage.x - 11} ${stateStage.y - 155}, ${stateStage.x + 31} ${stateStage.y - 5}`}
						class="recurrence"
						marker-end="url(#architecture-arrow)"
					/><text
						x={stateStage.x + 92}
						y={stateStage.y - 120}
						text-anchor="middle"
						class="recurrence-label">state t − 1 → state t</text
					><text x={stateStage.x + 92} y={stateStage.y - 98} text-anchor="middle" class="edge-label"
						>history in a recurrent state</text
					>{/if}
				{#if architecture === 'mla'}<text x="570" y="128" text-anchor="middle" class="cache-label"
						>CACHE · {trace.config.cacheCoordinatesPerToken ?? 'latent + rotary'} coordinates / token</text
					>{/if}
				{#each diagram.stages as stage}
					{@const operation = operationFor(stage, layer)}
					{@const data = preview(stage)}
					{#if stage.visual === 'tokens' || stage.visual === 'patches'}
						<g transform={`translate(${stage.x},${stage.y})`}>
							<text x={stage.w / 2} y="-35" class="stage-title" text-anchor="middle"
								>{stage.label}</text
							>
							{#if stage.visual === 'patches'}
								{#if imageUrl}<image
										href={imageUrl}
										x="0"
										y="20"
										width="150"
										height="150"
										preserveAspectRatio="xMidYMid slice"
									/>{:else}<rect x="0" y="20" width="150" height="150" fill="#edf3fb" />{/if}
								{#each Array.from({ length: patchCount }, (_, index) => index) as patch}
									<rect
										x={(patch % patchGrid) * (150 / patchGrid)}
										y={20 + Math.floor(patch / patchGrid) * (150 / patchGrid)}
										width={150 / patchGrid}
										height={150 / patchGrid}
										fill={(architecture === 'vit' ? tokenIndex - 1 : selectedPatch) === patch
											? '#9e67d975'
											: 'transparent'}
										stroke="#ffffffb5"
										stroke-width=".5"
										class="patch"
										role="button"
										tabindex="0"
										aria-label={`Select patch ${patch}`}
										on:click={() => choosePatch(patch)}
										on:keydown={(event) => {
											if (event.key === 'Enter' || event.key === ' ') {
												event.preventDefault();
												choosePatch(patch);
											}
										}}
									/>
								{/each}
								<text x="75" y="197" text-anchor="middle" class="edge-label"
									>{patchGrid} × {patchGrid} patches</text
								><text x="75" y="218" text-anchor="middle" class="edge-label"
									>{architecture === 'vit' ? 'CLS is position 0' : 'Recorded image input'}</text
								>
							{:else}
								{#each visibleTokens as token, index}<g
										class="token"
										role="button"
										tabindex="0"
										aria-label={`Select token ${tokenOffset + index}: ${token}`}
										on:click={() => (tokenIndex = tokenOffset + index)}
										on:keydown={(event) => {
											if (event.key === 'Enter' || event.key === ' ') {
												event.preventDefault();
												tokenIndex = tokenOffset + index;
											}
										}}
										><rect
											x="0"
											y={index * 31}
											width="150"
											height="27"
											rx="3"
											fill={tokenIndex === tokenOffset + index ? '#ede2f8' : '#f6f8fb'}
											stroke={tokenIndex === tokenOffset + index ? '#ac82ce' : 'transparent'}
										/><text x="9" y={index * 31 + 18} class="token-index"
											>{tokenOffset + index}</text
										><text x="34" y={index * 31 + 18} class="token-text"
											>{tokenLabel(token).slice(0, 15)}</text
										></g
									>{/each}
							{/if}
						</g>
					{:else if stage.visual === 'output' && example.output?.predictions?.length && !trace.provenance.partial}
						<g
							transform={`translate(${stage.x},${stage.y})`}
							class="output-stage"
							role="button"
							tabindex="0"
							aria-label="Inspect recorded output probabilities"
							on:click={() => inspect(stage)}
							on:keydown={(event) => {
								if (event.key === 'Enter' || event.key === ' ') {
									event.preventDefault();
									inspect(stage);
								}
							}}
						>
							<rect
								x="-8"
								y="-55"
								width={stage.w + 16}
								height="328"
								fill="transparent"
								pointer-events="all"
							/>
							<text x="0" y="-35" class="stage-title"
								>{architecture === 'vit'
									? 'Class probabilities'
									: architecture === 'encoder-decoder'
										? 'Next after translation'
										: 'First answer token'}</text
							>
							{#each example.output.predictions.slice(0, 7) as prediction, index}<text
									x="0"
									y={index * 33 + 18}
									class:top-prediction={index === 0}
									class="prediction-label">{prediction.label.slice(0, 14)}</text
								><rect
									x="103"
									y={index * 33 + 7}
									width={Math.max(1, prediction.probability * 43)}
									height="5"
									rx="1"
									fill={index === 0 ? '#9257dd' : '#d0c7db'}
								/><text
									x="154"
									y={index * 33 + 17}
									class:top-prediction={index === 0}
									class="prediction-value">{(prediction.probability * 100).toFixed(1)}%</text
								>{/each}
							<text x="0" y="263" class="edge-label">Actual distribution · top entries</text>
						</g>
					{:else}
						<g
							transform={`translate(${stage.x},${stage.y})`}
							class="operation-stage"
							class:interactive={!!operation}
							class:highlighted={replayStage === stage.id ||
								(selectedOperation === operation && !!operation)}
							role="button"
							tabindex={operation ? 0 : -1}
							aria-disabled={!operation}
							aria-label={`${stage.label}${operation ? ': inspect recorded values' : ': architecture context'}`}
							on:click={() => inspect(stage)}
							on:keydown={(event) => {
								if (event.key === 'Enter' || event.key === ' ') {
									event.preventDefault();
									inspect(stage);
								}
							}}
						>
							<rect class="stage-back" x="5" y="7" width={stage.w} height={stage.h} rx="3" /><rect
								class="stage-surface"
								width={stage.w}
								height={stage.h}
								rx="3"
							/>
							<text x={stage.w / 2} y="-16" class="stage-title" text-anchor="middle"
								>{stage.label}</text
							>
							{#if data.values.length}
								{#each data.values as values, row}<g
										>{#each values as value, column}<rect
												x={20 + column * ((stage.w - 40) / Math.max(1, values.length))}
												y={15 + row * (84 / data.values.length)}
												width={(stage.w - 40) / Math.max(1, values.length) - 3}
												height={Math.min(17, 84 / data.values.length - 3)}
												rx="2"
												fill={tensorColor(value, data.scale, data.probability)}
												stroke={data.start + row === data.selectedRow ? '#8957bb' : 'transparent'}
												stroke-width="1.2"
												><title
													>{data.tensor?.name} · {data.tensor?.rowLabels?.[data.start + row] ??
														`saved row ${data.start + row}`} · column {column} = {numeric(
														value
													)}</title
												></rect
											>{/each}</g
									>{/each}
								<text x={stage.w / 2} y={stage.h - 13} text-anchor="middle" class="recorded-label"
									>{data.tensor?.role === 'attention'
										? `Head ${(data.tensor.head ?? headIndex) + 1} · recorded slice`
										: 'Recorded tensor slice'} ↗</text
								>
							{:else}
								{#if stage.id === 'text-input'}<text
										x={stage.w / 2}
										y="40"
										text-anchor="middle"
										class="prompt-text"
										>{#each promptLines as line, index}<tspan
												x={stage.w / 2}
												dy={index === 0 ? 0 : 20}>{line.trim()}</tspan
											>{/each}</text
									>{:else}<text x={stage.w / 2} y="61" text-anchor="middle" class="concept-symbol"
										>{stage.id === 'memory'
											? 'K · V'
											: stage.id === 'query'
												? 'Q'
												: stage.id === 'selective-state'
													? 'hₜ'
													: stage.id === 'decoder-input'
														? '〈start〉'
														: '→'}</text
									>{/if}<text
									x={stage.w / 2}
									y={stage.h - 17}
									text-anchor="middle"
									class="recorded-label">Architecture context</text
								>
							{/if}
							<text x={stage.w / 2} y={stage.h + 27} text-anchor="middle" class="stage-note"
								>{stageNote(stage)}</text
							>
						</g>
					{/if}
				{/each}
			</svg>
		</div>
		<div class="recorded-operations">
			<span>Inspect a recorded operation</span>
			<div>
				{#each operations as operation}<button
						class:active={selectedOperation === operation}
						on:click={() => inspectOperation(operation)}>{operation.label} ↗</button
					>{/each}
			</div>
		</div>
		{#if example.output?.text && !trace.provenance.partial}<div class="recorded-output">
				<span>Recorded output</span><strong>{example.output.text}</strong>
			</div>{/if}
		<div class="provenance">
			<div>
				<span class="recording-dot"></span><strong
					>{trace.provenance.partial
						? 'Pretrained block recording'
						: 'Pretrained model recording'}</strong
				><a href={provenanceUrl} target="_blank" rel="noreferrer">{trace.provenance.modelId} ↗</a>
			</div>
			<p>{trace.provenance.scope}</p>
			<span
				>Replay uses saved tensors. These architecture examples do not run new prompts in the
				browser.</span
			>
		</div>
	{/if}
</section>

{#if selectedOperation && layer && example}<TensorInspector
		operation={selectedOperation}
		selectedToken={inspectorToken}
		selectedTokenLabel={inspectorTokenLabel}
		selectedHead={headIndex}
		context={`${trace?.provenance.modelId} · ${layer.label ?? `block ${layer.index + 1}`} · ${example.label}`}
		on:close={() => (selectedOperation = null)}
	/>{/if}

<style>
	.architecture-explorer {
		font-family: Arial, sans-serif;
		color: #7b638d;
		width: 100%;
		padding: 14px 34px 28px;
		box-sizing: border-box;
		background: #fff;
	}
	.trace-toolbar {
		display: flex;
		align-items: flex-end;
		gap: 15px;
		min-height: 57px;
	}
	.example-control {
		display: flex;
		align-items: center;
		gap: 13px;
		min-width: 0;
	}
	.example-control > span {
		font-size: 12px;
		color: #a18baa;
		white-space: nowrap;
	}
	.segments {
		display: flex;
		padding: 3px;
		border: 1px solid #e5d9ef;
		border-radius: 7px;
		background: #fcfaff;
		flex-wrap: wrap;
	}
	.segments button {
		padding: 8px 13px;
		font-size: 12px;
		border-radius: 4px;
		color: #9474ad;
		min-height: 32px;
	}
	.segments button.active {
		background: #a37bc8;
		color: white;
	}
	.segments button:hover:not(.active) {
		background: #f0e7f8;
	}
	.replay {
		margin-left: auto;
		border: 1px solid #d9c8e8;
		border-radius: 7px;
		padding: 10px 15px;
		font-size: 13px;
		white-space: nowrap;
		background: white;
		color: #8c64ad;
	}
	.replay span {
		margin-left: 14px;
	}
	.replay.active {
		background: #f2e8fb;
	}
	.trace-link {
		font-size: 12px;
		color: #9c81b4;
		text-decoration: none;
		padding: 10px 0;
		white-space: nowrap;
	}
	.title-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 20px;
		margin-top: 32px;
	}
	.title-row h1 {
		font-size: 25px;
		font-weight: 600;
		color: #5e4a73;
		margin: 0 0 9px;
	}
	.scope-badge {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-top: 11px;
		font-size: 11px;
	}
	.scope-badge strong {
		font-weight: 500;
		padding: 4px 7px;
		background: #f4eef9;
		border: 1px solid #e7dbee;
		border-radius: 4px;
		color: #9470ac;
	}
	.scope-badge a {
		color: #9b85ac;
		text-decoration: none;
	}
	.title-row p {
		margin: 0;
		color: #a18cad;
		font-size: 13px;
		line-height: 1.5;
	}
	.layer-controls {
		display: flex;
		align-items: center;
		gap: 7px;
		white-space: nowrap;
		color: #806991;
		font-size: 15px;
	}
	.layer-controls button,
	.position-controls > button {
		width: 29px;
		height: 29px;
		border: 1px solid #d9cbea;
		border-radius: 5px;
		background: white;
		color: #996fba;
		font-size: 24px;
		line-height: 1;
	}
	.layer-controls button:disabled,
	.position-controls > button:disabled {
		opacity: 0.35;
	}
	.layer-controls small {
		font-size: 11px;
		color: #a492af;
		margin-left: 5px;
	}
	.position-row {
		display: flex;
		align-items: center;
		gap: 25px;
		margin-top: 22px;
		font-size: 12px;
	}
	.position-controls,
	.head-control {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.position-controls label,
	.head-control label {
		color: #8d789d;
	}
	.position-controls select,
	.head-control select {
		border: 1px solid #ded0ed;
		border-radius: 5px;
		padding: 6px 10px;
		background: #fcfaff;
		color: #9475b0;
		max-width: 250px;
		height: 31px;
		font-size: 12px;
	}
	.position-controls select {
		min-width: 145px;
	}
	.head-note {
		color: #a18cad;
	}
	.dimensions {
		margin-left: auto;
		color: #a490b3;
	}
	.diagram-scroll {
		width: 100%;
		overflow-x: auto;
		margin-top: 16px;
		min-height: 400px;
	}
	.architecture-diagram {
		width: 100%;
		height: clamp(400px, calc(100dvh - 450px), 610px);
		min-width: 1120px;
		display: block;
		overflow: visible;
	}
	.flow-ribbon {
		fill: none;
		stroke: #bda0e1;
		stroke-width: 21;
		opacity: 0.12;
	}
	.flow-ribbon.illuminated {
		opacity: 0.42;
		stroke: #9d63d4;
	}
	.flow-line {
		fill: none;
		stroke: #d0bce5;
		stroke-width: 1.5;
	}
	.stage-title {
		font-size: 16px;
		font-weight: 500;
		fill: #8c759f;
	}
	.stage-note {
		font-size: 11px;
		fill: #aa95b9;
	}
	.stage-back {
		fill: #f6f3fa;
		stroke: #ede8f3;
		stroke-width: 1;
	}
	.stage-surface {
		fill: #fff;
		stroke: #e7def0;
		stroke-width: 1.4;
	}
	.interactive {
		cursor: pointer;
	}
	.interactive:hover .stage-surface,
	.highlighted .stage-surface {
		fill: #fcf9ff;
		stroke: #ad84ce;
		stroke-width: 2;
	}
	.recorded-label {
		font-size: 10px;
		fill: #a58eb7;
	}
	.edge-label {
		font-size: 10px;
		fill: #a993bb;
	}
	.lane-title {
		font-size: 11px;
		letter-spacing: 1.2px;
		fill: #b19cc2;
	}
	.token {
		cursor: pointer;
	}
	.token-text {
		font-size: 13px;
		fill: #697285;
	}
	.token-index {
		font-size: 10px;
		fill: #b5a9c2;
	}
	.patch {
		cursor: pointer;
	}
	.patch:hover {
		fill: #ab73d783;
	}
	.prediction-label {
		font-size: 13px;
		fill: #a9a0b3;
	}
	.prediction-value {
		font-size: 11px;
		fill: #aca1b6;
	}
	.top-prediction {
		fill: #9656df;
		font-weight: 600;
	}
	.output-stage {
		cursor: pointer;
	}
	.prompt-text {
		font-size: 12px;
		fill: #9174a8;
	}
	.concept-symbol {
		font:
			italic 29px Georgia,
			serif;
		fill: #aa8ebf;
	}
	.recurrence {
		stroke: #b99ad7;
		stroke-width: 2;
		fill: none;
	}
	.recurrence-label {
		font-size: 13px;
		fill: #9872b6;
	}
	.cache-label {
		font-size: 11px;
		letter-spacing: 0.5px;
		fill: #9d78b7;
	}
	.recorded-operations {
		display: flex;
		align-items: center;
		gap: 16px;
		margin-top: 0;
		position: relative;
	}
	.recorded-operations > span {
		font-size: 11px;
		color: #a28eae;
		white-space: nowrap;
	}
	.recorded-operations > div {
		display: flex;
		gap: 7px;
		flex-wrap: wrap;
	}
	.recorded-operations button {
		font-size: 11px;
		padding: 7px 10px;
		color: #9876b1;
		background: #fcfaff;
		border: 1px solid #e5daef;
		border-radius: 5px;
	}
	.recorded-operations button:hover,
	.recorded-operations button.active {
		background: #eee2f7;
		border-color: #c5a8dc;
	}
	.recorded-output {
		display: flex;
		align-items: baseline;
		gap: 15px;
		padding-top: 20px;
		font-size: 14px;
		color: #8c65ad;
	}
	.recorded-output span {
		font-size: 11px;
		color: #a18cad;
	}
	.recorded-output strong {
		font-weight: 500;
	}
	.provenance {
		border-top: 1px solid #f0eaf5;
		margin-top: 20px;
		padding-top: 15px;
		font-size: 11px;
		color: #aa99b4;
		line-height: 1.5;
	}
	.provenance > div {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.provenance strong {
		font-size: 11px;
		font-weight: 500;
		color: #8d749f;
	}
	.provenance a {
		color: #9a7caf;
		text-decoration: none;
	}
	.provenance p {
		margin: 5px 0;
		max-width: 1200px;
	}
	.provenance > span {
		font-size: 10px;
		color: #b2a4bb;
	}
	.recording-dot {
		width: 6px;
		height: 6px;
		background: #b08bd0;
		border-radius: 50%;
		display: inline-block;
	}
	.loading,
	.error {
		padding: 100px 30px;
		color: #9a7eb0;
		font-size: 16px;
		text-align: center;
	}
	.loading-line {
		width: 160px;
		height: 4px;
		margin: 0 auto 25px;
		background: #e6d8f3;
		border-radius: 5px;
	}
	.error strong {
		font-size: 20px;
	}
	.error p {
		font-size: 14px;
	}
	.error button {
		padding: 8px 20px;
		border: 1px solid #d5bee8;
		border-radius: 6px;
		margin-top: 12px;
	}
	button:focus-visible,
	select:focus-visible,
	a:focus-visible,
	.operation-stage:focus-visible,
	.token:focus-visible,
	.patch:focus-visible,
	.output-stage:focus-visible {
		outline: 2px solid #945cce;
		outline-offset: 4px;
	}
	@media (min-width: 1800px) {
		.architecture-explorer {
			padding-left: 50px;
			padding-right: 50px;
		}
		.diagram-scroll {
			margin-top: 16px;
		}
		.recorded-operations {
			margin-top: 0;
		}
	}
	@media (max-width: 1000px) {
		.architecture-explorer {
			padding: 12px 18px 25px;
		}
		.trace-toolbar {
			flex-wrap: wrap;
		}
		.example-control {
			flex-wrap: wrap;
		}
		.title-row {
			align-items: flex-start;
			flex-wrap: wrap;
			margin-top: 22px;
		}
		.title-row h1 {
			font-size: 22px;
		}
		.position-row {
			flex-wrap: wrap;
			gap: 12px;
		}
		.dimensions {
			margin-left: 0;
		}
		.diagram-scroll {
			margin-top: 16px;
		}
		.recorded-operations {
			margin-top: 0;
			align-items: flex-start;
			flex-direction: column;
		}
		.trace-link {
			margin-left: 0;
		}
	}
</style>
