<script lang="ts">
	import {
		tokens,
		modelMeta,
		rootRem,
		weightPopover,
		tooltip,
		attentionHeadIdx,
		blockIdx,
		moeModel
	} from '~/store';
	import * as d3 from 'd3';
	import { onMount, tick } from 'svelte';
	import resolveConfig from 'tailwindcss/resolveConfig';
	import tailwindConfig from '../../tailwind.config';
	import { gradientMap } from '~/constants/gradient';
	import {
		ATTENTION_HEAD_1,
		ATTENTION_HEAD_BACK,
		ATTENTION_OUT,
		EMBEDDING,
		LOGIT,
		MLP,
		TRANSFORMER_BLOCKS
	} from '~/constants/opacity';
	import { textPages } from '~/utils/textbookPages';
	import { hybridRecording, hybridInspector } from '$lib/hybrid/store';
	import { t5Inspector } from '$lib/seq2seq/store';
	import { getRenderedEndpointPairs } from '~/utils/sankeyGeometry';

	const { theme } = resolveConfig(tailwindConfig);

	let svgBackEl: HTMLOrSVGElement;
	let svgEl: HTMLOrSVGElement;

	let resizeObserver: ResizeObserver;
	let screenWidth: number;
	let mounted = false;

	$: curveFactor = Math.floor(screenWidth / 1000) || 1;
	$: defaultCurveOffset = 40 * (curveFactor - 1) + 80;

	type PathMap = Record<
		string,
		{
			from: string;
			to: string;
			gradientId?: string;
			fill?: string;
			opacity?: number;
			curve?: number;
			type?: 'stroke' | 'shape';
			id?: string;
			pathGenerator?: (source: DOMRect, target: DOMRect, curve: number) => string;
		}[]
	>;

	$: isMLA = $moeModel?.architecture === 'mla';
	$: isT5Decoder = $moeModel?.architecture === 'encoder-decoder' && $moeModel?.stack === 'decoder';
	$: isMambaOnly =
		$moeModel?.architecture === 'hybrid' &&
		$hybridRecording?.document.layers[$blockIdx].type === 'mamba';
	$: usesRoPE =
		!!$moeModel &&
		!['mla', 'vit', 'encoder-decoder'].includes($moeModel.architecture ?? '') &&
		$moeModel.maskMode !== 'bidirectional';
	$: backPaths = [
		{
			from: '.qkv .qkv-column .query',
			to: usesRoPE ? '.head-block .query .rotary-port' : '.head-block .query .vector',
			// fill: theme.colors.blue[defaultGradientBrightness],
			gradientId: 'blue-blue2',
			opacity: ATTENTION_HEAD_BACK
		},
		{
			from: '.qkv .qkv-column .key',
			to: usesRoPE ? '.head-block .key .rotary-port' : '.head-block .key .vector',
			// fill: theme.colors.red[defaultGradientBrightness],
			gradientId: 'red-red2',
			opacity: ATTENTION_HEAD_BACK
		},
		{
			from: '.qkv .qkv-column .value',
			to: '.head-block .value .vector',
			gradientId: 'green-green2',
			// fill: theme.colors.green[defaultGradientBrightness],
			opacity: ATTENTION_HEAD_BACK
		},
		{
			from: isT5Decoder
				? '.cross-attention .cross-head-out .vector'
				: '.attention .head-out .vector',
			to: '.mlp .initial .vector',
			gradientId: 'transparent-purple',
			opacity: ATTENTION_HEAD_BACK
		}
	];
	$: backPathMap = {
		attention: (isMLA ? backPaths.slice(3) : backPaths).map((d) => ({
			...d,
			from: `.block-steps.main ${d.from}`,
			to: `.block-steps.main ${d.to}`
		}))
	};

	$: qkvPaths = [
		{
			from: '.qkv .vector-column .column.vectors .vector',
			to: isMLA ? '.qkv .mla-hidden-column .vector' : '.qkv .qkv-column .vector',
			gradientId: $blockIdx === 0 ? 'gray-blue' : 'transparent-blue',
			opacity: EMBEDDING,
			pathGenerator: (source, target, curve: number) => {
				const scrollTop = window.scrollY;
				const scrollLeft = window.scrollX;

				const rightOffset = 30;
				const { curveOffset } = pathAdjustor(source, target, curve);
				return `
					M ${source.right + scrollLeft},${source.top + scrollTop}
					L ${source.right + rightOffset + scrollLeft},${source.top + scrollTop}
					C ${source.right + rightOffset + scrollLeft + curveOffset},${source.top + scrollTop} ${target.left + scrollLeft - curveOffset},${target.top + scrollTop} ${target.left + scrollLeft},${target.top + scrollTop}
					L ${target.left + scrollLeft},${target.bottom + scrollTop}
					C ${target.left + scrollLeft - curveOffset},${target.bottom + scrollTop} ${source.right + rightOffset + scrollLeft + curveOffset},${source.bottom + scrollTop} ${source.right + rightOffset + scrollLeft},${source.bottom + scrollTop}
					L ${source.right + scrollLeft},${source.bottom + scrollTop}
					Z
				`;
			},
			onMouseOver: () => {
				const paths = d3.select(`g.qkv`).selectAll('path');
				paths.transition().duration(100).style('opacity', 1);
				tooltip.set(
					isMLA
						? 'Inspect MLA projections from normalized hidden states'
						: 'click to see QKV calculation'
				);
			},
			onMouseOut: () => {
				const paths = d3.select(`g.qkv`).selectAll('path');
				paths.transition().duration(100).style('opacity', EMBEDDING);
				tooltip.set(null);
			},
			onMouseClick: (e, d) => {
				e.stopPropagation();
				textPages.find((page) => page.id === 'qkv')?.complete();

				if ($weightPopover === 'qkv') weightPopover.set(null);
				else weightPopover.set('qkv');
			}
		}
	];
	$: selfAttentionPaths = [
		{
			from: '.qkv .qkv-column .query .sub-vector.head1',
			to: usesRoPE ? '.head-block .query .rotary-port' : '.head-block .query .vector',
			// fill: theme.colors.blue[200]
			gradientId: 'blue-blue',
			opacity: ATTENTION_HEAD_1
		},
		{
			from: '.qkv .qkv-column .key .sub-vector.head1',
			to: usesRoPE ? '.head-block .key .rotary-port' : '.head-block .key .vector',
			// fill: theme.colors.red[200]
			gradientId: 'red-red',
			opacity: ATTENTION_HEAD_1
		},
		{
			from: '.qkv .qkv-column .value .sub-vector.head1',
			to: '.head-block .value .vector',
			// fill: theme.colors.green[200]
			gradientId: 'green-green',
			opacity: ATTENTION_HEAD_1
		},
		{
			from: isT5Decoder
				? '.cross-attention .cross-head-out .vector'
				: '.attention .head-out .vector',
			to: '.mlp .initial .head1',
			gradientId: 'purple-purple',
			opacity: ATTENTION_HEAD_1
			// fill: theme.colors.purple[200]
		},
		{
			from: '.head-block .head1.key .vector',
			to: `.attention-matrix.attention-initial .main g.g-row-${$tokens.length - 1} rect`,
			id: 'key-to-attention',
			type: 'stroke',
			gradientId: 'red-purple',
			opacity: ATTENTION_HEAD_1,
			unique: true,
			curve: 20,
			pathGenerator: (source, target, curve) => {
				const scrollTop = window.scrollY;
				const scrollLeft = window.scrollX;

				const sourceMiddleY = source.top + scrollTop + source.height / 2;
				const targetMiddleX = target.left + scrollLeft + target.width / 2;
				// const { curveOffset } = pathAdjustor(source, target, curve);
				let curveOffset = curve;
				return `
			    M ${source.right + scrollLeft},${sourceMiddleY}
			    L ${targetMiddleX - curveOffset},${sourceMiddleY}
			    A ${curveOffset},${curveOffset} 0 0 1 ${targetMiddleX},${sourceMiddleY + curveOffset}
			    L ${targetMiddleX},${target.bottom + scrollTop}
			`;
			}
		},
		{
			from: '.head-block .query .vector',
			to: `.attention-initial .main g.g-row`,
			gradientId: 'blue-purple',
			id: 'query-to-attention',
			unique: true,
			type: 'stroke',
			opacity: ATTENTION_HEAD_1,
			curve: curveFactor * 30,
			pathGenerator: (source, target, curve) => {
				let curveOffset = curve;

				const scrollTop = window.scrollY;
				const scrollLeft = window.scrollX;

				const sourceMiddleY = source.top + scrollTop + source.height / 2;
				const targetMiddleY = target.top + scrollTop + target.height / 2;
				const controlPoint1X = source.right + scrollLeft + curveOffset;
				const controlPoint2X = target.left + scrollLeft - curveOffset;

				return `
			    M ${source.right + scrollLeft},${sourceMiddleY}
			    C ${controlPoint1X},${sourceMiddleY} ${controlPoint2X},${targetMiddleY} ${target.left + scrollLeft},${targetMiddleY}
					L ${target.right + scrollLeft},${targetMiddleY}
			`;
			}
		},
		{
			from: '.attention-matrix.attention-out .main svg',
			to: '.attention .head-out',
			gradientId: 'transparent-purple2',
			id: 'to-attention-out',
			unique: true,
			opacity: ATTENTION_OUT,
			curve: 20,
			pathGenerator: (source, target, curve) => {
				const { curveOffset } = pathAdjustor(source, target, curve);

				const scrollTop = window.scrollY;
				const scrollLeft = window.scrollX;

				return `
			    M ${source.right + scrollLeft},${source.top + scrollTop}
			    C ${source.right + scrollLeft + curveOffset},${source.top + scrollTop} ${target.left + scrollLeft - curveOffset},${target.top + scrollTop} ${target.left + scrollLeft},${target.top + scrollTop}
			    L ${target.left + scrollLeft},${target.bottom + scrollTop}
			    C ${target.left + scrollLeft - curveOffset},${target.bottom + scrollTop} ${source.right + scrollLeft + curveOffset},${source.bottom + scrollTop} ${source.right + scrollLeft},${source.bottom + scrollTop}
			    Z
			`;
			},
			onMouseOver: () => {
				d3.select('path.value-to-out').transition().duration(100).style('opacity', 0.8);
				d3.select('path.to-attention-out').transition().duration(100).style('opacity', 0.8);

				tooltip.set('click to see Attention Out calculation');
			},
			onMouseOut: () => {
				d3.select('path.value-to-out').transition().duration(100).style('opacity', ATTENTION_OUT);
				d3.select('path.to-attention-out')
					.transition()
					.duration(100)
					.style('opacity', ATTENTION_OUT);

				tooltip.set(null);
			},
			onMouseClick: (e, d) => {
				e.stopPropagation();
				textPages.find((page) => page.id === 'output-concatenation')?.complete();

				if ($weightPopover === 'attention') weightPopover.set(null);
				else weightPopover.set('attention');
			}
		},
		{
			from: '.head-block .value',
			to: '.attention .head-out',
			gradientId: 'green-purple',
			id: 'to-attention-out value-to-out',
			unique: true,
			opacity: ATTENTION_OUT,
			curve: curveFactor * 30,
			pathGenerator: (source, target, curve) => {
				const scrollTop = window.scrollY;
				const scrollLeft = window.scrollX;
				const { curveOffset } = pathAdjustor(source, target, curve);
				return `
        M ${source.right + scrollLeft},${source.top + scrollTop} 
        C ${target.left - curveOffset},${source.top + scrollTop} ${target.left + scrollLeft - curveOffset},${source.top + scrollTop} ${target.left + scrollLeft},${target.top + scrollTop}
        L ${target.left + scrollLeft},${target.bottom + scrollTop}
        C ${target.left + scrollLeft - curveOffset},${source.bottom + scrollTop} ${target.left - curveOffset},${source.bottom + scrollTop} ${source.right + scrollLeft},${source.bottom + scrollTop}
        Z
    `;
			},
			onMouseOver: () => {
				d3.select('path.value-to-out').transition().duration(100).style('opacity', 0.8);
				d3.select('path.to-attention-out').transition().duration(100).style('opacity', 0.8);

				tooltip.set('click to see Attention Out calculation');
			},
			onMouseOut: () => {
				d3.select('path.value-to-out').transition().duration(100).style('opacity', ATTENTION_OUT);
				d3.select('path.to-attention-out')
					.transition()
					.duration(100)
					.style('opacity', ATTENTION_OUT);

				tooltip.set(null);
			},
			onMouseClick: (e, d) => {
				e.stopPropagation();
				textPages.find((page) => page.id === 'output-concatenation')?.complete();

				if ($weightPopover === 'attention') weightPopover.set(null);
				else weightPopover.set('attention');
			}
		}
	];
	const mlaConnector = (source: DOMRect, target: DOMRect) => {
		const x1 = source.right + window.scrollX,
			y1 = (source.top + source.bottom) / 2 + window.scrollY;
		const x2 = target.left + window.scrollX,
			y2 = (target.top + target.bottom) / 2 + window.scrollY;
		const mid = (x1 + x2) / 2;
		return `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`;
	};
	// MLA has one projection pipeline: normalized hidden states enter the flow,
	// and its final Q′/K′/V outputs feed the existing attention token vectors.
	$: mlaPaths = [
		{
			from: '.qkv .mla-hidden-column',
			to: '.mla-input',
			gradientId: 'gray-blue',
			type: 'stroke',
			opacity: 0.8,
			id: 'mla-hidden-input',
			pathGenerator: mlaConnector
		},
		...['key', 'query', 'value'].map((kind, index) => ({
			from: `.mla-output-${kind}`,
			to: `.head-block .column.${kind}`,
			gradientId: ['red-red2', 'blue-blue2', 'green-green2'][index],
			type: 'stroke',
			opacity: 0.9,
			id: `mla-${kind}-output`,
			pathGenerator: mlaConnector
		}))
	];
	$: attentionPaths = [
		...(isMLA ? [...mlaPaths, ...selfAttentionPaths.slice(3)] : selfAttentionPaths),
		...(isT5Decoder
			? [
					{
						from: '.attention .head-out .vector',
						to: '.cross-attention .cross-input .vector',
						gradientId: 'purple-purple',
						opacity: ATTENTION_HEAD_1
					},
					{
						...selfAttentionPaths[4],
						from: '.cross-attention .cross-key .vector',
						to: `.cross-attention .cross-matrix g.g-row-${$tokens.length - 1} rect`,
						id: 'cross-key-to-attention'
					},
					{
						...selfAttentionPaths[5],
						from: '.cross-attention .cross-query .vector',
						to: '.cross-attention .cross-matrix g.g-row',
						id: 'cross-query-to-attention'
					},
					{
						...selfAttentionPaths[6],
						from: '.cross-attention .cross-matrix svg',
						to: '.cross-attention .cross-head-out',
						id: 'cross-to-attention-out',
						onMouseClick: (e) => {
							e.stopPropagation();
							t5Inspector.set('cross');
						}
					},
					{
						...selfAttentionPaths[7],
						from: '.cross-attention .cross-value',
						to: '.cross-attention .cross-head-out',
						id: 'cross-value-to-out',
						onMouseClick: (e) => {
							e.stopPropagation();
							t5Inspector.set('cross');
						}
					}
				]
			: [])
	];

	$: mlpUpPaths = [
		{
			from: '.mlp .column.initial .vector',
			to: '.mlp .second-layer .vector',
			gradientId: 'purple-indigo',
			curve: 50 + (curveFactor - 1) * 20,
			opacity: MLP,
			pathGenerator: (source, target, curve: number) => {
				const scrollTop = window.scrollY;
				const scrollLeft = window.scrollX;

				const rightOffset = rootRem * 3;
				const { curveOffset } = pathAdjustor(source, target, curve);
				return `
					M ${source.right + scrollLeft},${source.top + scrollTop}
					L ${source.right + rightOffset + scrollLeft},${source.top + scrollTop}
					C ${source.right + rightOffset + scrollLeft + curveOffset},${source.top + scrollTop} ${target.left + scrollLeft - curveOffset},${target.top + scrollTop} ${target.left + scrollLeft},${target.top + scrollTop}
					L ${target.left + scrollLeft},${target.bottom + scrollTop}
					C ${target.left + scrollLeft - curveOffset},${target.bottom + scrollTop} ${source.right + rightOffset + scrollLeft + curveOffset},${source.bottom + scrollTop} ${source.right + rightOffset + scrollLeft},${source.bottom + scrollTop}
					L ${source.right + scrollLeft},${source.bottom + scrollTop}
					Z
				`;
			},
			onMouseOver: () => {
				const paths = d3.select(`g.mlpUp`).selectAll('path');
				paths.transition().duration(100).style('opacity', 0.6);
				tooltip.set('click to see MLP process');
			},
			onMouseOut: () => {
				const paths = d3.select(`g.mlpUp`).selectAll('path');
				paths.transition().duration(100).style('opacity', MLP);
				tooltip.set(null);
			},
			onMouseClick: (e, d) => {
				e.stopPropagation();

				if ($weightPopover === 'mlpUp') weightPopover.set(null);
				else weightPopover.set('mlpUp');
			}
		}
	];
	$: mlpDownPaths = [
		{
			from: '.mlp .second-layer .vector',
			to: '.mlp .column.out .vector',
			gradientId: 'indigo-blue',
			curve: 50 + (curveFactor - 1) * 20,
			opacity: MLP,
			pathGenerator: (source, target, curve: number) => {
				const scrollTop = window.scrollY;
				const scrollLeft = window.scrollX;

				const leftOffset = rootRem * 1.5;
				const { curveOffset } = pathAdjustor(source, target, curve);
				return `
					M ${source.right + scrollLeft},${source.top + scrollTop}
					C ${source.right + scrollLeft + curveOffset},${source.top + scrollTop} ${target.left - leftOffset + scrollLeft - curveOffset},${target.top + scrollTop} ${target.left - leftOffset + scrollLeft},${target.top + scrollTop}
					L ${target.left + scrollLeft},${target.top + scrollTop}
					L ${target.left + scrollLeft},${target.bottom + scrollTop}
					L ${target.left - leftOffset + scrollLeft},${target.bottom + scrollTop}
					C ${target.left - leftOffset + scrollLeft - curveOffset},${target.bottom + scrollTop} ${source.right + scrollLeft + curveOffset},${source.bottom + scrollTop} ${source.right + scrollLeft},${source.bottom + scrollTop}
					L ${source.right + scrollLeft},${source.bottom + scrollTop}
					Z
				`;
			},
			onMouseOver: () => {
				const paths = d3.select(`g.mlpDown`).selectAll('path');
				paths.transition().duration(100).style('opacity', 0.6);
				tooltip.set('click to see MLP process');
			},
			onMouseOut: () => {
				const paths = d3.select(`g.mlpDown`).selectAll('path');
				paths.transition().duration(100).style('opacity', MLP);
				tooltip.set(null);
			},
			onMouseClick: (e, d) => {
				e.stopPropagation();

				if ($weightPopover === 'mlpDown') weightPopover.set(null);
				else weightPopover.set('mlpDown');
			}
		}
	];

	$: pathMap = {
		mamba: isMambaOnly
			? [
					{
						from: '.hybrid-projection .column.vectors .vector',
						to: '.mamba-core .state-input',
						gradientId: 'blue-blue',
						opacity: ATTENTION_HEAD_1
					},
					{
						from: '.mamba-core .gated',
						to: '.hybrid-output .column.initial .vector',
						gradientId: 'purple-purple',
						opacity: ATTENTION_HEAD_1
					},
					{
						from: '.hybrid-output .column.initial .vector',
						to: '.hybrid-output .out .vector',
						gradientId: 'purple-purple',
						opacity: ATTENTION_HEAD_BACK,
						onMouseClick: () => hybridInspector.set('output')
					}
				]
			: [],
		embedding: [
			{
				from: '.embedding .vector-column .column.vectors .vector',
				to: '.block-steps.main .qkv .vector-column .column.vectors .vector',
				gradientId: 'gray-white-blue',
				opacity: TRANSFORMER_BLOCKS
			}
		],
		qkv: qkvPaths.map((d) => ({
			...d,
			from: `.block-steps.main ${d.from}`,
			to: `.block-steps.main ${d.to}`
		})),
		attention: attentionPaths.map((d) => ({
			...d,
			from: `.block-steps.main ${d.from}`,
			to: `.block-steps.main ${d.to}`
		})),
		'transformer-blocks': [
			{
				from: '.block-steps.main .mlp .column.out .vector',
				to: '.transformer-blocks .column.final .vector',
				gradientId: $blockIdx === $modelMeta.layer_num - 1 ? 'blue' : 'blue-white-blue',
				opacity: $blockIdx === $modelMeta.layer_num - 1 ? MLP : TRANSFORMER_BLOCKS,
				pathGenerator: (source, target, curve) => {
					const scrollTop = window.scrollY;
					const scrollLeft = window.scrollX;
					const { curveOffset } = pathAdjustor(source, target, curve);
					return `
        M ${source.right + scrollLeft},${source.top + scrollTop}
        C ${source.right + scrollLeft + curveOffset},${source.top + scrollTop} ${target.left + scrollLeft - curveOffset},${target.top + scrollTop} ${target.left + scrollLeft},${target.top + scrollTop}
        L ${target.left + scrollLeft},${target.bottom + scrollTop}
        C ${target.left + scrollLeft - curveOffset},${target.bottom + scrollTop} ${source.right + scrollLeft + curveOffset},${source.bottom + scrollTop} ${source.right + scrollLeft},${source.bottom + scrollTop}
        Z
    `;
				}
			}
		],
		softmax: [
			{
				from: '.transformer-blocks .final .vector.last-token',
				to: '.softmax .content .vector',
				gradientId: 'blue-gray',
				opacity: LOGIT,
				unique: true,
				onMouseOver: () => {
					const paths = d3.select(`g.softmax`).selectAll('path');
					paths.transition().duration(100).style('opacity', 1);
					tooltip.set('click to see Logits calculation');
					// showTooltip('');
				},
				onMouseOut: () => {
					const paths = d3.select(`g.softmax`).selectAll('path');
					paths.transition().duration(100).style('opacity', LOGIT);
					// hideTooltip();
					tooltip.set(null);
				},
				onMouseClick: (e, d) => {
					e.stopPropagation();
					textPages.find((page) => page.id === 'output-logit')?.complete();

					if ($weightPopover === 'softmax') weightPopover.set(null);
					else weightPopover.set('softmax');
				}
			}
		]
	};

	const createGradients = () => {
		const svg = d3.select(svgEl);
		const defs = svg.append('defs');

		Object.keys(gradientMap).forEach((key) => {
			const stops = gradientMap[key];
			const grad = defs
				.append('linearGradient')
				.attr('id', key)
				.attr('class', key)
				.attr('x1', '0%')
				.attr('y1', '0%')
				.attr('x2', '100%')
				.attr('y2', '0%');

			const gradClone = defs
				.append('linearGradient')
				.attr('id', key + '-last')
				.attr('class', key)
				.attr('x1', '0%')
				.attr('y1', '0%')
				.attr('x2', '100%')
				.attr('y2', '0%');

			Object.keys(stops).forEach((stop) => {
				let color, opacity;
				if (typeof stops[stop] !== 'string') {
					color = stops[stop].color;
					opacity = stops[stop].opacity;
				} else {
					color = stops[stop];
					opacity = 1;
				}

				grad
					.append('stop')
					.attr('offset', `${stop}%`)
					.attr('stop-color', color)
					.attr('stop-opacity', opacity);

				gradClone
					.append('stop')
					.attr('offset', `${stop}%`)
					.attr('stop-color', color)
					.attr('stop-opacity', opacity);
			});
		});
	};

	const drawPath = async () => {
		await tick();
		if (!mounted) return;
		const svg = d3.select(svgEl);
		const svgBack = d3.select(svgBackEl);

		[
			{ dataMap: pathMap, svg },
			{ dataMap: backPathMap, svg: svgBack }
		].forEach(({ dataMap, svg }) => {
			const g = svg
				.selectAll('g.path-group')
				.data(Object.keys(dataMap))
				.join('g')
				.attr('class', (d) => `path-group ${d}`);

			g.selectAll('path.sankey-path')
				.data((d) => {
					const data = dataMap[d].map((item) => {
						const { from, to, curve, pathGenerator, gradientId, unique, ...rest } = item;
						const sources = d3.selectAll(from).nodes() as Element[];
						const targets = d3.selectAll(to).nodes() as Element[];

						return getRenderedEndpointPairs(sources, targets).map(
							({ source, target, index: i }) => {
								const curveOffset = curve || defaultCurveOffset;

								const generator = pathGenerator || defaultPathGenerator;
								const path = generator(source, target, curveOffset);

								const isLast = targets.length > 1 && i === sources.length - 1;
								let gradUrl = gradientId;

								if (isLast && !unique && document.getElementById(gradientId + '-last')) {
									gradUrl = gradientId + '-last';
								}

								return {
									isLast: i === sources.length - 1,
									path,
									fill:
										item.type === 'stroke'
											? 'none'
											: item.gradientId
												? `url(#${gradUrl})`
												: item.fill,
									opacity: item.opacity,
									stroke:
										item.type === 'stroke'
											? item.gradientId
												? `url(#${item.gradientId})`
												: item.fill
											: 'none',
									clickable: !!item.onMouseClick,
									...rest
								};
							}
						);
					});

					return data.flat();
				})
				.join('path')
				.attr('class', (d) => `sankey-path ${d.id || ''} ${d.isLast ? 'last' : ''}`)
				.attr('fill', (d) => d.fill)
				.attr('stroke', (d) => d.stroke)
				.attr('stroke-width', 2)
				.attr('opacity', (d) => d.opacity)
				.attr('cursor', (d) => d.clickable && 'pointer')
				.attr('d', (d) => d.path)
				.on('mouseenter', (e, d) => d.onMouseOver?.(d))
				.on('mouseleave', (e, d) => d.onMouseOut?.(d))
				.on('click', (e, d) => d.onMouseClick?.(e, d));
		});
	};

	const defaultPathGenerator = (source, target, curve: number) => {
		const scrollTop = window.scrollY;
		const scrollLeft = window.scrollX;
		const { curveOffset } = pathAdjustor(source, target, curve);

		return `
        M ${source.right + scrollLeft},${source.top + scrollTop}
        C ${source.right + scrollLeft + curveOffset},${source.top + scrollTop} ${target.left + scrollLeft - curveOffset},${target.top + scrollTop} ${target.left + scrollLeft},${target.top + scrollTop}
        L ${target.left + scrollLeft},${target.bottom + scrollTop}
        C ${target.left + scrollLeft - curveOffset},${target.bottom + scrollTop} ${source.right + scrollLeft + curveOffset},${source.bottom + scrollTop} ${source.right + scrollLeft},${source.bottom + scrollTop}
        Z
    `;
	};

	const pathAdjustor = (source, target, curve: number) => {
		const distance = target.left - source.right;
		const maxDistance = 100;

		const curveOffset = distance > maxDistance ? curve : curve * (distance / maxDistance);

		return { curveOffset };
	};

	onMount(() => {
		createGradients();
		mounted = true;

		const updateSvgs = () => {
			drawPath();
			drawResidualPath();
		};

		// animate during resizing
		resizeObserver = new ResizeObserver(updateSvgs);

		const elements = document?.querySelectorAll('.resize-watch');
		elements.forEach((el) => resizeObserver.observe(el));

		return () => {
			mounted = false;
			resizeObserver.disconnect();
		};
	});

	// Redraw after Svelte commits a block change, even when no resize occurs.
	$: if (mounted && pathMap && $tokens && $attentionHeadIdx >= 0) {
		drawPath();
		drawResidualPath();
	}

	const drawResidualPath = async () => {
		await tick();
		if (!mounted) return;
		const svg = d3.select(svgEl);

		const starts = d3.selectAll(`.residual-start path.head`).nodes() as Element[];
		const ends = d3.selectAll(`.residual-end path.head`).nodes() as Element[];

		const lineData = getRenderedEndpointPairs(starts, ends).map(
			({ source: startEl, target: endEl, index }) => {
				const x1 = startEl.right;
				const y1 = startEl.top;
				const x2 = endEl.left;
				const y2 = endEl.top;

				return { x1, y1, x2, y2, id: starts[index].id };
			}
		);
		svg
			.selectAll('line.residual-connector')
			.data(lineData)
			.join('line')
			.attr('class', (d) => `residual-connector ${d.id}`)
			.attr('x1', (d) => d.x1)
			.attr('y1', (d) => d.y1)
			.attr('x2', (d) => d.x2)
			.attr('y2', (d) => d.y2)
			.attr('stroke', theme.colors.gray[400])
			.attr('stroke-width', 1)
			.attr('stroke-dasharray', '8,4')
			.style('opacity', 0);
	};
</script>

<svelte:window bind:innerWidth={screenWidth} />

<div bind:clientWidth={screenWidth} class="h-full w-full">
	<svg
		bind:this={svgBackEl}
		id="back"
		class="sankey-back absolute left-0 top-0 h-full w-full"
		style={`z-index:${$modelMeta.attention_head_num - 1};`}
	></svg>
	<svg
		bind:this={svgEl}
		class="sankey-top absolute left-0 top-0 h-full w-full"
		style={`z-index:${$modelMeta.attention_head_num};`}
	/>
</div>
