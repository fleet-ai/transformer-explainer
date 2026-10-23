export interface Box {
	left: number;
	right: number;
	top: number;
	bottom: number;
	cx: number;
	cy: number;
	width: number;
	height: number;
}
export interface Link {
	/** `data-b` of the source elements; each pairs with the target element at the same `data-i`. */
	from: string;
	to: string;
	color: string;
	/** Opacity of the band, or a function of the token row. */
	opacity?: number | ((row: number) => number);
	/** Draw one band between the first element of each name instead of one band per token. */
	group?: boolean;
	/** With `group`, start from the last source element instead of the first. */
	last?: boolean;
	path?: (source: Box, target: Box) => string;
	width?: (source: Box, target: Box) => number;
	label?: { text: string; at: (source: Box, target: Box) => { x: number; y: number } };
}
