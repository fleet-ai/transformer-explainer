/** Shared sizes for the encoder–decoder page, laid out like the Dense explainer. */
export const STEP_TITLE = 30;
export const ROW_HEADING = 20;
export const STRIP_W = 12;

export const FILL: Record<string, string> = {
	purple: '#e9d5ff',
	blue: '#bfdbfe',
	red: '#fecaca',
	green: '#bbf7d0',
	indigo: '#c7d2fe',
	gray: '#d1d5db'
};

/** Positions inside a Dense-style head card: Key list, then Query beside the matrix, then Value. */
export function headCardGeometry(rows: number, cols: number, pitch: number, cell: number) {
	const labelRight = 78;
	const dashX = labelRight + 4;
	const lineX = dashX + 12;
	const mLeft = lineX + 34;
	const mRight = mLeft + cols * cell;
	const outX = mRight + 30;
	const width = outX + 12 + 16;
	const kTop = 24;
	const midTop = kTop + cols * pitch + 22;
	const qHeight = rows * pitch;
	const mHeight = rows * cell;
	const midHeight = Math.max(qHeight, mHeight);
	const qTop = midTop + (midHeight - qHeight) / 2;
	const mTop = midTop + (midHeight - mHeight) / 2;
	const vTop = midTop + midHeight + 30;
	const height = vTop + cols * pitch + 14;
	return { labelRight, dashX, lineX, mLeft, mRight, outX, width, kTop, qTop, mTop, vTop, height };
}
export const headCardHeight = (rows: number, cols: number, pitch: number, cell: number) =>
	headCardGeometry(rows, cols, pitch, cell).height;
