/**
 * UI-only progress. Each cell represents a group, never a fabricated trace value.
 * @param {number} elapsed
 * @param {number} outputSize
 */
export function projectionSweep(elapsed, outputSize, cells = 32, duration = 3500) {
	const completed = Math.min(cells, Math.max(0, Math.floor((elapsed / duration) * cells)));
	const complete = completed === cells;
	return {
		completed,
		complete,
		active: complete ? -1 : completed,
		coordinate: Math.min(outputSize - 1, Math.ceil(((completed + 1) * outputSize) / cells) - 1)
	};
}
