/** Pair rendered endpoints without shifting token indices when one is hidden. */
export function getRenderedEndpointPairs(
	sources: Element[],
	targets: Element[]
): { source: DOMRect; target: DOMRect; index: number }[] {
	const pairs: { source: DOMRect; target: DOMRect; index: number }[] = [];
	sources.forEach((sourceElement, index) => {
		const targetElement = targets[index];
		if (
			!targetElement ||
			!sourceElement.getClientRects().length ||
			!targetElement.getClientRects().length
		)
			return;
		const source = sourceElement.getBoundingClientRect();
		const target = targetElement.getBoundingClientRect();
		if (source.height <= 0 || target.height <= 0) return;
		// A vertical anchor can have zero width; a fully collapsed pair cannot.
		if (source.width <= 0 && target.width <= 0) return;
		pairs.push({ source, target, index });
	});
	return pairs;
}
