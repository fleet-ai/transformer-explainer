import type { TokenTrace } from './types';

export interface CoordinateTerm {
	expert: number;
	weight: number;
	value: number;
	contribution: number;
	shared?: boolean;
}

export interface CoordinateSum {
	terms: CoordinateTerm[];
	reconstructed: number;
	recorded: number;
	/** Signed browser reconstruction minus the captured float32 result. */
	difference: number;
	/** Derived residual-branch contribution, before adding the residual stream. */
	scaled: number;
}

/**
 * Reconstruct one captured MoE output coordinate from the selected experts.
 * Expert previews and routing weights are aligned by selection rank, not expert ID.
 * Keep the captured pre-residual result separate from the browser's float64 sum.
 */
export function sumCoordinate(
	trace: TokenTrace,
	coordinate: number,
	residualMultiplier: number
): CoordinateSum {
	if (!Number.isInteger(coordinate) || coordinate < 0 || coordinate >= trace.combinedOutputPreview.length) {
		throw new RangeError('Coordinate must index a captured output preview.');
	}
	if (!Number.isFinite(residualMultiplier)) {
		throw new RangeError('Residual multiplier must be finite.');
	}
	const count = trace.selectedExperts.length;
	if (!count || trace.routingWeights.length !== count || trace.expertOutputPreview.length !== count) {
		throw new Error('Selected experts, routing weights, and output previews must align.');
	}
	const recorded = trace.combinedOutputPreview[coordinate];
	const terms = trace.selectedExperts.map((expert, rank): CoordinateTerm => {
		const weight = trace.routingWeights[rank];
		const value = trace.expertOutputPreview[rank][coordinate];
		const contribution = weight * value;
		if (!Number.isInteger(expert) || expert < 0 || ![weight, value, contribution].every(Number.isFinite)) {
			throw new Error('Expert IDs, routing weights, and coordinate values must be valid.');
		}
		return { expert, weight, value, contribution };
	});
	if (trace.sharedExpertOutputPreview) {
		const value = trace.sharedExpertOutputPreview[coordinate];
		if (!Number.isFinite(value)) throw new Error('Shared expert output must be finite.');
		// The always-active shared branch is added once, without a router weight.
		terms.push({ expert: trace.routerProbabilities.length, weight: 1, value, contribution: value, shared: true });
	}
	const reconstructed = terms.reduce((sum, term) => sum + term.contribution, 0);
	const difference = reconstructed - recorded;
	const scaled = recorded * residualMultiplier;
	if (![recorded, reconstructed, difference, scaled].every(Number.isFinite)) {
		throw new Error('Recorded output and coordinate arithmetic must be finite.');
	}
	return { terms, reconstructed, recorded, difference, scaled };
}
