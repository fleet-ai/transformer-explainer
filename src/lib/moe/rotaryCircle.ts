export interface RotaryCircle {
	/** Counterclockwise angle wrapped to [0, 2π), rather than accumulated turns. */
	radians: number;
	degrees: number;
	/** Unmodified recorded cosine and negated sine for SVG's downward y axis. */
	x: number;
	y: number;
	largeArc: boolean;
	showArc: boolean;
}

/** Place a recorded cos/sin pair on SVG unit axes without normalizing its values. */
export function rotaryCircle(cosine: number, sine: number): RotaryCircle {
	const limit = 1 + 1e-6;
	if (![cosine, sine].every((value) => Number.isFinite(value) && Math.abs(value) <= limit)) {
		throw new RangeError('Recorded cosine and sine must be finite values between −1 and 1.');
	}
	const fullTurn = 2 * Math.PI;
	let radians = Math.atan2(sine, cosine);
	if (radians < 0) radians += fullTurn;
	// atan2 can return -0; adding 2π to a tiny negative angle can round to 2π.
	if (radians === 0 || radians >= fullTurn) radians = 0;
	return {
		radians,
		degrees: radians * 180 / Math.PI,
		x: cosine,
		y: -sine,
		largeArc: radians > Math.PI,
		showArc: radians !== 0
	};
}

export interface RotaryPoint {
	x: number;
	y: number;
}

export interface RotaryValueVectors {
	/** Shared plotted radius in the recorded values' original units. */
	scale: number;
	before: RotaryPoint;
	after: RotaryPoint;
	/** Unit directions are for the angle arc only; the value arrows use the common scale. */
	beforeDirection: RotaryPoint | null;
	afterDirection: RotaryPoint | null;
}

/** Plot actual before/after pair values together, preserving their relative magnitudes. */
export function rotaryValueVectors(
	before: [number, number], after: [number, number]
): RotaryValueVectors {
	if (![before, after].every((pair) => Array.isArray(pair) && pair.length === 2 && pair.every(Number.isFinite))) {
		throw new RangeError('RoPE value vectors must each contain exactly two finite coordinates.');
	}
	const beforeNorm = Math.hypot(...before);
	const afterNorm = Math.hypot(...after);
	const scale = Math.max(beforeNorm, afterNorm) || 1;
	if (!Number.isFinite(scale)) throw new RangeError('RoPE value vector magnitudes must be finite.');
	const point = (pair: [number, number], divisor: number): RotaryPoint => ({
		x: pair[0] / divisor,
		y: -pair[1] / divisor
	});
	return {
		scale,
		before: point(before, scale),
		after: point(after, scale),
		beforeDirection: beforeNorm === 0 ? null : point(before, beforeNorm),
		afterDirection: afterNorm === 0 ? null : point(after, afterNorm)
	};
}
