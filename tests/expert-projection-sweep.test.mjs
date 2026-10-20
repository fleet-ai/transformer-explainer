import test from 'node:test';
import assert from 'node:assert/strict';
import { projectionSweep } from '../src/lib/moe/projection-sweep.js';

test('sweep fills all displayed cells through the last coordinate for expert and MLA projection widths', () => {
	for (const size of [512, 576, 1024, 1408, 2048, 2816, 3072, 10944]) {
		const start = projectionSweep(0, size);
		assert.equal(start.completed, 0);
		assert.equal(start.active, 0);
		const partial = projectionSweep(1000, size);
		assert.ok(partial.completed >= 8 && partial.completed < 32);
		assert.equal(partial.complete, false);
		assert.deepEqual(projectionSweep(3500, size), {
			completed: 32,
			complete: true,
			active: -1,
			coordinate: size - 1
		});
	}
});
test('delayed animation frames still finish, and replay resets progress', () => {
	assert.equal(projectionSweep(60_000, 512).completed, 32);
	assert.equal(projectionSweep(0, 512).completed, 0);
	assert.equal(projectionSweep(0, 512).complete, false);
});
