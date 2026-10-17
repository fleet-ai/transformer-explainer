import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const compiled = ts.transpileModule(
  readFileSync(new URL('../src/lib/moe/rotaryCircle.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }
).outputText;
const exports = {};
runInNewContext(compiled, { exports });
const { rotaryCircle, rotaryValueVectors } = exports;
const near = (value, expected, tolerance = 1e-12) => assert.ok(Math.abs(value - expected) <= tolerance, `${value} != ${expected}`);

test('cardinal directions use counterclockwise angles and the SVG downward y axis', () => {
  for (const [cosine, sine, degrees] of [[1, 0, 0], [0, 1, 90], [-1, 0, 180], [0, -1, 270]]) {
    const circle = rotaryCircle(cosine, sine);
    near(circle.degrees, degrees);
    near(circle.radians, degrees * Math.PI / 180);
    assert.equal(circle.x, cosine);
    assert.equal(circle.y, -sine);
    assert.equal(circle.largeArc, degrees > 180);
    assert.equal(circle.showArc, degrees !== 0);
  }
});

test('all four quadrants use the actual pair and wrap negative angles modulo one turn', () => {
  const diagonal = Math.SQRT1_2;
  for (const [cosine, sine, degrees] of [
    [diagonal, diagonal, 45], [-diagonal, diagonal, 135],
    [-diagonal, -diagonal, 225], [diagonal, -diagonal, 315]
  ]) {
    const circle = rotaryCircle(cosine, sine);
    near(circle.degrees, degrees);
    near(Math.cos(circle.radians), cosine);
    near(Math.sin(circle.radians), sine);
    assert.equal(circle.largeArc, degrees > 180);
    assert.equal(circle.showArc, true);
  }
});

test('positive and negative zero produce no zero-angle arc while preserving captured coordinates', () => {
  for (const sine of [0, -0]) {
    const circle = rotaryCircle(1, sine);
    assert.equal(circle.radians, 0);
    assert.equal(circle.degrees, 0);
    assert.equal(circle.showArc, false);
    assert.equal(circle.largeArc, false);
    assert.ok(Object.is(circle.y, -sine));
    assert.equal(rotaryCircle(-1, sine).radians, Math.PI);
  }
  assert.ok(Object.is(rotaryCircle(-0, 1).x, -0));
  // Tiny negative angles must not round to the excluded endpoint 2π / 360°.
  const tiny = rotaryCircle(1, -Number.MIN_VALUE);
  assert.ok(tiny.radians >= 0 && tiny.radians < 2 * Math.PI);
  assert.ok(tiny.degrees >= 0 && tiny.degrees < 360);
});

test('invalid coordinates are rejected and float32 rounding does not normalize recorded values', () => {
  for (const bad of [NaN, Infinity, -Infinity, 1.000002, -1.000002]) {
    assert.throws(() => rotaryCircle(bad, 0), /finite values/);
    assert.throws(() => rotaryCircle(0, bad), /finite values/);
  }
  const cosine = Math.fround(Math.cos(0.7));
  const sine = Math.fround(Math.sin(0.7));
  const circle = rotaryCircle(cosine, sine);
  assert.equal(circle.x, cosine);
  assert.equal(circle.y, -sine);
  assert.notEqual(Math.hypot(circle.x, circle.y), 1, 'The recorded float32 point is not renormalized');
  assert.equal(rotaryCircle(1 + 5e-7, 0).x, 1 + 5e-7);
});

test('every saved Granite and Qwen cos/sin pair retains its coordinates and correct quadrant', () => {
  const root = new URL('../static/traces/', import.meta.url);
  const files = ['attention-operations-language.json', 'attention-operations-code.json', 'attention-operations-math.json', 'rope-example-128.json'];
  const quadrants = new Set();
  let checked = 0;
  for (const file of files) {
    const document = JSON.parse(readFileSync(new URL(file, root), 'utf8'));
    for (const [token, cosines] of document.rotary.cos.entries()) {
      for (const [index, cosine] of cosines.entries()) {
        const sine = document.rotary.sin[token][index];
        const circle = rotaryCircle(cosine, sine);
        checked++;
        assert.equal(circle.x, cosine);
        assert.equal(circle.y, -sine);
        assert.ok(Number.isFinite(circle.radians) && Number.isFinite(circle.degrees));
        assert.ok(circle.radians >= 0 && circle.radians < 2 * Math.PI);
        assert.ok(circle.degrees >= 0 && circle.degrees < 360);
        near(Math.cos(circle.radians), cosine, 1e-6);
        near(Math.sin(circle.radians), sine, 1e-6);
        assert.equal(circle.largeArc, sine < 0);
        if (cosine !== 0 && sine !== 0) {
          const quadrant = cosine > 0 ? (sine > 0 ? 1 : 4) : (sine > 0 ? 2 : 3);
          quadrants.add(quadrant);
          assert.ok(circle.degrees > (quadrant - 1) * 90 && circle.degrees < quadrant * 90);
        }
        if (document.positions[token] === 0) {
          assert.equal(circle.radians, 0);
          assert.equal(circle.showArc, false);
        }
      }
    }
  }
  assert.ok(checked > 1000);
  assert.deepEqual([...quadrants].sort(), [1, 2, 3, 4]);
});

test('value arrows use one radius and retain original magnitudes while arc directions are unit length', () => {
  const before = [3, 4], after = [-8, 6];
  const result = rotaryValueVectors(before, after);
  assert.equal(result.scale, 10);
  near(result.before.x, 0.3);
  near(result.before.y, -0.4);
  near(result.after.x, -0.8);
  near(result.after.y, -0.6);
  near(Math.hypot(result.before.x, result.before.y), 0.5);
  near(Math.hypot(result.after.x, result.after.y), 1);
  near(result.beforeDirection.x, 0.6);
  near(result.beforeDirection.y, -0.8);
  near(Math.hypot(result.beforeDirection.x, result.beforeDirection.y), 1);
  near(Math.hypot(result.afterDirection.x, result.afterDirection.y), 1);
  for (const [raw, plotted] of [[before, result.before], [after, result.after]]) {
    near(plotted.x * result.scale, raw[0]);
    near(-plotted.y * result.scale, raw[1]);
  }
  assert.deepEqual(before, [3, 4]);
  assert.deepEqual(after, [-8, 6]);
});

test('zero vectors have no invented direction and tiny finite vectors remain usable', () => {
  const zero = rotaryValueVectors([0, -0], [-0, 0]);
  assert.equal(zero.scale, 1);
  assert.equal(zero.beforeDirection, null);
  assert.equal(zero.afterDirection, null);
  assert.ok(Object.is(zero.before.y, 0));
  assert.ok(Object.is(zero.after.x, -0));
  const oneZero = rotaryValueVectors([0, 0], [0, 5]);
  assert.equal(oneZero.scale, 5);
  assert.equal(oneZero.beforeDirection, null);
  assert.equal(oneZero.afterDirection.x, 0);
  assert.equal(oneZero.afterDirection.y, -1);
  const tiny = rotaryValueVectors([1e-300, 0], [0, -1e-300]);
  assert.equal(tiny.scale, 1e-300);
  assert.equal(tiny.before.x, 1);
  assert.equal(tiny.after.y, 1);
  for (const point of [zero.before, zero.after, oneZero.before, oneZero.after, tiny.before, tiny.after]) {
    assert.ok(Number.isFinite(point.x) && Number.isFinite(point.y));
  }
});

test('value vectors reject malformed pairs and non-finite values', () => {
  for (const invalid of [null, [], [1], [1, 2, 3], [NaN, 0], [0, Infinity], [-Infinity, 0], ['1', 0]]) {
    assert.throws(() => rotaryValueVectors(invalid, [1, 2]), /exactly two finite coordinates/);
    assert.throws(() => rotaryValueVectors([1, 2], invalid), /exactly two finite coordinates/);
  }
  assert.throws(() => rotaryValueVectors([Number.MAX_VALUE, Number.MAX_VALUE], [1, 2]), /magnitudes must be finite/);
});

test('real Granite and Qwen pairs yield distinct value lines and preserve float32 before/after lengths', () => {
  const root = new URL('../static/traces/', import.meta.url);
  const files = ['attention-operations-language.json', 'attention-operations-code.json', 'attention-operations-math.json', 'rope-example-128.json'];
  let checked = 0;
  let differentLengths = 0;
  for (const file of files) {
    const data = JSON.parse(readFileSync(new URL(file, root), 'utf8'));
    const headSize = data.model.headSize;
    const vectors = data.kind === 'rope-example' ? [data] : [0, 23].flatMap(layer => [0, 7].map(head => ({
      queryBefore: data.layers[layer].queryBefore[head], queryAfter: data.layers[layer].queryAfter[head],
      keyBefore: data.layers[layer].keyBefore[head], keyAfter: data.layers[layer].keyAfter[head]
    })));
    for (const vector of vectors) {
      for (const kind of ['query', 'key']) {
        for (const token of [0, data.positions.length - 1]) {
          const points = [];
          for (let index = 0; index < headSize / 2; index++) {
            const before = [vector[`${kind}Before`][token][index], vector[`${kind}Before`][token][index + headSize / 2]];
            const after = [vector[`${kind}After`][token][index], vector[`${kind}After`][token][index + headSize / 2]];
            const result = rotaryValueVectors(before, after);
            const beforeNorm = Math.hypot(...before), afterNorm = Math.hypot(...after);
            checked++;
            assert.equal(result.scale, Math.max(beforeNorm, afterNorm) || 1);
            for (const [raw, point] of [[before, result.before], [after, result.after]]) {
              near(point.x * result.scale, raw[0], 1e-10);
              near(-point.y * result.scale, raw[1], 1e-10);
              assert.ok(Math.hypot(point.x, point.y) <= 1 + 1e-12);
            }
            near(beforeNorm, afterNorm, 1e-5 + 2e-6 * beforeNorm);
            if (beforeNorm !== afterNorm) differentLengths++;
            if (beforeNorm && afterNorm) {
              near(Math.hypot(result.before.x, result.before.y) / Math.hypot(result.after.x, result.after.y), beforeNorm / afterNorm);
              const cos = data.rotary.cos[token][index], sin = data.rotary.sin[token][index];
              near(result.before.x * cos + result.before.y * sin, result.after.x, 1e-5);
              near(result.before.y * cos - result.before.x * sin, result.after.y, 1e-5);
            }
            points.push(JSON.stringify(result.before));
          }
          assert.ok(new Set(points).size > 1, 'Different recorded coordinate pairs must produce different value lines');
        }
      }
    }
  }
  assert.ok(checked > 1000);
  assert.ok(differentLengths > 0, 'Recorded float32 rounding is retained, rather than forcing identical unit-length arrows');
});
