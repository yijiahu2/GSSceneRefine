import test from 'node:test';
import assert from 'node:assert/strict';
import { upFromAxis } from '../geometry.mjs';

test('legacy up-axis helper fails explicitly on missing metadata', () => {
  assert.throws(() => upFromAxis(undefined), /Legacy up_axis/);
  assert.throws(() => upFromAxis(null), /Legacy up_axis/);
  assert.throws(() => upFromAxis('z'), /Legacy up_axis/);
  assert.deepEqual(upFromAxis('+z'), [0, 0, 1]);
});
