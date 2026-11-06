import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const app=fs.readFileSync(new URL('../app.mjs',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('../index.html',import.meta.url),'utf8');
test('Poisson preview slider replaces old camera depth clipping',()=>{
 assert.match(app,/async function applyPoissonDisplayFilter\(/);
 assert.match(app,/\/api\/display-filter\?multiplier=/);
 assert.doesNotMatch(app,/if\(byId\('clip-toggle'\)\.checked\)\{\s*const fraction/);
 assert.match(app,/byId\('clip-scale'\)\.onchange/);
 assert.match(html,/Poisson临时显示过滤/);
 assert.match(html,/安全距离倍数/);
});
test('R3 output is not modified by display selection',()=>{
 assert.match(app,/replaceDisplayGaussian\(/);
 assert.match(app,/source_ply:meta.source_ply/);
 assert.match(html,/roi-terminal-handoff-20261106/);
});
