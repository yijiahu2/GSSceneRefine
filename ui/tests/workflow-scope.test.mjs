import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const source = readFileSync(new URL('../app.mjs', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');

test('workflow functions exist at module scope, visible to init before wireControls', () => {
  // Exports can only name module-scope bindings: this failed with the earlier
  // checkWorkflow ReferenceError even though ordinary `node --check` passed.
  const check = spawnSync(process.execPath, ['--check', '--input-type=module'], {
    input: `${source}\nexport { checkWorkflow, monitorWorkflow, renderWorkflowStatus };\n`,
    encoding: 'utf8',
  });
  assert.equal(check.status, 0, check.stderr);
  assert.ok(source.indexOf('async function checkWorkflow(') < source.indexOf('function wireControls(){'));
  assert.ok(source.indexOf('function wireControls(){') < source.indexOf('async function init(){'));
});

test('HTML points to current workflow fix rather than a previously cached JS bundle', () => {
  assert.match(html, /app\.mjs\?v=roi-terminal-handoff-20261106/);
});
test('terminal handoff does not leave a browser status polling loop running', () => {
  assert.match(source, /job\?\.status==='running'&&!meta\.workflow\?\.terminal_handoff\)monitorWorkflow\(savedPath\)/);
  assert.match(source, /浏览器服务即将关闭/);
});
