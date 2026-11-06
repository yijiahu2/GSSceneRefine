import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../app.mjs', import.meta.url), 'utf8');
const page = readFileSync(new URL('../index.html', import.meta.url), 'utf8');

test('opening a scene never assigns saved canonical ROI into active drawing', () => {
  const startup = app.split('async function init(){')[1].split('const canvas=byId')[0];
  assert.match(startup, /meta\.canonical_roi/);
  assert.doesNotMatch(startup, /drawn\.top\s*=/);
  assert.doesNotMatch(startup, /drawn\.facades\.(?:set|clear)/);
  assert.match(startup, /load-existing-roi'\)\.disabled/);
});

test('loading previous ROI is only available through a user click', () => {
  assert.match(page, /id="load-existing-roi" disabled/);
  assert.match(app, /byId\('load-existing-roi'\)\.onclick=\(\)=>/);
  assert.match(app, /载入已有ROI会丢弃当前未保存/);
});

test('successful save clears top and all front polygons, failed save does not', () => {
  assert.match(app, /function clearAnnotationAfterSave\(\)\{\s*drawn\.top=null;drawn\.facades\.clear\(\)/);
  assert.match(app, /if\(!answer\.saved_path\|\|answer\.verified_on_disk!==true\)/);
  assert.match(app, /clearAnnotationAfterSave\(\)/);
});
