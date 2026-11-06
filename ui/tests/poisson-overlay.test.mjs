import test from 'node:test';
import assert from 'node:assert/strict';
import {projectPoissonVertices,poissonOverlayTriangles} from '../poisson-overlay.mjs';

test('Poisson projection respects arbitrary orthographic axes and model units',()=>{
 const up=[0,-0.6,-0.8],right=[1,0,0];
 const pts=[1,-0.6,-0.8, 0,0,0, -1,0.6,0.8];
 const a=projectPoissonVertices(pts,{width:800,height:600},[0,0,0],3,[right,up]);
 assert.ok(Math.abs(a[0]-500)<1e-5);assert.ok(Math.abs(a[1]-200)<1e-5);
 assert.ok(Math.abs(a[2]-400)<1e-5);assert.ok(Math.abs(a[3]-300)<1e-5);
 assert.ok(Math.abs(a[4]-300)<1e-5);assert.ok(Math.abs(a[5]-400)<1e-5);
});
test('Poisson overlay does not assume vertical +Z or a real DOM',()=>{
 const calls=[];
 const ctx={beginPath:()=>calls.push('begin'),moveTo:()=>calls.push('move'),
   lineTo:()=>calls.push('line'),closePath:()=>calls.push('close'),
   fill:()=>calls.push('fill'),stroke:()=>calls.push('stroke')};
 const n=poissonOverlayTriangles(ctx,new Float32Array([1,1,2,2,3,3]),new Uint32Array([0,1,2]));
 assert.equal(n,1);assert.ok(calls.includes('fill')&&calls.includes('stroke'));
});
