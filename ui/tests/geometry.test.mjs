import test from 'node:test';
import assert from 'node:assert/strict';
import {dot, cross, norm, poseForView, upFromAxis, imagePixelToPlane, planeToImagePixel, panFocus, viewAngle, validPolygon, makeViewRecord} from '../geometry.mjs';
const near=(a,b,eps=1e-9)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
for(const upAxis of ['+z','-z','+y','-y','+x','-x']) {
 test(`orthonormal top/front for ${upAxis}`,()=>{
   const up=upFromAxis(upAxis);
   for(const kind of ['top','front'])for(const deg of [0,45,90,135,180,270]){
     const p=poseForView(kind,up,deg);
     near(norm(p.direction),1);near(norm(p.right),1);near(norm(p.up),1);
     near(dot(p.direction,p.right),0);near(dot(p.direction,p.up),0);near(dot(p.up,p.right),0);
     const r=cross(p.direction,p.up);near(norm(r),1);
   }
 });
}
test('pixel/model roundtrip retains coordinates',()=>{
 const p=poseForView('top',[0,0,1]);const focus=[0.4,1.2,-0.7],origin=[0,0,0];
 for(const pos of [[0,0],[300,400],[1024,768],[100,765]]){
  const q=imagePixelToPlane(...pos,{width:1024,height:768},focus,3.1,origin,p.axes);
  const r=planeToImagePixel(q,{width:1024,height:768},focus,3.1,origin,p.axes);
  near(pos[0],r[0]);near(pos[1],r[1]);
 }
});
test('pan shifts focal plane predictably',()=>{
 const p=poseForView('front',[0,0,1]); const rect={width:700,height:500},focus=[0,0,0];
 const center=[350,250]; const before=imagePixelToPlane(...center,rect,focus,2,[0,0,0],p.axes);
 const updated=panFocus(focus,40,-25,rect,2,p.axes);
 const after=imagePixelToPlane(center[0]+40,center[1]-25,rect,updated,2,[0,0,0],p.axes);
 near(before[0],after[0]);near(before[1],after[1]);
});
test('8 angles and proper JSON view',()=>{
 assert.deepEqual(Array.from({length:8},(_,i)=>viewAngle(i,20)),[20,65,110,155,200,245,290,335]);
 const p=poseForView('front',[0,0,1],90);
 const polygon=[[0,0],[1,0],[1,1],[0,1]];assert.ok(validPolygon(polygon));
 const rec=makeViewRecord('front_90',p,[0,0,0],polygon);
 near(dot(rec.axes[0],rec.axes[1]),0);assert.equal(rec.polygon.length,4);
});

// Module 2 must reject self-crossing contours before the user confirms.
test('rejects nonzero-area self-intersecting polygon',()=>{
 const crossed=[[0,0],[3,0],[0,2],[3,2],[1.5,-1]];
 assert.equal(validPolygon(crossed),false);
});

test('non-axis-aligned COLMAP gravity yields true orthographic bases',()=>{
 const u=[.45,-.67,.59];const length=norm(u);const up=u.map(v=>v/length);
 for(const kind of ['top','front'])for(const deg of [0,45,90,180,315]){
   const q=poseForView(kind,up,deg);
   near(dot(q.right,q.up),0);near(dot(q.direction,q.right),0);
   near(dot(q.direction,q.up),0);
   if(kind==='top'){near(dot(cross(q.axes[0],q.axes[1]),up),1,1e-8);}
   else {near(dot(q.axes[1],up),1,1e-8);}
 }
});
