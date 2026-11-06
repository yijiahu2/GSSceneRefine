/** Coordinate-only helpers. No rendering API, intentionally unit-testable with Node.
 * World coordinates always remain the original COLMAP/GSPL coordinate system.
 */
export const sum=(a,b)=>a.map((x,i)=>x+b[i]);
export const sub=(a,b)=>a.map((x,i)=>x-b[i]);
export const mul=(a,k)=>a.map(x=>x*k);
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const norm=a=>Math.sqrt(dot(a,a));
export function unit(a) { const n=norm(a); if(n<1e-12) throw Error('Degenerate vector'); return mul(a,1/n); }
export function upFromAxis(axis) {
 // Legacy helper retained for old tests/clients only. Never silently use +Z
 // when the live metadata contract supplies an arbitrary SfM up_vector.
 if(typeof axis!=='string'|| !/^[+-][xyz]$/.test(axis))
   throw Error('Legacy up_axis is absent or invalid. The roi-init frontend must use coordinate_frame.up_vector; hard-refresh the browser.');
 const i='xyz'.indexOf(axis.slice(-1));const a=[0,0,0];a[i]=axis.startsWith('-')?-1:1;return a;
}
export function horizontalAxes(up) { const preferred=Math.abs(dot(up,[1,0,0]))<.9?[1,0,0]:[0,1,0]; const h1=unit(sub(preferred,mul(up,dot(preferred,up))));return [h1,unit(cross(up,h1))]; }
export function poseForView(kind, up, angleDeg=0) {
  const [h1,h2]=horizontalAxes(up); let d; let desiredUp;
  if(kind==='top') { d=up;desiredUp=h2; }
  else { const a=angleDeg*Math.PI/180; d=unit(sum(mul(h2,-Math.cos(a)),mul(h1,Math.sin(a))));desiredUp=up; }
  // PlayCanvas camera looks toward focus using forward=-d; screen right=forward x up.
  const right=unit(cross(mul(d,-1),desiredUp));
  return { direction:d, right, up:desiredUp, axes:[right,desiredUp] };
}
export function imagePixelToPlane(px,py,rect,focus,halfHeight,origin,axes) {
  if(!(rect.width>0&&rect.height>0&&halfHeight>0))throw Error('Invalid viewport');
  const scale=2*halfHeight/rect.height;
  const u=dot(sub(focus,origin),axes[0])+(px-rect.width/2)*scale;
  const v=dot(sub(focus,origin),axes[1])+(rect.height/2-py)*scale;
  return [u,v];
}
export function planeToImagePixel(pt,rect,focus,halfHeight,origin,axes) {
  const scale=2*halfHeight/rect.height;
  return [rect.width/2+(pt[0]-dot(sub(focus,origin),axes[0]))/scale,
          rect.height/2-(pt[1]-dot(sub(focus,origin),axes[1]))/scale];
}
export function panFocus(focus,dx,dy,rect,halfHeight,axes) {
  const s=2*halfHeight/rect.height;
  return sub(focus,sum(mul(axes[0],dx*s),mul(axes[1],-dy*s)));
}
export function zoomHalfHeight(before, deltaY) {return Math.max(1e-5,Math.min(1e8,before*Math.exp(deltaY*.0015)));}
export function polygonArea(poly){let a=0;for(let i=0;i<poly.length;i++){let j=(i+1)%poly.length;a+=poly[i][0]*poly[j][1]-poly[j][0]*poly[i][1];}return a*.5;}
function segmentsCross(a,b,c,d) {
 const orient=(p,q,r)=>(q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]);
 const inside=(p,q,r)=>Math.min(p[0],r[0])-1e-9<=q[0]&&q[0]<=Math.max(p[0],r[0])+1e-9&&
                           Math.min(p[1],r[1])-1e-9<=q[1]&&q[1]<=Math.max(p[1],r[1])+1e-9;
 const x=orient(a,b,c),y=orient(a,b,d),z=orient(c,d,a),w=orient(c,d,b);
 if((x>1e-10&&y< -1e-10||x< -1e-10&&y>1e-10)&&
    (z>1e-10&&w< -1e-10||z< -1e-10&&w>1e-10))return true;
 return (Math.abs(x)<=1e-10&&inside(a,c,b))||(Math.abs(y)<=1e-10&&inside(a,d,b))||
        (Math.abs(z)<=1e-10&&inside(c,a,d))||(Math.abs(w)<=1e-10&&inside(c,b,d));
}
export function validPolygon(poly) {
 if(poly.length<3||poly.length>2000||Math.abs(polygonArea(poly))<=1e-10||
    !poly.every(q=>q.length===2&&q.every(Number.isFinite)))return false;
 for(let i=0;i<poly.length;i++)for(let j=i+1;j<poly.length;j++){
   if(Math.abs(i-j)===1||(i===0&&j===poly.length-1))continue;
   if(segmentsCross(poly[i],poly[(i+1)%poly.length],poly[j],poly[(j+1)%poly.length]))return false;
 }
 return true;
}
export function makeViewRecord(name, view, origin, poly) {
  return {name, origin:origin.map(Number),axes:view.axes.map(a=>a.map(Number)),polygon:poly.map(p=>p.map(Number))};
}
export function viewAngle(i,baseAngle=0){return (baseAngle+i*45+3600)%360;}
