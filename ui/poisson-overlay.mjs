/** Visualization ONLY: projected Poisson mesh is not a depth test or 3D subject classifier.
 * Orthographic coordinates are always original model units. No CAD up-axis assumption.
 */
export function projectPoissonVertices(positions, viewport, focus, halfHeight, axes) {
 if(!(viewport.width>0&&viewport.height>0&&halfHeight>0))throw Error('Invalid orthographic viewport');
 if(positions.length%3!==0)throw Error('Poisson positions must be flat xyz triples');
 const s=viewport.height/(2*halfHeight);
 const focusU=focus[0]*axes[0][0]+focus[1]*axes[0][1]+focus[2]*axes[0][2];
 const focusV=focus[0]*axes[1][0]+focus[1]*axes[1][1]+focus[2]*axes[1][2];
 const xy=new Float32Array(positions.length/3*2);
 for(let i=0,j=0;i<positions.length;i+=3,j+=2){
  const x=positions[i],y=positions[i+1],z=positions[i+2];
  xy[j]=viewport.width/2+(x*axes[0][0]+y*axes[0][1]+z*axes[0][2]-focusU)*s;
  xy[j+1]=viewport.height/2-(x*axes[1][0]+y*axes[1][1]+z*axes[1][2]-focusV)*s;
 }
 return xy;
}
export function poissonOverlayTriangles(ctx,xy,indices,maxFaces=5500){
 const count=Math.floor(indices.length/3),stride=Math.max(1,Math.ceil(count/maxFaces));
 ctx.beginPath();
 let used=0;
 for(let k=0;k<count;k+=stride){
  const a=indices[k*3]*2,b=indices[k*3+1]*2,c=indices[k*3+2]*2;
  if(a<0||b<0||c<0||a+1>=xy.length||b+1>=xy.length||c+1>=xy.length)continue;
  ctx.moveTo(xy[a],xy[a+1]);ctx.lineTo(xy[b],xy[b+1]);ctx.lineTo(xy[c],xy[c+1]);ctx.closePath();
  used++;
 }
 ctx.fillStyle='rgba(32, 230, 175, 0.105)';ctx.fill();
 ctx.strokeStyle='rgba(58, 249, 183, 0.55)';ctx.lineWidth=0.6;ctx.stroke();
 return used;
}
