import * as pc from './node_modules/playcanvas/build/playcanvas.mjs';
import { projectPoissonVertices, poissonOverlayTriangles } from './poisson-overlay.mjs?v=roi-ux-20261104';
import {
  sum,sub,mul,dot,norm,poseForView,viewAngle,
  imagePixelToPlane,planeToImagePixel,panFocus,zoomHalfHeight,
  validPolygon,makeViewRecord
} from './geometry.mjs?v=orientation-vector-20261104';

const byId=id=>document.getElementById(id);
const msg=(t,error=false)=>{byId('status-message').textContent=t;byId('status-message').classList.toggle('warn',error);};
const vec=a=>new pc.Vec3(...a);
const toArr=v=>[v.x,v.y,v.z];
const round=(a,n=8)=>a.map(x=>+x.toFixed(n));
const stage=byId('stage');
let meta,app,gsplatEntity,poissonEntity;
let poissonPreview=null;
let poissonVisible=false;
const poissonDrawQueued={top:false,front:false};
let roiRequestBusy=false;
let workflowTimer=null;
let workflowNoticeKey=null;
let intersectionPreview=null;
let up=[0,0,1],baseYaw=0,currentAngle=0,frontIdx=0;
const drawn={top:null,facades:new Map()};
const panels={
 top:{id:'top-pane',svg:byId('top-svg'),focus:[0,0,0],halfHeight:1,mode:'pan',draft:[],camera:null,pose:null},
 front:{id:'front-pane',svg:byId('front-svg'),focus:[0,0,0],halfHeight:1,mode:'pan',draft:[],camera:null,pose:null}
};
let activePane='top';
function rect(kind){let el=byId(panels[kind].id);return {width:el.clientWidth,height:el.clientHeight};}
function currentFront(){return drawn.facades.get(frontIdx)||null;}
function angleName(deg){return `facade_${String(Math.round(deg*100)/100).replace('.','p')}deg`;}
function pose(kind){return poseForView(kind,up,kind==='front'?currentAngle:0);}
function cameraDistance(){return Math.max(meta.scene_diagonal*3,1);}
function ensureCamera(kind){
 const s=panels[kind], p=pose(kind);s.pose=p;
 const d=cameraDistance();s.camera.setPosition(vec(sum(s.focus,mul(p.direction,d))));
 s.camera.lookAt(vec(s.focus),vec(p.up));
 s.camera.camera.orthoHeight=s.halfHeight;
 s.camera.camera.nearClip=Math.max(meta.scene_diagonal*.0001,d-meta.scene_diagonal*2);
 s.camera.camera.farClip=d+meta.scene_diagonal*2;
 if(byId('clip-toggle').checked){
   const fraction=Number(byId('clip-scale').value)/100;
   const halfRange=Math.max(meta.scene_diagonal*.01,meta.scene_diagonal*fraction*.75);
   s.camera.camera.nearClip=Math.max(meta.scene_diagonal*.0001,d-halfRange);
   s.camera.camera.farClip=d+halfRange;
 }
}
function refit(kind){
 const s=panels[kind];const bounds=meta.reference_bounds;
 const min=bounds.min,max=bounds.max;
 const pts=[];
 for(const x of [min[0],max[0]])for(const y of [min[1],max[1]])for(const z of [min[2],max[2]])pts.push([x,y,z]);
 const p=pose(kind),left=pts.map(q=>dot(sub(q,s.focus),p.right)),height=pts.map(q=>dot(sub(q,s.focus),p.up));
 const spanX=Math.max(...left)-Math.min(...left),spanY=Math.max(...height)-Math.min(...height);
 const viewport=rect(kind), ratio=Math.max(.2,viewport.width/Math.max(1,viewport.height));
 s.halfHeight=Math.max(0.01,Math.max(spanY,spanX/ratio,meta.scene_diagonal*.04)*.68);
 ensureCamera(kind);drawOverlay(kind);
}
function redrawAll(){for(const kind of ['top','front']){ensureCamera(kind);drawOverlay(kind);}refreshInfo();}
function snappedCandidate(){return Math.abs(((viewAngle(frontIdx,baseYaw)-currentAngle+540)%360)-180)<0.1;}
function currentPolygon(kind){return kind==='top'?drawn.top:(snappedCandidate()?currentFront():null);}
function screenPolyline(kind,record){
 if(!record)return [];
 const s=panels[kind],r=rect(kind);
 return record.polygon.map(q=>planeToImagePixel(q,r,s.focus,s.halfHeight,record.origin,record.axes));
}
function drawingPoints(kind){
 const s=panels[kind];const r=rect(kind);return s.draft.map(p=>planeToImagePixel(p,r,s.focus,s.halfHeight,[0,0,0],s.pose.axes));
}
function drawOverlay(kind){
 const s=panels[kind],svg=s.svg,r=rect(kind);
 svg.setAttribute('viewBox',`0 0 ${Math.max(r.width,1)} ${Math.max(r.height,1)}`);
 const record=currentPolygon(kind),pol=record?screenPolyline(kind,record):[];
 const draft=drawingPoints(kind);
 let content='';
 function polyline(points,color,closed){
  if(!points.length)return '';
  let p=points.map(q=>q.map(v=>v.toFixed(2)).join(',')).join(' ');
  return `<polyline points="${p}${closed?' '+points[0].map(v=>v.toFixed(2)).join(','):''}" fill="${closed?color+'22':'none'}" stroke="${color}" stroke-width="2.6" stroke-linejoin="round" stroke-dasharray="${closed?'':'6 4'}" />`;
 }
 content+=polyline(pol,'#63dfad',true);
 content+=polyline(draft,'#ffcb76',false);
 if(intersectionPreview?.sample_xyz){
  for(const point of intersectionPreview.sample_xyz){
   const rec=record||makeViewRecord(kind==='top'?'top':'active_facade',s.pose,[0,0,0],[]);
   const xy=[dot(sub(point,rec.origin),rec.axes[0]),dot(sub(point,rec.origin),rec.axes[1])];
   const pix=planeToImagePixel(xy,r,s.focus,s.halfHeight,rec.origin,rec.axes);
   if(pix[0]>=0&&pix[0]<=r.width&&pix[1]>=0&&pix[1]<=r.height){
     content+=`<circle cx="${pix[0].toFixed(1)}" cy="${pix[1].toFixed(1)}" r="1.6" fill="#8cffac" opacity="0.65"/>`;
   }
  }
 }
 for(const q of draft)content+=`<circle cx="${q[0]}" cy="${q[1]}" r="4" fill="#ffd688" stroke="#1a2930" stroke-width="1.4"/>`;
 if(draft.length>1)content+=`<text x="12" y="${r.height-18}" font-size="12" fill="#ffcb76">草稿顶点 ${draft.length} · 点击完成圈选</text>`;
 svg.innerHTML=content;
 schedulePoissonDraw(kind);
}
function schedulePoissonDraw(kind) {
 if(poissonDrawQueued[kind])return;
 poissonDrawQueued[kind]=true;
 requestAnimationFrame(()=>{
  poissonDrawQueued[kind]=false;
  const c=byId(`${kind}-poisson-layer`);
  const r=rect(kind);
  const w=Math.max(1,Math.round(r.width)),h=Math.max(1,Math.round(r.height));
  if(c.width!==w||c.height!==h){c.width=w;c.height=h;}
  const ctx=c.getContext('2d');ctx.clearRect(0,0,w,h);
  if(!poissonVisible||!poissonPreview||!panels[kind].pose)return;
  const projected=projectPoissonVertices(poissonPreview.positions,r,panels[kind].focus,
      panels[kind].halfHeight,panels[kind].pose.axes);
  // Transparent projection guide: intentionally drawn over splats for visibility.
  // This view-independent overlay is NOT a depth-tested, trustworthy Poisson mask.
  poissonOverlayTriangles(ctx,projected,poissonPreview.indices,5500);
 });
}
function showFeedback(type,title,description,path=''){
 const overlay=byId('feedback-overlay');
 byId('feedback-title').textContent=title;
 byId('feedback-description').textContent=description;
 byId('feedback-path').textContent=path;
 byId('feedback-path-row').hidden=!path;
 overlay.classList.toggle('error',type==='error');
 overlay.hidden=false;
 byId('feedback-close').focus();
}
function clearAnnotationAfterSave(){
 drawn.top=null;drawn.facades.clear();intersectionPreview=null;
 for(const k of ['top','front']){
  panels[k].draft=[];
  setMode(k,'pan');
  drawOverlay(k);
 }
 updateChoices();refreshInfo();
}
function setMode(kind,mode){
 panels[kind].mode=mode;byId(panels[kind].id).classList.toggle('drawing',mode==='draw');
 byId(`${kind}-pan`).classList.toggle('active',mode==='pan');
 byId(`${kind}-draw`).classList.toggle('active',mode==='draw');
 activePane=kind;
}
function onFinish(kind){
 if(kind==='front'&&!snappedCandidate()){msg('当前方位不是8个固定候选角度：先点击下方候选角度按钮，再绘制立面。',true);return;}
 const s=panels[kind];if(!validPolygon(s.draft)){msg('需要至少3个不共线顶点才能完成多边形。',true);return;}
 const record=makeViewRecord(kind==='top'?'top':angleName(currentAngle),s.pose,[0,0,0],s.draft);
 intersectionPreview=null;
 if(kind==='top'){drawn.top=record;}else{
  if(!drawn.facades.has(frontIdx)&&drawn.facades.size>=4){msg('最多保存4个立面ROI，请先删除已有立面。',true);return;}
  drawn.facades.set(frontIdx,record);
 }
 s.draft=[];setMode(kind,'pan');drawOverlay(kind);updateChoices();refreshInfo();
 msg(kind==='top'?'俯视ROI已完成，继续完成至少一个立面。':`已保存立面 ${currentAngle}°（${drawn.facades.size}/4）。`);
}
function onClear(kind){
 const s=panels[kind];s.draft=[];intersectionPreview=null;
 if(kind==='top')drawn.top=null;else drawn.facades.delete(frontIdx);
 drawOverlay(kind);updateChoices();refreshInfo();msg(`${kind==='top'?'俯视':'当前立面'}多边形已清除。`);
}
function refreshInfo(){byId('details').textContent=`俯视 ${drawn.top?'✓':'—'} · 立面 ${drawn.facades.size}/4 · Gaussian ${meta?.gaussians?.toLocaleString()||'...'}`;}
function updateChoices(){
 const host=byId('az-buttons');host.innerHTML='';
 for(let i=0;i<8;i++){
  const b=document.createElement('button'),a=viewAngle(i,baseYaw);
  b.className=`az ${i===frontIdx?'on':''} ${drawn.facades.has(i)?'saved':''}`;
  b.textContent=`${i} · ${Math.round(a)}°`;
  b.onclick=()=>{
   if(panels.front.draft.length){if(!confirm('当前立面草稿未完成，放弃草稿并切换方向？'))return;panels.front.draft=[];}
   frontIdx=i;currentAngle=a;byId('heading').value=Math.round(a);byId('heading-value').textContent=`${Math.round(a)}°`;
   setMode('front','pan');ensureCamera('front');drawOverlay('front');updateChoices();
  };
  host.appendChild(b);
 }
}
function mouseXY(event,el){const r=el.getBoundingClientRect();return [event.clientX-r.left,event.clientY-r.top];}
function wirePane(kind){
 const el=byId(panels[kind].id);let drag=null;
 el.addEventListener('pointerdown',ev=>{
  if(ev.target.closest('.pane-bar')||ev.target.closest('#azimuth-panel'))return;
  activePane=kind;
  if(panels[kind].mode==='pan'&&ev.button===0){
    drag={xy:mouseXY(ev,el),focus:[...panels[kind].focus]};el.setPointerCapture(ev.pointerId);
  }
 });
 el.addEventListener('pointermove',ev=>{
  if(!drag)return;
  const xy=mouseXY(ev,el);let dx=xy[0]-drag.xy[0],dy=xy[1]-drag.xy[1];
  panels[kind].focus=panFocus(drag.focus,dx,dy,rect(kind),panels[kind].halfHeight,panels[kind].pose.axes);
  ensureCamera(kind);drawOverlay(kind);
 });
 el.addEventListener('pointerup',ev=>{if(drag){drag=null;return;}
  if(ev.button!==0||panels[kind].mode!=='draw')return;
  if(ev.target.closest('.pane-bar')||ev.target.closest('#azimuth-panel'))return;
  const xy=mouseXY(ev,el),s=panels[kind];
  const q=imagePixelToPlane(...xy,rect(kind),s.focus,s.halfHeight,[0,0,0],s.pose.axes);
  s.draft.push(round(q));drawOverlay(kind);
 });
 el.addEventListener('wheel',ev=>{
  if(ev.target.closest('.pane-bar')||ev.target.closest('#azimuth-panel'))return;
  ev.preventDefault();const s=panels[kind];const xy=mouseXY(ev,el);
  const before=imagePixelToPlane(...xy,rect(kind),s.focus,s.halfHeight,[0,0,0],s.pose.axes);
  s.halfHeight=zoomHalfHeight(s.halfHeight,ev.deltaY);
  const after=imagePixelToPlane(...xy,rect(kind),s.focus,s.halfHeight,[0,0,0],s.pose.axes);
  s.focus=sum(s.focus,sum(mul(s.pose.axes[0],before[0]-after[0]),mul(s.pose.axes[1],before[1]-after[1])));
  ensureCamera(kind);drawOverlay(kind);
 },{passive:false});
 byId(`${kind}-pan`).onclick=()=>setMode(kind,'pan');
 byId(`${kind}-draw`).onclick=()=>setMode(kind,'draw');
 byId(`${kind}-finish`).onclick=()=>onFinish(kind);
 byId(`${kind}-clear`).onclick=()=>onClear(kind);
}
function exportROI(){
 if(!drawn.top||drawn.facades.size<1)throw Error('需要完成俯视ROI以及至少一个立面ROI。');
 const views=[drawn.top,...[...drawn.facades].sort((a,b)=>a[0]-b[0]).map(x=>x[1])];
 return {
  coordinate_system:'original_COLMAP_world',polygon_units:'COLMAP_model_units_not_pixels',
  views,
  annotation_metadata:{version:'0.4.0', created_utc:new Date().toISOString(),
    original_gaussian:meta.gaussian_name,gaussian_count:meta.gaussians,model_scope:meta.model_scope,
    up_vector:meta.coordinate_frame.up_vector,
    orientation_method:meta.coordinate_frame.orientation_method,
    orientation_source_sha256:meta.coordinate_frame.orientation_source_sha256,
    frame_id:meta.coordinate_frame.frame_id,
    orientation_verified:meta.coordinate_frame.verified,
    scene:meta.scene,method:meta.method,source_ply:meta.source_ply,base_azimuth_deg:baseYaw,
    reference_center:meta.reference_center,view_count:views.length,
    notes:'Display-only Poisson overlay / temporary depth clipping did not change ROI or Gaussian model.'}
 };
}
function updateHeading(deg){
 if(panels.front.draft.length){panels.front.draft=[];msg('立面角度变化：已取消未完成的立面草稿，请在固定候选方位绘制。');}
 // Changes active front camera continuously; current saved polygon remains fixed
 // to its original captured projection axes and is not transformed or edited.
 currentAngle=(deg+360)%360;
 byId('heading-value').textContent=`${Math.round(currentAngle)}°`;
 const snap=[...Array(8)].map((_,i)=>({i,dist:Math.abs(((viewAngle(i,baseYaw)-currentAngle+540)%360)-180)})).sort((a,b)=>a.dist-b.dist)[0];
 // Saved facade label only corresponds to exactly one candidate position.
 frontIdx=snap.i;
 ensureCamera('front');drawOverlay('front');updateChoices();
}
function wireControls(){
 wirePane('top');wirePane('front');
 // Original COLMAP vertical is data-derived, not a manually selected axis.

 byId('heading').oninput=()=>updateHeading(Number(byId('heading').value));
 byId('base-set').onclick=()=>{
  if((drawn.facades.size||panels.front.draft.length)&&!confirm('重新生成8个基准方位会清除已有立面ROI，是否继续？'))return;
  baseYaw=currentAngle;frontIdx=0;drawn.facades.clear();panels.front.draft=[];updateChoices();drawOverlay('front');refreshInfo();
  msg(`已以 ${Math.round(baseYaw)}° 为参考方向生成其余7个候选立面。`);
 };
 byId('reset-camera').onclick=()=>{for(const k of ['top','front']){panels[k].focus=[...meta.reference_center];refit(k);}msg('已恢复模型参考范围。');};
 const clipChange=()=>redrawAll();byId('clip-toggle').onchange=clipChange;byId('clip-scale').oninput=clipChange;
 byId('poisson-toggle').onchange=async()=>{
  const checked=byId('poisson-toggle').checked;
  if(checked){
   try{
    msg('正在读取Poisson辅助曲面...');
    await loadPoisson();poissonVisible=true;
    if(poissonEntity)poissonEntity.enabled=true;
    for(const kind of ['top','front'])schedulePoissonDraw(kind);
    const n=poissonPreview?.preview_faces||Math.floor((poissonPreview?.indices?.length||0)/3);
    msg(`Poisson已显示：${n.toLocaleString()}个抽样面片。叠加线框为辅助投影，不代表可信度或遮挡关系。`);
   }catch(e){
    poissonVisible=false;byId('poisson-toggle').checked=false;
    for(const kind of ['top','front'])schedulePoissonDraw(kind);
    showFeedback('error','Poisson叠加失败',e.message);
    msg('Poisson叠加失败：'+e.message,true);
   }
  }else{
   poissonVisible=false;
   if(poissonEntity)poissonEntity.enabled=false;
   for(const kind of ['top','front'])schedulePoissonDraw(kind);
   msg('已关闭Poisson半透明辅助叠加。');
  }
 };
 byId('feedback-close').onclick=()=>{byId('feedback-overlay').hidden=true;};
 byId('feedback-overlay').addEventListener('click',ev=>{
  if(ev.target===byId('feedback-overlay'))byId('feedback-overlay').hidden=true;
 });
 byId('feedback-copy').onclick=async()=>{
  const pathname=byId('feedback-path').textContent;
  if(!pathname)return;
  try{await navigator.clipboard.writeText(pathname);byId('feedback-copy').textContent='已复制';}
  catch(e){msg('浏览器未允许复制，请在弹窗中手动选中路径。',true);}
 };
 byId('load-existing-roi').onclick=()=>{
  const saved=meta?.latest_roi || meta?.canonical_roi;
  if(!saved?.views?.length){showFeedback('error','没有可载入的ROI','当前场景没有可载入的历史ROI。');return;}
  if((drawn.top||drawn.facades.size||panels.top.draft.length||panels.front.draft.length) &&
     !confirm('载入已有ROI会丢弃当前未保存的多边形和草稿，确定继续吗？'))return;
  try{
    const top=saved.views.find(v=>v.name==='top');
    if(!top||!validPolygon(top.polygon))throw Error('历史ROI缺少有效俯视多边形。');
    const yaw=Number(saved.annotation_metadata?.base_azimuth_deg??0);
    if(!Number.isFinite(yaw))throw Error('历史ROI参考方位无效。');
    const facades=new Map();
    for(const v of saved.views.filter(v=>v.name!=='top')){
      if(!validPolygon(v.polygon))throw Error(`历史立面 ${v.name} 的多边形无效。`);
      const match=/^(?:facade|front)_([0-9p.]+)(?:deg)?$/.exec(v.name);
      if(!match)throw Error(`无法识别历史立面方位：${v.name}`);
      const angle=Number(match[1].replace('p','.'));
      const idx=Array.from({length:8},(_,i)=>i).find(i=>
        Math.abs(((viewAngle(i,yaw)-angle+540)%360)-180)<0.2);
      if(idx===undefined)throw Error(`历史立面方位 ${angle}° 不属于当前8个候选方向。`);
      if(facades.has(idx))throw Error(`历史ROI中重复的立面方向：${angle}°`);
      facades.set(idx,v);
    }
    if(facades.size<1||facades.size>4)throw Error('历史ROI立面数量不在1–4之间。');
    // All validation above must finish before changing the active annotation.
    drawn.top=top;drawn.facades.clear();for(const [i,v] of facades)drawn.facades.set(i,v);
    baseYaw=yaw;frontIdx=[...facades.keys()][0];currentAngle=viewAngle(frontIdx,baseYaw);
    byId('heading').value=Math.round(currentAngle);
    byId('heading-value').textContent=`${Math.round(currentAngle)}°`;
    intersectionPreview=null;
    for(const kind of ['top','front']){
      panels[kind].draft=[];setMode(kind,'pan');ensureCamera(kind);drawOverlay(kind);
    }
    updateChoices();refreshInfo();
    msg('已按用户操作载入历史ROI。此操作只恢复绘图，不修改原始PLY或已保存的文件。');
  }catch(err){showFeedback('error','载入历史ROI失败',err.message);}
 };
 byId('roi-status').onclick=async()=>{
  try{
   const res=await fetch('/api/roi-status',{cache:'no-store'});
   const status=await res.json();
   if(!res.ok)throw Error(status.error||`HTTP ${res.status}`);
   showFeedback('success',status.exists?'最新已确认ROI':'尚无已确认ROI',
      status.exists?`服务器已核实文件存在。Gaussian中心保留数：${status.gaussian_roi_count ?? '未记录'}。点击下方复制路径。`:
      '当前配置的统一ROI成果目录中没有该场景的已确认ROI。',status.saved_path||'');
  }catch(err){showFeedback('error','读取保存状态失败',err.message);}
 };
// Status belongs to the server-side job; closing the feedback modal does not
// stop computation. Do not show a success message until Module 4 PLY exists.
async function checkWorkflow(expectedRoi=null){
 try{
  const response=await fetch('/api/workflow-status',{cache:'no-store'});
  const job=await response.json();
  if(!response.ok)throw Error(job.error||`HTTP ${response.status}`);
  if(!job.enabled||job.status==='idle')return job;
  if(expectedRoi&&job.roi!==expectedRoi)return job;
  const label=job.roi?.split('/').pop()||'当前ROI';
  if(job.status==='running'){
   msg(`已保存 ${label}；模块三Poisson恢复与R3安全外扩、模块四PLY导出正在自动运行...`);
   byId('details').textContent='自动处理正在运行 · '+label;
  }else if(job.status==='completed'){
   const file=job.final_ply;
   msg(`R3最终ROI Gaussian模型已导出：${file}`);
   byId('details').textContent=`最终 ${Number(job.final_gaussians).toLocaleString()} 个Gaussian`;
   const key=`completed:${job.roi_sha256}`;
   if(workflowNoticeKey!==key){
    workflowNoticeKey=key;
    showFeedback('success','四模块流程完成',
      `模块一加载 → 模块二粗选ROI → 模块三局部主体恢复与自适应保护 → 模块四PLY已导出。\n原始ROI：${Number(job.R0_gaussians).toLocaleString()} 个Gaussian；主体恢复：${job.core_restored_gaussians==null?'见报告':Number(job.core_restored_gaussians).toLocaleString()} 个；安全外扩：${job.safety_added_gaussians==null?'见报告':Number(job.safety_added_gaussians).toLocaleString()} 个；最终：${Number(job.final_gaussians).toLocaleString()} 个。\n可直接将最终PLY导入SuperSplat。`,file);
   }
   if(workflowTimer){clearInterval(workflowTimer);workflowTimer=null;}
  }else if(job.status==='failed'){
   const key=`failed:${job.roi_sha256}:${job.return_code}`;
   msg(`ROI已保存，但后续自动计算失败：${job.error||'查看日志'}`,true);
   byId('details').textContent='自动计算未完成 · ROI已保留';
   if(workflowNoticeKey!==key){
    workflowNoticeKey=key;
    showFeedback('error','自动处理失败（初始ROI已保存）',
      `${job.error||'请查看运行日志'}\n可通过日志排查，或使用roi-auto命令重跑；不需要重新圈选ROI。`,job.log||'');
   }
   if(workflowTimer){clearInterval(workflowTimer);workflowTimer=null;}
  }
  return job;
 }catch(e){console.warn('[roi-init] Cannot query automatic workflow:',e);return null;}
}
function monitorWorkflow(expectedRoi=null){
 if(workflowTimer)clearInterval(workflowTimer);
 workflowNoticeKey=null;
 void checkWorkflow(expectedRoi);
 workflowTimer=setInterval(()=>void checkWorkflow(expectedRoi),2500);
}
const previewOrConfirm=async(confirmSave)=>{

  if(roiRequestBusy)return;
  roiRequestBusy=true;
  for(const id of ['confirm-roi','roi-preview'])byId(id).disabled=true;
  try{
   const obj=exportROI();
   const endpoint=confirmSave?'/api/confirm-roi':'/api/roi-preview';
   msg(confirmSave?'正在服务器检查并保存ROI...':'正在计算完整Gaussian中心交集...');
   const response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)});
   const answer=await response.json();
   if(!response.ok||answer.ok===false)throw Error(answer.error||'ROI交集计算失败');
   intersectionPreview=answer.preview;
   drawOverlay('top');drawOverlay('front');
   const number=intersectionPreview.gaussian_roi_count.toLocaleString();
   const ratio=(100*intersectionPreview.roi_fraction).toFixed(2);
   if(confirmSave){
    if(!answer.saved_path||answer.verified_on_disk!==true)
      throw Error('服务器没有返回已核实的保存文件路径；为安全起见，保留当前多边形。');
    const savedPath=answer.saved_path;
    // Latest confirmed scene ROI becomes the default for downstream refinement.
    meta.latest_roi={...obj,path:answer.latest_path||savedPath};
    byId('load-existing-roi').disabled=false;
    if(!answer.saved_as_revision){
      meta.canonical_roi={...obj,path:answer.canonical_path||savedPath};
      byId('load-existing-roi').disabled=false;
    }
    const caption=`${answer.saved_as_revision?'已自动保存为新编号ROI，原场景标准ROI保持不变':'已保存为场景首份标准ROI'}。\nGaussian中心交集：${number}/${meta.gaussians.toLocaleString()}（${ratio}%）。\n模块二只检查投影计算有效性，不判断主体结构是否误切。\n多边形已从工作区清除，文件仍保存在磁盘。\n已确认ROI会自动进入模块三的R3自适应恢复，并由模块四导出最终PLY。`;
    clearAnnotationAfterSave();
    byId('details').textContent=`已保存 · ${savedPath.split('/').pop()}`;
    msg(`ROI已保存（下游默认最新）：${savedPath}；${number}个中心保留。`);
    byId('feedback-copy').textContent='复制保存路径';
    const job=answer.workflow;
    if(job?.status==='failed'){
     showFeedback('error','ROI已保存，但自动任务未启动',
       `${caption}\n${job.error||'请检查服务器日志'}。`,savedPath);
    }else{
     const processing=job?.status==='running'?'\n已自动启动模块三R3恢复与模块四PLY导出，完成后会显示最终模型路径。':'';
     showFeedback('success',answer.saved_as_revision?'ROI新版本保存成功':'ROI保存成功',caption+processing,savedPath);
     if(job?.status==='running')monitorWorkflow(savedPath);
    }
   } else {
    msg(`初始ROI交集核查完成：完整Gaussian中心保留 ${number}/${meta.gaussians.toLocaleString()}（${ratio}%）。。`);
    byId('details').textContent=`ROI ${number} / ${meta.gaussians.toLocaleString()} (${ratio}%)`;
   }
  }catch(err){
   msg(`ROI ${confirmSave?'确认保存':'交集预览'}失败：${err.message}`,true);
   if(confirmSave)showFeedback('error','ROI保存未成功',`服务器未确认保存。当前多边形已完整保留，可根据以下错误修正后重试：\n${err.message}`);
  }finally{
   roiRequestBusy=false;
   for(const id of ['confirm-roi','roi-preview'])byId(id).disabled=false;
  }
 };
 byId('roi-preview').onclick=()=>previewOrConfirm(false);
 byId('confirm-roi').onclick=()=>previewOrConfirm(true);
 byId('download-json').onclick=()=>{
  try{const d=JSON.stringify(exportROI(),null,2),url=URL.createObjectURL(new Blob([d],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download=`${meta.scene}_${meta.method}_unsaved.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);msg('仅导出草稿JSON。正式成果请使用“确认并保存ROI”。');}
  catch(e){msg(e.message,true);}
 };
 window.addEventListener('keydown',ev=>{
  if(ev.target.closest('input,select,textarea'))return;
  const s=panels[activePane];if(!s)return;
  if(ev.key==='Escape'){s.draft=[];drawOverlay(activePane);}
  if(ev.key==='Backspace'&&s.draft.length){ev.preventDefault();s.draft.pop();drawOverlay(activePane);}
 });
 const observer=new ResizeObserver(()=>{if(app){app.resizeCanvas();redrawAll();}});observer.observe(stage);
}
async function loadGaussian(){
 const displayURL=meta.display_url||'/api/model.ply';
 const start=performance.now();
 console.info('[GSSceneRefine] Rendering source:',meta.display_format,displayURL,
   'localhost transfer bytes:',meta.display_bytes,'original PLY bytes:',meta.source_ply_bytes);
 const asset=new pc.Asset('GSSceneRefine Gaussian display','gsplat',{url:displayURL});
 app.assets.add(asset);
 await new Promise((resolve,reject)=>{asset.on('load',resolve);asset.on('error',reject);app.assets.load(asset);});
 gsplatEntity=new pc.Entity('Raw Gaussian (unaltered)');
 // No Euler rotation. Gaussian PLY coordinates remain the original COLMAP frame.
 gsplatEntity.addComponent('gsplat',{asset});app.root.addChild(gsplatEntity);
 console.info(`[GSSceneRefine] Gaussian asset ready in ${((performance.now()-start)/1000).toFixed(2)} s; type=${meta.display_format}`);
}
async function loadPoisson(){
 if(poissonPreview)return;
 if(!meta.poisson_loaded)throw Error('服务器未提供原始Poisson网格。');
 const res=await fetch('/api/poisson-preview');if(!res.ok)throw Error(`HTTP ${res.status}`);
 const data=await res.json();
 if(!Array.isArray(data.positions)||!Array.isArray(data.indices)||data.indices.length<3)throw Error('Poisson响应缺少三角网格数据');
 poissonPreview=data;
 try{
 const mesh=new pc.Mesh(app.graphicsDevice);
 const positions=new Float32Array(data.positions),indices=new Uint32Array(data.indices);
 mesh.setPositions(positions);mesh.setIndices(indices);mesh.update();
 const mat=new pc.StandardMaterial();mat.diffuse=new pc.Color(.28,.97,.73);mat.emissive=new pc.Color(.25,.8,.63);
 mat.opacity=.26;mat.blendType=pc.BLEND_NORMAL;mat.cull=pc.CULLFACE_NONE;mat.depthWrite=false;
 mat.useLighting=false;mat.update();
 poissonEntity=new pc.Entity('Poisson translucent reference');
 poissonEntity.addComponent('render',{meshInstances:[new pc.MeshInstance(mesh,mat)]});
 app.root.addChild(poissonEntity);
 }catch(e){
  poissonEntity=null;
  console.warn('[roi-init] GPU translucent mesh unavailable; 2D orthographic Poisson projection remains active:',e);
 }
}
async function init(){
 const res=await fetch('/api/meta');if(!res.ok)throw Error(`读取输入数据失败 HTTP ${res.status}`);
 meta=await res.json();
 const sourceUp=meta?.coordinate_frame?.up_vector;
 if(!Array.isArray(sourceUp)||sourceUp.length!==3)
   throw Error('前后端方向接口不匹配：/api/meta 必须返回 coordinate_frame.up_vector[3]。请重新启动 roi-init 并按 Ctrl+Shift+R 强制刷新。');
 up=sourceUp.map(Number);
 if(!up.every(Number.isFinite)||Math.abs(norm(up)-1)>1e-5)throw Error('无效COLMAP竖直向量');
 const diag=meta.coordinate_frame.orientation_diagnostics||{};
 byId('up-axis').textContent=`自动 [${up.map(x=>x.toFixed(3)).join(', ')}] (${meta.coordinate_frame.orientation_method}, ${diag.camera_count||'?'}相机)`;
 for(const k of ['top','front'])panels[k].focus=[...meta.reference_center];
 // A saved scene canonical ROI is metadata, NOT an active drawing session.
 // Every new browser session starts with empty annotations. Never silently load
 // historical polygons into only one viewport or let them affect a new experiment.
 if((meta.latest_roi||meta.canonical_roi)?.views?.length){
  msg('发现该场景已有历史ROI，但未自动载入：本次绘图从空白开始。可使用“查看保存记录”或手动点击“载入已有ROI”。');
 }
 byId('load-existing-roi').disabled=!(meta.latest_roi||meta.canonical_roi)?.views?.length;

 const canvas=byId('render-canvas');
 // Official PlayCanvas WebGPU initialization: synchronous Application previously used WebGL.
 const device=await pc.createGraphicsDevice(canvas,{
   deviceTypes:[pc.DEVICETYPE_WEBGPU],antialias:false,alpha:false,
   depth:true,powerPreference:'high-performance'
 });
 if(!device.isWebGPU)throw Error('本工作区要求Windows Edge WebGPU。请开启硬件加速并检查edge://gpu，不能默默回退到WSL软件渲染。');
 device.maxPixelRatio=Math.min(window.devicePixelRatio||1,1.5);
 const opts=new pc.AppOptions();opts.graphicsDevice=device;
 opts.componentSystems=[pc.RenderComponentSystem,pc.CameraComponentSystem,pc.GSplatComponentSystem];
 opts.resourceHandlers=[pc.TextureHandler,pc.ContainerHandler,pc.GSplatHandler];
 app=new pc.AppBase(canvas);app.init(opts);
 app.setCanvasFillMode(pc.FILLMODE_NONE);
 app.setCanvasResolution(pc.RESOLUTION_AUTO);
 app.start();
 console.info('[GSSceneRefine roi-init] WebGPU active; single shared model asset for both cameras');
 for(const k of ['top','front']){
  const s=panels[k],cam=new pc.Entity(`${k} ortho camera`);
  cam.addComponent('camera',{projection:pc.PROJECTION_ORTHOGRAPHIC,
    rect:k==='top'?new pc.Vec4(0,0,.5,1):new pc.Vec4(.5,0,.5,1),
    clearColor:new pc.Color(.045,.075,.083),
    priority:k==='top'?0:1, orthoHeight:1, nearClip:.001,farClip:100});
  app.root.addChild(cam);s.camera=cam;
 }
 app.resizeCanvas();
 for(const k of ['top','front'])refit(k);
 wireControls();updateChoices();refreshInfo();
 msg(`正在通过Windows原生文件服务加载${meta.display_format==='SOG'?'SOG缓存（全量Gaussian，视觉压缩）':meta.display_is_sampled?'显示抽样':'原始全精度PLY'}：${meta.display_gaussians.toLocaleString()}/${meta.gaussians.toLocaleString()}；浏览器传输约${(meta.display_bytes/1048576).toFixed(1)} MiB。几何计算始终使用原始PLY。`);
 await loadGaussian();
 msg(`Windows WebGPU已加载 ${meta.display_gaussians.toLocaleString()} 个Gaussian；请绘制俯视与至少一立面，执行“交集核查”后再“确认并保存ROI”。`);
 if(meta.workflow?.auto_after_confirm){
  byId('confirm-roi').textContent='② 确认ROI并自动生成最终PLY';
  const job=await checkWorkflow();
  if(job?.status==='running')monitorWorkflow(job.roi);
 }
}
init().catch(err=>{console.error(err);msg(`初始化失败：${err.message}。请检查npm install、模型路径及浏览器控制台。`,true);});
