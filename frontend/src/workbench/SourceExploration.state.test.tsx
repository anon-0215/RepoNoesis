// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SourceExploration } from './SourceExploration';
import { SourceFileContext } from './SourceFileReader';
import { explore, type Exploration, type Pair, type PairSummary, type TracebackResult } from '../lib/exploration';
vi.mock('../lib/exploration',async original=>({...await original<typeof import('../lib/exploration')>(),explore:vi.fn()}));
const api=vi.mocked(explore);
(globalThis as {IS_REACT_ACT_ENVIRONMENT?:boolean}).IS_REACT_ACT_ENVIRONMENT=true;
const summary=(id='p1'): PairSummary=>({id,title:`pair ${id}`,target:'package.render',status:'partial',differences:[{field:'value',a:"'a\\n'",b:"b'bytes'"}]});
const pair=(id='p1'): Pair=>({...summary(id),common:['flag = False'],reading:[],scenarios:['one','two'].map((title,i)=>({id:title,title,status:'partial',fields:{value:i ? "b'bytes'" : "'a\\n'"},conditions:[{text:'automatic fixture unresolved',source:{path:'conftest.py',start:1,end:3,sha256:'hash'}}],inputs:[],expectations:[{text:i ? '返回值 == None' : '返回值 == False',source:{path:'test.py',start:5,end:5,sha256:'hash'}}],assertions:[],implementation:[],unknowns:['unknown fixture binding']}))});
const data=(extra:Partial<Exploration>={}): Exploration=>({identity:{project_id:'p',workspace_id:'w',revision:'r',activation_version:1,repo:'neutral'},data_version:'v1',algorithm:'v1',coverage:{files:2,skipped_count:0,same_revision_inventory:true,parse_errors:[],tests:{test_files:1,tests:2,scenarios:2,pairs:2,exclusion_reasons:{}}},...extra});
const a=data({status:'EXACT_CANDIDATE',total:41,offset:0,limit:20,candidates:[{id:'c',path:'library.py',line:4,end_line:4,symbol:'oldCandidate',exception_expression:'ValueError',raise_source:'raise ValueError("old")',template:'old',match_mode:'exact',type_name_relation:'NOT_SUPPLIED',controls:[],sha256:'hash'}]});
const b=data({status:'READY',total:41,all_pairs:41,offset:0,limit:20,pairs:[summary(),summary('p2')]});
const trace:TracebackResult={status:'COMPLETE',complete:true,algorithm:'ordinary-traceback-v1',frames_detected:2,boundary:'仅对应当前快照；运行版本尚未核验。',issues:[],frames:[1,2].map(index=>({index,input_path:'C:\\demo\\pkg\\core.py',line:index+2,input_line:String(index+2),function:'run',status:'MATCHED',issues:[],matches_total:1,matches:[{path:'pkg/core.py',line:index+2,sha256:'hash',status:'MATCHED',checks:[],source_preview:index===2 ? 'raise ValueError("bad")' : 'return run()',readable:true,scope:{symbol:'run',start:1,end:5,kind:'function'},relation:index===2 ? 'RAISE_RANGE' : 'FUNCTION_RANGE',candidate_ids:index===2 ? ['c'] : []}]}))};
let cleanup: (()=>Promise<void>)|undefined;
afterEach(async()=>{await cleanup?.();cleanup=undefined;vi.clearAllMocks();});
async function mount(){
 const host=document.createElement('div');document.body.append(host);const root=createRoot(host);const open=vi.fn();
 await act(async()=>root.render(<SourceFileContext.Provider value={open}><SourceExploration projectId="p" workspaceId="w" revision="r" /></SourceFileContext.Provider>));
 cleanup=async()=>{await act(async()=>root.unmount());host.remove();};
 async function click(label:string){const el=Array.from(host.querySelectorAll('button')).find(b=>b.textContent===label || b.textContent?.startsWith(label));expect(el).toBeTruthy();await act(async()=>el!.click());}
 async function input(id:string,value:string){const el=host.querySelector<HTMLInputElement|HTMLTextAreaElement>('#'+id)!;await act(async()=>{Object.getOwnPropertyDescriptor(el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype,'value')!.set!.call(el,value);el.dispatchEvent(new Event('input',{bubbles:true}));});}
 async function submit(id:string){await act(async()=>host.querySelector('#'+id)!.closest('form')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));}
 async function rerender(ctx:{projectId:string;workspaceId:string;revision:string}) { await act(async()=>root.render(<SourceFileContext.Provider value={open}><SourceExploration {...ctx} /></SourceFileContext.Provider>)); }
 return {host,click,input,submit,open,rerender};
}
describe('exploration requests own their visible result',()=>{
 it('traceback adds ordered readable call positions, a range badge, and exact shared-reader selection',async()=>{
  api.mockResolvedValue(data());const v=await mount();api.mockResolvedValue({...a,traceback:trace,candidates:[{...a.candidates![0],trace_frames:[2]}]});await v.input('se-error','Traceback sample');await v.submit('se-error');
  const frames=v.host.querySelectorAll('.se-trace-frames summary');expect(frames).toHaveLength(2);expect(frames[0].textContent).toContain('#1');expect(frames[1].textContent).toContain('#2');expect(v.host.textContent).toContain('运行版本尚未核验');expect(v.host.textContent).toContain('堆栈帧 #2');expect(v.host.textContent).toContain('这是调用/源码位置');
  await v.click('打开当前快照位置');expect(v.open).toHaveBeenCalledWith(expect.objectContaining({projectId:'p',workspaceId:'w',revision:'r',path:'pkg/core.py',start:3,end:3}),expect.any(HTMLButtonElement));
 });
 it('traceback is cleared on failed search and cannot reappear from a late mode response',async()=>{
  api.mockResolvedValue(data());const v=await mount();api.mockResolvedValue({...a,traceback:trace});await v.input('se-error','first');await v.submit('se-error');expect(v.host.querySelector('.se-traceback')).toBeTruthy();
  api.mockRejectedValue(new Error('503'));await v.input('se-error','second');await v.submit('se-error');expect(v.host.querySelector('.se-traceback')).toBeNull();
  let done!:(d:Exploration)=>void;api.mockImplementationOnce(()=>new Promise(resolve=>{done=resolve;}));await v.input('se-error','late');await v.submit('se-error');api.mockResolvedValue(b);await v.click('用例对照');await act(async()=>done({...a,traceback:trace}));expect(v.host.querySelector('.se-traceback')).toBeNull();expect(v.host.querySelectorAll('.se-pair-list button')).toHaveLength(2);
 });
 it.each([{projectId:'other',workspaceId:'other-workspace',revision:'other-revision'},{projectId:'p',workspaceId:'w',revision:'new-revision'}])('late trace response cannot survive context change %j',async(ctx)=>{
  api.mockResolvedValue(data());const v=await mount();let done!:(d:Exploration)=>void;api.mockImplementationOnce(()=>new Promise(resolve=>{done=resolve;}));await v.input('se-error','slow trace');await v.submit('se-error');api.mockResolvedValue(data());await v.rerender(ctx);await act(async()=>done({...a,traceback:trace}));expect(v.host.querySelector('.se-traceback')).toBeNull();expect(v.host.querySelector('.se-candidate')).toBeNull();
 });
 it('does not silently truncate an oversized paste or query with the rejected content',async()=>{
  api.mockResolvedValue(data());const v=await mount();await v.input('se-error','original');const el=v.host.querySelector('textarea')!;const event=new Event('paste',{bubbles:true,cancelable:true});Object.defineProperty(event,'clipboardData',{value:{getData:()=> 'x'.repeat(4097)}});await act(async()=>{el.dispatchEvent(event);});expect(event.defaultPrevented).toBe(true);expect(el.value).toBe('original');expect(v.host.textContent).toContain('本次粘贴未接收');expect(api).toHaveBeenCalledTimes(1);
 });
 it('shows ambiguity, conflict and incomplete fallback without claiming a root cause',async()=>{
  api.mockResolvedValue(data());const v=await mount();const limited={...trace,status:'LIMITED',complete:false,issues:['后续帧未处理'],frames:trace.frames.map((f,i)=>({...f,status:i ? 'CONFLICT' : 'AMBIGUOUS',matches_total:i ? 1 : 3,matches:f.matches.map(m=>({...m,status:'UNVERIFIED',scope:undefined,relation:undefined,candidate_ids:[],checks:['未确认']}))}))};api.mockResolvedValue({...a,traceback:limited});await v.input('se-error','limited');await v.submit('se-error');expect(v.host.textContent).toContain('未完成堆栈关联');expect(v.host.textContent).toContain('没有自动选定文件');expect(v.host.textContent).toContain('与当前源码不一致');expect(v.host.querySelectorAll('.se-trace-frames details[open]')).toHaveLength(0);
 });
 it('new A search failure clears previous candidates and retry replays failed query',async()=>{
  api.mockResolvedValue(data());const v=await mount();api.mockResolvedValue(a);await v.input('se-error','old');await v.submit('se-error');
  expect(v.host.textContent).toContain('oldCandidate');api.mockRejectedValue(new Error('503 A'));await v.input('se-error','new');await v.submit('se-error');
  expect(v.host.querySelectorAll('.se-candidate')).toHaveLength(0);expect(v.host.textContent).not.toContain('oldCandidate');
  await v.input('se-error','unsubmitted');api.mockResolvedValue(data({status:'NO_MATCH',total:0,candidates:[]}));await v.click('重试查询');
  expect(api.mock.lastCall?.[2]?.body).toEqual({revision:'r',error:'new',offset:0,limit:20});
 });
 it('new B search failure clears previous list and detail and retries the submitted query',async()=>{
  api.mockResolvedValue(data());const v=await mount();api.mockResolvedValue(b);await v.click('用例对照');api.mockResolvedValue(data({pair:pair()}));await v.click('pair p1');
  expect(v.host.querySelector('.se-comparison')).toBeTruthy();api.mockRejectedValue(new Error('503 B'));await v.input('se-search','new');await v.submit('se-search');
  expect(v.host.querySelectorAll('.se-pair-list button')).toHaveLength(0);expect(v.host.querySelector('.se-comparison')).toBeNull();
  await v.input('se-search','unsubmitted');api.mockResolvedValue(b);await v.click('重试搜索');expect(api.mock.lastCall?.[2]?.query?.q).toBe('new');
 });
 it('detail failure preserves list but clears old detail and retries the selected id',async()=>{
  api.mockResolvedValue(data());const v=await mount();api.mockResolvedValue(b);await v.click('用例对照');api.mockResolvedValue(data({pair:pair()}));await v.click('pair p1');
  api.mockRejectedValue(new Error('503 detail'));await v.click('pair p2');expect(v.host.querySelectorAll('.se-pair-list button')).toHaveLength(2);expect(v.host.querySelector('.se-comparison')).toBeNull();
  api.mockResolvedValue(data({pair:pair('p2')}));await v.click('重试读取对照');expect(api.mock.lastCall?.[1]).toBe('/comparisons/p2');expect(v.host.querySelector('.se-comparison h2')?.textContent).toBe('pair p2');
 });
 it('A and B paging use submitted queries rather than edited drafts',async()=>{
  api.mockResolvedValue(data());const v=await mount();api.mockResolvedValue(a);await v.input('se-error','submitted A');await v.submit('se-error');await v.input('se-error','draft A');await v.click('下一页');
  expect(api.mock.lastCall?.[2]?.body).toEqual({revision:'r',error:'submitted A',offset:20,limit:20});
  api.mockResolvedValue(b);await v.click('用例对照');await v.input('se-search','submitted B');await v.submit('se-search');await v.input('se-search','draft B');await v.click('下一页');
  expect(api.mock.lastCall?.[2]?.query).toEqual(expect.objectContaining({q:'submitted B',offset:'20'}));
 });
 it('late successful response cannot overwrite a newer failed search',async()=>{
  api.mockResolvedValue(data());const v=await mount();let done!: (d:Exploration)=>void;
  api.mockImplementationOnce(()=>new Promise(resolve=>{done=resolve;}));await v.input('se-error','first');await v.submit('se-error');
  api.mockRejectedValue(new Error('new failed'));await v.input('se-error','second');await v.submit('se-error');await act(async()=>done(a));
  expect(v.host.textContent).toContain('new failed');expect(v.host.textContent).not.toContain('oldCandidate');
 });
 it('overview preserves exact input and expectation text and retains full evidence and unknowns',async()=>{
  api.mockResolvedValue(data());const v=await mount();api.mockResolvedValue(b);await v.click('用例对照');api.mockResolvedValue(data({pair:pair()}));await v.click('pair p1');
  const overview=v.host.querySelector('.se-overview')!;expect(overview.textContent).toContain("'a\\n'");expect(overview.textContent).toContain("b'bytes'");expect(overview.textContent).toContain('返回值 == None');expect(overview.textContent).toContain('返回值 == False');
  expect(v.host.textContent).toContain('未执行');expect(v.host.textContent).toContain('部分条件未解析');expect(v.host.textContent).toContain('unknown fixture binding');expect(v.host.textContent).toContain('automatic fixture unresolved');
 });
 it('rejects detail from a different material version while retaining the valid list',async()=>{
  api.mockResolvedValue(data());const v=await mount();api.mockResolvedValue(b);await v.click('用例对照');
  api.mockResolvedValue(data({data_version:'changed',pair:pair()}));await v.click('pair p1');
  expect(v.host.querySelector('.se-comparison')).toBeNull();expect(v.host.querySelectorAll('.se-pair-list button')).toHaveLength(2);expect(v.host.textContent).toContain('源码材料已更新');
 });
 it('material retry refreshes the selected mode and retains the failed material path',async()=>{
  api.mockResolvedValue(data());const v=await mount();api.mockResolvedValue(b);await v.click('用例对照');await v.input('se-path','original-path');
  api.mockRejectedValue(new Error('material unavailable'));await v.submit('se-path');await v.input('se-path','draft-path');
  api.mockResolvedValue(b);await v.click('重试材料读取');
  const calls=api.mock.calls;expect(calls[calls.length-2][1]).toBe('/materials');expect(calls[calls.length-2][2]?.body).toEqual({revision:'r',local_path:'original-path'});expect(api.mock.lastCall?.[1]).toBe('/comparisons');expect(v.host.querySelectorAll('.se-pair-list button')).toHaveLength(2);
 });
 it('switching modes drops late A results even when the transport ignores abort',async()=>{
  api.mockResolvedValue(data());const v=await mount();let done!: (d:Exploration)=>void;
  api.mockImplementationOnce(()=>new Promise(resolve=>{done=resolve;}));await v.input('se-error','slow');await v.submit('se-error');api.mockResolvedValue(b);await v.click('用例对照');await act(async()=>done(a));
  expect(v.host.querySelectorAll('.se-pair-list button')).toHaveLength(2);expect(v.host.textContent).not.toContain('oldCandidate');
 });

});
