// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SourceExploration } from './SourceExploration';
import { SourceFileContext } from './SourceFileReader';
import { explore, type Exploration } from '../lib/exploration';
vi.mock('../lib/exploration', async importOriginal => ({ ...await importOriginal<typeof import('../lib/exploration')>(), explore: vi.fn() }));
const mock=vi.mocked(explore);
const data=(id='one',extra: Partial<Exploration>={}): Exploration => ({ identity:{project_id:id,workspace_id:'w',revision:'r',activation_version:1,repo:id},data_version:id,algorithm:'v1',coverage:{files:3,skipped_count:0,same_revision_inventory:false,parse_errors:[],tests:{test_files:0,tests:0,scenarios:0,pairs:0,exclusion_reasons:{}}},...extra });
let cleanup: (()=>Promise<void>) | undefined;
afterEach(async()=>{await cleanup?.();cleanup=undefined;vi.clearAllMocks();});
async function mount() {
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);const open=vi.fn();
  const render=async(id='one')=>{await act(async()=>{root.render(<SourceFileContext.Provider value={open}><SourceExploration key={id} projectId={id} workspaceId="w" revision="r" /></SourceFileContext.Provider>);});};
  cleanup=async()=>{await act(async()=>root.unmount());host.remove();};await render();
  const click=async(text:string)=>{const button=Array.from(host.querySelectorAll('button')).find(b=>b.textContent===text);expect(button).toBeTruthy();await act(async()=>button!.click());};
  return {host,render,click,open};
}
describe('source exploration state and source identity',()=>{
  it('discards old repository responses even when transport ignores abort',async()=>{
    let resolve!: (v:Exploration)=>void;mock.mockImplementationOnce(()=>new Promise(done=>{resolve=done;}));mock.mockResolvedValue(data('two'));
    const view=await mount();await view.render('two');await act(async()=>resolve(data('one')));
    expect(view.host.textContent).toContain('two');expect(view.host.textContent).not.toContain('当前仓库 one');
  });
  it('explains absent snapshot tests and allows a failed feature to leave the page',async()=>{
    mock.mockResolvedValue(data());const view=await mount();mock.mockResolvedValue(data('one',{status:'MATERIALS_MISSING',pairs:[],total:0,all_pairs:0}));await view.click('用例对照');
    expect(view.host.textContent).toContain('不能据此判断原仓库没有测试');
    mock.mockRejectedValue(new Error('material failure'));await view.click('搜索');expect(view.host.querySelector('[role="alert"]')?.textContent).toContain('material failure');
    mock.mockResolvedValue(data());await view.click('报错定位');expect(view.host.querySelector('textarea')).toBeTruthy();
  });
  it('opens exact candidate lines through the shared reader and retains the query',async()=>{
    mock.mockResolvedValue(data());const view=await mount();
    const candidate={id:'c',path:'lib.py',line:12,end_line:13,symbol:'f',exception_expression:'ValueError',raise_source:'raise ValueError("bad")',template:'bad',match_mode:'exact',type_name_relation:'NOT_SUPPLIED',controls:[{kind:'if',line:11,source:'if invalid:'}],sha256:'a'.repeat(64)};
    const input=view.host.querySelector('textarea')!;
    await act(async()=>{Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value')!.set!.call(input,'bad');input.dispatchEvent(new Event('input',{bubbles:true}));});
    mock.mockResolvedValue(data('one',{status:'EXACT_CANDIDATE',candidates:[candidate],total:1}));
    await act(async()=>view.host.querySelector('form')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
    const sourceButton=Array.from(view.host.querySelectorAll('button')).find(b=>b.textContent?.startsWith('打开抛错源码'))!;await act(async()=>sourceButton.click());
    expect(view.open).toHaveBeenCalledWith(expect.objectContaining({projectId:'one',workspaceId:'w',revision:'r',path:'lib.py',start:12,end:13}),sourceButton);
    expect(input.value).toBe('bad');expect(view.host.textContent).toContain('if invalid:');
  });
  it('keeps material failure visible instead of hiding it with an automatic refresh',async()=>{
    mock.mockResolvedValue(data());const view=await mount();const input=view.host.querySelector<HTMLInputElement>('#se-path')!;
    await act(async()=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value')!.set!.call(input,'missing');input.dispatchEvent(new Event('input',{bubbles:true}));});
    mock.mockRejectedValue(new Error('revision unavailable'));
    await act(async()=>input.closest('form')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
    expect(view.host.querySelector('[role="alert"]')?.textContent).toContain('revision unavailable');
  });
  it('preserves submitted search when changing support and representative filters',async()=>{
    mock.mockResolvedValue(data('one',{status:'READY',pairs:[],total:0,all_pairs:0}));const view=await mount();await view.click('用例对照');
    const input=view.host.querySelector<HTMLInputElement>('#se-search')!;
    await act(async()=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value')!.set!.call(input,'render');input.dispatchEvent(new Event('input',{bubbles:true}));});
    await view.click('搜索');
    const select=view.host.querySelector<HTMLSelectElement>('#se-status')!;
    await act(async()=>{select.value='partial';select.dispatchEvent(new Event('change',{bubbles:true}));});
    expect(mock.mock.lastCall?.[2]?.query).toEqual(expect.objectContaining({q:'render',status:'partial'}));
    const check=view.host.querySelector<HTMLInputElement>('input[type="checkbox"]')!;await act(async()=>check.click());
    expect(mock.mock.lastCall?.[2]?.query).toEqual(expect.objectContaining({q:'render',representative:'false'}));
  });
  it('moves focus to selected comparison once so readers can start at its heading',async()=>{
    mock.mockResolvedValue(data());const view=await mount();
    const summary={id:'pair-1',title:'render input difference',target:'package.render',status:'supported',differences:[]};
    mock.mockResolvedValue(data('one',{status:'READY',pairs:[summary],total:1,all_pairs:1}));await view.click('用例对照');
    mock.mockResolvedValue(data('one',{pair:{...summary,common:[],scenarios:[],reading:[]}}));
    const button=view.host.querySelector<HTMLButtonElement>('.se-pair-list button')!;await act(async()=>button.click());
    expect(document.activeElement).toBe(view.host.querySelector('.se-comparison h2'));
  });
  it.each(['PARSE_FAILED','UNSUPPORTED','ZERO_COMPARISONS'])('gives a real next step for %s without fake results',async(status)=>{
    mock.mockResolvedValue(data());const view=await mount();mock.mockResolvedValue(data('one',{status,pairs:[],total:0,all_pairs:0}));await view.click('用例对照');
    expect(view.host.textContent).toContain('项目地图');expect(view.host.querySelectorAll('.se-pair-list button')).toHaveLength(0);expect(view.host.querySelector('.se-comparison')).toBeNull();
  });

});
