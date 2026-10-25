import { afterEach, describe, expect, it, vi } from 'vitest';
import { explore } from './exploration';
afterEach(()=>vi.unstubAllGlobals());
describe('exploration response binding',()=>{
  it('reports an oversized A response with A wording',async()=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify({detail:{code:'exploration_result_too_large',message:'private',retryable:false}}),{status:413})));
    await expect(explore({projectId:'p',workspaceId:'w',revision:'r'},'/errors',{body:{error:'bad',revision:'r'}})).rejects.toThrow('报错定位结果超过显示上限');
  });
  it.each([{project_id:'other',workspace_id:'w',revision:'r'}, {project_id:'p',workspace_id:'other',revision:'r'}, {project_id:'p',workspace_id:'w',revision:'old'}])('rejects a mismatched response %j',async(identity)=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify({identity,data_version:'v'}),{status:200})));
    await expect(explore({projectId:'p',workspaceId:'w',revision:'r'})).rejects.toThrow('仓库或版本不符');
  });
  it('passes revision, pagination, cancellation and the active project',async()=>{
    const fetcher=vi.fn().mockResolvedValue(new Response(JSON.stringify({identity:{project_id:'p',workspace_id:'w',revision:'r'},data_version:'v'}),{status:200}));vi.stubGlobal('fetch',fetcher);const controller=new AbortController();
    await explore({projectId:'p',workspaceId:'w',revision:'r'},'/comparisons',{query:{offset:'20'},signal:controller.signal});
    expect(fetcher.mock.calls[0][0]).toContain('/api/projects/p/exploration/comparisons?revision=r&offset=20');
    expect(fetcher.mock.calls[0][1].signal).toBe(controller.signal);
  });
});
