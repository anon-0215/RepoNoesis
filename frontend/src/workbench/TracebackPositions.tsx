import type { ReactNode } from 'react';
import type { SourceRef, TracebackResult } from '../lib/exploration';

const labels: Record<string,string> = { MATCHED:'对应当前快照', AMBIGUOUS:'路径有歧义或信息不足', MISSING:'当前材料无法对应', INVALID:'路径或行号无效', OUT_OF_RANGE:'行号超出当前正文', CONFLICT:'与当前源码不一致', UNVERIFIED:'范围未确认' };
export function TracebackPositions({ result, source }: { result: TracebackResult; source: (ref:SourceRef,label:string)=>ReactNode }) {
  if (result.status==='NOT_PROVIDED') return null;
  const relevant=result.frames.filter(f=>f.status==='MATCHED');
  const preferred=new Set([...relevant].sort((a,b)=>Number(b.matches.some(m=>m.relation==='RAISE_RANGE'))-Number(a.matches.some(m=>m.relation==='RAISE_RANGE')) || b.index-a.index).slice(0,3).map(f=>f.index));
  return <section className="se-traceback" aria-label="堆栈关联">
    <h2>堆栈中的调用位置</h2>
    <p>{result.complete ? '普通帧格式已识别' : '未完成堆栈关联'} · 识别 {result.frames_detected} 帧 · {relevant.length} 帧可明确对应当前快照</p>
    <p className="se-boundary">{result.boundary}</p>
    {result.issues.length>0 && <ul className="se-trace-issues">{result.issues.map((issue,i)=><li key={i}>{issue}</li>)}</ul>}
    {result.status==='UNSUPPORTED' && <p>以下候选仅使用末行报错文本，不代表完整支持该堆栈。</p>}
    {result.frames.length>0 && <><p className="se-trace-hint">按输入调用顺序排列；优先展开最多三个相关位置，其余可逐项展开。</p>
      <ol className="se-trace-frames">{result.frames.map(frame=><li key={frame.index}><details open={preferred.has(frame.index)}>
        <summary><strong>#{frame.index} · {labels[frame.status] || frame.status}</strong><span>{frame.input_path} · 行 {frame.line ?? frame.input_line} · {frame.function}</span></summary>
        {frame.issues.map((issue,i)=><p key={i}>{issue}</p>)}
        {frame.status==='AMBIGUOUS' && <p>共 {frame.matches_total} 个可能文件；没有自动选定文件或确认抛错关系。</p>}
        {frame.matches.map((location,i)=><div key={i} className="se-trace-location">
          <p><strong>{location.path} · 行 {location.line}</strong> · {labels[location.status] || location.status}</p>
          {location.source_preview!==undefined && <pre>{location.source_preview}</pre>}
          {location.source_preview_truncated && <small>源码预览已摘要，完整正文可在阅读器查看。</small>}
          {location.scope && <p>该位置落在当前源码{location.scope.kind==='function' ? '函数' : '语法'}范围：<code>{location.scope.symbol}</code>（行 {location.scope.start}–{location.scope.end}）。</p>}
          {location.relation==='RAISE_RANGE' ? <p>落在当前抛错语句范围{location.candidate_ids.length ? '，与消息候选关联（候选可分页查看）' : '；未匹配到本次消息候选'}。</p> : location.status==='MATCHED' && <p>这是调用/源码位置，不是已确认的抛错语句。</p>}
          {location.source_check==='AGREES' && <small>提供的源码展示行与当前该行文本一致。</small>}
          {location.checks.map((check,n)=><p className="se-boundary" key={n}>{check}</p>)}
          {location.readable && source({path:location.path,start:location.line,end:location.line,sha256:location.sha256},'打开当前快照位置')}
        </div>)}
      </details></li>)}</ol></>}
  </section>;
}
