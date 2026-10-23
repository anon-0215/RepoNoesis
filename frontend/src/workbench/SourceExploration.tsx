import { useEffect, useRef, useState, type FormEvent } from 'react';
import { explore, type Candidate, type Entry, type Exploration, type ExplorationContext, type Pair, type SourceRef } from '../lib/exploration';
import { useSourceFile } from './SourceFileReader';
import './source-exploration.css';
import { TracebackPositions } from './TracebackPositions';

const messages: Record<string,string> = {
  MATERIALS_MISSING: '当前快照未收录可识别的作者测试文件，不能据此判断原仓库没有测试。可在下方补充同版本材料。',
  PARSE_FAILED: '已收录测试材料，但相关文件无法静态解析。请展开“材料范围与诊断”查看路径，再从项目地图阅读原始源码。',
  UNSUPPORTED: '已收录材料中没有符合当前静态规则的测试；复杂 fixture、动态设置等需直接阅读源码。可展开不支持的测试示例，或从项目地图继续阅读。',
  ZERO_COMPARISONS: '已提取测试场景，但没有可配对的输入条件。可从项目地图阅读原测试；材料不足时在下方补充同版本源码。',
  NO_MATCH: '没有找到匹配的静态候选。动态消息、未收录源码或不支持的格式可能无法匹配。',
  NO_SUPPORTED_SOURCE: '当前材料中没有可支持的抛错语句。可检查源码收录范围。',
};

export function SourceExploration({ projectId, workspaceId, revision }: ExplorationContext) {
  const ctx = { projectId, workspaceId, revision };
  const openFile = useSourceFile();
  const [mode, setMode] = useState<'a'|'b'>('a');
  const [meta, setMeta] = useState<Exploration | null>(null);
  const [result, setResult] = useState<Exploration | null>(null);
  const [pair, setPair] = useState<Pair | null>(null);
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const [input, setInput] = useState(''); const [submitted, setSubmitted] = useState('');
  const [pasteWarning, setPasteWarning] = useState('');
  const [search, setSearch] = useState(''); const [status, setStatus] = useState('');
  const [appliedSearch, setAppliedSearch] = useState('');
  const [representative, setRepresentative] = useState(true);
  const [localPath, setLocalPath] = useState('');
  type Options = { query?: Record<string,string>; body?: unknown };
  type Request = { route: string; options: Options; detail: boolean };
  const [failedRequest, setFailedRequest] = useState<Request | null>(null);
  const [selectedPairId, setSelectedPairId] = useState<string | null>(null);
  const listQuery = useRef<Record<string,string>>({ q: '', status: '', representative: 'true' });
  const pending = useRef<AbortController | null>(null); const generation = useRef(0);
  const pairHeading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (!pair) return;
    pairHeading.current?.focus({ preventScroll: true });
    pairHeading.current?.scrollIntoView?.({ block: 'start' });
  }, [pair?.id]);
  const action = async (route: string, options: Options = {}, detail = false): Promise<boolean> => {
    pending.current?.abort(); const controller = new AbortController(); pending.current = controller;
    const request: Request = { route, options: { ...options, query: options.query ? { ...options.query } : undefined }, detail };
    const id = ++generation.current; setBusy(true); setError(''); setFailedRequest(null);
    // A new search owns its result. A detail request keeps only the valid list.
    setPair(null);
    if (detail) setSelectedPairId(route.split('/').pop() || null);
    else { setResult(null); setSelectedPairId(null); }
    if (route === '/comparisons') listQuery.current = { ...options.query };
    try {
      const data = await explore(ctx, route, { ...request.options, signal: controller.signal });
      if (controller.signal.aborted || id !== generation.current) return false;
      if (detail) {
        if (meta && meta.data_version !== data.data_version) throw new Error('源码材料已更新，请重新搜索并选择对照。');
        setPair(data.pair || null);
      } else {
        setMeta(data); setResult(route === '' || route === '/materials' ? null : data);
        if (route === '/materials') {
          // Successful retries must refresh exactly as the initial material action does.
          return await action(mode === 'a' ? '' : '/comparisons', { query: { ...listQuery.current, offset: '0' } });
        }
      }
      return true;
    } catch (reason) {
      if (!controller.signal.aborted && id === generation.current) {
        setError(reason instanceof Error ? reason.message : '读取失败，请重试。');
        setFailedRequest(request);
      }
      return false;
    } finally { if (id === generation.current) setBusy(false); }
  };
  useEffect(() => {
    setSubmitted('');
    void action(mode === 'a' ? '' : '/comparisons', { query: { q: appliedSearch, representative: String(representative), status } });
    return () => { ++generation.current; pending.current?.abort(); };
  }, [projectId, workspaceId, revision, mode, representative, status]);
  const retryLabel = failedRequest?.detail ? '重试读取对照' : failedRequest?.route === '/errors' ? '重试查询'
    : failedRequest?.route === '/comparisons' ? '重试搜索' : failedRequest?.route === '/materials' ? '重试材料读取' : '重新加载探索';

  function source(ref: SourceRef, label: string) {
    return <button type="button" onClick={event => openFile?.({ projectId, workspaceId, revision, path: ref.path, start: ref.start, end: ref.end }, event.currentTarget)}>{label}<small>{ref.path} · 行 {ref.start}–{ref.end}</small></button>;
  }
  function entries(title: string, rows: Entry[]) {
    return <section className="se-entries"><h4>{title}</h4>{rows.length ? rows.map((entry,i) => <div key={i}><p>{entry.text || entry.label}</p>{entry.reason && <small>{entry.reason}</small>}{source(entry.source, title === '候选实现' ? '阅读候选实现' : title === '原始断言' ? '查看原始断言' : '查看来源')}</div>) : <p>未提取到此项。</p>}</section>;
  }
  function queryErrors(event?: FormEvent, offset = 0) {
    event?.preventDefault(); const text = event ? input : submitted;
    if (!text.trim()) return; setSubmitted(text);
    void action('/errors', { body: { revision, error: text, offset, limit: 20 } });
  }
  function queryPairs(event?: FormEvent, offset = 0) {
    event?.preventDefault();
    const query = event ? { q: search, status, representative: String(representative) } : listQuery.current;
    if (event) setAppliedSearch(search);
    void action('/comparisons', { query: { ...query, offset: String(offset) } });
  }
  function candidate(item: Candidate) {
    return <article key={item.id} className="se-card se-candidate">
      <p className="se-location">可能位置 · {item.path} · 行 {item.line}–{item.end_line}</p>
      <h3>{item.symbol}</h3><pre>{item.raise_source}</pre>
      {item.trace_frames?.length ? <p className="se-trace-badge">当前快照范围关联：堆栈帧 {item.trace_frames.map(i=>`#${i}`).join('、')}；运行版本尚未核验。</p> : null}
      <h4>包围抛错语句的局部条件 / 分支</h4>
      {item.controls.length ? <ol>{item.controls.map((control,i) => <li key={i}><code>{control.source}</code> <small>行 {control.line}</small></li>)}</ol> : <p>所属局部作用域内没有包围它的条件或分支。</p>}
      <div className="se-candidate-foot">{source({ path: item.path, start: item.line, end: item.end_line, sha256: item.sha256 }, '打开抛错源码')}<small>{item.match_mode === 'template' ? '有限格式化文本匹配' : '字面量文本匹配'} · 异常类型运行绑定未确认</small></div>
    </article>;
  }
  const inputNames = Array.from(new Set(pair?.scenarios.flatMap(scene => Object.keys(scene.fields)) || []));
  return <section className="wb-page se-page" aria-label="源码探索">
    <header className="se-header"><div><h1>源码探索</h1><p>当前仓库 {meta?.identity.repo || ''} · 版本 {revision.slice(0,12)}</p></div>
      <div className="se-tabs" role="group" aria-label="探索方式"><button type="button" aria-pressed={mode==='a'} onClick={() => setMode('a')}>报错定位</button><button type="button" aria-pressed={mode==='b'} onClick={() => setMode('b')}>用例对照</button></div>
    </header>
    {mode === 'a' ? <form onSubmit={queryErrors} className="se-search se-error-search">
      <label htmlFor="se-error">报错文本</label><textarea id="se-error" rows={3} maxLength={4096} value={input} onChange={e => { setInput(e.target.value); setPasteWarning(''); }} onPaste={e=>{const el=e.currentTarget;const length=input.length-(el.selectionEnd-el.selectionStart)+e.clipboardData.getData('text').length;if(length>4096){e.preventDefault();setPasteWarning('内容超过4096字符，本次粘贴未接收，输入框保持原内容；请缩短后重试，不会自动截取堆栈。');}else setPasteWarning('');}} placeholder="粘贴普通 Python traceback，或只输入报错末行" />
      <small>{input.length}/4096 字符 · 普通File帧最多核对32项；复杂异常链与ExceptionGroup不完整支持。</small>
      {pasteWarning && <p role="alert">{pasteWarning}</p>}
      {input.length===4096 && <p className="se-boundary">已达输入上限，请核对是否包含完整末行；超过上限的粘贴会被拒绝。</p>}
      <div className="se-search-foot"><button type="submit" disabled={!input.trim()}>查找源码候选</button><small>静态候选与局部结构，不能确定现场根因。</small></div>
    </form> : <><p className="se-boundary">测试预期未执行；候选实现不证明运行绑定或因果。</p>
      <form className="se-search se-pair-search" onSubmit={queryPairs}>
        <label className="se-query-field" htmlFor="se-search">搜索对照<input id="se-search" value={search} maxLength={200} onChange={e => setSearch(e.target.value)} placeholder="函数、输入条件或预期" /></label>
        <label className="se-status-field" htmlFor="se-status">支持情况<select id="se-status" value={status} onChange={e => setStatus(e.target.value)}><option value="">全部支持情况</option><option value="supported">符合静态规则</option><option value="partial">存在未解析条件</option></select></label>
        <label className="se-check"><input type="checkbox" checked={representative} onChange={e => setRepresentative(e.target.checked)} />优先显示代表对照</label><button type="submit">搜索</button>
      </form></>}
    {busy && <p className="se-loading" role="status">{selectedPairId ? '正在读取所选对照…' : '正在读取当前版本的源码材料…'}</p>}
    {error && <div role="alert" className="se-error"><p>{error}</p><button type="button" disabled={busy || !failedRequest} onClick={() => { if (failedRequest) void action(failedRequest.route, failedRequest.options, failedRequest.detail); }}>{retryLabel}</button><small>可继续使用源码问答、历史和项目地图。</small></div>}
    {result && <div aria-live="polite">
      {mode==='a' && result.traceback && <TracebackPositions key={`${submitted}:${result.offset || 0}`} result={result.traceback} source={source} />}
      {result.status && messages[result.status] && <p className="se-empty">{messages[result.status]}</p>}
      {mode==='a' && result.status==='UNSUPPORTED' && <p className="se-empty">当前材料中的抛错消息不符合支持规则；动态变量、任意格式化等需直接阅读源码。</p>}
      {mode==='a' && result.total!>0 && <p className="se-result-count">找到 {result.total} 个可能位置，均为静态候选。</p>}
      {mode==='a' && result.candidates?.map(candidate)}
      {mode==='b' && <><p className="se-result-count">已提交关键词：<code>{appliedSearch || '全部'}</code> · 当前筛选 {result.total} 组 / 已整理 {result.all_pairs} 组</p>
        {result.status==='READY' && result.total===0 && <p>没有符合当前搜索或筛选条件的对照。</p>}
        <div className="se-pair-list">{result.pairs?.map(item => <button type="button" key={item.id} aria-pressed={selectedPairId===item.id} onClick={() => void action(`/comparisons/${encodeURIComponent(item.id)}`,{},true)}>
          <strong>{item.title}</strong><small>{item.target} · {item.status==='partial' ? '部分条件未解析' : '符合静态规则'}</small>
          <span className="se-preview-label">输入差异预览 · 完整值见详情</span>
          {item.differences.map((difference,i) => <span className="se-preview" key={i}><b>{difference.field}</b><code>{difference.a}</code><span>→</span><code>{difference.b}</code></span>)}
        </button>)}</div></>}
      {(result.total || 0)>20 && <div className="se-pagination"><button type="button" disabled={busy || !result.offset} onClick={() => mode==='a' ? queryErrors(undefined,Math.max(0,result.offset!-20)) : queryPairs(undefined,Math.max(0,result.offset!-20))}>上一页</button><span>第 {Math.floor((result.offset || 0)/20)+1} 页</span><button type="button" disabled={busy || (result.offset || 0)+20>=result.total!} onClick={() => mode==='a' ? queryErrors(undefined,(result.offset || 0)+20) : queryPairs(undefined,(result.offset || 0)+20)}>下一页</button></div>}
    </div>}
    {mode==='b' && pair && <section className="se-comparison" aria-label="对照详情">
      <h2 ref={pairHeading} tabIndex={-1}>{pair.title}</h2>
      <p className="se-boundary"><strong>测试预期未执行</strong> · {pair.status==='partial' ? '部分条件未解析，完整条件与未确认项见下方。' : '符合静态规则，运行绑定与分支仍未确认。'}</p>
      <div className="se-overview">{pair.scenarios.map((scene,i) => <article key={scene.id} className="se-overview-card">
        <h3>场景{i===0 ? '一' : '二'} · {scene.title}</h3>
        <section className="se-overview-inputs"><h4>显式输入条件</h4>{inputNames.length ? <dl>{inputNames.map(name => <div key={name}><dt>{name}</dt><dd><code>{Object.prototype.hasOwnProperty.call(scene.fields,name) ? scene.fields[name] : pair.differences.find(d => d.field===name)?.[i===0 ? 'a' : 'b'] || '未提取到显式输入'}</code></dd></div>)}</dl> : <p>未提取到显式输入。</p>}</section>
        <section className="se-overview-expectations"><h4>作者测试预期（未执行）</h4>{scene.expectations.length ? scene.expectations.map((entry,n) => <p key={n}><code>{entry.text}</code></p>) : <p>未提取到测试预期。</p>}</section>
        <small>{scene.status==='partial' ? '部分条件未解析' : '运行绑定未确认'} · {scene.unknowns.length} 项未确认说明，可展开下方完整依据核对。</small>
      </article>)}</div>
      <div className="se-evidence-columns">{pair.scenarios.map((scene,i) => <details key={scene.id} className="se-evidence"><summary>场景{i===0 ? '一' : '二'}：原始断言与候选实现</summary>{entries('原始断言',scene.assertions)}{entries('候选实现',scene.implementation)}</details>)}</div>
      <details className="se-context-details"><summary>完整条件、来源与未确认项</summary>
        <p>静态指向相同目标：{pair.target}；不等于完整前置条件相同。</p><h3>共同项</h3>{pair.common?.length ? <ul>{pair.common.map((text,i) => <li key={i}><code>{text}</code></li>)}</ul> : <p>未提取额外共同项。</p>}
        <h3>完整输入差异</h3><table><thead><tr><th>字段</th><th>场景一</th><th>场景二</th></tr></thead><tbody>{pair.differences.map(diff => <tr key={diff.field}><th>{diff.field}</th><td><code>{diff.a}</code></td><td><code>{diff.b}</code></td></tr>)}</tbody></table>
        <div className="se-scenes">{pair.scenarios.map((scene,i) => <article key={scene.id} className="se-card"><h3>场景{i===0 ? '一' : '二'} · {scene.title}</h3>{entries('输入条件',scene.conditions)}{entries('显式输入及来源',scene.inputs)}{entries('测试预期及来源',scene.expectations)}<h4>未确认项</h4><ul>{scene.unknowns.map((unknown,n) => <li key={n}>{unknown}</li>)}</ul></article>)}</div>
        {pair.reasons?.map((reason,i) => <p key={i}>{reason}</p>)}
      </details>
    </section>}
    {mode==='b' && result?.unsupported?.length ? <details className="se-materials"><summary>不支持的测试示例与缺失条件</summary>{result.unsupported.map((item,i) => <div key={i}><p>{item.title}：{item.reason}</p>{source(item.source,'阅读测试')}</div>)}</details> : null}
    <details className="se-materials"><summary>补充同版本源码材料</summary><p>如果快照缺少 tests、conftest 或辅助实现，可提供包含该精确版本的本地 Git 仓库。只读取该提交的 Git 文件对象，不读取其他版本的工作树，不改变原快照或问答索引。</p><form onSubmit={e => { e.preventDefault(); void action('/materials',{ body:{ revision,local_path:localPath } }); }}><label htmlFor="se-path">本地 Git 仓库目录</label><input id="se-path" value={localPath} onChange={e => setLocalPath(e.target.value)} placeholder="本地仓库完整路径" /><button type="submit" disabled={busy || !localPath.trim()}>读取同版本材料</button></form><p>无法取得该版本或内容冲突时会拒绝补充；不会下载源码。最多 500 个 Python 文件、每文件 1 MiB、共 20 MB。</p></details>
    {meta && <details className="se-diagnostics"><summary>材料范围与诊断</summary><p>已收录 {meta.coverage.files} 个 Python 文件，解析失败 {meta.coverage.parse_errors.length} 个，跳过 {meta.coverage.skipped_count} 个。{meta.coverage.same_revision_inventory ? '已核对同版本 Git 文件清单。' : '仅包含保存快照；原仓库的完整测试数量未知。'}</p><p>缺少辅助实现会显示未解析或不支持，不生成替代案例。</p><pre>{JSON.stringify({ identity:meta.identity, data_version:meta.data_version, algorithm:meta.algorithm, coverage:meta.coverage },null,2)}</pre></details>}
  </section>;
}
