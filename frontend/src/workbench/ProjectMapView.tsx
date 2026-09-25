import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { ChevronDown, ChevronRight, File, Folder, LocateFixed, Minus, Plus, Search, X } from 'lucide-react';
import type { ProjectMap } from '../types';
import { ancestorIds, indexProjectMap, visibleMap, type MapIndex, type MapNode } from './projectMapModel';

interface Props { projectId: string; revision: string; projectMap: ProjectMap | null; error: string | null; loading: boolean; onRetry: () => void; }
const STEP_X = 225;
const STEP_Y = 59;
const NODE_WIDTH = 194;
const NODE_HEIGHT = 43;
const MIN_ZOOM = 0.35;
const MAX_ZOOM = 2.2;

function useCanvas(index: MapIndex | null) {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(index ? [index.rootId] : []));
  const [shown, setShown] = useState<Map<string, number>>(() => new Map());
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [view, setView] = useState(() => window.innerWidth <= 650 ? { x: 12, y: 18, zoom: 0.7 } : { x: 30, y: 28, zoom: 1 });
  const canvas = useRef<HTMLDivElement>(null);
  const drag = useRef<{ pointerId: number; x: number; y: number; initialX: number; initialY: number } | null>(null);
  const selectionTrigger = useRef<HTMLElement | null>(null);
  const focusTarget = useRef<HTMLDivElement>(null);
  const structure = useMemo(() => index ? visibleMap(index, expanded, shown) : { nodes: [], more: [] }, [index, expanded, shown]);
  const matches = useMemo(() => {
    if (!index || !query.trim()) return [];
    const needle = query.trim().toLocaleLowerCase();
    return [...index.nodes.values()].filter((node) => node.path.toLocaleLowerCase().includes(needle) || node.name.toLocaleLowerCase().includes(needle));
  }, [index, query]);

  useEffect(() => { if (index) setExpanded(new Set([index.rootId])); }, [index]);

  function locate(id: string) {
    if (!index) return;
    const ancestors = ancestorIds(index, id);
    setExpanded((old) => new Set([...old, ...ancestors]));
    setShown((old) => {
      const next = new Map(old);
      for (const parentId of ancestors) {
        const parent = index.nodes.get(parentId);
        const child = parent?.children.find((candidate) => candidate === id || ancestorIds(index, id).includes(candidate));
        const position = child ? parent?.children.indexOf(child) ?? -1 : -1;
        if (position >= 0) next.set(parentId, Math.floor(position / 24) * 24);
      }
      return next;
    });
    setSelected(id);
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
      const node = [...(canvas.current?.querySelectorAll<HTMLElement>('[data-map-node-id]') ?? [])].find((item) => item.dataset.mapNodeId === id);
      const host = canvas.current;
      if (!node || !host) return;
      const x = Number(node.dataset.x || 0);
      const y = Number(node.dataset.y || 0);
      setView((old) => ({ ...old, x: host.clientWidth / 2 - (x + NODE_WIDTH / 2) * old.zoom, y: host.clientHeight / 2 - (y + NODE_HEIGHT / 2) * old.zoom }));
      if (window.innerWidth <= 650) focusTarget.current?.scrollIntoView?.({ block: 'nearest' });
    }));
  }

  function fit() {
    const host = canvas.current;
    if (!host || !structure.nodes.length) return;
    const maxDepth = Math.max(...structure.nodes.map((entry) => entry.depth), ...structure.more.map((entry) => entry.depth), 0);
    const width = maxDepth * STEP_X + NODE_WIDTH + 36;
    const height = (structure.nodes.length + structure.more.length) * STEP_Y + 28;
    const zoom = Math.max(MIN_ZOOM, Math.min(1.4, (host.clientWidth - 30) / width, (host.clientHeight - 30) / height));
    setView({ x: 15, y: 15, zoom });
  }

  function zoomBy(factor: number) { setView((old) => ({ ...old, zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, old.zoom * factor)) })); }
  useEffect(() => {
    const host = canvas.current;
    if (!host || !index) return;
    function onWheel(event: globalThis.WheelEvent) {
      if (!host || event.defaultPrevented || event.ctrlKey || event.metaKey || drag.current) return;
      if (!Number.isFinite(event.deltaX) || !Number.isFinite(event.deltaY) || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
      const rect = host.getBoundingClientRect();
      const x = event.clientX - rect.left - host.clientLeft;
      const y = event.clientY - rect.top - host.clientTop;
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? Math.max(host.clientHeight, 360) : 1;
      const delta = Math.max(-160, Math.min(160, event.deltaY * unit));
      const factor = Math.exp(-delta * 0.0015);
      event.preventDefault();
      setView((old) => {
        const zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, old.zoom * factor));
        if (zoom === old.zoom) return old;
        const ratio = zoom / old.zoom;
        return { x: x - (x - old.x) * ratio, y: y - (y - old.y) * ratio, zoom };
      });
    }
    host.addEventListener('wheel', onWheel, { passive: false });
    return () => host.removeEventListener('wheel', onWheel);
  }, [index]);
  function stopDrag(pointerId?: number) {
    const active = drag.current;
    if (!active || (pointerId !== undefined && active.pointerId !== pointerId)) return;
    drag.current = null;
    const host = canvas.current;
    if (host?.hasPointerCapture?.(active.pointerId)) host.releasePointerCapture(active.pointerId);
  }
  useEffect(() => {
    window.addEventListener('blur', onBlur);
    function onBlur() { stopDrag(); }
    return () => { window.removeEventListener('blur', onBlur); stopDrag(); };
  }, []);
  function pointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.target !== event.currentTarget || event.button !== 0) return;
    event.preventDefault();
    drag.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, initialX: view.x, initialY: view.y };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function pointerMove(event: PointerEvent<HTMLDivElement>) {
    const active = drag.current;
    if (!active || active.pointerId !== event.pointerId) return;
    const x = active.initialX + event.clientX - active.x;
    const y = active.initialY + event.clientY - active.y;
    setView((old) => ({ ...old, x, y }));
  }
  function close() { setSelected(null); window.setTimeout(() => selectionTrigger.current?.focus(), 0); }

  return { expanded, setExpanded, shown, setShown, selected, setSelected, query, setQuery, view, canvas, structure, matches, selectionTrigger, focusTarget, locate, fit, zoomBy, pointerDown, pointerMove, stopDrag, close };
}

export function ProjectMapView({ projectId, revision, projectMap, error, loading, onRetry }: Props) {
  const index = useMemo(() => projectMap ? indexProjectMap(projectMap, projectId, revision) : null, [projectMap, projectId, revision]);
  const state = useCanvas(index);
  const selectedNode: MapNode | null = state.selected ? index?.nodes.get(state.selected) ?? null : null;
  const declaredCount = projectMap?.file_count;
  const partial = Boolean(index && (index.incomplete || (typeof declaredCount === 'number' && declaredCount !== index.fileCount)));
  const scope = projectMap?.coverage === 'analyzed_files' ? '已分析文件' : '旧接口返回的分析文件（覆盖范围未声明）';
  const matchedIds = useMemo(() => new Set(state.matches.map((node) => node.id)), [state.matches]);

  useEffect(() => {
    if (!selectedNode) return;
    function onKey(event: globalThis.KeyboardEvent) { if (event.key === 'Escape') { event.preventDefault(); state.close(); } }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [selectedNode]);

  return <section className="pm" aria-label="项目地图">
    <header className="pm-heading"><div><p>PROJECT MAP</p><h1>项目地图</h1><span>{scope} · revision {revision ? revision.slice(0, 12) : '未提供'}</span></div>{index && <strong>{index.fileCount} 个文件 · {index.nodes.size - index.fileCount} 个目录</strong>}</header>
    {loading && !projectMap ? <div className="pm-state" role="status">正在加载当前项目的地图…</div> : error ? <div className="pm-state" role="alert">{error}<button type="button" onClick={onRetry}>重试</button></div> : !projectMap ? <div className="pm-state" role="status">地图尚未加载。</div> : !index ? <div className="pm-state" role="alert">地图结构缺失或格式无效，无法判断项目是否为空。<button type="button" onClick={onRetry}>重试</button></div> : <>
      {partial && <p className="pm-warning" role="status">地图文件数与接口记录不一致，或部分路径无法安全展示；当前仅展示已解析的节点。</p>}
      {index.fileCount === 0 ? <div className="pm-state">{partial ? '返回的结构不完整，无法确认分析文件是否为空。' : '当前分析文件结构为空；不代表仓库没有文件。'}</div> : <>
        <div className="pm-search"><Search aria-hidden="true" /><input aria-label="搜索已分析文件或目录" value={state.query} onChange={(event) => state.setQuery(event.target.value)} placeholder="搜索名称或相对路径" />{state.query && <button type="button" aria-label="清空搜索" onClick={() => state.setQuery('')}><X /></button>}</div>
        {state.query.trim() && <div className="pm-results" aria-label="地图搜索结果"><p>在{scope}中找到 {state.matches.length} 项{state.matches.length > 50 ? '，显示前 50 项' : ''}</p>{state.matches.length === 0 ? <span>没有匹配的文件或目录。</span> : <div>{state.matches.slice(0, 50).map((node) => <button key={node.id} type="button" onClick={(event) => { state.selectionTrigger.current = event.currentTarget; state.locate(node.id); }}><strong>{node.name}</strong><small>{node.path || '项目根目录'}</small></button>)}</div>}</div>}
        <div className="pm-tools"><span>拖动画布平移 · 滚轮缩放</span><div><button type="button" aria-label="缩小地图" onClick={() => state.zoomBy(0.8)}><Minus /></button><output aria-label="地图缩放">{Math.round(state.view.zoom * 100)}%</output><button type="button" aria-label="放大地图" onClick={() => state.zoomBy(1.25)}><Plus /></button><button type="button" onClick={state.fit}><LocateFixed />适应画布</button></div></div>
        <div className="pm-map-area"><div ref={state.canvas} className="pm-canvas" aria-label="目录包含关系图" onPointerDown={state.pointerDown} onPointerMove={state.pointerMove} onPointerUp={(event) => state.stopDrag(event.pointerId)} onPointerCancel={(event) => state.stopDrag(event.pointerId)} onLostPointerCapture={(event) => state.stopDrag(event.pointerId)}>
          <div className="pm-world" style={{ transform: `translate(${state.view.x}px, ${state.view.y}px) scale(${state.view.zoom})` }}>
            <svg className="pm-edges" width={Math.max(500, (Math.max(0, ...state.structure.nodes.map((entry) => entry.depth)) + 1) * STEP_X)} height={(state.structure.nodes.length + state.structure.more.length + 1) * STEP_Y} aria-hidden="true">{state.structure.nodes.filter((entry) => entry.parentRow !== null).map((entry) => <path key={entry.node.id} d={`M ${((entry.depth - 1) * STEP_X) + NODE_WIDTH} ${(entry.parentRow! * STEP_Y) + NODE_HEIGHT / 2} H ${entry.depth * STEP_X - 14} V ${(entry.row * STEP_Y) + NODE_HEIGHT / 2} H ${entry.depth * STEP_X}`} />)}</svg>
            {state.structure.nodes.map(({ node, depth, row }) => <div className={`pm-node ${node.type} ${state.selected === node.id ? 'selected' : ''} ${matchedIds.has(node.id) ? 'matched' : ''}`} key={node.id} data-map-node-id={node.id} data-x={depth * STEP_X} data-y={row * STEP_Y} style={{ left: depth * STEP_X, top: row * STEP_Y }}><button type="button" className="pm-node-select" title={node.path || '项目根目录'} aria-label={`${node.type === 'directory' ? '目录' : '文件'} ${node.path || node.name}`} aria-pressed={state.selected === node.id} onClick={(event) => { state.selectionTrigger.current = event.currentTarget; state.setSelected(node.id); }} >{node.type === 'directory' ? <Folder /> : <File />}<span>{node.name}</span></button>{node.type === 'directory' && <button type="button" className="pm-node-toggle" aria-label={`${state.expanded.has(node.id) ? '折叠' : '展开'}目录 ${node.path || node.name}`} aria-expanded={state.expanded.has(node.id)} onClick={() => state.setExpanded((old) => { const next = new Set(old); if (next.has(node.id)) next.delete(node.id); else next.add(node.id); return next; })}>{state.expanded.has(node.id) ? <ChevronDown /> : <ChevronRight />}</button>}</div>)}
            {state.structure.more.map((entry) => <button className="pm-more" key={`${entry.parentId}:${entry.direction}`} type="button" style={{ left: entry.depth * STEP_X, top: entry.row * STEP_Y }} onClick={() => state.setShown((old) => new Map(old).set(entry.parentId, Math.max(0, (old.get(entry.parentId) ?? 0) + (entry.direction === 'next' ? 24 : -24))))}>{entry.direction === 'next' ? '下一组' : '上一组'} 24 项（剩余 {entry.remaining}）</button>)}
          </div>
        </div>
        {selectedNode && <aside ref={state.focusTarget} className="pm-detail" aria-label="地图节点详情"><div><h2>{selectedNode.type === 'directory' ? '目录详情' : '文件详情'}</h2><button type="button" aria-label="关闭地图详情" onClick={state.close}><X /></button></div><dl><dt>名称</dt><dd>{selectedNode.name}</dd><dt>类型</dt><dd>{selectedNode.type === 'directory' ? '目录' : '文件'}</dd><dt>相对路径</dt><dd>{selectedNode.path || '项目根目录'}</dd><dt>revision</dt><dd>{revision || '未提供'}</dd>{selectedNode.type === 'directory' && <><dt>直接子项</dt><dd>{selectedNode.children.length}（当前分析结构）</dd></>}{selectedNode.type === 'file' && selectedNode.isCore && <><dt>分析标记</dt><dd>核心文件</dd></>}</dl><p>此节点来自项目结构，不属于问答引用。</p></aside>}
        </div>
        <p className="pm-note">连线仅表示目录包含关系。初始折叠目录，每个目录一次显示最多 24 项；搜索覆盖本次接口返回的整棵分析文件树。</p>
      </>}
    </>}
  </section>;
}
