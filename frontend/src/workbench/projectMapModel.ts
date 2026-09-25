import type { ProjectMap, TreeNode } from '../types';

export interface MapNode {
  id: string;
  name: string;
  path: string;
  type: 'directory' | 'file';
  parentId: string | null;
  children: string[];
  isCore?: boolean;
}

export interface MapIndex {
  rootId: string;
  nodes: Map<string, MapNode>;
  fileCount: number;
  incomplete: boolean;
}

export function nodeId(projectId: string, revision: string, type: string, path: string): string {
  return JSON.stringify([projectId, revision, type, path]);
}

export function indexProjectMap(map: ProjectMap, projectId: string, revision: string): MapIndex | null {
  const root = map.tree;
  if (!root || root.type !== 'directory' || root.path !== '' || !Array.isArray(root.children)) return null;
  const nodes = new Map<string, MapNode>();
  let incomplete = false;
  let fileCount = 0;
  function visit(input: TreeNode, parent: MapNode | null): void {
    const path = input.path;
    if (typeof path !== 'string' || typeof input.name !== 'string' ||
        (parent && (path !== `${parent.path ? `${parent.path}/` : ''}${input.name}` || input.name === '.' || input.name === '..' || input.name.includes('\\'))) ||
        (input.type !== 'file' && input.type !== 'directory')) { incomplete = true; return; }
    const id = nodeId(projectId, revision, input.type, path);
    if (nodes.has(id)) { incomplete = true; return; }
    const node: MapNode = { id, name: input.name, path, type: input.type, parentId: parent?.id ?? null, children: [], isCore: input.is_core };
    nodes.set(id, node);
    parent?.children.push(id);
    if (input.type === 'file') { fileCount += 1; return; }
    if (!Array.isArray(input.children)) { incomplete = true; return; }
    input.children.forEach((child) => visit(child, node));
  }
  visit(root, null);
  return { rootId: nodeId(projectId, revision, 'directory', ''), nodes, fileCount, incomplete };
}

export interface VisibleNode { node: MapNode; depth: number; row: number; parentRow: number | null; }
export interface MoreRow { parentId: string; depth: number; row: number; remaining: number; direction: 'previous' | 'next'; }

export function visibleMap(index: MapIndex, expanded: Set<string>, shown: Map<string, number>): { nodes: VisibleNode[]; more: MoreRow[] } {
  const nodes: VisibleNode[] = [];
  const more: MoreRow[] = [];
  function visit(id: string, depth: number, parentRow: number | null) {
    const node = index.nodes.get(id);
    if (!node) return;
    const row = nodes.length + more.length;
    nodes.push({ node, depth, row, parentRow });
    if (node.type !== 'directory' || !expanded.has(id)) return;
    const start = Math.min(shown.get(id) ?? 0, Math.max(0, node.children.length - 1));
    const end = Math.min(node.children.length, start + 24);
    if (start > 0) more.push({ parentId: id, depth: depth + 1, row: nodes.length + more.length, remaining: start, direction: 'previous' });
    node.children.slice(start, end).forEach((child) => visit(child, depth + 1, row));
    if (end < node.children.length) more.push({ parentId: id, depth: depth + 1, row: nodes.length + more.length, remaining: node.children.length - end, direction: 'next' });
  }
  visit(index.rootId, 0, null);
  return { nodes, more };
}

export function ancestorIds(index: MapIndex, id: string): string[] {
  const result: string[] = [];
  let parent = index.nodes.get(id)?.parentId;
  while (parent) { result.unshift(parent); parent = index.nodes.get(parent)?.parentId; }
  return result;
}
