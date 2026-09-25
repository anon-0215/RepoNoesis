import type { ChatAnswer, Citation } from '../types';

export interface EvidenceBinding { evidenceId: string; citations: Citation[]; indices: number[] }

export function evidenceBindings(result: ChatAnswer, projectId?: string, revision?: string): Map<string, EvidenceBinding> {
  const bindings = new Map<string, EvidenceBinding>();
  if (!result.evidence || !projectId || !revision) return bindings;
  const evidenceById = new Map(result.evidence.map((item) => [item.evidence_id, item]));
  result.citations.forEach((citation, index) => {
    const id = citation.evidence_id;
    if (!id || !/^E[1-9]\d*$/.test(id)) return;
    const evidence = evidenceById.get(id);
    if (!evidence || evidence.project_id !== projectId || evidence.repository_revision !== revision ||
        evidence.path !== citation.path || (evidence.qualified_name || evidence.symbol_name) !== citation.qualified_name ||
        citation.start_line < evidence.start_line || citation.end_line > evidence.end_line ||
        citation.start_line <= 0 || citation.end_line < citation.start_line) return;
    const binding = bindings.get(id) || { evidenceId: id, citations: [], indices: [] };
    binding.citations.push(citation);
    binding.indices.push(index);
    bindings.set(id, binding);
  });
  return bindings;
}
