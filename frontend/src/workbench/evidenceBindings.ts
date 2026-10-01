import type { ChatAnswer, ReadableCitation, RecoveryMetadata, SavedCitationBinding } from '../types';

export interface EvidenceBinding { evidenceId: string; citations: ReadableCitation[]; indices: number[]; source: 'live_evidence' | 'saved_binding' }

export function evidenceBindings(result: ChatAnswer, projectId?: string, revision?: string): Map<string, EvidenceBinding> {
  return bindCitations(result.citations, result.evidence, projectId, revision, 'live_evidence');
}

export function savedCitationBindings(citations: ReadableCitation[], metadata: RecoveryMetadata | null,
  projectId: string | undefined, revision: string | undefined, reliable: boolean): Map<string, EvidenceBinding> {
  return reliable ? bindCitations(citations, metadata?.citation_bindings, projectId, revision, 'saved_binding') : new Map();
}

function bindCitations(citations: ReadableCitation[], identities: Array<Omit<SavedCitationBinding, 'symbol_name'> & { symbol_name?: string }> | undefined,
  projectId: string | undefined, revision: string | undefined, source: EvidenceBinding['source']): Map<string, EvidenceBinding> {
  const bindings = new Map<string, EvidenceBinding>();
  if (!identities || !projectId || !revision) return bindings;
  const evidenceById = new Map(identities.map((item) => [item.evidence_id, item]));
  if (evidenceById.size !== identities.length) return bindings;
  citations.forEach((citation, index) => {
    const id = citation.evidence_id;
    if (!id || !/^E[1-9]\d*$/.test(id)) return;
    const evidence = evidenceById.get(id);
    if (!evidence || evidence.project_id !== projectId || evidence.repository_revision !== revision ||
        evidence.path !== citation.path || (evidence.qualified_name || evidence.symbol_name) !== citation.qualified_name ||
        !Number.isInteger(citation.start_line) || !Number.isInteger(citation.end_line) ||
        citation.start_line! < evidence.start_line || citation.end_line! > evidence.end_line ||
        citation.start_line! <= 0 || citation.end_line! < citation.start_line!) return;
    const binding = bindings.get(id) || { evidenceId: id, citations: [], indices: [], source };
    binding.citations.push(citation);
    binding.indices.push(index);
    bindings.set(id, binding);
  });
  return bindings;
}
