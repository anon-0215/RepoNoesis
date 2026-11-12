import { describe, expect, it } from 'vitest';
import { validEvidenceKinds } from '../lib/api';
import { evidenceBindings, savedCitationBindings } from './evidenceBindings';
import type { ChatAnswer, RecoveryMetadata } from '../types';

const citation = { evidence_id: 'E1', source_kind: 'document' as const, path: 'README.md',
  qualified_name: 'README.md', summary: 'Project documentation', snippet: '# Acorn\n', start_line: 1, end_line: 2 };
const evidence = { ...citation, project_id: 'p', repository_revision: 'rev', symbol_name: '',
  code_chunk_id: null, document_file_id: 7, chunk_type: 'document', language: 'text' };
const result: ChatAnswer = { answer: 'Document statement [E1]', citations: [citation], evidence: [evidence],
  answer_mode: 'llm_grounded', grounding_status: 'grounded', warnings: [] };

describe('document citation identities', () => {
  it('binds genuine document and restores saved binding', () => {
    expect(validEvidenceKinds(result as unknown as Record<string, unknown>)).toBe(true);
    expect(evidenceBindings(result, 'p', 'rev').has('E1')).toBe(true);
    const metadata: RecoveryMetadata = { format_version: 1, citation_bindings: [evidence], metadata_truncated: false };
    expect(savedCitationBindings([citation], metadata, 'p', 'rev', true).has('E1')).toBe(true);
    expect(savedCitationBindings([citation], metadata, 'p', 'old', true).size).toBe(0);
  });
  it('rejects fake document chunks, unknown kinds and changed citation kinds', () => {
    expect(validEvidenceKinds({ ...result, evidence: [{ ...evidence, code_chunk_id: 7 }] })).toBe(false);
    expect(validEvidenceKinds({ ...result, evidence: [{ ...evidence, source_kind: 'other' }] })).toBe(false);
    expect(evidenceBindings({ ...result, citations: [{ ...citation, source_kind: 'code' }] }, 'p', 'rev').size).toBe(0);
    expect(evidenceBindings(result, 'other-project', 'rev').size).toBe(0);
  });
  it('keeps legacy source-only bindings compatible', () => {
    const source = { ...citation, path: 'a.py', qualified_name: 'compute', source_kind: undefined };
    expect(evidenceBindings({ ...result, citations: [source], evidence: [{ ...source,
      project_id: 'p', repository_revision: 'rev' }] }, 'p', 'rev').has('E1')).toBe(true);
  });
});
