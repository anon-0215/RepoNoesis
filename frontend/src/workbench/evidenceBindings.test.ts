import { describe, expect, it } from 'vitest';
import type { ChatAnswer } from '../types';
import { evidenceBindings } from './evidenceBindings';

const revision = 'a'.repeat(40);
const result: ChatAnswer = {
  answer: '[E1] [E5]', answer_mode: 'llm_grounded', grounding_status: 'grounded', warnings: [],
  evidence: [
    { evidence_id: 'E1', project_id: 'p', repository_revision: revision, path: 'src/shared.py', qualified_name: 'first', start_line: 10, end_line: 20 },
    { evidence_id: 'E5', project_id: 'p', repository_revision: revision, path: 'src/shared.py', qualified_name: 'fifth', start_line: 30, end_line: 40 }
  ],
  citations: [
    { evidence_id: 'E5', path: 'src/shared.py', qualified_name: 'fifth', summary: '', snippet: 'fifth', start_line: 30, end_line: 40 },
    { evidence_id: 'E1', path: 'src/shared.py', qualified_name: 'first', summary: '', snippet: 'first', start_line: 10, end_line: 20 }
  ]
};

describe('authoritative response binding', () => {
  it('uses E identities across reordered citations and distinct ranges in one file', () => {
    const bindings = evidenceBindings(result, 'p', revision);
    expect(bindings.get('E1')?.citations[0].snippet).toBe('first');
    expect(bindings.get('E5')?.indices).toEqual([0]);
  });
  it('leaves old and mismatched responses unbound', () => {
    expect(evidenceBindings({ ...result, evidence: undefined }, 'p', revision).size).toBe(0);
    expect(evidenceBindings(result, 'other', revision).size).toBe(0);
    expect(evidenceBindings(result, 'p', 'other').size).toBe(0);
    expect(evidenceBindings({ ...result, citations: [{ ...result.citations[0], start_line: 99 }] }, 'p', revision).has('E5')).toBe(false);
  });
});
