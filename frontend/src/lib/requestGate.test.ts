import { describe, expect, it } from 'vitest';

import { createRequestGate } from './requestGate';

describe('request gate', () => {
  it('rejects a second submit until the in-flight request leaves', () => {
    const gate = createRequestGate();
    const first = gate.tryEnter('ask');
    expect(first).not.toBeNull();
    expect(gate.tryEnter('ask')).toBeNull();
    expect(gate.finish(first!, true)).toBe(true);
    expect(gate.tryEnter('ask')).not.toBeNull();
  });

  it.each(['success', 'failure', 'exception'])('releases ask after %s', async (outcome) => {
    const gate = createRequestGate();
    const token = gate.tryEnter('ask');
    expect(token).not.toBeNull();
    try {
      if (outcome === 'exception') throw new Error('synthetic');
      await Promise.resolve(outcome);
    } catch {}
    finally {
      expect(gate.finish(token!, true)).toBe(true);
    }
    expect(gate.tryEnter('ask')).not.toBeNull();
  });

  it('drops stale ask success and failure after switching from A to B', async () => {
    const gate = createRequestGate();
    const openA = gate.beginContextChange('context', 'workspace-a');
    expect(
      gate.commitContext(openA, {
        workspaceId: 'workspace-a',
        projectId: 'project-a',
        revision: 'revision-a'
      })
    ).toBe(true);
    gate.finish(openA);

    const staleSuccess = gate.tryEnter('ask');
    expect(staleSuccess).not.toBeNull();
    const openB = gate.beginContextChange('context', 'workspace-b');
    expect(
      gate.commitContext(openB, {
        workspaceId: 'workspace-b',
        projectId: 'project-b',
        revision: 'revision-b'
      })
    ).toBe(true);
    gate.finish(openB);

    const answers: string[] = [];
    const errors: string[] = [];
    await Promise.resolve('answer-a').then((answer) => {
      if (gate.isCurrent(staleSuccess!)) answers.push(answer);
    });
    await Promise.reject(new Error('error-a')).catch((error: Error) => {
      if (gate.isCurrent(staleSuccess!)) errors.push(error.message);
    });
    expect(answers).toEqual([]);
    expect(errors).toEqual([]);
  });

  it('stale A finally cannot release or clear an in-flight B ask', () => {
    const gate = createRequestGate();
    const openA = gate.beginContextChange('context', 'workspace-a');
    gate.commitContext(openA, {
      workspaceId: 'workspace-a',
      projectId: 'project-a',
      revision: 'revision-a'
    });
    gate.finish(openA);
    const askA = gate.tryEnter('ask')!;

    const openB = gate.beginContextChange('context', 'workspace-b');
    gate.commitContext(openB, {
      workspaceId: 'workspace-b',
      projectId: 'project-b',
      revision: 'revision-b'
    });
    gate.finish(openB);
    const askB = gate.tryEnter('ask')!;

    expect(gate.finish(askA, true)).toBe(false);
    expect(gate.tryEnter('ask')).toBeNull();
    expect(gate.isCurrent(askB)).toBe(true);
    expect(gate.finish(askB, true)).toBe(true);
  });

  it('only the newest overlapping workspace open may commit success or failure', () => {
    const gate = createRequestGate();
    const openA = gate.beginContextChange('context', 'workspace-a');
    const openB = gate.beginContextChange('context', 'workspace-b');

    expect(
      gate.commitContext(openA, {
        workspaceId: 'workspace-a',
        projectId: 'project-a',
        revision: 'revision-a'
      })
    ).toBe(false);
    expect(gate.clearContext(openA)).toBe(false);
    expect(gate.finish(openA)).toBe(false);
    expect(gate.isActive(openB)).toBe(true);
    expect(
      gate.commitContext(openB, {
        workspaceId: 'workspace-b',
        projectId: 'project-b',
        revision: 'revision-b'
      })
    ).toBe(true);
    expect(gate.getContext()).toMatchObject({
      workspaceId: 'workspace-b',
      projectId: 'project-b',
      revision: 'revision-b'
    });
  });

  it('restores the previous project only for the current failed analysis', () => {
    const gate = createRequestGate();
    const openA = gate.beginContextChange('context', 'workspace-a');
    gate.commitContext(openA, {
      workspaceId: 'workspace-a',
      projectId: 'project-a',
      revision: 'revision-a'
    });
    gate.finish(openA);

    const analyze = gate.beginContextChange('context', '');
    expect(gate.restoreContext(analyze)).toBe(true);
    expect(gate.getContext()).toMatchObject({
      workspaceId: 'workspace-a',
      projectId: 'project-a',
      revision: 'revision-a'
    });
    expect(gate.finish(analyze)).toBe(true);
  });
});
