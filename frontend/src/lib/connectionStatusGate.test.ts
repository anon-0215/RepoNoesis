import { describe, expect, it } from 'vitest';

import { createConnectionStatusGate } from './connectionStatusGate';

describe('connection status gate', () => {
  it('allows a newer success to clear an older connection failure', () => {
    const gate = createConnectionStatusGate();
    const failed = gate.begin(0);
    const recovered = gate.begin(0);
    expect(gate.settle(failed)).toBe(true);
    expect(gate.settle(recovered)).toBe(true);
  });

  it('does not let an older late success override a newer failure', () => {
    const gate = createConnectionStatusGate();
    const stale = gate.begin(0);
    const current = gate.begin(0);
    expect(gate.settle(current)).toBe(true);
    expect(gate.settle(stale)).toBe(false);
  });

  it('isolates probes after a workspace generation change', () => {
    const gate = createConnectionStatusGate();
    const oldWorkspace = gate.begin(0);
    gate.changeContext(1);
    const newWorkspace = gate.begin(1);
    expect(gate.settle(oldWorkspace)).toBe(false);
    expect(gate.settle(newWorkspace)).toBe(true);
  });
});
