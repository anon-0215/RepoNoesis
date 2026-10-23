// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { ProjectMaterials } from './ProjectMaterials';
import { SourceFileContext } from './SourceFileReader';
import type { ProjectResponse } from '../types';
const project = { project: { id: 'p' }, overview: 'existing overview', core_files: [{ path: 'a.py', summary: 'existing summary' }, { path: '../invalid.py', summary: '' }], modules: [] } as unknown as ProjectResponse;
describe('collapsible project materials', () => {
  it('keeps unavailable identity and unsafe file paths explicitly unavailable', () => {
    const host = document.createElement('div');
    host.innerHTML = renderToStaticMarkup(<SourceFileContext.Provider value={vi.fn()}><ProjectMaterials project={project} workspaceId="w" revision="" loading={false} /></SourceFileContext.Provider>);
    expect(host.textContent).toContain('暂不能打开源码');
    expect(host.querySelectorAll('button:disabled')).toHaveLength(2);
    expect(host.querySelector<HTMLDetailsElement>('details')?.open).toBe(false);
  });
  it('opens only a valid file through the current snapshot reader', async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    const open = vi.fn(), host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
    try {
      await act(async () => root.render(<SourceFileContext.Provider value={open}><ProjectMaterials project={project} workspaceId="w" revision="rev" loading={false} /></SourceFileContext.Provider>));
      expect(host.querySelectorAll('button:disabled')).toHaveLength(1);
      await act(async () => host.querySelector<HTMLButtonElement>('button')!.click());
      expect(open).toHaveBeenCalledWith({ projectId: 'p', workspaceId: 'w', revision: 'rev', path: 'a.py' }, expect.any(HTMLButtonElement));
    } finally { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); }
  });
});
