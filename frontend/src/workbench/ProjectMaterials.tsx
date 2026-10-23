import type { ProjectResponse } from '../types';
import { useSourceFile } from './SourceFileReader';
const validCitationPath = (path: string) => Boolean(path && !/^[\\/]|^[a-zA-Z]:|\0/.test(path) && path.split(/[\\/]/).every(part => part && part !== '.' && part !== '..'));

export function ProjectMaterials({ project, workspaceId, revision, loading }: { project: ProjectResponse; workspaceId: string; revision: string; loading: boolean }) {
  const openFile = useSourceFile();
  const identity = Boolean(project.project.id && workspaceId && revision && !loading);
  function file(path: string) {
    const available = identity && validCitationPath(path) && Boolean(openFile);
    return <button type="button" key={path} disabled={!available} title={available ? '阅读已分析快照的源码' : '缺少有效的项目、工作区、版本或路径，无法打开源码。'} onClick={event => openFile?.({ projectId: project.project.id, workspaceId, revision, path }, event.currentTarget)}>{path}</button>;
  }
  return <details className="wb-project-materials"><summary>项目资料</summary><div>
    <p>{project.overview || '仓库简介未提供。'}</p>
    <dl><dt>语言</dt><dd>{project.project.primary_language || '未提供'}</dd><dt>版本</dt><dd>{revision || '未提供'}</dd><dt>来源</dt><dd>{project.project.source_location || project.project.repo_url || '未提供'}</dd></dl>
    {!identity && <p>缺少有效的当前快照身份，暂不能打开源码。</p>}
    <h3>核心文件</h3>{project.core_files.length ? project.core_files.map(item => <div key={item.path}>{file(item.path)}<p>{item.summary}</p></div>) : <p>分析记录未提供核心文件。</p>}
    <h3>模块</h3>{project.modules.length ? project.modules.map(module => <div key={module.name}><strong>{module.name}</strong><p>{module.responsibility}</p>{module.files.map(path => file(path))}</div>) : <p>分析记录未提供模块。</p>}
  </div></details>;
}
