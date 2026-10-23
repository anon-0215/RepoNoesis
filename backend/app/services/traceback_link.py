"""Bounded ordinary traceback linking. Input paths are text, never filesystem IO."""
from __future__ import annotations

import ast
import re
from typing import Any
from app.source_lines import source_lines

MAX_FRAMES = 32
MAX_LOCATIONS = 5
MAX_PATH = 1024
MAX_LINE = 10_000_000
FRAME = re.compile(r'^\s*File "([^"\r\n]*)", line ([^,\s]+), in (.{1,200})\s*$')
EXCEPTION = re.compile(r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::(?:\s.*)?)?$')
CARET = re.compile(r'^[ \t]*[~^][~^ \t]*$')
HEADER = 'Traceback (most recent call last):'
BOUNDARY = '仅对应当前快照；用户实际运行版本尚未核验。调用位置不等于抛错位置，静态关联不证明现场根因。'


def parse(text: str) -> dict[str, Any]:
    lines = source_lines(text.strip())
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    terminal = lines[nonempty[-1]] if nonempty else ''
    detected = any(line.strip() == HEADER or line.lstrip().startswith('File ') for line in lines)
    complex_input = any(line.strip() in (
        'During handling of the above exception, another exception occurred:',
        'The above exception was the direct cause of the following exception:')
        or 'ExceptionGroup' in line and ('Traceback' in line or line.lstrip().startswith(('ExceptionGroup:', 'BaseExceptionGroup:')))
        or line.lstrip().startswith(('[Previous line repeated ', '+ Exception Group Traceback', '| ExceptionGroup', '| Traceback'))
        for line in lines)
    detected = detected or complex_input
    if not detected:
        if len(nonempty)>1:
            return {'status':'UNSUPPORTED','complete':False,'frames':[], 'frames_detected':0,
                    'issues':['未识别普通Python堆栈格式；未完成堆栈关联，仅尝试末行报错匹配。'],
                    'error_text':terminal, 'boundary':BOUNDARY}
        return {'status':'NOT_PROVIDED','complete':False,'frames':[], 'frames_detected':0,
                'issues':[], 'error_text':terminal, 'boundary':BOUNDARY}
    ordinary = [(i, FRAME.fullmatch(line)) for i, line in enumerate(lines)]
    count = sum(match is not None for _, match in ordinary)
    if complex_input or sum(line.strip() == HEADER for line in lines) > 1:
        return {'status':'UNSUPPORTED','complete':False,'frames':[], 'frames_detected':count,
                'issues':['复杂异常链、ExceptionGroup或压缩递归格式不在支持范围；未完成堆栈关联，仅尝试末行报错匹配。'],
                'error_text':terminal if EXCEPTION.fullmatch(terminal) else '', 'boundary':BOUNDARY}
    issues=[]; frames=[]; current=None
    terminal_index=nonempty[-1] if nonempty and EXCEPTION.fullmatch(terminal) and terminal==terminal.lstrip() else -1
    if not any(line.strip()==HEADER for line in lines):issues.append('未见普通Traceback开头，输入可能不完整。')
    if terminal_index<0:issues.append('未识别末行异常信息，输入可能不完整。')
    for i, match in ordinary:
        line=lines[i]
        if match:
            current={'index':len(frames)+1,'input_path':match[1],'line_text':match[2],
                     'function':match[3].strip(),'display_source':None}
            frames.append(current)
        elif i==terminal_index or line.strip()==HEADER or not line.strip():
            continue
        elif CARET.fullmatch(line):
            if current is None:issues.append('位置提示行缺少所属帧。')
        elif current is not None and line[:1] in (' ', '\t') and not line.lstrip().startswith('File '):
            if current['display_source'] is None:current['display_source']=line.strip()
            else:issues.append('一帧包含多条源码展示行，格式未完整支持。')
        else:issues.append('存在无法识别的堆栈行，未完成堆栈关联。')
    if not count:issues.append('没有识别到普通File帧。')
    limited=count>MAX_FRAMES
    if limited:issues.append(f'识别到{count}帧，仅核对前{MAX_FRAMES}帧；其后帧未处理，非完整关联。')
    return {'status':'LIMITED' if limited else 'PARTIAL' if issues else 'COMPLETE',
            'complete':not issues,'frames':frames[:MAX_FRAMES],'frames_detected':count,
            'issues':list(dict.fromkeys(issues)),'error_text':terminal if terminal_index>=0 else '', 'boundary':BOUNDARY}


def path_matches(raw: str, paths: list[str]) -> tuple[str, list[str]]:
    if not raw or len(raw)>MAX_PATH or any(ord(c)<32 or ord(c)==127 for c in raw):return 'INVALID',[]
    normalized=raw.replace('\\','/')
    # Reject traversal instead of resolving/reading any user path.
    parts=normalized.split('/')
    if '..' in parts or any(':' in p for p in parts[1:]):return 'INVALID',[]
    absolute=normalized.startswith('/') or bool(re.match(r'^[A-Za-z]:/',normalized))
    if ':' in normalized and not re.match(r'^[A-Za-z]:/',normalized):return 'INVALID',[]
    normalized=normalized.removeprefix('./')
    if not absolute and normalized in paths and '/' in normalized:return 'EXACT',[normalized]
    if '/' not in normalized:
        found=[p for p in paths if p.rsplit('/',1)[-1]==normalized]
        return ('WEAK' if found else 'MISSING'),found
    # Full stored relative path must be a suffix, with at least two components.
    found=[p for p in paths if '/' in p and normalized.endswith('/'+p)]
    if found:
        return ('SUFFIX' if len(found)==1 else 'AMBIGUOUS'),found
    return 'MISSING',[]


def scopes(text: str) -> tuple[list[dict[str, Any]], list[tuple[int,int]], list[tuple[int,int]]]:
    tree=ast.parse(text); result=[]; raises=[]; uncertain=[]
    def walk(node: ast.AST, names: tuple[str,...]=()):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            # Definition headers/defaults/decorators run in an enclosing scope.
            # Same-line bodies cannot be separated using a traceback line alone.
            header_start=min([node.lineno]+[d.lineno for d in node.decorator_list])
            body_start=node.body[0].lineno if node.body else node.lineno
            uncertain.append((header_start,max(node.lineno,body_start-1)))
            names=names+(node.name,)
            result.append({'symbol':'.'.join(names),'name':node.name,'start':node.lineno,
                           'end':node.end_lineno,'kind':'class' if isinstance(node,ast.ClassDef) else 'function'})
        if isinstance(node,(ast.Lambda,ast.ListComp,ast.SetComp,ast.DictComp,ast.GeneratorExp)):
            uncertain.append((node.lineno,node.end_lineno))
        if isinstance(node,ast.Raise):raises.append((node.lineno,node.end_lineno))
        for child in ast.iter_child_nodes(node):walk(child,names)
    walk(tree)
    return result,raises,uncertain


def associate(data: dict[str,Any], parsed: dict[str,Any], candidates: list[dict[str,Any]]) -> dict[str,Any]:
    """Only read data['files']; never open, resolve, or execute an input path."""
    files=data['files']; paths=sorted(files); indexes={}; by_candidate={c['id']:c for c in candidates}
    for candidate in candidates:candidate['trace_frames']=[]
    for frame in parsed['frames']:
        raw_line=frame.pop('line_text'); valid_line=raw_line.isascii() and raw_line.isdigit() and len(raw_line)<=8 and 0<int(raw_line)<=MAX_LINE
        frame['line']=int(raw_line) if valid_line else None
        frame['input_line']=raw_line
        path_status, found=path_matches(frame['input_path'],paths)
        frame.update(path_status=path_status,matches_total=len(found),matches=[],status='MISSING',issues=[])
        if path_status=='INVALID' or not valid_line or not frame['function']:
            frame['status']='INVALID';frame['issues'].append('路径、行号或函数字段无效，未读取输入路径，也未移动行号。')
            parsed['complete']=False
            if parsed['status']=='COMPLETE':parsed['status']='PARTIAL'
            continue
        if not found:
            frame['issues'].append('当前已收录材料没有可明确对应的文件；不能据此判断是外部依赖。');continue
        ambiguous=path_status in ('WEAK','AMBIGUOUS') or len(found)>1
        if len(found)>MAX_LOCATIONS:frame['issues'].append(f'有{len(found)}个可能文件，仅列出前{MAX_LOCATIONS}项；没有自动选定文件。')
        for path in found[:MAX_LOCATIONS]:
            record=files[path]; lines=source_lines(record['content']); line=frame['line']
            location={'path':path,'line':line,'sha256':record['content_sha256'],'origin':record['origin'],
                      'total_lines':len(lines),'status':'UNVERIFIED','checks':[],'candidate_ids':[]}
            frame['matches'].append(location)
            if line>len(lines):location['checks'].append('行号超出当前正文范围；不能打开或调整到其他行。');location['status']='OUT_OF_RANGE';continue
            raw=lines[line-1];location.update(source_preview=raw[:400],source_preview_truncated=len(raw)>400,readable=True)
            conflict=False
            if frame['display_source'] is not None:
                location['source_check']='AGREES' if frame['display_source']==raw.strip() else 'DIFFERS'
                if location['source_check']=='DIFFERS':conflict=True;location['checks'].append('展示源码与当前快照该行不一致。')
            else:location['source_check']='NOT_SUPPLIED';location['checks'].append('未提供源码展示行，文本一致性未核对。')
            if path not in indexes:
                try:indexes[path]=scopes(record['content'])
                except (SyntaxError,ValueError,RecursionError):indexes[path]=None
            index=indexes[path]
            if index is None:
                location['checks'].append('当前文件无法AST解析，函数/抛错范围未核对。')
                if conflict:location['status']='CONFLICT'
                continue
            definitions,raises,uncertain=index
            enclosed=[s for s in definitions if s['start']<=line<=s['end']]
            scope=min(enclosed,key=lambda s:s['end']-s['start']) if enclosed else {'symbol':'<module>','name':'<module>','kind':'module','start':1,'end':len(lines)}
            if any(a<=line<=b for a,b in uncertain):
                location['checks'].append('该行涉及定义头、lambda或推导式作用域，函数关系未确认。')
                if conflict:location['status']='CONFLICT'
                continue
            agrees=frame['function'] in (scope['name'],scope['symbol'])
            location['function_check']='AGREES' if agrees else 'DIFFERS'
            if not agrees:conflict=True;location['checks'].append('帧函数与当前源码最内层语法范围不一致。')
            if conflict:location['status']='CONFLICT';continue
            if ambiguous:location['checks'].append('路径信息不足或存在多义文件，不确认函数/抛错关联。');continue
            location.update(status='MATCHED',scope=scope,relation='FUNCTION_RANGE' if scope['kind']!='module' else 'POSITION_ONLY')
            span=next(((a,b) for a,b in raises if a<=line<=b),None)
            if span:
                location.update(relation='RAISE_RANGE',raise_range={'start':span[0],'end':span[1]})
                for item in data['items']:
                    if item['path']==path and item['line']<=line<=item['end_line'] and item['id'] in by_candidate:
                        location['candidate_ids'].append(item['id']);by_candidate[item['id']]['trace_frames'].append(frame['index'])
        frame['status']='AMBIGUOUS' if ambiguous else frame['matches'][0]['status']
    return parsed
