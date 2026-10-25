"""Deterministic A/B analysis over one revision-bound source contract."""
from __future__ import annotations

import ast
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from threading import RLock
from typing import Any

from app.database import SourceFileError
from app.services import exploration_sources as sources
from app.services.error_locator import Extractor, BOUNDARY, matches
from app.services.use_case_extractor import Repository
from app.services.use_case_selection import choose_representatives
from app.services import traceback_link

ALGORITHM = 'source-exploration-3-direct-parametrization'
_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_lock = RLock()


def analysis(db: Any, runtime: Path, project_id: str, revision: str) -> dict[str, Any]:
    data = sources.read_sources(db,runtime,project_id,revision)
    ctx = data['identity']
    key = (str(Path(db.path).resolve()),ctx['workspace_id'],project_id,revision,
           ctx['activation_version'],data['fingerprint'],ALGORITHM)
    with _lock:
        if key not in _cache:
            items=[]; parse_errors=[]; stats={'raise_total':0,'supported':0,'unsupported':0}
            for path,record in data['files'].items():
                try:
                    ext = Extractor(record['content'],project_id,path,revision,record['content_sha256'])
                    ext.walk(ast.parse(record['content'],filename=path))
                    record['parse_status']='parsed'
                except (SyntaxError,ValueError,RecursionError) as error:
                    record['parse_status']='failed'
                    parse_errors.append({'path':path,'line':getattr(error,'lineno',None)})
                    continue
                for name,value in ext.stats.items(): stats[name]+=value
                items.extend(ext.items)
            repo=Repository(data['files'],project_id,revision)
            tests,_=repo.run()
            for pair in tests['comparisons']:
                pair['id']=sources.sha(json.dumps([project_id,revision,data['fingerprint'],ALGORITHM,pair['id']]))[:24]
                a,b=pair['scenarios']
                pair['common']=[f'{k} = {v}' for k,v in a['fields'].items() if b['fields'].get(k)==v]
            groups=defaultdict(list)
            for pair in tests['comparisons']: groups[pair['target']].append(pair)
            representatives={p['id'] for group in groups.values() for p in choose_representatives(group,3)}
            value={'items':items,'tests':tests,'representatives':representatives,
                   'parse_errors':parse_errors,'raise_stats':stats,
                   'file_status':[{'path':p,'content_sha256':f['content_sha256'],'origin':f['origin'],
                                   'included':True,'truncated':False,'parse_status':f['parse_status']}
                                  for p,f in data['files'].items()]}
            _cache[key]=value
            while len(_cache)>4: _cache.popitem(last=False)
        else:
            _cache.move_to_end(key)
        cached=_cache[key]
    # A repository update may finish during parsing; fail rather than return stale output.
    if sources.context(db,project_id,revision)['activation_version']!=ctx['activation_version']:
        raise SourceFileError('source_snapshot_inactive',409)
    return {**data,**cached,'algorithm':ALGORITHM}


def envelope(data: dict[str, Any]) -> dict[str, Any]:
    return {'identity':data['identity'],'data_version':data['fingerprint'],'algorithm':ALGORITHM,
            'coverage':{**data['coverage'],'parse_errors':data['parse_errors'],
                        'raise_stats':data['raise_stats'],'tests':data['tests']['inventory'],
                        'file_status':data['file_status']}}


def error_query(data: dict[str, Any], error: str, offset: int=0, limit: int=20) -> dict[str, Any]:
    trace=traceback_link.parse(error)
    text=trace.pop('error_text')
    prefix=re.fullmatch(r'([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):\s+(.*)',text)
    variants=[('',text)]
    if prefix: variants.append(prefix.groups())
    candidates=[]
    for item in data['items']:
        p=item['pattern']
        if not text: continue
        matching=next(((kind,value) for kind,value in variants if matches(p,value)),None)
        if matching:
            error_type,_=matching
            candidate={k:v for k,v in item.items() if k!='pattern'}
            candidate.update(match_mode=p['mode'],template=p['template'],
                type_name_relation='NOT_SUPPLIED' if not error_type else 'SYNTACTIC_NAME_AGREES'
                if error_type.rsplit('.',1)[-1]==item['exception_expression'].rsplit('.',1)[-1]
                else 'NAME_DIFFERS_OR_UNKNOWN')
            candidates.append(candidate)
    status='NO_MATCH' if not candidates else 'MULTIPLE_CANDIDATES' if len(candidates)>1 else 'TEMPLATE_CANDIDATE' if candidates[0]['match_mode']=='template' else 'EXACT_CANDIDATE'
    if not data['items']: status='UNSUPPORTED' if data['raise_stats']['unsupported'] else 'NO_SUPPORTED_SOURCE'
    trace=traceback_link.associate(data,trace,candidates)
    trace['algorithm']='ordinary-traceback-v1'
    page=candidates[offset:offset+limit]
    result={**envelope(data),'status':status,'normalized_error':text,'boundary':BOUNDARY,
            'candidates':page,'total':len(candidates),'offset':offset,'limit':limit,'traceback':trace}
    if len(json.dumps(result,ensure_ascii=False).encode('utf-8'))>250_000:
        raise SourceFileError('exploration_result_too_large',413)
    return result


def comparisons(data: dict[str, Any], query: str='', status: str='', representative: bool=True,
                offset: int=0, limit: int=20) -> dict[str, Any]:
    tests=data['tests']; pairs=tests['comparisons']
    selected=[p for p in pairs if (not representative or p['id'] in data['representatives'])
              and (not status or p['status']==status)
              and (not query or query.casefold() in json.dumps(p,ensure_ascii=False).casefold())]
    inventory=tests['inventory']
    physical_tests=sum(Repository.is_test(p) for p in data['files'])
    failed_tests=any(Repository.is_test(p['path']) for p in data['parse_errors'])
    state=('MATERIALS_MISSING' if not physical_tests else
           'PARSE_FAILED' if not inventory['tests'] and failed_tests else
           'UNSUPPORTED' if not inventory['scenarios'] else 'ZERO_COMPARISONS' if not pairs else 'READY')
    return {**envelope(data),'status':state,'total':len(selected),'all_pairs':len(pairs),
            'offset':offset,'limit':limit,'pairs':[
                {**{k:p[k] for k in ('id','title','target','status')},
                 'differences':[{'field':d['field'],'a':d['a'][:200],'b':d['b'][:200]} for d in p['differences']]}
                for p in selected[offset:offset+limit]],
            'unsupported':tests['unsupported'][:20]}


def comparison(data: dict[str, Any], pair_id: str) -> dict[str, Any]:
    pair=next((p for p in data['tests']['comparisons'] if p['id']==pair_id),None)
    if pair is None: raise SourceFileError('exploration_pair_not_found',404)
    # Full file bodies are fetched only through the existing reader endpoint.
    if len(json.dumps(pair,ensure_ascii=False).encode('utf-8'))>250_000:
        raise SourceFileError('exploration_pair_too_large',413)
    return {**envelope(data),'pair':pair}
