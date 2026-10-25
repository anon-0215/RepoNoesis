"""Generic, non-executing Python test comparison discovery. Python stdlib only.

Input: ordinary source roots and revision metadata. No evaluation manifest is read.
All implementation links describe static reference candidates, never runtime binding.
"""
from __future__ import annotations
import ast
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

class Unknown(ValueError): pass

def dotted(node):
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return base + '.' + node.attr if base else ''
    return ''

def constant(node, env):
    """Deliberately small expression language; never imports or executes source."""
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Name) and node.id in env: return env[node.id][0]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [constant(x, env) for x in node.elts]
        if isinstance(node, ast.Tuple): return tuple(values)
        if isinstance(node, ast.Set): raise Unknown('集合顺序不纳入支持范围')
        return values
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys): raise Unknown('字典展开不支持')
        return {constant(k, env): constant(v, env) for k,v in zip(node.keys,node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = constant(node.operand,env)
        if type(value) not in (int,float): raise Unknown('非数字一元运算')
        return -value if isinstance(node.op,ast.USub) else value
    raise Unknown('不是字面量或已知常量局部变量')

def digest(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

class Repository:
    def __init__(self, files, name='Python 仓库', revision='未提供'):
        self.name=name; self.revision=revision
        self.files={}; self.modules={}; self.defs=defaultdict(list); self.exports={}
        self.sources={}; self.scenarios=[]; self.unsupported=[]; self.tests=0
        self.discovery=[]; self.exclusions=Counter(); self.parse_errors=[]
        self.scene_work=0; self.statement_work=0; self.budget_exhausted=False
        for rel,record in sorted(files.items()):
            if not rel.endswith('.py'): continue
            text=record['content']
            try: tree=ast.parse(text,filename=rel)
            except (SyntaxError,ValueError,RecursionError) as error:
                self.parse_errors.append({'path':rel,'line':getattr(error,'lineno',None)})
                self.exclusions['无法静态解析的 Python 文件']+=1; continue
            self.files[rel]=(text,tree,record['content_sha256'])
            parts=Path(rel).with_suffix('').parts
            if parts[0]=='src': parts=parts[1:]
            if parts[-1]=='__init__': parts=parts[:-1]
            module='.'.join(parts)
            self.modules[module]=rel
        for module,rel in self.modules.items():
            tree=self.files[rel][1]
            if self.is_test_context(rel) or any(x in Path(rel).parts for x in ('examples','docs')): continue
            aliases=self.imports(tree.body,module,Path(rel).name=='__init__.py')
            self.exports[module]=aliases
            for node in tree.body:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    self.defs[module+'.'+node.name].append((rel,node,module,node.name))
                    if isinstance(node,ast.ClassDef):
                        for member in node.body:
                            if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
                                self.defs[module+'.'+node.name+'.'+member.name].append((rel,member,module,node.name+'.'+member.name))

    @staticmethod
    def is_test(rel):
        return any(x in ('tests','test') for x in Path(rel).parts) and Path(rel).name.startswith('test_')

    @staticmethod
    def is_test_context(rel):
        # Helpers throughout a test tree are test infrastructure, even if not named test_*.
        return any(x in ('tests','test') for x in Path(rel).parts) or Path(rel).name.startswith('test_')

    @staticmethod
    def diverse_prefix(scenes, limit=64):
        families=defaultdict(list)
        for scene in scenes: families[(scene['path'],scene['function'])].append(scene)
        interleaved=[scene for row in itertools.zip_longest(*families.values()) for scene in row if scene is not None]
        return interleaved[:limit]

    def imports(self, body, module='', package=False):
        aliases={}
        for node in body:
            if isinstance(node,ast.Import):
                for item in node.names: aliases[item.asname or item.name.split('.')[0]]=item.name if item.asname else item.name.split('.')[0]
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    prefix=module.split('.') if package else module.split('.')[:-1]
                    prefix=prefix[:len(prefix)-node.level+1]
                    base='.'.join(prefix+([base] if base else []))
                for item in node.names:
                    if item.name=='*':
                        other=self.modules.get(base)
                        if other:
                            for stmt in self.files[other][1].body:
                                if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__all__' for t in stmt.targets):
                                    try:
                                        for name in constant(stmt.value,{}):
                                            if isinstance(name,str): aliases[name]=base+'.'+name
                                    except (Unknown,TypeError): pass
                    else: aliases[item.asname or item.name]=base+'.'+item.name
        return aliases

    def resolve(self, name, depth=0):
        if depth>12: return None
        if name in self.defs:
            values=self.defs[name]
            return values[0] if len(values)==1 else None
        bits=name.split('.')
        for i in range(len(bits)-1,0,-1):
            module='.'.join(bits[:i]); attr=bits[i]
            target=self.exports.get(module,{}).get(attr)
            if target:
                translated=target+('.'+'.'.join(bits[i+1:]) if i+1<len(bits) else '')
                if translated!=name: return self.resolve(translated,depth+1)
        return None

    def ref(self, rel, node, symbol=''):
        key=self.name+':'+rel
        text,_,sha=self.files[rel]
        self.sources[key]={'repository':self.name,'revision':self.revision,'path':rel,'sha256':sha,'content_sha256':sha}
        return {'key':key,'repository':self.name,'revision':self.revision,'path':rel,'symbol':symbol,
                'start':node.lineno,'end':node.end_lineno,'sha256':sha}

    def code(self,rel,node):
        text=ast.get_source_segment(self.files[rel][0],node) or ast.unparse(node)
        if len(text)>8192: raise Unknown('单项源码表达式超过8192字符显示上限；请直接阅读测试源码')
        return text

    def canonical(self,node,aliases,blocked):
        name=dotted(node)
        if not name: return None
        first,*rest=name.split('.')
        if first in blocked or first not in aliases: return None
        target=aliases[first]+('.'+'.'.join(rest) if rest else '')
        result=self.resolve(target)
        return result

    def implementations(self, resolved, observers):
        rel,node,module,symbol=resolved
        todo=[]; unknown=[]
        if isinstance(node,ast.ClassDef):
            wanted=['__init__']+[x[1:] for x in observers if x.startswith('.')]
            wanted += [x for x in observers if x in ('__str__','__repr__')]
            for name in dict.fromkeys(wanted):
                member=self.resolve(module+'.'+symbol+'.'+name)
                if member: todo.append((member,0))
                else: unknown.append('未找到本类直接声明的 '+name+'；继承或动态协议未确认')
        else: todo=[(resolved,0)]
        entries=[]; seen=set(); limit=16
        while todo and len(entries)<limit:
            item,depth=todo.pop(0); r,n,m,s=item
            ident=(r,s)
            if ident in seen: continue
            seen.add(ident)
            entries.append({'label':s,'status':'candidate','source':self.ref(r,n,s),
                            'reason':'按导入、定义及显式静态引用定位的候选；未确认运行时绑定或分支，非执行轨迹'})
            used={x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}
            for stmt in self.files[r][1].body:
                targets=stmt.targets if isinstance(stmt,ast.Assign) else [stmt.target] if isinstance(stmt,ast.AnnAssign) else []
                names={x.id for t in targets for x in ast.walk(t) if isinstance(x,ast.Name)}
                if names & used and (r,stmt.lineno) not in seen:
                    seen.add((r,stmt.lineno))
                    entries.append({'label':'模块级定义 '+', '.join(sorted(names & used)), 'status':'candidate',
                                    'source':self.ref(r,stmt,', '.join(sorted(names))),
                                    'reason':'候选实现直接引用的模块级定义；可能存在后续修改，请结合完整文件阅读。'})
            if depth>=2: continue
            aliases=self.exports.get(m,{})
            class_prefix=s.rsplit('.',1)[0] if '.' in s else None
            # Only unqualified module functions/imports and explicit same-class self references.
            for call in [x for x in ast.walk(n) if isinstance(x,ast.Call)]:
                name=dotted(call.func); candidate=None
                if name and '.' not in name:
                    candidate=self.resolve(aliases.get(name,m+'.'+name))
                elif name.startswith('self.') and class_prefix and name.count('.')==1:
                    candidate=self.resolve(m+'.'+class_prefix+'.'+name[5:])
                elif name:
                    first,*rest=name.split('.')
                    if first in aliases: candidate=self.resolve(aliases[first]+'.'+'.'.join(rest))
                if candidate and not self.is_test_context(candidate[0]) and not isinstance(candidate[1],ast.ClassDef): todo.append((candidate,depth+1))
        if todo: unknown.append('候选引用阅读范围最多两层、16 个定义；其余需继续阅读')
        return entries,unknown

    def ambient_context(self,rel):
        """Report automatic test setup instead of inferring fixture-free execution."""
        context=[]
        test_parent=Path(rel).parent
        for path,(_,tree,_) in self.files.items():
            p=Path(path)
            conftest=p.name=='conftest.py' and (p.parent==test_parent or p.parent in test_parent.parents)
            if not conftest and path!=rel: continue
            aliases=self.imports(tree.body)
            for fn in tree.body:
                if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
                auto=False
                for deco in fn.decorator_list:
                    if isinstance(deco,ast.Call):
                        name=dotted(deco.func); first,*rest=name.split('.')
                        canonical=aliases.get(first,first)+('.'+'.'.join(rest) if rest else '')
                        if canonical in ('pytest.fixture','pytest_asyncio.fixture'):
                            for kw in deco.keywords:
                                if kw.arg=='autouse':
                                    try: auto=constant(kw.value,{}) is not False
                                    except Unknown: auto=True
                if auto or (path==rel and fn.name in ('setup_function','setup_module','teardown_function','teardown_module')):
                    context.append({'text':'自动测试设置 '+fn.name+'：作用未解析，本轮未执行。', 'source':self.ref(path,fn,fn.name)})
        return context

    def parameters(self,fn,rel,aliases):
        rows=[({},None)]
        for deco in fn.decorator_list:
            if not isinstance(deco,ast.Call) or not dotted(deco.func).endswith('.mark.parametrize'):
                # Marks without call targets are metadata, arbitrary decorators can change execution.
                if dotted(deco).startswith('pytest.mark.'): continue
                raise Unknown('非简单常量参数化装饰器')
            name=dotted(deco.func).split('.')[0]
            if aliases.get(name)!='pytest' or len(deco.args)<2: raise Unknown('参数化导入或参数不明确')
            # Indirect values feed request.param, not the tested call. Do not
            # project them as concrete inputs without resolving the fixture.
            if any(kw.arg is None for kw in deco.keywords):
                raise Unknown('参数化 ** 选项未解析，无法确认 indirect 或直接输入')
            indirect_nodes=deco.args[2:3]+[kw.value for kw in deco.keywords if kw.arg=='indirect']
            if len(indirect_nodes)>1: raise Unknown('参数化 indirect 重复指定，不能确认输入')
            if indirect_nodes:
                try: indirect=constant(indirect_nodes[0],{})
                except (Unknown,TypeError,ValueError):
                    raise Unknown('参数化 indirect 无法静态确认，fixture 输入未解析') from None
                if not (indirect is False or isinstance(indirect,(list,tuple)) and not indirect):
                    raise Unknown('参数化 indirect 依赖未解析的 fixture，不是被测调用的直接输入')
            names=constant(deco.args[0],{})
            if isinstance(names,str): names=[x.strip() for x in names.split(',')]
            values=constant(deco.args[1],{})
            if not isinstance(names,(list,tuple)) or not isinstance(values,(list,tuple)): raise Unknown('参数化不是常量序列')
            if len(rows)*len(values)>128: raise Unknown('常量参数化超过128行上限')
            new=[]
            for old,old_index in rows:
                for i,value in enumerate(values):
                    vals=[value] if len(names)==1 else value
                    if len(vals)!=len(names): raise Unknown('参数化行宽不一致')
                    rownode=deco.args[1].elts[i]
                    env=dict(old)
                    env.update({k:(v,rownode) for k,v in zip(names,vals)})
                    new.append((env,i if old_index is None else str(old_index)+'.'+str(i)))
            rows=new
        args=[a.arg for a in fn.args.posonlyargs+fn.args.args+fn.args.kwonlyargs]
        if fn.args.vararg or fn.args.kwarg: raise Unknown('可变测试参数未解析')
        if any(set(args)-set(env) for env,_ in rows): raise Unknown('测试参数依赖未解析的 fixture 或外部状态')
        return rows

    def call_inputs(self,call,env,rel,fn,resolved):
        result=[]; values={}
        target=resolved[1]
        if isinstance(target,ast.ClassDef):
            target=next((x for x in target.body if isinstance(x,ast.FunctionDef) and x.name=='__init__'),target)
        sig=[x.arg for x in target.args.posonlyargs+target.args.args] if isinstance(target,(ast.FunctionDef,ast.AsyncFunctionDef)) else []
        if sig and sig[0] in ('self','cls'): sig=sig[1:]
        def add(name,node,value):
            if len(repr(value))>8192: raise Unknown('静态输入值超过8192字符显示上限；请直接阅读测试源码')
            origin=env[node.id][1] if isinstance(node,ast.Name) and node.id in env else node
            values[name]=repr(value)
            result.append({'name':name,'text':repr(value),'source':self.ref(rel,origin,fn.name)})
        for i,arg in enumerate(call.args):
            if isinstance(arg,ast.Starred): raise Unknown('位置参数动态展开')
            add(sig[i] if i<len(sig) else '参数'+str(i+1),arg,constant(arg,env))
        for kw in call.keywords:
            value=constant(kw.value,env)
            if kw.arg is None:
                if not isinstance(value,dict) or any(not isinstance(k,str) for k in value): raise Unknown('关键字展开不是常量字符串字典')
                for key,val in value.items(): add(key,kw.value,val)
            else: add(kw.arg,kw.value,value)
        return result,values

    def extract_test(self,rel,fn,aliases):
        self.tests+=1
        if self.tests > 4000 or len(self.scenarios) >= 2048:
            self.exclusions["测试或场景数量达到处理上限"]+=1; return
        try:
            rows=self.parameters(fn,rel,aliases)
            ambient=self.ambient_context(rel)
            allowed=(ast.Assign,ast.AnnAssign,ast.Assert,ast.Expr,ast.Pass)
            for stmt in fn.body:
                if not isinstance(stmt,allowed): raise Unknown('包含控制流、上下文管理、嵌套定义或动态测试设置')
                if isinstance(stmt,ast.Expr) and not isinstance(stmt.value,ast.Constant): raise Unknown('存在可能影响状态的调用语句')
            assigned={x.id for x in ast.walk(fn) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store)}
            blocked=assigned | {a.arg for a in fn.args.args}
            assertions=[n for n in ast.walk(fn) if isinstance(n,ast.Assert)]
            ordinals={id(n):i for i,n in enumerate(assertions)}
            for initial,row_index in rows:
                env=dict(initial); objects={}; pending=[]; obj_index=-1
                for stmt in fn.body:
                    self.statement_work+=1
                    if self.statement_work>16000:
                        self.budget_exhausted=True
                        raise Unknown('达到全局静态语句处理上限；该函数未完整提取')
                    if isinstance(stmt,(ast.Assign,ast.AnnAssign)):
                        targets=stmt.targets if isinstance(stmt,ast.Assign) else [stmt.target]
                        if len(targets)!=1 or not isinstance(targets[0],ast.Name): raise Unknown('复杂赋值或对象修改未解析')
                        name=targets[0].id; value=stmt.value
                        if value is None: raise Unknown('未赋值注解')
                        try:
                            env[name]=(constant(value,env),value); objects.pop(name,None)
                        except Unknown:
                            if not isinstance(value,ast.Call): raise Unknown('局部赋值不是常量或明确构造器')
                            target=self.canonical(value.func,aliases,blocked)
                            if not target or not isinstance(target[1],ast.ClassDef): raise Unknown('构造调用无法可靠静态定位')
                            inputs,fields=self.call_inputs(value,env,rel,fn,target)
                            obj_index+=1
                            scene=self.new_scene(rel,fn,row_index,None,obj_index,value,target,inputs,fields)
                            pending.append(scene); objects[name]=scene; env.pop(name,None)
                    elif isinstance(stmt,ast.Assert):
                        test=stmt.test
                        if not isinstance(test,ast.Compare) or len(test.ops)!=1 or len(test.comparators)!=1 or not isinstance(test.ops[0],(ast.Eq,ast.NotEq,ast.Is,ast.IsNot,ast.Lt,ast.LtE,ast.Gt,ast.GtE)):
                            raise Unknown('断言不是支持的单层直接比较')
                        expected=constant(test.comparators[0],env)
                        if len(repr(expected))>8192: raise Unknown('测试预期超过8192字符显示上限；请直接阅读测试源码')
                        left=test.left; scene=None; observer='返回值'
                        if isinstance(left,ast.Call):
                            target=self.canonical(left.func,aliases,blocked)
                            if target:
                                inputs,fields=self.call_inputs(left,env,rel,fn,target)
                                scene=self.new_scene(rel,fn,row_index,ordinals[id(stmt)],None,left,target,inputs,fields)
                                pending.append(scene)
                            elif isinstance(left.func,ast.Name) and left.func.id in ('str','repr') and left.func.id not in aliases and left.func.id not in blocked and len(left.args)==1 and not left.keywords and isinstance(left.args[0],ast.Name):
                                scene=objects.get(left.args[0].id); observer='__'+left.func.id+'__'
                        elif isinstance(left,ast.Attribute) and isinstance(left.value,ast.Name):
                            scene=objects.get(left.value.id); observer='.'+left.attr
                        if scene is None: raise Unknown('断言被测表达式依赖动态调用、辅助函数或未知对象')
                        if len(scene['assertions'])>=64:
                            self.budget_exhausted=True
                            raise Unknown('单场景断言达到64项上限；该函数未完整提取')
                        op={ast.Eq:'==',ast.NotEq:'!=',ast.Is:'is',ast.IsNot:'is not',ast.Lt:'<',ast.LtE:'<=',ast.Gt:'>',ast.GtE:'>='}[type(test.ops[0])]
                        rhs=test.comparators[0]
                        expected_origin=env[rhs.id][1] if isinstance(rhs,ast.Name) and rhs.id in env else stmt
                        scene['expectations'].append({'text':observer+' '+op+' '+repr(expected),'source':self.ref(rel,expected_origin,fn.name)})
                        scene['assertions'].append({'text':self.code(rel,stmt),'source':self.ref(rel,stmt,fn.name)})
                        scene['_observers'].append(observer)
                        scene['assert_ordinals'].append(ordinals[id(stmt)])
                for scene in pending:
                    if scene['assertions']:
                        impl,unknown=self.implementations(scene.pop('_resolved'),scene['_observers'])
                        scene['implementation']=impl; scene['unknowns']+=unknown
                        scene['conditions']+=ambient
                        if ambient: scene['unknowns'].append('存在自动 fixture 或测试设置，其外部状态影响未解析；只能部分对照。')
                        scene['schema']=tuple(scene.pop('_observers'))
                        scene['status']='partial' if unknown or ambient or not impl else 'supported'
                        self.scenarios.append(scene)
            self.discovery.append({'path':rel,'function':fn.name,'parameter_rows':len(rows),'result':'提取候选'})
        except (Unknown,TypeError,ValueError,KeyError) as error:
            # Atomic function transaction: do not expose partially parsed fixture/state context.
            self.scenarios=[s for s in self.scenarios if not (s['path']==rel and s['function']==fn.name)]
            reason=str(error)
            self.exclusions[reason]+=1
            self.unsupported.append({'title':fn.name,'path':rel,'function':fn.name,'reason':reason,'source':self.ref(rel,fn,fn.name)})
            self.discovery.append({'path':rel,'function':fn.name,'result':'不支持','reason':reason})

    def new_scene(self,rel,fn,row,assertion,obj,call,target,inputs,fields):
        self.scene_work+=1
        if self.scene_work>2048:
            self.budget_exhausted=True
            raise Unknown('达到全局2048场景处理上限；该函数未完整提取')
        identity=f'{rel}:{fn.name}:{row}:{assertion}:{obj}'
        return {'id':digest(identity),'title':fn.name+(' / 参数 '+str(row+1) if isinstance(row,int) else ''),
                'path':rel,'function':fn.name,'parameter_row':row,'assert_ordinal':assertion,'object_assignment_ordinal':obj,
                'assert_ordinals':[], 'target':target[2]+'.'+target[3],
                'conditions':[{'text':'测试中显式调用：'+self.code(rel,call),'source':self.ref(rel,call,fn.name)}],
                'inputs':inputs,'fields':fields,'expectations':[],'assertions':[],
                'implementation':[],'unknowns':['以下为静态源码整理；未执行测试，未确认运行时绑定、分支或外部副作用。'],
                '_resolved':target,'_observers':[]}

    def run(self):
        for rel,(_,tree,_) in self.files.items():
            if not self.is_test(rel): continue
            aliases=self.imports(tree.body)
            # A module-level reassignment/declaration can shadow imports. Never guess.
            for stmt in tree.body:
                if isinstance(stmt,(ast.Assign,ast.AnnAssign)):
                    targets=stmt.targets if isinstance(stmt,ast.Assign) else [stmt.target]
                    for t in targets:
                        if isinstance(t,ast.Name): aliases.pop(t.id,None)
                elif isinstance(stmt,(ast.FunctionDef,ast.ClassDef)): aliases.pop(stmt.name,None)
            for fn in tree.body:
                if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)) and fn.name.startswith('test_'): self.extract_test(rel,fn,aliases)
                elif isinstance(fn,ast.ClassDef) and fn.name.startswith('Test'):
                    for method in fn.body:
                        if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)) and method.name.startswith('test_'):
                            self.tests+=1; self.exclusions['测试类实例上下文未支持']+=1
                            self.unsupported.append({'title':fn.name+'.'+method.name,'path':rel,'function':method.name,'reason':'测试类实例上下文未支持','source':self.ref(rel,method,fn.name+'.'+method.name)})
        groups=defaultdict(list)
        for scene in self.scenarios: groups[(scene['target'],scene['schema'])].append(scene)
        comparisons=[]; truncated=0; discarded=[]
        for (target,schema),scenes in groups.items():
            if len(scenes)>64: truncated+=len(scenes)-64
            chosen=self.diverse_prefix(scenes)
            chosen_ids={s['id'] for s in chosen}
            discarded.extend({'id':s['id'],'path':s['path'],'function':s['function'],'parameter_row':s['parameter_row'],'target':target} for s in scenes if s['id'] not in chosen_ids)
            for a,b in itertools.combinations(chosen,2):
                if len(comparisons)>=3000: break
                keys=sorted(a['fields'].keys() | b['fields'].keys())
                diff=[{'field':k,'a':a['fields'].get(k,'未显式传入（请阅读默认值）'),'b':b['fields'].get(k,'未显式传入（请阅读默认值）')} for k in keys if a['fields'].get(k)!=b['fields'].get(k)]
                if not diff or len(diff)>2: continue
                id_=digest(a['id']+':'+b['id'])
                reading=[{'label':'先看场景'+label+'的测试条件与断言','source':s['assertions'][0]['source']} for label,s in [('一',a),('二',b)]]
                seen=set()
                for entry in a['implementation']+b['implementation']:
                    ref=entry['source']; key=(ref['key'],ref['start'])
                    if key not in seen:
                        seen.add(key); reading.append({'label':'再读候选实现 '+entry['label'],'source':ref})
                comparisons.append({'id':id_,'title':target.split('.')[-1]+'：'+ '、'.join(x['field'] for x in diff)+' 的差异',
                                     'target':target,'status':'supported' if a['status']==b['status']=='supported' else 'partial',
                                     'differences':diff,'scenarios':[a,b],'reading':reading,
                                     'reasons':['同一静态引用候选、相同断言观察项；显式输入有一至两项变化。自动设置与未解析影响另列，未确认运行时绑定。','阅读顺序用于查证源码，不代表运行顺序。']})
        return {'name':self.name,'revision':self.revision,
                'inventory':{'python_files':len(self.files),'test_files':sum(self.is_test(r) for r in self.files),
                             'tests':self.tests,'scenarios':len(self.scenarios),'pairs':len(comparisons),'excluded':sum(self.exclusions.values()),
                             'exclusion_reasons':dict(self.exclusions),'truncated_scenarios':truncated,
                             'pair_cap_reached':len(comparisons)>=3000,'scene_cap_reached':len(self.scenarios)>=2048,
                             'work_budget_reached':self.budget_exhausted,
                             'pairing_limit':{'per_group':64,'strategy':'round-robin across test path/function families','discarded_count':len(discarded),'discarded':discarded[:100]}},
                'comparisons':comparisons,'unsupported':self.unsupported},self.sources
