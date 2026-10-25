"""Representative selection retained from the verified reading UI."""
import ast,hashlib,json
BOILERPLATE="以下为静态源码整理；未执行测试，未确认运行时绑定、分支或外部副作用。"
def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def stable(prefix,value): return prefix+'_'+hashlib.sha256(canonical(value).encode()).hexdigest()[:24]
def literal_descriptor(text, present=True):
    """Tag existing repr syntax without executing it or inventing missing defaults."""
    if not present: return {'state':'omitted','type':'unknown_default','text':'未传入；默认值未知'}
    kind='unclassified'
    try:
        node=ast.parse(text,mode='eval').body
        if isinstance(node,ast.Constant): kind=type(node.value).__name__
        elif isinstance(node,(ast.List,ast.Tuple,ast.Dict,ast.Set)): kind=type(node).__name__.lower()
        elif isinstance(node,ast.UnaryOp) and isinstance(node.operand,ast.Constant): kind=type(node.operand.value).__name__
    except (SyntaxError,ValueError,TypeError): pass
    return {'state':'explicit','type':kind,'text':text}

def expectation_texts(scene): return [x['text'] for x in scene['expectations']]
def presentation_signature(pair):
    # Similar presentation is a navigation grouping, never a source-identity merge.
    return canonical({'fields':[[[k,literal_descriptor(v)] for k,v in sorted(s['fields'].items())] for s in pair['scenarios']],
                      'expectations':[expectation_texts(s) for s in pair['scenarios']],
                      'conditions':[[x['text'] for x in s['conditions']] for s in pair['scenarios']],
                      'status':pair['status'],'unknowns':[s['unknowns'] for s in pair['scenarios']]})

def choose_representatives(pairs,limit=3):
    """Deterministic coverage of limits/equal/different expectations, then diversity."""
    selected=[]; used=set()
    def same(p): return expectation_texts(p['scenarios'][0])==expectation_texts(p['scenarios'][1])
    def unknown_count(p): return len({u for s in p['scenarios'] for u in s['unknowns'] if u!=BOILERPLATE})
    def complexity(p): return sum(len(x['a'])+len(x['b']) for x in p['differences'])+sum(len(t) for s in p['scenarios'] for t in expectation_texts(s))
    def base(p): return (len(p['differences']),complexity(p),p['id'])
    def add(candidates,rank=None):
        available=[p for p in candidates if presentation_signature(p) not in used]
        if available and len(selected)<limit:
            chosen=min(available,key=rank or base);selected.append(chosen);used.add(presentation_signature(chosen))
    # Reserve visible space for a limitation if one exists; never score support upward.
    add([p for p in pairs if p['status']!='supported' or unknown_count(p)],lambda p:(-unknown_count(p),*base(p)))
    add([p for p in pairs if same(p)])
    add([p for p in pairs if not same(p)])
    while len(selected)<limit:
        old=len(selected)
        seen_scenes={s['id'] for p in selected for s in p['scenarios']}
        seen_fields={x['field'] for p in selected for x in p['differences']}
        add(pairs,lambda p:(-len({s['id'] for s in p['scenarios']}-seen_scenes),
                            -len({x['field'] for x in p['differences']}-seen_fields),*base(p)))
        if len(selected)==old: break
    return selected
