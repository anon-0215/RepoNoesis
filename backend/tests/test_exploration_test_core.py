import json
import tempfile
import unittest
from pathlib import Path
from app.services.use_case_extractor import Repository as ProductRepository
from app.services.exploration_sources import record

def Repository(root,name,revision):
    files={p.relative_to(root).as_posix():record(p.relative_to(root).as_posix(),p.read_text(encoding='utf-8'),'fixture') for p in Path(root).rglob('*.py')}
    repo=ProductRepository(files,name,revision); repo.root=Path(root)
    return repo

class StaticExtractionTests(unittest.TestCase):
    def make(self,test,library='def render(value, flag=False):\n    return value\n', init='from .core import render\n'):
        tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        for name,text in {'package/__init__.py':init,'package/core.py':library,'tests/test_cases.py':test}.items():
            p=root/name; p.parent.mkdir(exist_ok=True); p.write_text(text,encoding='utf-8')
        repo=Repository(root,'Unseen package','revision-1')
        data,sources=repo.run()
        return repo,data,sources

    def test_constant_parameters_and_dictionary_expansion_preserve_origins(self):
        repo,data,sources=self.make('import pytest\nimport package as p\n@pytest.mark.parametrize(("opts", "expected"), [({}, "a"), ({"flag": True}, "b")])\ndef test_diff(opts, expected):\n    assert p.render("x", **opts) == expected\n')
        self.assertEqual(len(data['comparisons']),1)
        pair=data['comparisons'][0]
        self.assertEqual(pair['differences'],[{'field':'flag','a':'未显式传入（请阅读默认值）','b':'True'}])
        self.assertEqual(pair['scenarios'][1]['expectations'][0]['source']['start'],3)
        self.assertEqual(pair['scenarios'][0]['implementation'][0]['source']['path'],'package/core.py')
        self.assertTrue(all(s['status']=='candidate' for s in pair['scenarios'][0]['implementation']))
        for source in sources.values(): self.assertEqual(len(source['sha256']),64)

    def test_fixture_mutation_and_nested_calls_are_rejected(self):
        for test in [
            'import package\ndef test_x(monkeypatch):\n    monkeypatch.setattr(package,"render",lambda x: x)\n    assert package.render("x")=="x"\n',
            'import package\ndef test_x():\n    package.render = lambda x: x\n    assert package.render("x")=="x"\n',
            'import package\ndef helper(x): return x\ndef test_x():\n    assert helper("x")=="x"\n']:
            with self.subTest(test=test):
                repo,data,_=self.make(test)
                self.assertEqual(repo.scenarios,[])
                self.assertTrue(data['unsupported'])

    def test_no_execution_and_no_same_name_guess(self):
        repo,data,_=self.make('raise RuntimeError("MUST NOT RUN")\nimport absent\ndef test_case():\n    assert absent.render("__import__(\\"os\\").system(\\"bad\\")") == "x"\n')
        self.assertEqual(data['comparisons'],[])
        self.assertTrue(data['unsupported'])

    def test_duplicate_definition_and_import_shadowing_rejected(self):
        repo,data,_=self.make('import package\ndef test_x():\n    assert package.render("x")=="x"\n',library='def render(x): return x\ndef render(x): return x+"!"\n')
        self.assertEqual(repo.scenarios,[])
        repo,data,_=self.make('import package\npackage = None\ndef test_x():\n    assert package.render("x")=="x"\n')
        self.assertEqual(repo.scenarios,[])

    def test_constructor_segments_and_observer_methods(self):
        library='class Box:\n    def __init__(self, text): self.text=text\n    def __str__(self): return self.text\n'
        test='from package.core import Box\ndef test_boxes():\n    box = Box("a")\n    assert str(box) == "a"\n    box = Box("b")\n    assert str(box) == "b"\n'
        repo,data,_=self.make(test,library,'from .core import Box\n')
        self.assertEqual(len(data['comparisons']),1)
        scenes=data['comparisons'][0]['scenarios']
        self.assertEqual([s['object_assignment_ordinal'] for s in scenes],[0,1])
        self.assertEqual(scenes[0]['assertions'][0]['source']['start'],4)
        self.assertEqual([x['label'] for x in scenes[0]['implementation']],['Box.__init__','Box.__str__'])

    def test_atomic_rejection_removes_partial_parameter_rows(self):
        test='import pytest\nimport package\n@pytest.mark.parametrize("x", ["a", "b"])\ndef test_x(x):\n    assert package.render(x) == "a"\n    assert package.render(unknown) == "b"\n'
        repo,data,_=self.make(test)
        self.assertEqual(repo.scenarios,[])

    def test_automatic_fixture_is_reported_even_without_test_parameters(self):
        repo,data,_=self.make('import package\ndef test_a():\n    assert package.render("a")=="a"\ndef test_b():\n    assert package.render("b")=="b"\n')
        (repo.root/'tests/conftest.py').write_text('import pytest\n@pytest.fixture(autouse=True)\ndef clear_env():\n    raise RuntimeError("never execute")\n',encoding='utf-8')
        repo=Repository(repo.root,'Unseen package','rev')
        data,_=repo.run()
        self.assertEqual(data['comparisons'][0]['status'],'partial')
        self.assertIn('clear_env',data['comparisons'][0]['scenarios'][0]['conditions'][-1]['text'])

    def test_imported_test_helper_cannot_be_product_implementation(self):
        repo,data,_=self.make('from tests.common import render\ndef test_a():\n    assert render("a")=="a"\ndef test_b():\n    assert render("b")=="b"\n')
        (repo.root/'tests/common.py').write_text('def render(x):\n    return x\n',encoding='utf-8')
        repo=Repository(repo.root,'Unseen package','rev'); data,_=repo.run()
        self.assertFalse(data['comparisons'])
        self.assertFalse(repo.scenarios)

    def test_bounded_pairing_budget_keeps_small_test_families(self):
        rows=repr(list(range(66)))
        test='import pytest\nimport package\n@pytest.mark.parametrize("x", '+rows+')\ndef test_many(x):\n    assert package.render(x)=="expected"\ndef test_small_a():\n    assert package.render("A")=="A"\ndef test_small_b():\n    assert package.render("B")=="B"\n'
        repo,data,_=self.make(test)
        self.assertEqual(data['inventory']['truncated_scenarios'],4)
        self.assertTrue(any({s['function'] for s in p['scenarios']}=={'test_small_a','test_small_b'} for p in data['comparisons']))
        self.assertEqual(len(data['inventory']['pairing_limit']['discarded']),4)

if __name__=='__main__': unittest.main()
