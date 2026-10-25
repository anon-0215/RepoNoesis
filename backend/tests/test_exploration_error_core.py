"""Deterministic synthetic probes of safety/semantic boundaries; no target code execution."""
import unittest
from app.services.error_locator import extract_text, message_pattern
import ast

class ExtractionTests(unittest.TestCase):
    def items(self, source):
        return extract_text(source).items

    def test_adjacent_if_is_not_a_guard(self):
        source = 'def f(x):\n if x:\n  value = 2\n raise ValueError("bad input")\n'
        item = self.items(source)[0]
        self.assertEqual(item['controls'], [])
        self.assertTrue(item['unconditional_in_local_scope'])

    def test_nested_if_excludes_neighbor_assignment_branches(self):
        source = 'def f(x,y):\n if x:\n  if y:\n   v=1\n  else:\n   v=2\n  if x+y:\n   raise ValueError("bad input")\n'
        controls = self.items(source)[0]['controls']
        self.assertEqual([x['line'] for x in controls], [2,7])

    def test_if_elif_else_are_distinct(self):
        source = 'def f(a,b):\n if a:\n  raise ValueError("first bad")\n elif b:\n  raise ValueError("second bad")\n else:\n  raise ValueError("third bad")\n'
        items = self.items(source)
        self.assertEqual([x['kind'] for x in items[1]['controls']], ['else','elif'])
        self.assertEqual([x['kind'] for x in items[2]['controls']], ['else','else'])
        self.assertIn('elif b:', items[2]['controls'][-1]['source'])

    def test_except_not_try_guard(self):
        source = 'def f():\n try:\n  value=external()\n except KeyError:\n  raise ValueError("bad external value")\n'
        c = self.items(source)[0]['controls']
        self.assertEqual([(x['kind'],x['line']) for x in c], [('except',4)])

    def test_function_definition_ancestors_not_call_conditions(self):
        source='if flag:\n def f():\n  raise ValueError("nested call error")\n'
        item=self.items(source)[0]
        self.assertEqual(item['symbol'],'f')
        self.assertEqual(item['controls'],[])

    def test_dynamic_message_and_reraise_unsupported(self):
        ext=extract_text('def f(msg):\n raise ValueError(msg)\n raise\n')
        self.assertEqual(ext.items,[])
        self.assertEqual(ext.stats['unsupported'],2)

    def test_fstring_does_not_execute_expressions(self):
        source='def f():\n raise ValueError(f"An external failure: {never_execute()!r}")\n'
        item=self.items(source)[0]
        self.assertEqual(item['pattern']['mode'],'template')
        self.assertIn('never_execute()',item['message_source'])

    def test_broad_template_rejected(self):
        value=ast.parse('f"{value}"',mode='eval').body
        self.assertIsNone(message_pattern(value))

    def test_regex_metacharacters_literal(self):
        value=ast.parse('f"Invalid [value]. ({x})"',mode='eval').body
        import re
        p=message_pattern(value)
        self.assertIsNotNone(re.fullmatch(p['regex'],'Invalid [value]. (abc)'))
        self.assertIsNone(re.fullmatch(p['regex'],'Invalid vvvvvX (abc)'))

    def test_fstring_slot_bound(self):
        import re
        value=ast.parse('f"Invalid input: {x}"',mode='eval').body
        p=message_pattern(value)
        self.assertIsNone(re.fullmatch(p['regex'],'Invalid input: '+'a'*513))

    def test_syntactic_binding_unknown_is_retained(self):
        item=self.items('def f(OtherError):\n raise OtherError("same input text")\n')[0]
        self.assertEqual(item['exception_expression'],'OtherError')
        self.assertIn('运行绑定',item['boundary'])

    def test_loop_and_with_structure(self):
        source='def f():\n for x in values:\n  with context():\n   raise ValueError("bad context value")\n'
        self.assertEqual([c['kind'] for c in self.items(source)[0]['controls']], ['for','with'])
