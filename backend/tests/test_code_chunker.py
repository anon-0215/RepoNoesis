import hashlib
import unittest

from app.services.code_chunker import extract_python_code_chunks


def _by_name(result):
    return {chunk.qualified_name: chunk for chunk in result.chunks}


class CodeChunkerTests(unittest.TestCase):
    def test_extracts_top_level_function_and_async_function(self):
        source = (
            "def login():\n"
            "    return True\n"
            "\n"
            "async def fetch():\n"
            "    return 1\n"
        )
        chunks = _by_name(extract_python_code_chunks("app/main.py", source, "abc123"))

        self.assertEqual(chunks["login"].chunk_type, "function")
        self.assertEqual(chunks["login"].start_line, 1)
        self.assertEqual(chunks["login"].end_line, 2)
        self.assertEqual(chunks["login"].repository_revision, "abc123")
        self.assertEqual(chunks["fetch"].chunk_type, "async_function")
        self.assertEqual(chunks["fetch"].start_line, 4)
        self.assertEqual(chunks["fetch"].end_line, 5)

    def test_extracts_class_methods_and_async_methods(self):
        source = (
            "class UserService:\n"
            "    def login(self):\n"
            "        return True\n"
            "\n"
            "    async def logout(self):\n"
            "        return True\n"
        )
        chunks = _by_name(extract_python_code_chunks("app/service.py", source))

        self.assertEqual(chunks["UserService"].chunk_type, "class")
        self.assertEqual(chunks["UserService"].start_line, 1)
        self.assertEqual(chunks["UserService"].end_line, 6)
        self.assertEqual(chunks["UserService.login"].chunk_type, "method")
        self.assertEqual(chunks["UserService.login"].parent_symbol, "UserService")
        self.assertEqual(chunks["UserService.login"].content, "    def login(self):\n        return True\n")
        self.assertEqual(chunks["UserService.logout"].chunk_type, "async_method")

    def test_extracts_nested_functions_and_nested_classes(self):
        source = (
            "def outer():\n"
            "    def inner():\n"
            "        return 'x'\n"
            "    return inner\n"
            "\n"
            "class OuterClass:\n"
            "    class InnerClass:\n"
            "        def method(self):\n"
            "            return 1\n"
        )
        chunks = _by_name(extract_python_code_chunks("nested.py", source))

        self.assertEqual(chunks["outer.inner"].chunk_type, "function")
        self.assertEqual(chunks["outer.inner"].parent_symbol, "outer")
        self.assertEqual(chunks["OuterClass.InnerClass"].chunk_type, "class")
        self.assertEqual(chunks["OuterClass.InnerClass"].parent_symbol, "OuterClass")
        self.assertEqual(chunks["OuterClass.InnerClass.method"].chunk_type, "method")

    def test_decorators_multiline_signature_precise_content_and_end_line(self):
        source = (
            "@decorator\n"
            "@factory(\n"
            "    'x',\n"
            ")\n"
            "def wrapped(\n"
            "    first,\n"
            "    second,\n"
            "):\n"
            "    return first + second\n"
        )
        chunks = _by_name(extract_python_code_chunks("decorated.py", source))
        chunk = chunks["wrapped"]

        self.assertEqual(chunk.start_line, 1)
        self.assertEqual(chunk.end_line, 9)
        self.assertEqual(chunk.content, source)
        self.assertEqual(chunk.content_hash, hashlib.sha256(source.encode("utf-8")).hexdigest())

    def test_class_decorator_start_line(self):
        source = (
            "@register\n"
            "class Plugin:\n"
            "    pass\n"
        )
        chunk = _by_name(extract_python_code_chunks("plugin.py", source))["Plugin"]

        self.assertEqual(chunk.start_line, 1)
        self.assertEqual(chunk.end_line, 3)

    def test_hash_stability_and_change_detection(self):
        base = "# outside one\n\ndef target():\n    return 1\n"
        outside_changed = "# outside two\n\ndef target():\n    return 1\n"
        inside_changed = "# outside one\n\ndef target():\n    return 2\n"

        base_chunk = _by_name(extract_python_code_chunks("hash.py", base))["target"]
        outside_chunk = _by_name(extract_python_code_chunks("hash.py", outside_changed))["target"]
        inside_chunk = _by_name(extract_python_code_chunks("hash.py", inside_changed))["target"]

        self.assertEqual(base_chunk.content, "def target():\n    return 1\n")
        self.assertEqual(base_chunk.content_hash, outside_chunk.content_hash)
        self.assertNotEqual(base_chunk.content_hash, inside_chunk.content_hash)

    def test_empty_and_no_symbol_files(self):
        empty = extract_python_code_chunks("empty.py", "")
        no_symbols = extract_python_code_chunks("constants.py", "VALUE = 1\n# 中文注释\n")

        self.assertEqual(empty.chunks, [])
        self.assertEqual(empty.warnings, [])
        self.assertEqual(no_symbols.chunks, [])
        self.assertEqual(no_symbols.warnings, [])

    def test_syntax_error_returns_warning_without_raising(self):
        result = extract_python_code_chunks("broken.py", "def broken(:\n    pass\n")

        self.assertEqual(result.chunks, [])
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(result.warnings[0].path, "broken.py")
        self.assertIn("Python syntax error", result.warnings[0].message)

    def test_unicode_crlf_and_file_without_final_newline(self):
        source = "def greet():\r\n    text = '你好'\r\n    return text"
        result = extract_python_code_chunks("windows\\greet.py", source)
        chunk = _by_name(result)["greet"]

        self.assertEqual(chunk.path, "windows/greet.py")
        self.assertEqual(chunk.start_line, 1)
        self.assertEqual(chunk.end_line, 3)
        self.assertEqual(chunk.content, source)
        self.assertEqual(chunk.content_hash, hashlib.sha256(source.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    unittest.main()
