"""Physical source lines: LF, CRLF and CR, matching Python's parser.

Unlike str.splitlines(), Unicode/control separators inside source text are not
line terminators. Slices preserve the original bytes after UTF-8 encoding.
"""
import re

_BREAK = re.compile(r"\r\n|\r|\n")

def source_lines(content: str, keepends: bool = False) -> list[str]:
    lines = []
    start = 0
    for match in _BREAK.finditer(content):
        lines.append(content[start:match.end() if keepends else match.start()])
        start = match.end()
    if start < len(content):
        lines.append(content[start:])
    return lines
