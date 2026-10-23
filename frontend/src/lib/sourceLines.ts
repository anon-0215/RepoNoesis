// Physical source lines match Python AST: LF, CRLF and CR only.
// Other Unicode/control separators stay inside their original source line.
export function sourceLines(content: string): string[] {
  if (!content) return [];
  const lines = content.split(/\r\n|\r|\n/);
  if (/[\r\n]$/.test(content)) lines.pop();
  return lines;
}
