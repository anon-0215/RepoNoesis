// Matches Python str.splitlines(), as used by the source citation chain.
export function sourceLines(content: string): string[] {
  if (!content) return [];
  const lines = content.split(/\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]/);
  if (/[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]$/.test(content)) lines.pop();
  return lines;
}
