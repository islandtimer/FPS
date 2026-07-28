// Merges narrative + critique into a round's data file, then regenerates the
// progress page and the phone artifact.
//   node tools/annotate.mjs <round> <<< '{"label":"...","changes":[...],"critique":[...]}'

import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const ROOT = resolve(import.meta.dirname, '..');
const round = String(process.argv[2] ?? '').padStart(2, '0');
const file = join(ROOT, 'progress', 'data', `round-${round}.json`);

const patch = JSON.parse(await new Promise((res, rej) => {
  let s = ''; process.stdin.setEncoding('utf8');
  process.stdin.on('data', (d) => (s += d));
  process.stdin.on('end', () => res(s || '{}'));
  process.stdin.on('error', rej);
}));

const data = JSON.parse(await readFile(file, 'utf8'));
Object.assign(data, patch);
await writeFile(file, JSON.stringify(data, null, 2));

for (const t of ['progress.mjs', 'artifact.mjs']) {
  const r = spawnSync('node', [join(ROOT, 'tools', t)], { cwd: ROOT, encoding: 'utf8' });
  process.stdout.write(r.stdout || '');
  if (r.status !== 0) process.stderr.write(r.stderr || '');
}
console.log(`round ${round} annotated`);
