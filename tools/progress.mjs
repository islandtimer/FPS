// Regenerates progress/index.html from progress/data/*.json + progress/shots/**.
// Phone-first: this is the page the user watches without interrupting the build.
// Run after every bench: node tools/progress.mjs

import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '..');
const DATA = join(ROOT, 'progress', 'data');

async function loadRounds() {
  if (!existsSync(DATA)) return [];
  const files = (await readdir(DATA)).filter((f) => /^round-\d+\.json$/.test(f)).sort();
  const out = [];
  for (const f of files) out.push(JSON.parse(await readFile(join(DATA, f), 'utf8')));
  return out.sort((a, b) => a.round - b.round);
}

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function head() {
  return `<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>OPERATION BLACKSAND — build log</title>
<style>
:root{
  --bg:#0a0c0f; --panel:#12161c; --panel2:#171d25; --line:#232b36;
  --ink:#e6ecf3; --dim:#8b97a6; --acc:#f0a030; --ok:#4ec97f; --bad:#e8574a; --warn:#e8b04a;
  color-scheme:dark;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-monospace,"SF Mono","Roboto Mono",Menlo,Consolas,monospace;
  -webkit-font-smoothing:antialiased;padding-bottom:64px}
a{color:var(--acc)}
.wrap{max-width:980px;margin:0 auto;padding:0 14px}
header{border-bottom:1px solid var(--line);background:
  linear-gradient(180deg,#141a22,#0a0c0f);padding:22px 0 18px;margin-bottom:18px}
h1{font-size:19px;letter-spacing:2.5px;font-weight:700}
h1 span{color:var(--acc)}
.sub{color:var(--dim);font-size:12px;letter-spacing:1px;margin-top:5px}
.grid{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));margin:14px 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px 13px}
.stat .k{color:var(--dim);font-size:10.5px;letter-spacing:1.2px;text-transform:uppercase}
.stat .v{font-size:25px;font-weight:700;margin-top:3px;line-height:1.1}
.stat .n{color:var(--dim);font-size:10.5px;margin-top:3px}
.ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)} .dim{color:var(--dim)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:16px;margin:14px 0}
.card h2{font-size:12px;letter-spacing:2px;color:var(--acc);text-transform:uppercase;margin-bottom:11px}
.note{background:#1b1408;border:1px solid #4a3413;border-left:3px solid var(--warn);
  border-radius:8px;padding:13px 15px;margin:14px 0;font-size:13px;color:#e9d9b8}
.note b{color:var(--warn)}
table{width:100%;border-collapse:collapse;font-size:12.5px}
td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--dim);font-weight:500;font-size:10.5px;letter-spacing:1px;text-transform:uppercase}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.shots{display:grid;gap:9px;grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.shot{border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#000}
.shot img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}
.shot .cap{padding:7px 9px;font-size:11px;color:var(--dim);background:var(--panel2)}
.shot .cap b{color:var(--ink);letter-spacing:1px}
.round{border:1px solid var(--line);border-radius:11px;margin:16px 0;overflow:hidden}
.round>summary{cursor:pointer;padding:13px 15px;background:var(--panel2);list-style:none;
  display:flex;align-items:center;gap:11px;flex-wrap:wrap}
.round>summary::-webkit-details-marker{display:none}
.round .body{padding:15px}
.pill{font-size:10px;letter-spacing:1.2px;padding:3px 8px;border-radius:999px;
  border:1px solid currentColor;text-transform:uppercase}
.bignum{font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}
.chg{font-size:13px;color:#c6d2df;margin:3px 0 3px 16px;text-indent:-16px}
.chg:before{content:"▸ ";color:var(--acc)}
.bar{height:5px;background:#1d232c;border-radius:3px;overflow:hidden;margin-top:5px}
.bar i{display:block;height:100%}
footer{color:var(--dim);font-size:11px;text-align:center;padding:26px 14px;line-height:1.8}
dialog{border:none;background:#000;padding:0;max-width:100vw;max-height:100vh}
dialog img{max-width:100vw;max-height:88vh;display:block}
dialog form{padding:9px;text-align:center;background:#12161c}
dialog button{background:#232b36;color:var(--ink);border:0;padding:8px 20px;border-radius:6px;font:inherit}
</style></head><body>`;
}

function statCard(k, v, n, cls = '') {
  return `<div class="stat"><div class="k">${esc(k)}</div><div class="v ${cls}">${v}</div><div class="n">${esc(n)}</div></div>`;
}

function render(rounds) {
  const latest = rounds.at(-1);
  const p = latest?.perf ?? {};
  const cpu = p.cpu ?? {};
  const est = cpu.estimate ?? {};
  const budgets = cpu.budgets ?? { rows: [], pass: false };

  const estFps = (hw) => est[hw]?.estFps ?? 0;
  const fpsCls = (f) => (f >= 60 ? 'ok' : f >= 45 ? 'warn' : 'bad');

  let h = head();
  h += `<header><div class="wrap">
    <h1>OPERATION <span>BLACKSAND</span></h1>
    <div class="sub">PROCEDURAL FPS · THREE.JS · BUILD LOG${latest ? ` · ROUND ${latest.round} · ${esc(latest.ts.slice(0, 16).replace('T', ' '))}Z` : ''}</div>
  </div></header><div class="wrap">`;

  if (!latest) {
    h += `<div class="card"><h2>no rounds yet</h2><p class="dim">Run <code>node tools/bench.mjs</code>.</p></div>`;
    return h + `</div></body></html>`;
  }

  // --- headline numbers
  h += `<div class="grid">
    ${statCard('M1 est. fps', estFps('m1'), '1080p · cost model', fpsCls(estFps('m1')))}
    ${statCard('GTX 1660 est. fps', estFps('gtx1660'), '1080p · cost model', fpsCls(estFps('gtx1660')))}
    ${statCard('logic p95', (cpu.logicP95 ?? 0).toFixed(2) + 'ms', 'measured · limit 4.00ms', (cpu.logicP95 ?? 9) <= 4 ? 'ok' : 'bad')}
    ${statCard('draw calls', cpu.stats?.calls ?? '—', 'limit 260', (cpu.stats?.calls ?? 999) <= 260 ? 'ok' : 'bad')}
    ${statCard('triangles', ((cpu.stats?.triangles ?? 0) / 1000).toFixed(0) + 'k', 'limit 950k', (cpu.stats?.triangles ?? 9e9) <= 950000 ? 'ok' : 'bad')}
    ${statCard('perf gate', budgets.pass && (cpu.logicP95 ?? 9) <= 4 ? 'PASS' : 'FAIL', 'all budgets', budgets.pass && (cpu.logicP95 ?? 9) <= 4 ? 'ok' : 'bad')}
  </div>`;

  h += `<div class="note"><b>How the fps numbers are produced.</b> The build container has
    <b>no GPU</b> — Chromium falls back to SwiftShader, a CPU rasteriser — so raw frame rate measured
    here would be meaningless for an M1 or a GTX 1660. Instead every round measures what actually
    transfers: JS logic time per frame, draw calls, triangles, fullscreen passes and shadow texels,
    all against fixed budgets, converted to an <b>estimated</b> ms/frame using published throughput
    figures for both reference GPUs. SwiftShader's own 1080p rate is recorded purely as a
    round-over-round regression signal${p.swiftshader1080p ? ` (<b>${p.swiftshader1080p.fps} fps</b> software)` : ''}.
    The game ships a live fps counter and an in-game benchmark, so the real number on real hardware is one keypress away.</div>`;

  // --- budget table
  h += `<div class="card"><h2>frame budget · 1920×1080</h2><table>
    <tr><th>metric</th><th class="num">value</th><th class="num">limit</th><th class="num">headroom</th></tr>`;
  for (const r of budgets.rows) {
    const pct = Math.min(100, (r.v / r.limit) * 100);
    h += `<tr><td>${esc(r.k)}</td><td class="num ${r.ok ? 'ok' : 'bad'}">${r.v}</td>
      <td class="num dim">${r.limit}</td><td class="num">
      <div class="bar"><i style="width:${pct}%;background:${r.ok ? '#4ec97f' : '#e8574a'}"></i></div></td></tr>`;
  }
  h += `</table></div>`;

  // --- estimate table
  h += `<div class="card"><h2>cost model · estimated ms per frame</h2><table>
    <tr><th>gpu</th><th class="num">fragment</th><th class="num">shadow</th><th class="num">vertex</th>
    <th class="num">draw cpu</th><th class="num">frame</th><th class="num">fps</th></tr>`;
  for (const [, e] of Object.entries(est)) {
    h += `<tr><td>${esc(e.name)}</td><td class="num">${e.fragMs}</td><td class="num">${e.shadowMs}</td>
      <td class="num">${e.triMs}</td><td class="num">${e.drawCpuMs}</td>
      <td class="num">${e.estMs}</td><td class="num ${fpsCls(e.estFps)}">${e.estFps}</td></tr>`;
  }
  h += `</table></div>`;

  // --- rounds newest first
  for (const r of [...rounds].reverse()) {
    const rp = r.perf?.cpu ?? {};
    const pass = r.verdict?.pass;
    h += `<details class="round" ${r === latest ? 'open' : ''}><summary>
      <span class="pill" style="color:${pass ? '#4ec97f' : '#e8574a'}">${pass ? 'perf pass' : 'perf fail'}</span>
      <b>ROUND ${r.round}</b>
      <span class="dim">${esc(r.label || '')}</span>
      <span class="bignum">· ${(rp.logicP95 ?? 0).toFixed(2)}ms logic · ${rp.stats?.calls ?? '—'} calls
      · est ${r.perf?.cpu?.estimate?.m1?.estFps ?? '—'}fps M1</span>
    </summary><div class="body">`;

    if (r.changes?.length) {
      h += `<div style="margin-bottom:13px">${r.changes.map((c) => `<div class="chg">${esc(c)}</div>`).join('')}</div>`;
    }
    if (r.critique?.length) {
      h += `<table><tr><th>piece</th><th class="num">score</th><th>biggest remaining gap</th></tr>`;
      for (const c of r.critique) {
        h += `<tr><td>${esc(c.piece)}</td><td class="num ${c.score >= 8 ? 'ok' : c.score >= 6 ? 'warn' : 'bad'}">${esc(c.score)}/10</td><td>${esc(c.gap)}</td></tr>`;
      }
      h += `</table>`;
    }
    const shots = Object.entries(r.shots || {});
    if (shots.length) {
      h += `<div class="shots" style="margin-top:13px">`;
      for (const [name, path] of shots) {
        h += `<div class="shot"><img loading="lazy" src="${esc(path)}" alt="${esc(name)}" data-full="${esc(path)}">
          <div class="cap"><b>${esc(name.toUpperCase())}</b></div></div>`;
      }
      h += `</div>`;
    }
    if (r.errors?.length) {
      h += `<div class="note" style="margin-top:12px"><b>${r.errors.length} console errors</b><br>${r.errors.slice(0, 6).map(esc).join('<br>')}</div>`;
    }
    h += `</div></details>`;
  }

  h += `</div>
<dialog id="lb"><img><form method="dialog"><button>close</button></form></dialog>
<footer>All art, audio, meshes and textures in this project are generated procedurally in code.<br>
No third-party game assets, audio, model names or trademarks are used.<br>
Page regenerated by <code>tools/progress.mjs</code>.</footer>
<script>
const lb=document.getElementById('lb');
document.addEventListener('click',e=>{const i=e.target.closest('.shot img');
  if(i){lb.querySelector('img').src=i.dataset.full;lb.showModal();}});
</script></body></html>`;
  return h;
}

const rounds = await loadRounds();
await mkdir(join(ROOT, 'progress'), { recursive: true });
await writeFile(join(ROOT, 'progress', 'index.html'), render(rounds));
console.log(`progress page → progress/index.html (${rounds.length} rounds)`);
