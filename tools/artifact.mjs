// Builds the phone-facing build log as a single self-contained page (images
// inlined, no external requests) for publishing as an Artifact.
//   node tools/artifact.mjs  ->  progress/artifact.html

import { readdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '..');
const DATA = join(ROOT, 'progress', 'data');
const PROG = join(ROOT, 'progress');

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function inline(rel) {
  const p = join(PROG, rel);
  if (!existsSync(p)) return null;
  const b = await readFile(p);
  const mime = rel.endsWith('.png') ? 'image/png' : 'image/jpeg';
  return `data:${mime};base64,${b.toString('base64')}`;
}

const rounds = existsSync(DATA)
  ? (await Promise.all(
      (await readdir(DATA)).filter((f) => /^round-\d+\.json$/.test(f))
        .map(async (f) => JSON.parse(await readFile(join(DATA, f), 'utf8')))
    )).sort((a, b) => a.round - b.round)
  : [];

const latest = rounds.at(-1);
const cpu = latest?.perf?.cpu ?? {};
const est = cpu.estimate ?? {};
const budgets = cpu.budgets ?? { rows: [], pass: false };
const pass = !!(budgets.pass && (cpu.logicP95 ?? 9) <= 4);

// Inline the latest round's contact sheet only — history stays as numbers so the
// page does not grow without bound as rounds accumulate.
const gallery = [];
for (const [name, rel] of Object.entries(latest?.shots ?? {})) {
  const src = await inline(rel);
  if (src) gallery.push({ name, src });
}

const sparkline = (vals, w = 92, h = 22) => {
  if (vals.length < 2) return '';
  const min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  const pts = vals.map((v, i) => [(i / (vals.length - 1)) * w, h - ((v - min) / span) * (h - 3) - 1.5]);
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
  const area = `${d} L${w} ${h} L0 ${h} Z`;
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" aria-hidden="true">
    <path class="sa" d="${area}"/><path class="sl" d="${d}"/>
    <circle class="sp" cx="${pts.at(-1)[0].toFixed(1)}" cy="${pts.at(-1)[1].toFixed(1)}" r="2"/></svg>`;
};

const fpsHist = rounds.map((r) => r.perf?.cpu?.estimate?.m1?.estFps ?? 0);
const logicHist = rounds.map((r) => r.perf?.cpu?.logicP95 ?? 0);
const callHist = rounds.map((r) => r.perf?.cpu?.stats?.calls ?? 0);

const sev = (ok) => (ok ? 'ok' : 'bad');
const fpsSev = (f) => (f >= 60 ? 'ok' : f >= 45 ? 'warn' : 'bad');

const instrument = (label, value, unit, note, severity, spark) => `
  <div class="inst ${severity}">
    <div class="inst-k">${esc(label)}</div>
    <div class="inst-v">${esc(value)}<em>${esc(unit || '')}</em></div>
    <div class="inst-f">${spark || ''}<span>${esc(note)}</span></div>
  </div>`;

const html = `<title>Operation Blacksand — build log</title>
<style>
  :root{
    --ground:#0B0D10; --panel:#14181E; --panel-2:#1A1F27; --rule:#252D38;
    --ink:#DFE6EE; --dim:#7C8899; --faint:#4C5867;
    --accent:#E8A33D; --ok:#5FCB8B; --warn:#E0B64A; --bad:#E2564B;
    --mono:ui-monospace,"SF Mono","JetBrains Mono","Roboto Mono",Menlo,Consolas,monospace;
    --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  }
  @media (prefers-color-scheme: light){
    :root{ --ground:#F2F0EC; --panel:#FFFFFF; --panel-2:#F7F5F1; --rule:#DFDAD2;
      --ink:#161A20; --dim:#6A7280; --faint:#A8AEB8; --accent:#B5761B; }
  }
  :root[data-theme="dark"]{ --ground:#0B0D10; --panel:#14181E; --panel-2:#1A1F27; --rule:#252D38;
    --ink:#DFE6EE; --dim:#7C8899; --faint:#4C5867; --accent:#E8A33D; }
  :root[data-theme="light"]{ --ground:#F2F0EC; --panel:#FFFFFF; --panel-2:#F7F5F1; --rule:#DFDAD2;
    --ink:#161A20; --dim:#6A7280; --faint:#A8AEB8; --accent:#B5761B; }

  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
    font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
  .wrap{max-width:900px;margin:0 auto;padding:0 16px 72px}

  .plate{border-bottom:1px solid var(--rule);padding:22px 0 16px;margin-bottom:20px;
    display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  .plate h1{font-family:var(--mono);font-size:15px;font-weight:700;letter-spacing:.32em;
    margin:0;text-transform:uppercase}
  .plate h1 b{color:var(--accent);font-weight:700}
  .plate .meta{font-family:var(--mono);font-size:11px;color:var(--dim);letter-spacing:.14em}
  .verdict{font-family:var(--mono);font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;
    padding:3px 10px;border:1px solid currentColor;border-radius:2px}
  .verdict.ok{color:var(--ok)} .verdict.bad{color:var(--bad)}

  section{margin:26px 0}
  .eyebrow{font-family:var(--mono);font-size:10px;letter-spacing:.26em;text-transform:uppercase;
    color:var(--dim);display:flex;align-items:center;gap:12px;margin-bottom:12px}
  .eyebrow::after{content:"";flex:1;height:1px;background:var(--rule)}

  .insts{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
    grid-template-columns:repeat(auto-fit,minmax(158px,1fr))}
  .inst{background:var(--panel);padding:13px 14px 11px;position:relative}
  .inst::before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--faint)}
  .inst.ok::before{background:var(--ok)} .inst.warn::before{background:var(--warn)}
  .inst.bad::before{background:var(--bad)}
  .inst-k{font-family:var(--mono);font-size:9.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--dim)}
  .inst-v{font-family:var(--mono);font-size:27px;font-weight:700;line-height:1.15;margin-top:2px;
    font-variant-numeric:tabular-nums}
  .inst-v em{font-style:normal;font-size:12px;font-weight:500;color:var(--dim);margin-left:3px}
  .inst-f{display:flex;align-items:center;gap:8px;margin-top:5px;
    font-family:var(--mono);font-size:9.5px;color:var(--dim);letter-spacing:.06em}
  .spark{width:92px;height:22px;flex:none;overflow:visible}
  .spark .sl{fill:none;stroke:var(--accent);stroke-width:1.4;vector-effect:non-scaling-stroke}
  .spark .sa{fill:var(--accent);opacity:.12;stroke:none}
  .spark .sp{fill:var(--accent)}

  .caution{border:1px solid var(--rule);border-top:2px solid var(--warn);background:var(--panel);
    padding:15px 16px;font-size:13.5px;line-height:1.62}
  .caution b{font-family:var(--mono);font-size:10px;letter-spacing:.2em;text-transform:uppercase;
    color:var(--warn);display:block;margin-bottom:7px}

  .gauge{border:1px solid var(--rule);background:var(--panel);overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px}
  th{font-size:9.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
    font-weight:500;text-align:left;padding:9px 12px;border-bottom:1px solid var(--rule)}
  td{padding:8px 12px;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}
  tr:last-child td{border-bottom:none}
  td.n{text-align:right}
  .track{position:relative;height:6px;background:var(--panel-2);min-width:88px;overflow:hidden}
  .track i{position:absolute;left:0;top:0;bottom:0;background:var(--ok)}
  .track.over i{background:var(--bad)}
  .track::after{content:"";position:absolute;right:0;top:0;bottom:0;width:14%;
    background:repeating-linear-gradient(45deg,transparent 0 3px,var(--bad) 3px 4px);opacity:.5}

  .sheet{display:grid;gap:2px;grid-template-columns:repeat(auto-fit,minmax(268px,1fr));
    background:var(--rule);border:1px solid var(--rule)}
  figure{margin:0;background:#000;position:relative}
  figure img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}
  figcaption{font-family:var(--mono);font-size:9.5px;letter-spacing:.18em;text-transform:uppercase;
    color:var(--ink);background:var(--panel);padding:7px 10px}
  figcaption span{color:var(--dim);letter-spacing:.04em;text-transform:none;font-size:10.5px}

  .log{border:1px solid var(--rule);background:var(--panel)}
  .log .row{padding:12px 14px;border-bottom:1px solid var(--rule)}
  .log .row:last-child{border-bottom:none}
  .log .rh{display:flex;gap:11px;align-items:baseline;flex-wrap:wrap;
    font-family:var(--mono);font-size:11px;letter-spacing:.1em}
  .log .rh b{font-size:12px;letter-spacing:.16em}
  .log .rh span{color:var(--dim)}
  .log ul{margin:8px 0 0;padding-left:18px;font-size:13.5px}
  .log li{margin:2px 0}
  footer{font-family:var(--mono);font-size:10.5px;color:var(--dim);text-align:center;
    padding-top:26px;border-top:1px solid var(--rule);margin-top:30px;line-height:2}
  @media (prefers-reduced-motion:no-preference){
    figure img{transition:opacity .2s}
  }
</style>

<div class="wrap">
  <div class="plate">
    <h1>Operation <b>Blacksand</b></h1>
    <span class="verdict ${pass ? 'ok' : 'bad'}">perf gate ${pass ? 'pass' : 'fail'}</span>
    <span class="meta">${latest ? `ROUND ${latest.round} · ${esc(latest.ts.slice(0, 16).replace('T', ' '))}Z` : 'NO ROUNDS'}</span>
  </div>

  <section>
    <div class="eyebrow">frame telemetry</div>
    <div class="insts">
      ${instrument('M1 · est. fps', est.m1?.estFps ?? '—', '', 'cost model @1080p', fpsSev(est.m1?.estFps ?? 0), sparkline(fpsHist))}
      ${instrument('GTX 1660 · est. fps', est.gtx1660?.estFps ?? '—', '', 'cost model @1080p', fpsSev(est.gtx1660?.estFps ?? 0), '')}
      ${instrument('JS logic p95', (cpu.logicP95 ?? 0).toFixed(2), 'ms', 'measured · limit 4.00', sev((cpu.logicP95 ?? 9) <= 4), sparkline(logicHist))}
      ${instrument('draw calls', cpu.stats?.calls ?? '—', '', 'limit 260', sev((cpu.stats?.calls ?? 999) <= 260), sparkline(callHist))}
      ${instrument('triangles', ((cpu.stats?.triangles ?? 0) / 1000).toFixed(0), 'k', 'limit 950k', sev((cpu.stats?.triangles ?? 9e9) <= 950000), '')}
      ${instrument('software rate', latest?.perf?.swiftshader1080p?.fps ?? '—', 'fps', 'regression signal only', 'neutral', '')}
    </div>
  </section>

  <section>
    <div class="caution">
      <b>What these numbers are</b>
      The build machine has <strong>no GPU</strong> — Chromium falls back to SwiftShader, a CPU
      rasteriser — so a frame rate measured here would say nothing about an M1 or a GTX 1660.
      Every round instead measures what actually transfers to real hardware: JavaScript logic
      time per frame, draw calls, triangles, full-screen passes and shadow-map texels, each against
      a fixed budget, then converts them to an <strong>estimated</strong> frame time on both
      reference GPUs using published throughput figures. The software rate above is kept only to
      catch round-over-round regressions. The game ships a live frame-rate readout and an in-game
      benchmark, so the real number on real hardware is one keypress away.
    </div>
  </section>

  <section>
    <div class="eyebrow">budget · 1920 × 1080</div>
    <div class="gauge"><table>
      <tr><th>metric</th><th class="n">value</th><th class="n">limit</th><th>headroom</th></tr>
      ${budgets.rows.map((r) => {
        const pct = Math.min(100, (r.v / r.limit) * 100);
        return `<tr><td>${esc(r.k)}</td>
          <td class="n" style="color:var(--${r.ok ? 'ok' : 'bad'})">${esc(r.v)}</td>
          <td class="n" style="color:var(--dim)">${esc(r.limit)}</td>
          <td><div class="track ${r.ok ? '' : 'over'}"><i style="width:${pct.toFixed(1)}%"></i></div></td></tr>`;
      }).join('')}
    </table></div>
  </section>

  ${gallery.length ? `<section>
    <div class="eyebrow">contact sheet · round ${latest.round}</div>
    <div class="sheet">
      ${gallery.map((g) => `<figure><img src="${g.src}" alt="${esc(g.name)}">
        <figcaption>${esc(g.name.replace(/_/g, ' '))}</figcaption></figure>`).join('')}
    </div>
  </section>` : ''}

  <section>
    <div class="eyebrow">round log</div>
    <div class="log">
      ${[...rounds].reverse().map((r) => `<div class="row">
        <div class="rh"><b>ROUND ${r.round}</b>
          <span style="color:var(--${r.verdict?.pass ? 'ok' : 'bad'})">${r.verdict?.pass ? 'PASS' : 'FAIL'}</span>
          <span>${esc(r.label || '')}</span>
          <span>${(r.perf?.cpu?.logicP95 ?? 0).toFixed(2)}ms · ${r.perf?.cpu?.stats?.calls ?? '—'} calls · est ${r.perf?.cpu?.estimate?.m1?.estFps ?? '—'}fps</span>
        </div>
        ${r.changes?.length ? `<ul>${r.changes.map((c) => `<li>${esc(c)}</li>`).join('')}</ul>` : ''}
        ${r.critique?.length ? `<ul>${r.critique.map((c) => `<li><strong>${esc(c.piece)}</strong> ${esc(c.score)}/10 — ${esc(c.gap)}</li>`).join('')}</ul>` : ''}
      </div>`).join('')}
    </div>
  </section>

  <footer>
    Every texture, mesh, animation and sound in this project is generated in code.<br>
    No third-party game assets, audio, model names or trademarks are used.
  </footer>
</div>`;

await writeFile(join(PROG, 'artifact.html'), html);
console.log(`artifact → progress/artifact.html (${(html.length / 1024).toFixed(0)} KB, ${gallery.length} shots inlined)`);
