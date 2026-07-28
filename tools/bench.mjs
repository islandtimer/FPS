// Benchmark + screenshot harness.
//
// IMPORTANT — read before citing any number this produces.
// The build container has no GPU. Chromium runs on SwiftShader (a CPU rasteriser),
// so full-resolution frame rate here is NOT a prediction of an M1 or a GTX 1660.
// This harness therefore produces four separate signals:
//
//   cpuMs        - JS main-thread ms/frame, measured with a 320x180 framebuffer so
//                  fragment cost is ~0 and what is left is game logic + draw-call
//                  submission. This DOES transfer to real hardware.
//   swFps        - full 1080p frame rate under SwiftShader. Relative regression
//                  signal between rounds only. Never quote it as "our fps".
//   budgets      - hardware-neutral counters vs. hard limits (draw calls, tris,
//                  fullscreen passes, shadow texels). The real 60fps gate.
//   estimate     - cost model -> estimated ms/frame on M1 / GTX 1660. Labelled
//                  as an estimate wherever it is displayed.
//
// Usage: node tools/bench.mjs [--round N] [--label "text"] [--shots a,b,c]

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { spawn } from 'node:child_process';

const ROOT = resolve(import.meta.dirname, '..');
const DIST = join(ROOT, 'dist');
const CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const PORT = 4319;

const args = process.argv.slice(2);
const argOf = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm', '.map': 'application/json',
};

function serve(dir) {
  return new Promise((res) => {
    const s = createServer(async (req, rq) => {
      try {
        let p = decodeURIComponent(req.url.split('?')[0]);
        if (p === '/' || p.endsWith('/')) p += 'index.html';
        const file = join(dir, p);
        if (!file.startsWith(dir)) { rq.writeHead(403).end(); return; }
        const body = await readFile(file);
        rq.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' });
        rq.end(body);
      } catch { rq.writeHead(404).end('nope'); }
    });
    s.listen(PORT, () => res(s));
  });
}

function run(cmd, cmdArgs, opts = {}) {
  return new Promise((res, rej) => {
    const p = spawn(cmd, cmdArgs, { cwd: ROOT, stdio: 'inherit', ...opts });
    p.on('exit', (c) => (c === 0 ? res() : rej(new Error(`${cmd} exited ${c}`))));
  });
}

/**
 * Downscale a PNG to a JPEG for the progress page and the phone artifact.
 * Playwright's bundled ffmpeg is a video-only build with no PNG decoder and no
 * image2 demuxer, so Chromium does the work instead — it already has both codecs.
 */
async function thumbnail(page, srcPath, dstPath, width) {
  try {
    const b64 = (await readFile(srcPath)).toString('base64');
    const out = await page.evaluate(async ({ b64, width }) => {
      const img = new Image();
      img.src = 'data:image/png;base64,' + b64;
      await img.decode();
      const c = document.createElement('canvas');
      c.width = width;
      c.height = Math.round((img.height / img.width) * width);
      const ctx = c.getContext('2d');
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, c.width, c.height);
      return c.toDataURL('image/jpeg', 0.74).split(',')[1];
    }, { b64, width });
    await writeFile(dstPath, Buffer.from(out, 'base64'));
    return true;
  } catch (e) {
    console.log(`  (thumbnail failed: ${String(e.message).slice(0, 80)})`);
    return false;
  }
}

async function nextRound() {
  const dir = join(ROOT, 'progress', 'data');
  if (!existsSync(dir)) return 1;
  const files = (await readdir(dir)).filter((f) => /^round-\d+\.json$/.test(f));
  return files.length
    ? Math.max(...files.map((f) => +f.match(/\d+/)[0])) + 1
    : 1;
}

async function main() {
  console.log('› building…');
  await run('npx', ['vite', 'build', '--logLevel', 'warn']);

  const round = +(argOf('--round', await nextRound()));
  const label = argOf('--label', '');
  const outDir = join(ROOT, 'progress', 'shots', `r${String(round).padStart(2, '0')}`);
  await mkdir(outDir, { recursive: true });
  await mkdir(join(ROOT, 'progress', 'data'), { recursive: true });

  const server = await serve(DIST);
  const browser = await chromium.launch({
    executablePath: CHROME,
    // No --disable-frame-rate-limit / --disable-gpu-vsync. They let the renderer
    // produce frames as fast as it can, which starves the compositor and makes
    // captureScreenshot hang indefinitely at 1080p. They also buy nothing: software
    // rasterisation runs at single-digit fps, nowhere near the 60fps vsync cap, so
    // the uncapped-fps measurement is identical either way.
    args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-gl=angle',
           '--use-angle=swiftshader', '--js-flags=--expose-gc'],
  });

  const errors = [];
  const result = { round, label, ts: new Date().toISOString(), shots: {}, errors, perf: {} };

  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  // SwiftShader at 1080p is slow enough that default 30s timeouts flake.
  page.setDefaultTimeout(180000);
  page.setDefaultNavigationTimeout(180000);
  page.on('pageerror', (e) => errors.push(String(e).slice(0, 400)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 400)); });

  // A second, deliberately empty page used only as an image-resizing surface.
  const thumbPage = await ctx.newPage();
  await thumbPage.setContent('<!doctype html><title>thumb</title>');

  const url = (q) => `http://127.0.0.1:${PORT}/index.html?bench=1&${q}`;
  const waitReady = () => page.waitForFunction('window.__game && window.__game.ready', null, { timeout: 120000 });

  // ---- 1. CPU probe (tiny framebuffer => fragment cost ≈ 0) --------------
  console.log('› cpu probe (320x180, 600 frames)…');
  await page.goto(url('w=320&h=180&shot=firefight'), { waitUntil: 'commit' });
  await waitReady();
  const cpu = await page.evaluate(async () => {
    const g = window.__game;
    for (let i = 0; i < 90; i++) await new Promise(requestAnimationFrame); // warm caches/JIT
    // Drive the simulation synchronously and time it directly. This measures pure
    // JS game-logic cost with zero rAF-scheduling or rasteriser contamination, so
    // the number transfers to real hardware.
    g.setPaused(true);
    const samples = [];
    for (let i = 0; i < 900; i++) {
      const t0 = performance.now();
      g.step(1, 1 / 60);
      samples.push(performance.now() - t0);
    }
    g.setPaused(false);
    for (let i = 0; i < 30; i++) await new Promise(requestAnimationFrame);
    samples.sort((a, b) => a - b);
    const at = (p) => +samples[Math.min(samples.length - 1, Math.floor(p * samples.length))].toFixed(4);
    const s = g.snapshot();
    return {
      logicMedian: at(0.5), logicP95: at(0.95), logicP99: at(0.99),
      logicMax: +samples.at(-1).toFixed(4),
      stats: s.stats, budgets: s.budgets, estimate: s.estimate,
    };
  });
  result.perf.cpu = cpu;
  console.log(`  logicMs median=${cpu.logicMedian} p95=${cpu.logicP95} | calls=${cpu.stats.calls} tris=${cpu.stats.triangles}`);

  // ---- 2. SwiftShader 1080p relative fps --------------------------------
  console.log('› swiftshader 1080p probe…');
  await page.goto(url('w=1920&h=1080&shot=firefight'), { waitUntil: 'commit' });
  await waitReady();
  const sw = await page.evaluate(async () => {
    for (let i = 0; i < 20; i++) await new Promise(requestAnimationFrame);
    const t0 = performance.now(); let n = 0;
    while (performance.now() - t0 < 6000) { await new Promise(requestAnimationFrame); n++; }
    const g = window.__game;
    return { fps: +((n * 1000) / (performance.now() - t0)).toFixed(2), snapshot: g.snapshot() };
  });
  result.perf.swiftshader1080p = { fps: sw.fps, note: 'CPU rasteriser — relative regression signal only' };
  result.perf.snapshot = sw.snapshot;
  console.log(`  swFps=${sw.fps} (software)`);

  // ---- 3. Screenshots ---------------------------------------------------
  const shotNames = (argOf('--shots', '') || '').split(',').filter(Boolean);
  const names = shotNames.length
    ? shotNames
    : await page.evaluate(() => Object.keys(window.__game.shots));

  // No navigation here on purpose. The page is already loaded at 1920x1080 from the
  // SwiftShader probe, and re-posing via applyShot costs one scene init instead of
  // eight. A navigation at this point stalls anyway: committing one needs a response
  // from a renderer process that is busy emitting a frame every ~200ms.

  for (const name of names) {
   try {
    process.stdout.write(`› shot ${name}… `);
    await page.evaluate((n) => {
      const g = window.__game;
      g.applyShot(n);
      // Drive a fixed number of rendered frames so temporal accumulation (TAA
      // history, eye adaptation) converges, then stop rendering entirely — the
      // capture below then reads a static surface instead of racing the compositor.
      g.settle(48, 1 / 60);
      // Transient effects (muzzle flash, sparks) live for a frame or two, so they
      // are triggered last — after the scene has settled, immediately before capture.
      g.preCapture(n);
    }, name);
    const file = join(outDir, `${name}.png`);
    let captured = false;
    for (let attempt = 0; attempt < 2 && !captured; attempt++) {
      try {
        await page.screenshot({ path: file, timeout: 150000, caret: 'initial' });
        captured = true;
      } catch (e) {
        if (attempt === 1) throw e;
        // Capture occasionally loses the race with the compositor. Nudge the page
        // into producing a fresh frame and try once more before giving up the pose.
        await page.evaluate(() => { window.__game.resume(); }).catch(() => {});
        await page.waitForTimeout(2000);
        await page.evaluate(() => window.__game.settle(8, 1 / 60)).catch(() => {});
      }
    }
    await thumbnail(thumbPage, file, join(outDir, `${name}.jpg`), 960);
    const dir = `shots/r${String(round).padStart(2, '0')}`;
    result.shots[name] = `${dir}/${name}.jpg`;
    result.shotsFull = result.shotsFull || {};
    result.shotsFull[name] = `${dir}/${name}.png`;
    console.log('ok');
    await page.evaluate(() => window.__game.resume());
   } catch (e) {
    await page.evaluate(() => window.__game.resume()).catch(() => {});
    // One bad pose must not cost us the whole round's data.
    console.log('FAILED: ' + String(e.message).split('\n')[0]);
    errors.push(`shot ${name}: ${String(e.message).slice(0, 200)}`);
   }
  }

  await browser.close();
  server.close();

  const verdict = cpu.budgets.pass && cpu.logicP95 <= 4.0;
  result.verdict = {
    budgetsPass: cpu.budgets.pass,
    cpuOk: cpu.logicP95 <= 4.0,
    pass: verdict,
    failing: cpu.budgets.rows.filter((r) => !r.ok).map((r) => `${r.k}=${r.v} > ${r.limit}`),
  };

  const out = join(ROOT, 'progress', 'data', `round-${String(round).padStart(2, '0')}.json`);
  await writeFile(out, JSON.stringify(result, null, 2));
  console.log(`\n› round ${round} → ${out}`);
  console.log(`› verdict: ${verdict ? 'PASS' : 'FAIL'} ${result.verdict.failing.join(', ')}`);
  if (errors.length) console.log(`› ${errors.length} console/page errors:\n  ` + errors.slice(0, 5).join('\n  '));
  process.exit(errors.length ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(2); });
