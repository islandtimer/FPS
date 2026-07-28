// Quick single-frame look at the current build, for eyeballing mid-wave.
//   node tools/peek.mjs <shot> [w] [h]
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
const ROOT = resolve(import.meta.dirname, '..'), DIST = join(ROOT, 'dist'), PORT = 4331;
const shot = process.argv[2] || 'establish';
const W = +(process.argv[3] || 1280), H = +(process.argv[4] || 720);
spawnSync('npx', ['vite', 'build', '--logLevel', 'error'], { cwd: ROOT, stdio: 'inherit' });
const MIME={'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png'};
const srv = createServer(async (rq, rs) => {
  try { let p = rq.url.split('?')[0]; if (p.endsWith('/')) p += 'index.html';
    const b = await readFile(join(DIST, p));
    rs.writeHead(200, { 'content-type': MIME[extname(p)] || 'text/plain' }); rs.end(b);
  } catch { rs.writeHead(404).end(); }
});
await new Promise((r) => srv.listen(PORT, r));
const br = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  args: ['--no-sandbox','--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader'] });
const pg = await br.newPage({ viewport: { width: W, height: H } });
pg.setDefaultTimeout(180000);
const errs = []; pg.on('pageerror', e => errs.push(String(e).slice(0,300)));
pg.on('console', m => { if (m.type()==='error') errs.push(m.text().slice(0,300)); });
await pg.goto(`http://127.0.0.1:${PORT}/index.html?bench=1&w=${W}&h=${H}&shot=${shot}`, { waitUntil:'domcontentloaded' });
await pg.waitForFunction('window.__game&&window.__game.ready');
await pg.evaluate(() => new Promise(r => { let n=0; const t=()=>(++n<20?requestAnimationFrame(t):r()); requestAnimationFrame(t); }));
await pg.screenshot({ path: join(ROOT, 'progress', 'peek.png'), timeout: 180000 });
const snap = await pg.evaluate(() => window.__game.snapshot());
console.log(JSON.stringify({ calls: snap.stats.calls, tris: snap.stats.triangles, passes: snap.pipelineCost.fullscreenPasses, programs: snap.stats.programs, errs }, null, 1));
await br.close(); srv.close();
