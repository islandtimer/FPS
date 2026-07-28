// Regenerate JPEG thumbnails from captured PNGs. Chromium does the decode/encode;
// Playwright's bundled ffmpeg is a video-only build with no PNG decoder.
//   node tools/thumbs.mjs <shotDir> [width]
import { chromium } from 'playwright';
import { readFile, writeFile, readdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';

const dir = resolve(process.argv[2]);
const width = +(process.argv[3] || 960);
const br = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  args: ['--no-sandbox'] });
const pg = await br.newPage();
await pg.setContent('<!doctype html><title>thumb</title>');
for (const f of (await readdir(dir)).filter((f) => f.endsWith('.png'))) {
  const b64 = (await readFile(join(dir, f))).toString('base64');
  const out = await pg.evaluate(async ({ b64, width }) => {
    const img = new Image(); img.src = 'data:image/png;base64,' + b64; await img.decode();
    const c = document.createElement('canvas');
    c.width = width; c.height = Math.round((img.height / img.width) * width);
    const ctx = c.getContext('2d'); ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, c.width, c.height);
    return c.toDataURL('image/jpeg', 0.74).split(',')[1];
  }, { b64, width });
  const dst = join(dir, f.replace(/\.png$/, '.jpg'));
  await writeFile(dst, Buffer.from(out, 'base64'));
  console.log(`${f} -> ${(Buffer.from(out, 'base64').length / 1024).toFixed(0)}KB`);
}
await br.close();
