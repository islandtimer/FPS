// OWNER: agent "tex" — procedural PBR texture synthesis. NO image files, ever.
// CONTRACT:
//   makeTextureSet(name, opts) -> { map, normalMap, roughnessMap, metalnessMap?, aoMap? }
//   TEXTURE_NAMES -> string[]  (names main/level/props may request)
//   textureBytes() -> approximate VRAM in bytes, for the perf budget

import * as THREE from 'three';

export const TEXTURE_NAMES = [
  'concrete', 'plaster', 'sand', 'asphalt', 'rustMetal', 'paintedMetal',
  'wood', 'fabric', 'gunmetal', 'polymer', 'rubber', 'glass', 'tile', 'brick',
];

const cache = new Map();
let bytes = 0;

function canvasTex(size, fill, { srgb = true, repeat = 1 } = {}) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  fill(ctx, size);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat, repeat);
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.anisotropy = 8;
  bytes += size * size * 4 * 1.34;
  return t;
}

export function makeTextureSet(name, opts = {}) {
  const key = name + JSON.stringify(opts);
  if (cache.has(key)) return cache.get(key);
  const size = opts.size || 512;
  const base = { concrete: '#8a8a86', sand: '#c3a97c', rustMetal: '#7a5340' }[name] || '#9a9a96';
  const set = {
    map: canvasTex(size, (ctx, s) => {
      ctx.fillStyle = base; ctx.fillRect(0, 0, s, s);
      for (let i = 0; i < s * 12; i++) {
        const v = Math.random() * 40 - 20;
        ctx.fillStyle = `rgba(${128 + v},${128 + v},${128 + v},0.06)`;
        ctx.fillRect(Math.random() * s, Math.random() * s, 2, 2);
      }
    }, { repeat: opts.repeat || 1 }),
  };
  cache.set(key, set);
  return set;
}

export function textureBytes() { return bytes; }
export function disposeTextures() {
  for (const set of cache.values()) for (const t of Object.values(set)) t?.dispose?.();
  cache.clear(); bytes = 0;
}
