// OWNER: agent "tex" — material library built on the procedural textures.
// CONTRACT:
//   buildMaterials(renderer) -> { get(name, opts) -> THREE.Material, names: string[], dispose() }
// Materials MUST be shared/cached by name so draw calls can batch.

import * as THREE from 'three';
import { makeTextureSet } from './textures.js';

export function buildMaterials(renderer) {
  const cache = new Map();
  const api = {
    names: ['concrete', 'sand', 'rustMetal', 'gunmetal', 'polymer', 'fabric', 'wood', 'glass'],
    get(name, opts = {}) {
      const key = name + JSON.stringify(opts);
      if (cache.has(key)) return cache.get(key);
      const set = makeTextureSet(name, opts);
      const m = new THREE.MeshStandardMaterial({
        map: set.map,
        normalMap: set.normalMap || null,
        roughnessMap: set.roughnessMap || null,
        roughness: opts.roughness ?? 0.85,
        metalness: opts.metalness ?? 0.0,
        color: opts.color ?? 0xffffff,
      });
      cache.set(key, m);
      return m;
    },
    dispose() { for (const m of cache.values()) m.dispose(); cache.clear(); },
  };
  return api;
}
