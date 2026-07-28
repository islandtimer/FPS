// OWNER: agent "tex" — material library built on the procedural textures.
// CONTRACT:
//   buildMaterials(renderer) -> { get(name, opts) -> THREE.Material, names: string[], dispose() }
// Materials MUST be shared/cached by name so draw calls can batch.
//
// Two things happen here beyond "make a MeshStandardMaterial".
//
// 1. CHANNEL PACKING. textures.js hands back one RGB texture carrying
//    AO / roughness / metalness (the glTF ORM convention, which three reads
//    natively off .r/.g/.b). A full PBR surface is therefore three texture
//    units — albedo, normal, ORM — not five.
//
// 2. MACRO VARIATION. A single shared 256px low-frequency map is layered over
//    every large surface at a frequency deliberately incommensurate with the
//    tile (2.37 / repeat), so the base tile and the variation never come back
//    into phase. That is what stops a 512px concrete texture repeated across a
//    40m wall reading as a grid. Cost: one extra texture fetch per fragment,
//    and it is off by default for small props and the viewmodel where the
//    surface is never large enough to show a repeat.
//
// Caller-supplied `roughness` / `metalness` / `color` are interpreted as TARGET
// values, not raw multipliers: they are divided through by the material's own
// mean so that `get('gunmetal', { roughness: 0.42 })` actually lands near 0.42
// instead of multiplying an already-0.4 map down to 0.17.

import * as THREE from 'three';
import {
  makeTextureSet, macroTexture, setTextureRenderer, disposeTextures, TEXTURE_NAMES,
} from './textures.js';

// meanRough / meanMetal / meanAlbedo describe what the baked maps average out
// to, and exist only so caller intent survives the map multiply.
const CFG = {
  concrete:     { rough: 0.90, metal: 0.00, albedo: 0x7a7873, normal: 1.00, ao: 1.0, macro: 0.9 },
  plaster:      { rough: 0.74, metal: 0.00, albedo: 0xb4b0a8, normal: 0.85, ao: 1.0, macro: 0.8 },
  sand:         { rough: 0.94, metal: 0.00, albedo: 0xa89068, normal: 1.10, ao: 0.8, macro: 1.0 },
  asphalt:      { rough: 0.84, metal: 0.00, albedo: 0x25252a, normal: 1.00, ao: 1.0, macro: 0.9 },
  rustMetal:    { rough: 0.74, metal: 0.45, albedo: 0x5c4030, normal: 1.10, ao: 1.0, macro: 0.6 },
  paintedMetal: { rough: 0.50, metal: 0.22, albedo: 0x3c4038, normal: 0.85, ao: 1.0, macro: 0.5 },
  gunmetal:     { rough: 0.40, metal: 0.90, albedo: 0x24262a, normal: 0.70, ao: 1.0, macro: 0.0 },
  polymer:      { rough: 0.62, metal: 0.00, albedo: 0x1b1b1c, normal: 0.80, ao: 1.0, macro: 0.0 },
  wood:         { rough: 0.80, metal: 0.00, albedo: 0x5a4028, normal: 0.95, ao: 1.0, macro: 0.7 },
  fabric:       { rough: 0.92, metal: 0.00, albedo: 0x4c4638, normal: 0.90, ao: 1.0, macro: 0.4 },
  rubber:       { rough: 0.84, metal: 0.00, albedo: 0x0e0e0f, normal: 0.80, ao: 1.0, macro: 0.0 },
  glass:        { rough: 0.10, metal: 0.00, albedo: 0xdadde0, normal: 0.35, ao: 0.4, macro: 0.0 },
  tile:         { rough: 0.42, metal: 0.00, albedo: 0xa8a49c, normal: 0.90, ao: 1.0, macro: 0.6 },
  brick:        { rough: 0.90, metal: 0.00, albedo: 0x6a4034, normal: 1.00, ao: 1.0, macro: 0.8 },
};

// Names other modules may reasonably ask for that are not texture names.
const ALIAS = {
  metal: 'gunmetal', steel: 'gunmetal', rust: 'rustMetal', paint: 'paintedMetal',
  dirt: 'sand', ground: 'sand', earth: 'sand', stone: 'concrete',
  cloth: 'fabric', plastic: 'polymer', road: 'asphalt', wall: 'plaster',
};

// ---------------------------------------------------------------- macro patch

const MACRO_DECL = /* glsl */`
uniform sampler2D uMacroMap;
uniform vec2 uMacroScale;
uniform vec3 uMacroAmt;   // x tint, y brightness, z roughness break
vec4 gMacro = vec4(0.5);
`;

const MACRO_MAP = /* glsl */`#include <map_fragment>
#ifdef USE_MAP
  // rotated off the tile axes as well as scaled off the tile period, so the
  // variation shares no symmetry with the thing it is hiding
  vec2 macroUv = mat2( 0.8660254, 0.5, -0.5, 0.8660254 ) * vMapUv * uMacroScale;
  gMacro = texture2D( uMacroMap, macroUv );
  float macroL = dot( gMacro.rgb, vec3( 0.55, 0.33, 0.12 ) );
  diffuseColor.rgb *= 1.0 + ( gMacro.rgb - 0.5 ) * uMacroAmt.x;
  diffuseColor.rgb *= 1.0 + ( macroL - 0.5 ) * uMacroAmt.y;
#endif
`;

const MACRO_ROUGH = /* glsl */`#include <roughnessmap_fragment>
roughnessFactor = clamp( roughnessFactor * ( 1.0 + ( gMacro.a - 0.5 ) * uMacroAmt.z ), 0.04, 1.0 );
`;

function applyMacro(mat, macroTex, scale, strength) {
  const uScale = new THREE.Vector2(scale, scale);
  const uAmt = new THREE.Vector3(0.34 * strength, 0.42 * strength, 0.55 * strength);
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uMacroMap = { value: macroTex };
    shader.uniforms.uMacroScale = { value: uScale };
    shader.uniforms.uMacroAmt = { value: uAmt };
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', MACRO_DECL + '\nvoid main() {')
      .replace('#include <map_fragment>', MACRO_MAP)
      .replace('#include <roughnessmap_fragment>', MACRO_ROUGH);
  };
  // Every macro-patched material injects byte-identical code, so they all share
  // one compiled program. Without this key three would reuse the UNPATCHED
  // program for them, silently dropping the injection.
  mat.customProgramCacheKey = () => 'tex:macro:1';
}

// ---------------------------------------------------------------- helpers

const _c = new THREE.Color();
const _t = new THREE.Color();

/** Tint that lands the surface on `target` given the map already averages `mean`. */
function tintFor(target, mean) {
  _c.setHex(mean, THREE.SRGBColorSpace);            // -> linear
  _t.setHex(target, THREE.SRGBColorSpace);          // -> linear
  const f = (a, b) => Math.min(4, Math.max(0.02, a / Math.max(1e-4, b)));
  return [f(_t.r, _c.r), f(_t.g, _c.g), f(_t.b, _c.b)];
}

function keyOf(name, opts) {
  const ks = Object.keys(opts).sort();
  let s = name;
  for (const k of ks) {
    const v = opts[k];
    s += '|' + k + '=' + (typeof v === 'object' && v !== null ? JSON.stringify(v) : v);
  }
  return s;
}

// ---------------------------------------------------------------- public

export function buildMaterials(renderer) {
  const cache = new Map();
  let macroTex = null;

  // textures.js bakes into this renderer's context, so it must be the renderer
  // that draws the scene. A null renderer degrades to flat fallback textures
  // rather than throwing during module init.
  if (renderer) {
    setTextureRenderer(renderer);
    macroTex = macroTexture();
  }

  const api = {
    names: TEXTURE_NAMES.slice(),

    /**
     * opts:
     *   repeat | repeatX/repeatY  uv tiling
     *   size                      override bake resolution (power of two)
     *   color                     TARGET surface colour (sRGB hex)
     *   roughness / metalness     TARGET values, normalised against the map
     *   normalScale               multiplier on the baked relief
     *   aoIntensity               0..1
     *   macro                     force the tiling-break layer on/off
     *   transparent / opacity / side / depthWrite / emissive / emissiveIntensity
     */
    get(name, opts = {}) {
      const key = keyOf(name, opts);
      const hit = cache.get(key);
      if (hit) return hit;

      const resolved = CFG[name] ? name : (ALIAS[name] || 'concrete');
      const cfg = CFG[resolved];

      const set = makeTextureSet(resolved, {
        size: opts.size,
        repeat: opts.repeat,
        repeatX: opts.repeatX,
        repeatY: opts.repeatY,
        rotation: opts.rotation,
      });

      const [tr, tg, tb] = opts.color != null
        ? tintFor(opts.color, cfg.albedo)
        : [1, 1, 1];

      const nScale = (opts.normalScale ?? 1) * cfg.normal;
      const params = {
        map: set.map,
        normalMap: set.normalMap,
        normalScale: new THREE.Vector2(nScale, nScale),
        roughnessMap: set.roughnessMap,
        metalnessMap: set.metalnessMap,
        aoMap: set.aoMap,
        aoMapIntensity: (opts.aoIntensity ?? 1) * cfg.ao,
        // divide the target through by what the map already averages
        roughness: clamp01((opts.roughness ?? cfg.rough) / cfg.rough),
        metalness: cfg.metal > 0.02
          ? clamp01((opts.metalness ?? cfg.metal) / cfg.metal)
          : (opts.metalness ?? 0),
        envMapIntensity: opts.envMapIntensity ?? 1.0,
        dithering: true,
      };
      if (opts.side != null) params.side = opts.side;
      if (opts.transparent) { params.transparent = true; params.opacity = opts.opacity ?? 1; }
      if (opts.depthWrite != null) params.depthWrite = opts.depthWrite;
      if (opts.emissive != null) {
        params.emissive = new THREE.Color(opts.emissive);
        params.emissiveIntensity = opts.emissiveIntensity ?? 1;
      }

      if (resolved === 'glass' && opts.transparent !== false) {
        params.transparent = true;
        params.opacity = opts.opacity ?? 0.28;
        params.depthWrite = false;
        params.side = opts.side ?? THREE.DoubleSide;
      }

      const m = new THREE.MeshStandardMaterial(params);
      m.color.setRGB(tr, tg, tb, THREE.LinearSRGBColorSpace);
      m.name = 'mat:' + resolved;

      const wantMacro = opts.macro ?? (cfg.macro > 0);
      if (wantMacro && macroTex && set.map) {
        // One macro cycle every 3.37 tiles: an irrational-ish ratio, so the base
        // tile and the variation never come back into phase.
        const scale = opts.macroScale ?? (1 / 3.37);
        applyMacro(m, macroTex, scale, opts.macroStrength ?? Math.max(cfg.macro, 0.35));
      }

      cache.set(key, m);
      return m;
    },

    /** Debug/perf: how many distinct materials exist (== state buckets). */
    count() { return cache.size; },

    dispose() {
      for (const m of cache.values()) m.dispose();
      cache.clear();
      disposeTextures();
      setTextureRenderer(null);
    },
  };

  return api;
}

function clamp01(v) { return Math.min(1, Math.max(0, v)); }
