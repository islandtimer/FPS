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
// 2. THE SHARED SURFACE LAYER. One shader injection, identical for every
//    material, carrying macro tiling-break, cavity grime, convex edge wear and
//    a world-anchored dirt splash at the foot of every wall. See the block
//    above applySurface() for what each does and why it cannot live in the
//    baked maps. Cost: two texture fetches per fragment (one 256px macro map,
//    one re-read of the already-bound ORM), and one compiled program in total —
//    the injection is byte-identical everywhere, so all fourteen families share
//    a single custom program cache key.
//
// Caller-supplied `roughness` / `metalness` / `color` are interpreted as TARGET
// values, not raw multipliers: they are divided through by the material's own
// mean so that `get('gunmetal', { roughness: 0.42 })` actually lands near 0.42
// instead of multiplying an already-0.4 map down to 0.17.

import * as THREE from 'three';
import {
  makeTextureSet, macroTexture, setTextureRenderer, disposeTextures, TEXTURE_NAMES,
} from './textures.js';

// rough / metal / albedo are the reference values a caller's TARGET is divided
// through by, so they must be kept in step with the compose pass in textures.js.
//
// `albedo` is the mean of the CLEAN material — intact plaster, unstained clay —
// NOT the mean of the whole map including its dirt. That distinction matters:
// `get('plaster', { color: 0xd8c8a9 })` means "this wall is painted this
// colour", and the stains, cavity grime and base splash then read as darker
// than it. Anchoring on the dirty mean instead lifts the clean plaster above
// the requested colour to compensate, which clips it to white.
//
// If `rough` drifts below what the map actually averages, every caller's
// requested roughness silently clamps to 1 and the whole family goes flat.
//
// grime  [ cavityAmt, concavityAmt, baseHeightM, baseAmt ]
//        cavityAmt   how much baked AO also means dirt
//        concavity   how much texture-scale concavity collects dirt
//        baseHeight  metres of dirt-splash gradient up from the ground (0 = off)
// gCol   multiply applied where grime is full
// wear   [ amount, roughnessDelta, metalnessDelta ]  on convex texture detail
// wCol   multiply applied where edge wear is full (>1 exposes brighter material)
const CFG = {
  concrete: {
    rough: 0.855, metal: 0.00, albedo: 0x7a7873, normal: 1.00, ao: 1.0, macro: 0.9,
    grime: [0.48, 0.48, 0.50, 0.75], gCol: [0.54, 0.520, 0.485],
    wear: [0.35, 0.05, 0.0], wCol: [1.22, 1.20, 1.16],
  },
  plaster: {
    rough: 0.870, metal: 0.00, albedo: 0xc8c1b2, normal: 0.85, ao: 1.0, macro: 0.85,
    grime: [0.34, 0.42, 0.55, 0.85], gCol: [0.52, 0.485, 0.435],
    wear: [0.30, 0.05, 0.0], wCol: [1.16, 1.145, 1.11],
  },
  sand: {
    rough: 0.940, metal: 0.00, albedo: 0xa28d68, normal: 1.10, ao: 0.8, macro: 1.0,
    grime: [0.35, 0.30, 0.0, 0.0], gCol: [0.74, 0.715, 0.665],
    wear: [0.18, 0.02, 0.0], wCol: [1.10, 1.09, 1.07],
  },
  asphalt: {
    rough: 0.840, metal: 0.00, albedo: 0x25252a, normal: 1.00, ao: 1.0, macro: 0.9,
    grime: [0.45, 0.40, 0.0, 0.0], gCol: [0.70, 0.70, 0.72],
    wear: [0.30, 0.02, 0.0], wCol: [1.25, 1.24, 1.22],
  },
  rustMetal: {
    rough: 0.740, metal: 0.45, albedo: 0x5c4030, normal: 1.10, ao: 1.0, macro: 0.6,
    grime: [0.50, 0.55, 0.30, 0.35], gCol: [0.62, 0.575, 0.53],
    wear: [0.55, -0.10, 0.10], wCol: [1.55, 1.58, 1.62],
  },
  paintedMetal: {
    rough: 0.500, metal: 0.22, albedo: 0x3c4038, normal: 0.85, ao: 1.0, macro: 0.5,
    grime: [0.45, 0.50, 0.25, 0.30], gCol: [0.66, 0.645, 0.62],
    wear: [0.60, -0.12, 0.35], wCol: [1.70, 1.73, 1.78],
  },
  gunmetal: {
    rough: 0.380, metal: 0.90, albedo: 0x24262a, normal: 0.70, ao: 1.0, macro: 0.22,
    grime: [0.30, 0.45, 0.0, 0.0], gCol: [0.80, 0.79, 0.78],
    wear: [0.70, -0.15, 0.08], wCol: [1.85, 1.88, 1.94],
  },
  polymer: {
    rough: 0.630, metal: 0.00, albedo: 0x1b1b1c, normal: 0.80, ao: 1.0, macro: 0.20,
    grime: [0.28, 0.40, 0.0, 0.0], gCol: [0.82, 0.815, 0.81],
    wear: [0.45, -0.10, 0.0], wCol: [1.34, 1.34, 1.36],
  },
  wood: {
    rough: 0.760, metal: 0.00, albedo: 0x57422e, normal: 0.95, ao: 1.0, macro: 0.7,
    grime: [0.42, 0.48, 0.35, 0.55], gCol: [0.56, 0.52, 0.47],
    wear: [0.40, -0.07, 0.0], wCol: [1.22, 1.19, 1.14],
  },
  fabric: {
    rough: 0.920, metal: 0.00, albedo: 0x4c4638, normal: 0.90, ao: 1.0, macro: 0.4,
    grime: [0.50, 0.45, 0.0, 0.0], gCol: [0.60, 0.575, 0.535],
    wear: [0.20, 0.02, 0.0], wCol: [1.12, 1.11, 1.09],
  },
  rubber: {
    rough: 0.840, metal: 0.00, albedo: 0x0e0e0f, normal: 0.80, ao: 1.0, macro: 0.0,
    grime: [0.25, 0.30, 0.0, 0.0], gCol: [0.86, 0.86, 0.87],
    wear: [0.25, -0.04, 0.0], wCol: [1.20, 1.20, 1.21],
  },
  glass: {
    rough: 0.100, metal: 0.00, albedo: 0xdadde0, normal: 0.35, ao: 0.4, macro: 0.0,
    grime: [0.20, 0.20, 0.0, 0.0], gCol: [0.88, 0.885, 0.89],
    wear: [0.0, 0.0, 0.0], wCol: [1.0, 1.0, 1.0],
  },
  tile: {
    rough: 0.420, metal: 0.00, albedo: 0xa8a49c, normal: 0.90, ao: 1.0, macro: 0.6,
    grime: [0.55, 0.60, 0.30, 0.40], gCol: [0.60, 0.585, 0.555],
    wear: [0.35, 0.04, 0.0], wCol: [1.16, 1.15, 1.13],
  },
  brick: {
    rough: 0.870, metal: 0.00, albedo: 0x785240, normal: 1.00, ao: 1.0, macro: 0.8,
    grime: [0.40, 0.45, 0.45, 0.70], gCol: [0.56, 0.530, 0.490],
    wear: [0.35, 0.04, 0.0], wCol: [1.20, 1.17, 1.13],
  },
};

// Names other modules may reasonably ask for that are not texture names.
const ALIAS = {
  metal: 'gunmetal', steel: 'gunmetal', rust: 'rustMetal', paint: 'paintedMetal',
  dirt: 'sand', ground: 'sand', earth: 'sand', stone: 'concrete',
  cloth: 'fabric', plastic: 'polymer', road: 'asphalt', wall: 'plaster',
};

// ---------------------------------------------------------------- surface patch
//
// One injection, applied to EVERY material this module makes, carrying three
// things the baked maps cannot express on their own:
//
//   MACRO BREAK   a 256px low-frequency field at a period deliberately
//                 incommensurate with the tile, so a 512px texture repeated
//                 across a 40m wall stops reading as a grid.
//   CAVITY GRIME  dirt multiplied in wherever the surface is occluded or
//                 concave. Driven off the ORM texture that is already bound —
//                 R is baked AO, A is baked curvature — so it costs one extra
//                 fetch of a texture that is already resident, and no new map.
//   EDGE WEAR     the mirror of it: convex texture detail gets rubbed brighter,
//                 smoother and (on metals) more metallic, because that is what
//                 hands and passing shoulders do to a raised edge.
//
// Plus a world-anchored dirt splash up the bottom half-metre of every wall.
// That one has to be world-space — anchored to the tile it would repeat every
// 2m up a 6m wall, which is worse than not having it.
//
// The code is byte-identical for every material; only uniforms differ. That is
// deliberate: one custom cache key means one extra compiled program across the
// entire game rather than one per family.

const SURFACE_KEY = 'tex:surface:2';

const SURFACE_DECL = /* glsl */`
uniform sampler2D uMacroMap;
uniform vec2 uMacroScale;
uniform vec3 uMacroAmt;   // x tint, y brightness, z roughness break
uniform vec4 uGrime;      // x cavity, y concavity, z base height (m), w base amount
uniform vec3 uGrimeCol;
uniform vec4 uWear;       // x amount, y roughness delta, z metalness delta
uniform vec3 uWearCol;
vec4  gMacro = vec4( 0.5 );
float gCav = 0.0;
float gWear = 0.0;
`;

const SURFACE_MAP = /* glsl */`#include <map_fragment>
#ifdef USE_MAP
  // rotated off the tile axes as well as scaled off the tile period, so the
  // variation shares no symmetry with the thing it is hiding
  vec2 macroUv = mat2( 0.8660254, 0.5, -0.5, 0.8660254 ) * vMapUv * uMacroScale;
  gMacro = texture2D( uMacroMap, macroUv );
  float macroL = dot( gMacro.rgb, vec3( 0.55, 0.33, 0.12 ) );
  // biased below the midpoint on purpose: weathering removes light. A symmetric
  // multiply on an already bright albedo clips its highlights instead of adding
  // variation to them.
  // Biased hard toward DARKENING, and that is not a stylistic choice. The level
  // asks for wall colours near 0.69 linear; a symmetric multiply on top of that
  // clips instead of varying, and a clipped wall is the flat beige this layer
  // exists to prevent. Weathering removes light anyway — it does not add it.
  diffuseColor.rgb *= 1.0 + ( gMacro.rgb - 0.72 ) * uMacroAmt.x;
  diffuseColor.rgb *= 1.0 + ( macroL - 0.72 ) * uMacroAmt.y;
#endif

#ifdef USE_ROUGHNESSMAP
  // ORM alpha is baked curvature: 0.5 flat, above it convex, below it concave.
  vec4 gOrm = texture2D( roughnessMap, vRoughnessMapUv );
  float gConcave = max( 0.0, 0.5 - gOrm.a ) * 2.0;
  float gConvex  = max( 0.0, gOrm.a - 0.5 ) * 2.0;
  gCav  = clamp( ( 1.0 - gOrm.r ) * uGrime.x + gConcave * uGrime.y, 0.0, 1.0 );
  // broken up by the macro field so wear is regional, never a uniform rim
  gWear = clamp( gConvex * uWear.x * ( 0.20 + 1.60 * gMacro.a ), 0.0, 1.0 );
#endif

// World position without a varying: viewPos = R * ( P - camera ), and R is
// orthonormal, so P = camera + viewPos * R. Three dot products, no interpolant.
float gBase = 0.0;
if ( uGrime.w > 0.0 ) {
  vec3 gWorld = cameraPosition + ( - vViewPosition ) * mat3( viewMatrix );
  #ifdef FLAT_SHADED
    float gFace = 1.0;
  #else
    float gFace = 1.0 - abs( ( normalize( vNormal ) * mat3( viewMatrix ) ).y );
  #endif
  gBase = uGrime.w * gFace * ( 0.45 + 1.10 * gMacro.a )
        * ( 1.0 - smoothstep( uGrime.z * 0.15, uGrime.z, max( gWorld.y, 0.0 ) ) );
}
float gDirt = clamp( gCav + gBase, 0.0, 1.0 );
diffuseColor.rgb *= mix( vec3( 1.0 ), uGrimeCol, gDirt );
diffuseColor.rgb *= mix( vec3( 1.0 ), uWearCol, gWear );
`;

const SURFACE_ROUGH = /* glsl */`#include <roughnessmap_fragment>
roughnessFactor *= 1.0 + ( gMacro.a - 0.5 ) * uMacroAmt.z;
// dirt is matte, wear is not; both move roughness, in opposite directions
roughnessFactor = clamp( roughnessFactor + gDirt * 0.085 + gWear * uWear.y, 0.04, 1.0 );
`;

const SURFACE_METAL = /* glsl */`#include <metalnessmap_fragment>
metalnessFactor = clamp( metalnessFactor + gWear * uWear.z, 0.0, 1.0 );
`;

/** Attach the shared surface shader to `mat`, and make it survive .clone(). */
function applySurface(mat, u) {
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uMacroMap = u.map;
    shader.uniforms.uMacroScale = u.scale;
    shader.uniforms.uMacroAmt = u.amt;
    shader.uniforms.uGrime = u.grime;
    shader.uniforms.uGrimeCol = u.gCol;
    shader.uniforms.uWear = u.wear;
    shader.uniforms.uWearCol = u.wCol;
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', SURFACE_DECL + '\nvoid main() {')
      .replace('#include <map_fragment>', SURFACE_MAP)
      .replace('#include <roughnessmap_fragment>', SURFACE_ROUGH)
      .replace('#include <metalnessmap_fragment>', SURFACE_METAL);
  };
  // Without this key three would reuse the UNPATCHED program and silently drop
  // the whole injection. Every material injects byte-identical source, so one
  // key is correct and costs one program, not fourteen.
  mat.customProgramCacheKey = () => SURFACE_KEY;
  // THREE.Material.copy() carries neither onBeforeCompile nor
  // customProgramCacheKey. level.js clones every material it takes from here in
  // order to turn vertexColors on, so without this the entire map — every wall,
  // road and roof — rendered with no macro break, no grime and no edge wear,
  // while the library material it was cloned from looked correct in isolation.
  // Re-patching inside clone() is the only place that can be fixed from here.
  // Carrying the hook by reference rather than re-running applySurface matters:
  // sky.js wraps onBeforeCompile again afterwards to inject its cascade
  // selection, and a clone that reinstated only THIS patch would silently drop
  // the shadow cascades from every cloned material in the level.
  mat.clone = function surfaceClone() {
    const c = THREE.Material.prototype.clone.call(this);
    c.onBeforeCompile = this.onBeforeCompile;
    c.customProgramCacheKey = this.customProgramCacheKey;
    c.clone = surfaceClone;
    return c;
  };
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
     *   macroStrength             override the family's tiling-break amount
     *   grimeStrength             0..1+ scale on cavity + base-of-wall grime
     *   wearStrength              0..1+ scale on convex edge wear
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

      if (macroTex && set.map) {
        // One macro cycle every 3.37 tiles — an irrational-ish ratio, so the base
        // tile and the variation never come back into phase. On a 2m plaster tile
        // that is a 6.7m field, which is the band a 6m wall needs and the tile
        // itself physically cannot carry.
        const scale = opts.macroScale ?? (1 / 3.37);
        const macroOn = (opts.macro ?? true) ? (opts.macroStrength ?? cfg.macro) : 0;
        const g = cfg.grime || [0, 0, 0, 0];
        const w = cfg.wear || [0, 0, 0];
        const gc = cfg.gCol || [1, 1, 1];
        const wc = cfg.wCol || [1, 1, 1];
        const gs = opts.grimeStrength ?? 1;
        const ws = opts.wearStrength ?? 1;
        applySurface(m, {
          map: { value: macroTex },
          scale: { value: new THREE.Vector2(scale, scale) },
          amt: { value: new THREE.Vector3(0.40 * macroOn, 0.46 * macroOn, 0.55 * macroOn) },
          grime: { value: new THREE.Vector4(g[0] * gs, g[1] * gs, g[2], g[3] * gs) },
          gCol: { value: new THREE.Vector3(...mixToward1(gc, gs)) },
          wear: { value: new THREE.Vector4(w[0] * ws, w[1], w[2], 0) },
          wCol: { value: new THREE.Vector3(...mixToward1(wc, ws)) },
        });
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

/** Scale a colour MULTIPLIER's distance from neutral, not the multiplier itself. */
function mixToward1(c, k) { return [1 + (c[0] - 1) * k, 1 + (c[1] - 1) * k, 1 + (c[2] - 1) * k]; }
