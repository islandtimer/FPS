// OWNER: agent "combat" — ballistics, hit resolution, impact FX, tracers, decals, shells.
// CONTRACT:
//   new Combat(scene, level, world)   world = { enemies: [] , player }
//   .update(dt)
//   .dispose()
// Subscribes EV.SHOT, emits EV.HIT / EV.KILL. All FX must be pooled + instanced;
// zero per-shot allocation in steady state.
//
// ---------------------------------------------------------------------------
// HOW THIS IS BUILT, and why.
//
// 1. THE BULLET IS ONE THING, SIMULATED TWO WAYS. Every round starts as a
//    hitscan segment out to `instantRange` (muzzleVelocity * 0.055, clamped to
//    18..45m) because inside that distance a real projectile's drop is under
//    two centimetres and the extra frame of latency is felt. If the instant
//    segment finds nothing, the round is handed to a pooled projectile that
//    carries velocity, gravity and drag and is stepped in update(). Same hit
//    resolution, same FX, same penetration chain — only the integration differs.
//
// 2. RAYCASTING IS OUR OWN, NOT THREE'S. level.raycastables are ~190 invisible
//    unit boxes. Raycaster.intersectObjects() allocates an array and an
//    intersection record per call, per shot, forever — exactly the thing the
//    brief forbids. Instead every proxy is cached once as {world centre, world
//    radius, inverse matrix, local half-extents, world axes} and tested with a
//    sphere reject followed by a slab test in the proxy's local space. That
//    yields entry t, EXIT t and an axis-aligned face normal from a single test,
//    which is what makes penetration a two-line calculation instead of a second
//    cast, and it allocates nothing.
//
// 3. PENETRATION IS A MATERIAL PROPERTY, NOT A FLAG. Each surface class has a
//    max thickness it will pass and a soak coefficient; damage retained through
//    a wall is exp(-soak * thickness). A 5cm plank keeps 78% of the round, a
//    2cm steel sheet 59%, 10cm of concrete 41% (and anything thicker stops it
//    dead — a 42cm wall never passes). Exit direction gets a small seeded
//    deflection so a penetrating kill is never a free shot.
//
// 4. THE WHOLE FX LAYER IS FIVE DRAW CALLS. One alpha-blended GPU particle
//    field, one additive one, one decal field, one tracer field, one instanced
//    shell mesh. Particles are never touched again after they are emitted: the
//    vertex shader integrates p = p0 + v0*(1-e^-kt)/k - g t^2/2 from a single
//    time uniform and collapses the quad to a degenerate triangle when the
//    instance is dead. Emission writes 17 floats into a ring buffer and marks a
//    contiguous update range, so a burst of automatic fire uploads a few KB.
//
// 5. DECALS MULTIPLY, THEY DO NOT LIGHT. A decal drawn with MultiplyBlending
//    darkens whatever light the surface already had, so a bullet hole reads
//    correctly in a black interior and in full sun without costing a lit
//    material or a second pass. The atlas is generated as raw bytes (no canvas:
//    canvas antialiasing is not identical across platforms and screenshots have
//    to reproduce), and values above 1.0 in the multiply are what give the
//    chipped concrete its bright aggregate ring.
//
// 6. NOTHING HERE MAKES A SOUND. The bullet-crack whip and the shell landing
//    are published as events for the audio module.
// ---------------------------------------------------------------------------

import * as THREE from 'three';
import { bus, EV } from '../core/bus.js';
import { makeRng, WORLD_SEED } from '../core/rng.js';

// Events this module publishes that are not in the canonical EV table yet.
// (Reported to the orchestrator as a requested bus.js addition.)
export const EV_CRACK = 'bulletCrack';   // {point, distance, speed} supersonic pass-by
export const EV_SHELL = 'shellLand';     // {point, surface, energy}

// ---------------------------------------------------------------------------
// tuning
// ---------------------------------------------------------------------------

const GRAV = 9.81;
const MAX_PEN = 2;              // walls a single round may pass through
const MIN_PEN_DMG = 0.16;       // below this fraction the round is spent
const DECAL_CAP = 288;
const DECAL_LIFE = 26.0;
const TRACER_CAP = 64;
const SHELL_CAP = 24;
const ALPHA_CAP = 1024;
const ADD_CAP = 640;
const CRACK_RADIUS = 3.2;       // how close a round passes before it whips

// Surface behaviour. `soak` is damage attenuation per metre of material
// (retain = e^-soak*thickness); `max` is the thickest slab the round will pass.
// Concrete's 0.11m is deliberate: it lets a plaster partition or a thin slab be
// shot through while the map's 0.42m walls remain hard cover.
const SURF = {
  concrete: {
    fx: 'dust', cell: 0, cellN: 2,
    dust: [0.66, 0.63, 0.58], chip: [0.52, 0.50, 0.47], chips: 6, sparks: 0,
    tint: [0.30, 0.29, 0.275],
    pen: { max: 0.11, soak: 9.0, deflect: 0.055 },
  },
  plaster: null,   // level maps plaster -> concrete; alias installed below
  sand: {
    fx: 'poof', cell: 10, cellN: 1,
    dust: [0.80, 0.66, 0.45], chip: [0.62, 0.51, 0.34], chips: 3, sparks: 0,
    tint: [0.46, 0.38, 0.26],
    pen: null,
  },
  metal: {
    fx: 'spark', cell: 4, cellN: 2,
    dust: [0.34, 0.32, 0.31], chip: [0.45, 0.42, 0.40], chips: 2, sparks: 16,
    tint: [0.20, 0.19, 0.19],
    pen: { max: 0.055, soak: 26.0, deflect: 0.105 },
  },
  wood: {
    fx: 'splinter', cell: 6, cellN: 2,
    dust: [0.62, 0.50, 0.35], chip: [0.55, 0.42, 0.26], chips: 8, sparks: 0,
    tint: [0.28, 0.20, 0.13],
    pen: { max: 0.34, soak: 5.0, deflect: 0.075 },
  },
  glass: {
    fx: 'glass', cell: 8, cellN: 2,
    dust: [0.74, 0.82, 0.84], chip: [0.80, 0.88, 0.92], chips: 10, sparks: 0,
    tint: [0.62, 0.68, 0.70],
    pen: { max: 0.26, soak: 2.2, deflect: 0.028 },
  },
  fabric: {
    fx: 'poof', cell: 11, cellN: 1,
    dust: [0.74, 0.66, 0.50], chip: [0.58, 0.50, 0.38], chips: 3, sparks: 0,
    tint: [0.38, 0.33, 0.25],
    pen: { max: 0.10, soak: 22.0, deflect: 0.13 },
  },
  flesh: {
    fx: 'blood', cell: 12, cellN: 1,
    dust: [0.42, 0.045, 0.035], chip: [0.30, 0.03, 0.03], chips: 0, sparks: 0,
    tint: [0.34, 0.10, 0.09],
    pen: { max: 0.30, soak: 14.0, deflect: 0.02 },
  },
};
SURF.plaster = SURF.concrete;
SURF.brick = SURF.concrete;
SURF.road = SURF.concrete;
SURF.ground = SURF.sand;

// ---------------------------------------------------------------------------
// procedural decal atlas — 4x4 cells of 128px, RGBA
//   R = hole/darkening mask   G = bright aggregate/rim mask   A = coverage
// Written as bytes rather than drawn on a canvas so it is byte-identical on
// every machine, which is what "screenshots reproduce between rounds" requires.
// ---------------------------------------------------------------------------

const ATLAS_CELL = 128;
const ATLAS_N = 4;
const ATLAS_SIZE = ATLAS_CELL * ATLAS_N;

function buildDecalAtlas(rng) {
  const S = ATLAS_SIZE;
  const data = new Uint8Array(S * S * 4);

  // Per-cell shape parameters, drawn from the seeded stream once.
  const cells = [];
  for (let i = 0; i < ATLAS_N * ATLAS_N; i++) {
    const spokes = 5 + ((i * 7) % 4);
    const ph = [];
    for (let k = 0; k < 8; k++) ph.push(rng() * Math.PI * 2);
    const amp = [];
    for (let k = 0; k < 8; k++) amp.push(rng());
    cells.push({ spokes, ph, amp, jitter: rng() });
  }

  // kind per cell index: 0-1 concrete, 2-3 concrete big, 4-5 metal scorch,
  // 6-7 wood, 8-9 glass, 10 sand, 11 fabric, 12 blood, 13-15 spare variants.
  const KIND = ['hole', 'hole', 'hole', 'hole', 'scorch', 'scorch', 'wood', 'wood',
    'glass', 'glass', 'sand', 'pock', 'blood', 'hole', 'scorch', 'glass'];

  for (let ci = 0; ci < ATLAS_N * ATLAS_N; ci++) {
    const cx = (ci % ATLAS_N) * ATLAS_CELL;
    const cy = ((ci / ATLAS_N) | 0) * ATLAS_CELL;
    const P = cells[ci];
    const kind = KIND[ci];

    for (let y = 0; y < ATLAS_CELL; y++) {
      for (let x = 0; x < ATLAS_CELL; x++) {
        const u = (x + 0.5) / ATLAS_CELL * 2 - 1;
        const v = (y + 0.5) / ATLAS_CELL * 2 - 1;
        let r = Math.sqrt(u * u + v * v);
        const a = Math.atan2(v, u);

        // wobble the radius so nothing is a perfect circle
        let wob = 0;
        for (let k = 0; k < 4; k++) wob += Math.sin(a * (k + 2) + P.ph[k]) * (0.045 * P.amp[k]) / (k + 1);
        r *= 1 + wob * 2.4;

        let dark = 0, bright = 0, cov = 0;

        if (kind === 'hole' || kind === 'pock') {
          const core = kind === 'pock' ? 0.24 : 0.30;
          dark = smoothstep(core * 1.30, core * 0.5, r);
          // exposed aggregate: a chalky annulus, brighter than the surface
          bright = Math.max(0, smoothstep(core * 2.2, core * 1.1, r) - dark * 0.8);
          // radial cracks
          let cr = 0;
          for (let k = 0; k < P.spokes; k++) {
            const sa = P.ph[k % 8] + (k / P.spokes) * Math.PI * 2;
            const da = Math.abs(wrapPi(a - sa));
            const len = 0.55 + 0.4 * P.amp[k % 8];
            const w = 0.045 + 0.03 * P.amp[(k + 3) % 8];
            const line = smoothstep(w, 0.0, da * Math.max(0.12, r)) * smoothstep(len, len * 0.2, r);
            cr = Math.max(cr, line);
          }
          dark = Math.max(dark, cr * 0.72);
          cov = Math.max(dark, bright * 0.95);
          cov *= smoothstep(1.05, 0.6, r);
        } else if (kind === 'scorch') {
          const n = 0.5 + 0.5 * Math.sin(u * 9.3 + P.ph[0]) * Math.sin(v * 11.7 + P.ph[1]);
          dark = smoothstep(0.85, 0.05, r) * (0.55 + 0.45 * n);
          bright = smoothstep(0.13, 0.0, r) * 0.9;
          cov = Math.max(dark, bright);
        } else if (kind === 'wood') {
          let sp = 0;
          for (let k = 0; k < P.spokes + 3; k++) {
            const sa = P.ph[k % 8] * 1.7 + (k / (P.spokes + 3)) * Math.PI * 2;
            const da = Math.abs(wrapPi(a - sa));
            const len = 0.65 + 0.35 * P.amp[k % 8];
            const w = 0.07 + 0.06 * P.amp[(k + 2) % 8];
            sp = Math.max(sp, smoothstep(w, 0.0, da * Math.max(0.1, r)) * smoothstep(len, 0.1, r));
          }
          dark = Math.max(smoothstep(0.26, 0.07, r), sp * 0.72);
          bright = sp * 0.35 * smoothstep(0.3, 0.6, r);
          cov = Math.max(dark, bright);
        } else if (kind === 'glass') {
          let cr = 0;
          for (let k = 0; k < 9; k++) {
            const sa = P.ph[k % 8] * 2.1 + (k / 9) * Math.PI * 2;
            const da = Math.abs(wrapPi(a - sa));
            const w = 0.013 + 0.012 * P.amp[k % 8];
            cr = Math.max(cr, smoothstep(w, 0.0, da * Math.max(0.08, r)) * smoothstep(0.95, 0.1, r));
          }
          // concentric fracture rings
          for (let k = 1; k <= 3; k++) {
            const rr = 0.2 * k + 0.06 * P.amp[k];
            cr = Math.max(cr, smoothstep(0.022, 0.0, Math.abs(r - rr)) * 0.8);
          }
          dark = Math.max(smoothstep(0.1, 0.02, r), cr * 0.42);
          bright = cr * 0.85;
          cov = Math.max(dark, bright);
        } else if (kind === 'sand') {
          dark = smoothstep(0.55, 0.12, r) * 0.85;
          bright = Math.max(0, smoothstep(0.9, 0.55, r) - dark * 0.6) * 0.8;
          cov = Math.max(dark, bright) * smoothstep(1.05, 0.6, r);
        } else { // blood
          dark = smoothstep(0.6, 0.05, r);
          let sp = 0;
          for (let k = 0; k < 7; k++) {
            const sa = P.ph[k % 8] * 3.1 + (k / 7) * Math.PI * 2;
            const da = Math.abs(wrapPi(a - sa));
            const len = 0.5 + 0.45 * P.amp[k % 8];
            sp = Math.max(sp, smoothstep(0.08, 0.0, da * Math.max(0.1, r)) * smoothstep(len, 0.1, r));
          }
          dark = Math.max(dark, sp * 0.8);
          bright = 0;
          cov = dark;
        }

        const o = ((cy + y) * S + (cx + x)) * 4;
        data[o] = clamp255(dark * 255);
        data[o + 1] = clamp255(bright * 255);
        data[o + 2] = 0;
        data[o + 3] = clamp255(cov * 255);
      }
    }
  }

  const tex = new THREE.DataTexture(data, S, S, THREE.RGBAFormat);
  tex.needsUpdate = true;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.anisotropy = 4;
  return tex;
}

function smoothstep(e0, e1, x) {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0 || 1e-6)));
  return t * t * (3 - 2 * t);
}
function wrapPi(a) {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}
function clamp255(v) { return v < 0 ? 0 : v > 255 ? 255 : v | 0; }

// ---------------------------------------------------------------------------
// GPU particle field — one draw call, zero CPU work after emission
// ---------------------------------------------------------------------------

const PART_VERT = /* glsl */`
uniform float uTime;
attribute vec3 aPos;
attribute vec3 aVel;
attribute vec3 aCol;
attribute vec2 aLife;   // birth, 1/life
attribute vec2 aSize;   // size at birth, size at death
attribute vec4 aCtl;    // gravity, drag, shape, shape param
varying vec3 vCol;
varying vec2 vUv;
varying float vA;
varying float vShape;
varying float vSeed;

void main() {
  float t = uTime - aLife.x;
  float u = t * aLife.y;
  vUv = uv; vCol = aCol; vShape = aCtl.z; vSeed = aCtl.w; vA = 0.0;
  if (u < 0.0 || u >= 1.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  // closed-form drag + gravity: no per-frame CPU integration for any particle
  float k = max(aCtl.y, 0.02);
  float e = exp(-k * t);
  vec3 p = aPos + aVel * ((1.0 - e) / k);
  p.y -= 0.5 * aCtl.x * t * t;
  vec3 vel = aVel * e;
  vel.y -= aCtl.x * t;

  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  float sz = mix(aSize.x, aSize.y, u);
  vec2 off = position.xy * sz;

  if (vShape > 0.5 && vShape < 1.5) {
    // spark streak: stretch along screen-space velocity so it reads as a trail
    vec3 vv = (modelViewMatrix * vec4(vel, 0.0)).xyz;
    vec2 d = vv.xy;
    float L = length(d);
    if (L > 1e-4) {
      d /= L;
      float st = 1.0 + min(aCtl.w * L, 26.0);
      off = d * (position.x * sz * st) + vec2(-d.y, d.x) * (position.y * sz);
    }
  } else if (vShape > 1.5 && vShape < 2.5) {
    // chip / shard: spin about the view axis
    float a = aCtl.w * 6.2831 + t * (4.0 + aCtl.w * 22.0);
    float c = cos(a), s = sin(a);
    off = vec2(off.x * c - off.y * s, off.x * s + off.y * c);
  }

  mv.xy += off;
  gl_Position = projectionMatrix * mv;

  // life curve: puffs bloom then dissipate, sparks decay hard
  float fadeIn = smoothstep(0.0, 0.10, u);
  float fadeOut = 1.0 - smoothstep(vShape > 0.5 && vShape < 1.5 ? 0.35 : 0.45, 1.0, u);
  vA = fadeIn * fadeOut;
  if (vShape > 0.5 && vShape < 1.5) vA *= 0.55 + 0.45 * sin(t * 96.0 + aCtl.w * 40.0);
  // distant FX must not out-punch the aerial perspective the sky module sets up
  vA *= 1.0 - smoothstep(55.0, 130.0, -mv.z);
}
`;

const PART_FRAG = /* glsl */`
varying vec3 vCol;
varying vec2 vUv;
varying float vA;
varying float vShape;
varying float vSeed;

void main() {
  vec2 q = vUv - 0.5;
  float a;
  if (vShape < 0.5) {
    // soft puff, slightly lumpy so it is not a gaussian blob
    float d = length(q) * 2.0;
    float lump = 1.0 + 0.18 * sin(atan(q.y, q.x) * 3.0 + vSeed * 6.2831);
    a = smoothstep(1.0, 0.05, d / lump);
    // dust you can see through: a fully opaque puff reads as a paper cut-out
    a *= a * 0.62;
  } else if (vShape < 1.5) {
    float core = smoothstep(0.5, 0.0, abs(q.y) * 2.0);
    a = core * smoothstep(0.5, 0.12, abs(q.x));
    a *= a;
  } else if (vShape < 2.5) {
    vec2 s = abs(q);
    a = step(s.x + s.y * 1.6, 0.42);
  } else {
    // muzzle flash: petal count varies per shot
    float ang = atan(q.y, q.x);
    float r = length(q);
    float petals = 3.0 + floor(vSeed * 4.0);
    float lobe = 0.26 + 0.17 * sin(ang * petals + vSeed * 6.2831);
    a = smoothstep(lobe, 0.0, r);
    a = a * a + smoothstep(0.15, 0.0, r);
  }
  if (a <= 0.001 || vA <= 0.001) discard;
  gl_FragColor = vec4(vCol, a * vA);
}
`;

const QUAD_POS = new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]);
const QUAD_UV = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
const QUAD_IDX = [0, 1, 2, 0, 2, 3];

function quadGeometry() {
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(QUAD_POS.slice(), 3));
  g.setAttribute('uv', new THREE.BufferAttribute(QUAD_UV.slice(), 2));
  g.setIndex(QUAD_IDX.slice());
  g.instanceCount = 0;
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
  return g;
}

class ParticleField {
  constructor(cap, additive, uTime, order) {
    this.cap = cap;
    this.i = 0;
    this.used = 0;
    this._lo = 1e9; this._hi = -1;

    const g = quadGeometry();
    const mk = (name, size) => {
      const a = new THREE.InstancedBufferAttribute(new Float32Array(cap * size), size);
      a.setUsage(THREE.DynamicDrawUsage);
      g.setAttribute(name, a);
      return a;
    };
    this.aPos = mk('aPos', 3);
    this.aVel = mk('aVel', 3);
    this.aCol = mk('aCol', 3);
    this.aLife = mk('aLife', 2);
    this.aSize = mk('aSize', 2);
    this.aCtl = mk('aCtl', 4);
    this.attrs = [this.aPos, this.aVel, this.aCol, this.aLife, this.aSize, this.aCtl];
    // dead instances are collapsed by the vertex shader, but a birth of -1e9
    // keeps them dead before anything has ever been emitted
    for (let i = 0; i < cap; i++) this.aLife.array[i * 2] = -1e9;

    this.geometry = g;
    this.material = new THREE.ShaderMaterial({
      uniforms: { uTime },
      vertexShader: PART_VERT,
      fragmentShader: PART_FRAG,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    this.mesh = new THREE.Mesh(g, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.renderOrder = order;
    this.mesh.castShadow = this.mesh.receiveShadow = false;
  }

  /** 17 floats into a ring slot. No allocation, no branching on capacity. */
  emit(x, y, z, vx, vy, vz, r, g, b, s0, s1, life, grav, drag, shape, param, now) {
    const i = this.i;
    this.i = (i + 1) % this.cap;
    if (i >= this.used) this.used = i + 1;

    let o = i * 3;
    const P = this.aPos.array, V = this.aVel.array, C = this.aCol.array;
    P[o] = x; P[o + 1] = y; P[o + 2] = z;
    V[o] = vx; V[o + 1] = vy; V[o + 2] = vz;
    C[o] = r; C[o + 1] = g; C[o + 2] = b;
    o = i * 2;
    const L = this.aLife.array, S = this.aSize.array;
    L[o] = now; L[o + 1] = 1 / life;
    S[o] = s0; S[o + 1] = s1;
    o = i * 4;
    const K = this.aCtl.array;
    K[o] = grav; K[o + 1] = drag; K[o + 2] = shape; K[o + 3] = param;

    if (i < this._lo) this._lo = i;
    if (i > this._hi) this._hi = i;
  }

  flush() {
    if (this._hi < this._lo) return;
    const start = this._lo, count = this._hi - this._lo + 1;
    for (const a of this.attrs) {
      if (a.clearUpdateRanges) { a.clearUpdateRanges(); a.addUpdateRange(start * a.itemSize, count * a.itemSize); }
      a.needsUpdate = true;
    }
    this.geometry.instanceCount = this.used;
    this._lo = 1e9; this._hi = -1;
  }

  dispose() { this.geometry.dispose(); this.material.dispose(); }
}

// ---------------------------------------------------------------------------
// decals — one instanced quad field, multiply-blended, ring buffered
// ---------------------------------------------------------------------------

const DECAL_VERT = /* glsl */`
uniform float uTime;
attribute vec3 aCenter;
attribute vec3 aTan;
attribute vec3 aBit;
attribute vec3 aTint;
attribute vec4 aParam;   // birth, 1/life, cellX, cellY
varying vec2 vUv;
varying vec3 vTint;
varying float vFade;

void main() {
  float u = (uTime - aParam.x) * aParam.y;
  vUv = vec2(0.0); vTint = aTint; vFade = 0.0;
  if (u < 0.0 || u >= 1.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec3 p = aCenter + aTan * position.x + aBit * position.y;
  // 3% inset per cell: mip levels must not bleed a neighbouring hole in
  vUv = (uv * 0.94 + 0.03 + vec2(aParam.z, aParam.w)) * 0.25;
  vFade = 1.0 - smoothstep(0.72, 1.0, u);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}
`;

const DECAL_FRAG = /* glsl */`
uniform sampler2D uAtlas;
varying vec2 vUv;
varying vec3 vTint;
varying float vFade;

void main() {
  vec4 t = texture2D(uAtlas, vUv);
  float m = t.a * vFade;
  if (m <= 0.004) discard;
  // multiply space: 1.0 = untouched, <1 darkens, >1 brightens (exposed aggregate)
  vec3 c = mix(vec3(1.0), vTint, t.r) + vec3(t.g) * 0.55 * vFade;
  gl_FragColor = vec4(mix(vec3(1.0), c, m), 1.0);
}
`;

class DecalField {
  constructor(cap, atlas, uTime) {
    this.cap = cap; this.i = 0; this.used = 0;
    this._lo = 1e9; this._hi = -1;

    const g = quadGeometry();
    const mk = (name, size) => {
      const a = new THREE.InstancedBufferAttribute(new Float32Array(cap * size), size);
      a.setUsage(THREE.DynamicDrawUsage);
      g.setAttribute(name, a);
      return a;
    };
    this.aCenter = mk('aCenter', 3);
    this.aTan = mk('aTan', 3);
    this.aBit = mk('aBit', 3);
    this.aTint = mk('aTint', 3);
    this.aParam = mk('aParam', 4);
    this.attrs = [this.aCenter, this.aTan, this.aBit, this.aTint, this.aParam];
    for (let i = 0; i < cap; i++) this.aParam.array[i * 4] = -1e9;

    this.geometry = g;
    this.material = new THREE.ShaderMaterial({
      uniforms: { uTime, uAtlas: { value: atlas } },
      vertexShader: DECAL_VERT,
      fragmentShader: DECAL_FRAG,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.MultiplyBlending,
      // three's MultiplyBlending path is only defined for premultiplied alpha;
      // the shader writes alpha 1.0 so dst alpha is preserved either way.
      premultipliedAlpha: true,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -4,
    });
    this.mesh = new THREE.Mesh(g, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.renderOrder = 5;
    this.mesh.castShadow = this.mesh.receiveShadow = false;
  }

  place(px, py, pz, nx, ny, nz, tx, ty, tz, size, cellX, cellY, tr, tg, tb, life, now) {
    const i = this.i;
    this.i = (i + 1) % this.cap;
    if (this.used < this.cap && i >= this.used) this.used = i + 1;

    // bitangent = n x t, both scaled to the decal's world size
    const bx = ny * tz - nz * ty, by = nz * tx - nx * tz, bz = nx * ty - ny * tx;
    let o = i * 3;
    const C = this.aCenter.array, T = this.aTan.array, B = this.aBit.array, N = this.aTint.array;
    C[o] = px + nx * 0.012; C[o + 1] = py + ny * 0.012; C[o + 2] = pz + nz * 0.012;
    T[o] = tx * size; T[o + 1] = ty * size; T[o + 2] = tz * size;
    B[o] = bx * size; B[o + 1] = by * size; B[o + 2] = bz * size;
    N[o] = tr; N[o + 1] = tg; N[o + 2] = tb;
    o = i * 4;
    const P = this.aParam.array;
    P[o] = now; P[o + 1] = 1 / life; P[o + 2] = cellX; P[o + 3] = cellY;

    if (i < this._lo) this._lo = i;
    if (i > this._hi) this._hi = i;
  }

  flush() {
    if (this._hi < this._lo) return;
    const start = this._lo, count = this._hi - this._lo + 1;
    for (const a of this.attrs) {
      if (a.clearUpdateRanges) { a.clearUpdateRanges(); a.addUpdateRange(start * a.itemSize, count * a.itemSize); }
      a.needsUpdate = true;
    }
    this.geometry.instanceCount = this.used;
    this._lo = 1e9; this._hi = -1;
  }

  dispose() { this.geometry.dispose(); this.material.dispose(); }
}

// ---------------------------------------------------------------------------
// tracers — camera-facing ribbon that travels the bullet path
// ---------------------------------------------------------------------------

const TRACER_VERT = /* glsl */`
uniform float uTime;
attribute vec3 aStart;
attribute vec3 aDir;
attribute vec3 aCol;
attribute vec4 aParam;   // length, birth, 1/life, speed
varying vec3 vCol;
varying float vA;
varying float vT;

void main() {
  float t = uTime - aParam.y;
  float u = t * aParam.z;
  vCol = aCol; vA = 0.0; vT = 0.0;
  if (u < 0.0 || u >= 1.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float headD = min(aParam.x, aParam.w * t);
  float trail = min(9.0, aParam.x * 0.65);
  float tailD = max(0.0, headD - trail);
  vec4 mvH = modelViewMatrix * vec4(aStart + aDir * headD, 1.0);
  vec4 mvT = modelViewMatrix * vec4(aStart + aDir * tailD, 1.0);
  float s = position.x + 0.5;
  vec4 mv = mix(mvT, mvH, s);

  vec3 seg = mvH.xyz - mvT.xyz;
  vec3 toCam = -mv.xyz;
  vec3 side = cross(seg, toCam);
  float L = length(side);
  // a tracer thinner than a pixel flickers; widen it a little with distance
  float w = 0.016 + 0.0018 * (-mv.z);
  if (L > 1e-6) mv.xyz += side * (position.y * w * 2.0 / L);

  gl_Position = projectionMatrix * mv;
  vT = s;
  vA = (1.0 - u) * (1.0 - u);
}
`;

const TRACER_FRAG = /* glsl */`
varying vec3 vCol;
varying float vA;
varying float vT;
void main() {
  float taper = smoothstep(0.0, 0.45, vT);
  float a = vA * taper;
  if (a <= 0.002) discard;
  gl_FragColor = vec4(vCol * (0.55 + 0.45 * vT), a);
}
`;

class TracerField {
  constructor(cap, uTime) {
    this.cap = cap; this.i = 0; this.used = 0;
    this._lo = 1e9; this._hi = -1;
    const g = quadGeometry();
    const mk = (name, size) => {
      const a = new THREE.InstancedBufferAttribute(new Float32Array(cap * size), size);
      a.setUsage(THREE.DynamicDrawUsage);
      g.setAttribute(name, a);
      return a;
    };
    this.aStart = mk('aStart', 3);
    this.aDir = mk('aDir', 3);
    this.aCol = mk('aCol', 3);
    this.aParam = mk('aParam', 4);
    this.attrs = [this.aStart, this.aDir, this.aCol, this.aParam];
    for (let i = 0; i < cap; i++) this.aParam.array[i * 4 + 1] = -1e9;

    this.geometry = g;
    this.material = new THREE.ShaderMaterial({
      uniforms: { uTime },
      vertexShader: TRACER_VERT,
      fragmentShader: TRACER_FRAG,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.AdditiveBlending,
    });
    this.mesh = new THREE.Mesh(g, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.renderOrder = 12;
    this.mesh.castShadow = this.mesh.receiveShadow = false;
  }

  fire(x, y, z, dx, dy, dz, len, speed, r, g, b, life, now) {
    const i = this.i;
    this.i = (i + 1) % this.cap;
    if (this.used < this.cap && i >= this.used) this.used = i + 1;
    let o = i * 3;
    const S = this.aStart.array, D = this.aDir.array, C = this.aCol.array;
    S[o] = x; S[o + 1] = y; S[o + 2] = z;
    D[o] = dx; D[o + 1] = dy; D[o + 2] = dz;
    C[o] = r; C[o + 1] = g; C[o + 2] = b;
    o = i * 4;
    const P = this.aParam.array;
    P[o] = len; P[o + 1] = now; P[o + 2] = 1 / life; P[o + 3] = speed;
    if (i < this._lo) this._lo = i;
    if (i > this._hi) this._hi = i;
  }

  flush() {
    if (this._hi < this._lo) return;
    const start = this._lo, count = this._hi - this._lo + 1;
    for (const a of this.attrs) {
      if (a.clearUpdateRanges) { a.clearUpdateRanges(); a.addUpdateRange(start * a.itemSize, count * a.itemSize); }
      a.needsUpdate = true;
    }
    this.geometry.instanceCount = this.used;
    this._lo = 1e9; this._hi = -1;
  }

  dispose() { this.geometry.dispose(); this.material.dispose(); }
}

// ---------------------------------------------------------------------------
// shells — instanced brass, CPU physics (there are only ever 24 of them)
// ---------------------------------------------------------------------------

class ShellPool {
  constructor(cap, rng) {
    this.cap = cap; this.i = 0; this.rng = rng;
    // ~26 triangles each; a case, not a cylinder — the rim reads at 0.5m
    const geo = new THREE.CylinderGeometry(0.0046, 0.0052, 0.0235, 7, 1, false);
    geo.rotateX(Math.PI * 0.5);   // axis along +Z so tumble looks right
    const mat = new THREE.MeshStandardMaterial({
      color: 0xb08a3c, metalness: 0.92, roughness: 0.34,
    });
    this.mesh = new THREE.InstancedMesh(geo, mat, cap);
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    this.s = new Float32Array(cap * 12);  // x y z vx vy vz qx qy qz qw floorY state
    this.age = new Float32Array(cap);
    this.surf = new Array(cap).fill('concrete');
    this.wx = new Float32Array(cap * 3);  // angular velocity

    this._m = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._dq = new THREE.Quaternion();
    this._p = new THREE.Vector3();
    this._one = new THREE.Vector3(1, 1, 1);
    this._zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < cap; i++) this.mesh.setMatrixAt(i, this._zero);
    this.mesh.instanceMatrix.needsUpdate = true;
    this.active = 0;
  }

  spawn(x, y, z, vx, vy, vz, floorY, surface) {
    const i = this.i;
    this.i = (i + 1) % this.cap;
    const o = i * 12, r = this.rng;
    const S = this.s;
    S[o] = x; S[o + 1] = y; S[o + 2] = z;
    S[o + 3] = vx; S[o + 4] = vy; S[o + 5] = vz;
    S[o + 6] = 0; S[o + 7] = 0; S[o + 8] = 0; S[o + 9] = 1;
    S[o + 10] = floorY; S[o + 11] = 1;   // 1 = flying, 2 = at rest, 0 = free
    const w = i * 3;
    this.wx[w] = (r() - 0.5) * 46;
    this.wx[w + 1] = (r() - 0.5) * 40;
    this.wx[w + 2] = (r() - 0.5) * 52;
    this.age[i] = 0;
    this.surf[i] = surface;
    return i;
  }

  update(dt, onLand) {
    const S = this.s, W = this.wx;
    let any = false;
    for (let i = 0; i < this.cap; i++) {
      const o = i * 12;
      const st = S[o + 11];
      if (st === 0 || st === 3) continue;   // 3 = settled, matrix already final
      any = true;
      this.age[i] += dt;

      if (st === 1) {
        S[o + 4] -= GRAV * dt;
        // light air drag so short brass does not sail
        const d = 1 - 1.1 * dt;
        S[o + 3] *= d; S[o + 4] *= d; S[o + 5] *= d;
        S[o] += S[o + 3] * dt; S[o + 1] += S[o + 4] * dt; S[o + 2] += S[o + 5] * dt;

        const floor = S[o + 10] + 0.006;
        if (S[o + 1] <= floor && S[o + 4] < 0) {
          const impact = -S[o + 4];
          S[o + 1] = floor;
          S[o + 4] = impact * 0.34;
          S[o + 3] *= 0.55; S[o + 5] *= 0.55;
          W[i * 3] *= 0.5; W[i * 3 + 1] *= 0.5; W[i * 3 + 2] *= 0.5;
          if (impact > 0.55 && onLand) onLand(i, S[o], S[o + 1], S[o + 2], this.surf[i], Math.min(1, impact / 3.4));
          if (impact < 0.6) { S[o + 11] = 2; S[o + 4] = 0; S[o + 3] = 0; S[o + 5] = 0; }
        }
        // rotate
        const q = this._q.set(S[o + 6], S[o + 7], S[o + 8], S[o + 9]);
        const w3 = i * 3;
        this._dq.set(W[w3] * dt * 0.5, W[w3 + 1] * dt * 0.5, W[w3 + 2] * dt * 0.5, 1).normalize();
        q.multiply(this._dq).normalize();
        S[o + 6] = q.x; S[o + 7] = q.y; S[o + 8] = q.z; S[o + 9] = q.w;
      }

      // brass stays put once it settles; the pool recycles it, nothing fades
      this._p.set(S[o], S[o + 1], S[o + 2]);
      this._q.set(S[o + 6], S[o + 7], S[o + 8], S[o + 9]);
      this._m.compose(this._p, this._q, this._one);
      this.mesh.setMatrixAt(i, this._m);
      if (st === 2) S[o + 11] = 3;
    }
    if (any) this.mesh.instanceMatrix.needsUpdate = true;
  }

  dispose() { this.mesh.geometry.dispose(); this.mesh.material.dispose(); }
}

// ---------------------------------------------------------------------------
// Combat
// ---------------------------------------------------------------------------

export class Combat {
  constructor(scene, level, world) {
    this.scene = scene;
    this.level = level;
    this.world = world;
    this.time = 0;
    this._rng = makeRng('combat:' + WORLD_SEED);

    // three's Raycaster is kept only because the enemy hitbox API takes one
    this.ray = new THREE.Raycaster();
    this.ray.far = 300;

    this._uTime = { value: 0 };

    // ---- FX layer: five draw calls total
    this.atlas = buildDecalAtlas(this._rng.fork('decals'));
    this.fxA = new ParticleField(ALPHA_CAP, false, this._uTime, 10);
    this.fxB = new ParticleField(ADD_CAP, true, this._uTime, 11);
    this.decals = new DecalField(DECAL_CAP, this.atlas, this._uTime);
    this.tracers = new TracerField(TRACER_CAP, this._uTime);
    this.shells = new ShellPool(SHELL_CAP, this._rng.fork('shells'));
    scene.add(this.fxA.mesh, this.fxB.mesh, this.decals.mesh, this.tracers.mesh, this.shells.mesh);

    // Added once and never toggled: changing a light's visibility changes the
    // light count and recompiles every lit program in the scene.
    this.flash = new THREE.PointLight(0xffcc88, 0, 11, 2.0);
    this.flash.castShadow = false;
    scene.add(this.flash);
    this._flashT = 0;
    this._flashPeak = 0;

    // ---- scratch (never reallocated). Must exist before the proxy cache is
    // built — _buildProxyCache borrows _v and _mw.
    this._v = new THREE.Vector3();
    this._v2 = new THREE.Vector3();
    this._v3 = new THREE.Vector3();
    this._mw = new THREE.Matrix4();
    this._hit = {
      t: 0, exit: 0, px: 0, py: 0, pz: 0, nx: 0, ny: 1, nz: 0, obj: null, surface: 'concrete',
    };
    this._lastFloorSurface = 'sand';
    this._port = null;

    // ---- proxy cache for our own raycaster
    this._proxyN = 0;
    this._pc = null;
    this._buildProxyCache();

    // ---- projectile pool (position, velocity, damage scale, distance, pens)
    this._pj = new Float32Array(32 * 11);
    this._pjW = new Array(32).fill(null);   // weapon config reference per slot
    this._pjI = 0;
    this._pjLive = 0;

    // ---- pooled event payloads
    this._hitPool = [];
    for (let i = 0; i < 8; i++) {
      this._hitPool.push({
        point: new THREE.Vector3(), normal: new THREE.Vector3(), dir: new THREE.Vector3(),
        distance: 0, surface: 'concrete', victim: null, headshot: false, damage: 0,
        penetrated: false, material: 'concrete',
      });
    }
    this._hitI = 0;
    this._killPool = [];
    for (let i = 0; i < 4; i++) {
      this._killPool.push({
        victim: null, headshot: false, distance: 0,
        dir: new THREE.Vector3(), point: new THREE.Vector3(), weapon: null, penetrated: false,
      });
    }
    this._killI = 0;
    this._crackMsg = { point: new THREE.Vector3(), distance: 0, speed: 0 };
    this._shellMsg = { point: new THREE.Vector3(), surface: 'concrete', energy: 0 };

    // ai.js emits EV.KILL from enemy.damage(); we must not double-count it, so
    // we watch the bus and only emit our own richer payload if nobody else did.
    this._killEcho = 0;
    this._offKill = bus.on(EV.KILL, () => { this._killEcho++; });
    this._off = bus.on(EV.SHOT, (e) => this.onShot(e));

    this.stats = { shots: 0, hits: 0, pens: 0, drawCalls: 5, particlesLive: 0 };
    if (typeof window !== 'undefined') window.__combat = this;
  }

  // ------------------------------------------------------------------ proxies
  /** Cache every hitscan proxy as sphere + inverse matrix + local extents. */
  _buildProxyCache() {
    const list = this.level?.raycastables || [];
    const n = list.length;
    this._proxyN = n;
    this._objs = list;
    // 16 (inverse) + 3 (centre) + 1 (radius) + 3 (half extents) + 9 (world axes)
    const stride = 32;
    const pc = new Float32Array(n * stride);
    this._pcStride = stride;
    const pos = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
    const box = new THREE.Box3();
    for (let i = 0; i < n; i++) {
      const o = list[i];
      o.updateWorldMatrix(true, false);
      const m = o.matrixWorld;
      this._mw.copy(m).invert();
      const b = i * stride;
      for (let k = 0; k < 16; k++) pc[b + k] = this._mw.elements[k];
      const g = o.geometry;
      if (g && !g.boundingBox) g.computeBoundingBox();
      if (g && g.boundingBox) box.copy(g.boundingBox); else box.set(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5));
      const hx = (box.max.x - box.min.x) * 0.5, hy = (box.max.y - box.min.y) * 0.5, hz = (box.max.z - box.min.z) * 0.5;
      const cx = (box.max.x + box.min.x) * 0.5, cy = (box.max.y + box.min.y) * 0.5, cz = (box.max.z + box.min.z) * 0.5;
      m.decompose(pos, q, sc);
      // local centre -> world
      this._v.set(cx, cy, cz).applyMatrix4(m);
      pc[b + 16] = this._v.x; pc[b + 17] = this._v.y; pc[b + 18] = this._v.z;
      pc[b + 19] = Math.hypot(hx * sc.x, hy * sc.y, hz * sc.z) + 1e-3;
      pc[b + 20] = hx; pc[b + 21] = hy; pc[b + 22] = hz;
      pc[b + 23] = cx; pc[b + 24] = cy; pc[b + 25] = cz;
      const e = m.elements;
      const l0 = Math.hypot(e[0], e[1], e[2]) || 1;
      const l1 = Math.hypot(e[4], e[5], e[6]) || 1;
      const l2 = Math.hypot(e[8], e[9], e[10]) || 1;
      pc[b + 26] = e[0] / l0; pc[b + 27] = e[1] / l0; pc[b + 28] = e[2] / l0;
      pc[b + 29] = e[4] / l1; pc[b + 30] = e[5] / l1; pc[b + 31] = e[6] / l1;
      // third axis is reconstructed on demand as ax0 x ax1 (proxies are rigid)
    }
    this._pc = pc;
  }

  /**
   * Ray vs every proxy. Fills this._hit and returns entry distance, or -1.
   * Sphere reject first, then a slab test in the proxy's own space, which also
   * hands back the exit distance — that is the wall thickness, for free.
   */
  _castWorld(ox, oy, oz, dx, dy, dz, maxDist) {
    const pc = this._pc, stride = this._pcStride, n = this._proxyN;
    let bestT = maxDist, found = -1, bestExit = 0, bestAxis = 0, bestSign = 1;
    for (let i = 0; i < n; i++) {
      const b = i * stride;
      // --- bounding sphere reject
      const cx = pc[b + 16] - ox, cy = pc[b + 17] - oy, cz = pc[b + 18] - oz;
      const r = pc[b + 19];
      const proj = cx * dx + cy * dy + cz * dz;
      if (proj < -r || proj > bestT + r) continue;
      const d2 = cx * cx + cy * cy + cz * cz - proj * proj;
      if (d2 > r * r) continue;

      // --- ray into local space
      const lox = pc[b] * ox + pc[b + 4] * oy + pc[b + 8] * oz + pc[b + 12] - pc[b + 23];
      const loy = pc[b + 1] * ox + pc[b + 5] * oy + pc[b + 9] * oz + pc[b + 13] - pc[b + 24];
      const loz = pc[b + 2] * ox + pc[b + 6] * oy + pc[b + 10] * oz + pc[b + 14] - pc[b + 25];
      const ldx = pc[b] * dx + pc[b + 4] * dy + pc[b + 8] * dz;
      const ldy = pc[b + 1] * dx + pc[b + 5] * dy + pc[b + 9] * dz;
      const ldz = pc[b + 2] * dx + pc[b + 6] * dy + pc[b + 10] * dz;

      let tmin = -1e30, tmax = 1e30, axis = 0, sign = -1;
      // x slab
      let h = pc[b + 20];
      if (ldx > -1e-9 && ldx < 1e-9) { if (lox < -h || lox > h) continue; }
      else {
        const inv = 1 / ldx;
        let tn = (-h - lox) * inv, tf = (h - lox) * inv, s = -1;
        if (tn > tf) { const q = tn; tn = tf; tf = q; s = 1; }
        if (tn > tmin) { tmin = tn; axis = 0; sign = s; }
        if (tf < tmax) tmax = tf;
        if (tmin > tmax) continue;
      }
      // y slab
      h = pc[b + 21];
      if (ldy > -1e-9 && ldy < 1e-9) { if (loy < -h || loy > h) continue; }
      else {
        const inv = 1 / ldy;
        let tn = (-h - loy) * inv, tf = (h - loy) * inv, s = -1;
        if (tn > tf) { const q = tn; tn = tf; tf = q; s = 1; }
        if (tn > tmin) { tmin = tn; axis = 1; sign = s; }
        if (tf < tmax) tmax = tf;
        if (tmin > tmax) continue;
      }
      // z slab
      h = pc[b + 22];
      if (ldz > -1e-9 && ldz < 1e-9) { if (loz < -h || loz > h) continue; }
      else {
        const inv = 1 / ldz;
        let tn = (-h - loz) * inv, tf = (h - loz) * inv, s = -1;
        if (tn > tf) { const q = tn; tn = tf; tf = q; s = 1; }
        if (tn > tmin) { tmin = tn; axis = 2; sign = s; }
        if (tf < tmax) tmax = tf;
        if (tmin > tmax) continue;
      }

      if (tmax < 1e-4) continue;
      const enter = tmin > 1e-4 ? tmin : 1e-4;   // starting inside counts as a hit at 0
      if (enter >= bestT) continue;
      bestT = enter; bestExit = tmax; found = i; bestAxis = axis; bestSign = sign;
    }

    if (found < 0) return -1;
    const b = found * stride;
    const H = this._hit;
    H.t = bestT;
    H.exit = bestExit;
    H.px = ox + dx * bestT; H.py = oy + dy * bestT; H.pz = oz + dz * bestT;
    // world normal from the proxy's own axis
    let ax, ay, az;
    if (bestAxis === 0) { ax = pc[b + 26]; ay = pc[b + 27]; az = pc[b + 28]; }
    else if (bestAxis === 1) { ax = pc[b + 29]; ay = pc[b + 30]; az = pc[b + 31]; }
    else {
      ax = pc[b + 27] * pc[b + 31] - pc[b + 28] * pc[b + 30];
      ay = pc[b + 28] * pc[b + 29] - pc[b + 26] * pc[b + 31];
      az = pc[b + 26] * pc[b + 30] - pc[b + 27] * pc[b + 29];
    }
    H.nx = ax * bestSign; H.ny = ay * bestSign; H.nz = az * bestSign;
    H.obj = this._objs[found];
    H.surface = this.level.surfaceOf ? this.level.surfaceOf(H.obj) : 'concrete';
    return bestT;
  }

  /** Nearest live enemy hitbox along the segment, or null. */
  _castEnemies(ox, oy, oz, dx, dy, dz, maxDist) {
    const list = this.world?.enemies;
    if (!list || !list.length) return null;
    const R = this.ray.ray;
    R.origin.set(ox, oy, oz);
    R.direction.set(dx, dy, dz);
    let best = null, bestE = null;
    for (let i = 0; i < list.length; i++) {
      const e = list[i];
      if (!e || !e.alive || !e.raycast) continue;
      const h = e.raycast(this.ray);
      if (!h || h.distance > maxDist || h.distance < 0) continue;
      if (!best || h.distance < best.distance) { best = h; bestE = e; }
    }
    if (!best) return null;
    this._enemyHit = bestE;
    return best;
  }

  // ------------------------------------------------------------------ shot
  onShot(msg) {
    if (!msg) return;
    const { origin, dir, spread, weapon, muzzle } = msg;
    const cfg = weapon || {};
    const rng = this._rng;
    this.stats.shots++;

    // --- seeded cone spread in a basis perpendicular to the bore
    const dx0 = dir.x, dy0 = dir.y, dz0 = dir.z;
    let ux, uy, uz;
    if (Math.abs(dy0) < 0.9) { ux = -dz0; uy = 0; uz = dx0; }
    else { ux = 1; uy = 0; uz = 0; }
    let l = Math.hypot(ux, uy, uz) || 1;
    ux /= l; uy /= l; uz /= l;
    const vx = dy0 * uz - dz0 * uy, vy = dz0 * ux - dx0 * uz, vz = dx0 * uy - dy0 * ux;
    const s = (spread || 0) * 0.62;
    const a = rng.gauss() * s, b2 = rng.gauss() * s;
    let dx = dx0 + ux * a + vx * b2;
    let dy = dy0 + uy * a + vy * b2;
    let dz = dz0 + uz * a + vz * b2;
    l = Math.hypot(dx, dy, dz) || 1;
    dx /= l; dy /= l; dz /= l;

    // --- muzzle flash + brass, off the weapon's own attachment points
    const mz = this._muzzleWorld(muzzle, origin, dx, dy, dz);
    this._muzzleFlash(mz.x, mz.y, mz.z, dx, dy, dz, cfg);
    this._ejectShell(muzzle, mz.x, mz.y, mz.z, dx, dy, dz);

    // --- tracer on roughly one round in three, deterministic per shot index
    const tracer = (this.stats.shots % 3) === 1;

    // A round fired by anything other than the player gets full FX but never
    // damages the enemy list — otherwise AI crossfire would kill AI. Hostility
    // rides on the SIGN of the damage scale so it survives into the projectile
    // pool without widening the already-wide _resolve signature.
    const hostile = !!(msg.source && msg.source !== 'player');

    const mv = cfg.muzzleVelocity || 780;
    const instant = Math.min(45, Math.max(18, mv * 0.055));
    this._resolve(origin.x, origin.y, origin.z, dx, dy, dz, instant, cfg,
      hostile ? -1 : 1, 0, 0, tracer, mz.x, mz.y, mz.z, mv);
  }

  /**
   * Walk a segment, resolving hits and penetrations. If it runs out of segment
   * without hitting anything and the round still has range, hand it to a
   * projectile so the rest of the flight has drop and travel time.
   */
  _resolve(ox, oy, oz, dx, dy, dz, maxDist, cfg, signedScale, travelled, pens,
    tracer, tox, toy, toz, speed) {
    const hostile = signedScale < 0;
    const dmgScale = hostile ? -signedScale : signedScale;
    const gT = this._castWorld(ox, oy, oz, dx, dy, dz, maxDist);
    const geoT = gT < 0 ? maxDist : gT;
    const eh = this._castEnemies(ox, oy, oz, dx, dy, dz, geoT);

    // bullet-crack whip for anything that passes close to the player
    this._maybeCrack(ox, oy, oz, dx, dy, dz, gT < 0 ? maxDist : gT, speed);

    if (eh) {
      const enemy = this._enemyHit;
      const dist = travelled + eh.distance;
      const head = eh.name === 'head';
      const mult = head ? (cfg.headMult || eh.mult || 1.9) : (eh.mult || 1);
      const dmg = (cfg.damage || 25) * mult * falloff(cfg.falloff, dist) * dmgScale;
      const px = eh.point ? eh.point.x : ox + dx * eh.distance;
      const py = eh.point ? eh.point.y : oy + dy * eh.distance;
      const pz = eh.point ? eh.point.z : oz + dz * eh.distance;

      if (tracer) this._tracer(tox, toy, toz, px, py, pz, speed, true);
      this._impact('flesh', px, py, pz, -dx, -dy, -dz, dx, dy, dz, head ? 1.7 : 1.0);
      this.stats.hits++;

      // Stamp the hit on the victim so ai.js's own KILL emission carries it.
      enemy.lastHit = enemy.lastHit || { headshot: false, distance: 0, dirX: 0, dirY: 0, dirZ: 0, damage: 0, penetrated: false };
      enemy.lastHit.headshot = head;
      enemy.lastHit.distance = dist;
      enemy.lastHit.dirX = dx; enemy.lastHit.dirY = dy; enemy.lastHit.dirZ = dz;
      enemy.lastHit.damage = dmg;
      enemy.lastHit.penetrated = pens > 0;

      const echo = this._killEcho;
      const wasAlive = enemy.alive;
      if (!hostile) enemy.damage(dmg, head);
      const died = wasAlive && !enemy.alive;

      this._emitHit(px, py, pz, -dx, -dy, -dz, dx, dy, dz, dist, 'flesh', enemy, head, dmg, pens > 0);

      if (died && this._killEcho === echo) {
        const k = this._killPool[this._killI = (this._killI + 1) & 3];
        k.victim = enemy; k.headshot = head; k.distance = dist;
        k.dir.set(dx, dy, dz); k.point.set(px, py, pz);
        k.weapon = cfg; k.penetrated = pens > 0;
        bus.emit(EV.KILL, k);
      }
      return;
    }

    if (gT >= 0) {
      // Copy the hit out before anything else runs: _emitHit reaches the bus,
      // and a listener that casts a ray would otherwise overwrite it underneath us.
      const H = this._hit;
      const px = H.px, py = H.py, pz = H.pz;
      const hnx = H.nx, hny = H.ny, hnz = H.nz;
      const hExit = H.exit;
      const surface = H.surface || 'concrete';
      const dist = travelled + gT;
      const energy = Math.min(2.0, Math.max(0.45, (cfg.damage || 25) / 28)) * dmgScale;

      if (tracer) this._tracer(tox, toy, toz, px, py, pz, speed, true);
      this._impact(surface, px, py, pz, hnx, hny, hnz, dx, dy, dz, energy);
      this._decal(surface, px, py, pz, hnx, hny, hnz, energy);
      this._emitHit(px, py, pz, hnx, hny, hnz, dx, dy, dz, dist, surface, null, false, 0, pens > 0);

      // --- penetration
      const S = SURF[surface];
      const pen = S && S.pen;
      const thick = hExit - gT;
      if (!pen || pens >= MAX_PEN || thick > pen.max || dmgScale < MIN_PEN_DMG) return;
      const retain = Math.exp(-pen.soak * thick) * 0.92;
      const nextScale = dmgScale * retain;
      if (nextScale < MIN_PEN_DMG) return;
      this.stats.pens++;

      // exit side: a smaller spall puff, and the round is bent a little
      const ex = ox + dx * hExit, ey = oy + dy * hExit, ez = oz + dz * hExit;
      this._impact(surface, ex, ey, ez, dx, dy, dz, dx, dy, dz, energy * 0.55);
      this._decal(surface, ex, ey, ez, dx, dy, dz, energy * 0.8);

      const r = this._rng;
      let ndx = dx + r.gauss() * pen.deflect;
      let ndy = dy + r.gauss() * pen.deflect;
      let ndz = dz + r.gauss() * pen.deflect;
      const nl = Math.hypot(ndx, ndy, ndz) || 1;
      ndx /= nl; ndy /= nl; ndz /= nl;
      const rest = maxDist - hExit;
      const nextSigned = hostile ? -nextScale : nextScale;
      if (rest > 0.5) {
        this._resolve(ex + ndx * 0.01, ey + ndy * 0.01, ez + ndz * 0.01, ndx, ndy, ndz,
          rest, cfg, nextSigned, travelled + hExit, pens + 1, false, ex, ey, ez, speed);
      } else if (speed > 0) {
        // penetrated on the last centimetres of a projectile substep: let the
        // round carry on as a projectile rather than vanishing inside the wall
        this._spawnProjectile(ex + ndx * 0.01, ey + ndy * 0.01, ez + ndz * 0.01,
          ndx * speed, ndy * speed, ndz * speed, cfg, nextSigned, travelled + hExit, pens + 1);
      }
      return;
    }

    // --- nothing inside the instant window: continue as a real projectile
    if (tracer) {
      this._tracer(tox, toy, toz, ox + dx * 140, oy + dy * 140, oz + dz * 140, speed, false);
    }
    this._spawnProjectile(ox + dx * maxDist, oy + dy * maxDist, oz + dz * maxDist,
      dx * speed, dy * speed, dz * speed, cfg, signedScale, travelled + maxDist, pens);
  }

  _emitHit(px, py, pz, nx, ny, nz, dx, dy, dz, dist, surface, victim, head, dmg, penetrated) {
    const h = this._hitPool[this._hitI = (this._hitI + 1) & 7];
    h.point.set(px, py, pz);
    h.normal.set(nx, ny, nz);
    h.dir.set(dx, dy, dz);
    h.distance = dist;
    h.surface = surface;
    h.material = surface;
    h.victim = victim;
    h.headshot = head;
    h.damage = dmg;
    h.penetrated = penetrated;
    bus.emit(EV.HIT, h);
  }

  // ------------------------------------------------------------- projectiles
  _spawnProjectile(x, y, z, vx, vy, vz, cfg, dmgScale, travelled, pens) {
    const i = this._pjI;
    this._pjI = (i + 1) % 32;
    const o = i * 11;
    const P = this._pj;
    P[o] = x; P[o + 1] = y; P[o + 2] = z;
    P[o + 3] = vx; P[o + 4] = vy; P[o + 5] = vz;
    P[o + 6] = dmgScale; P[o + 7] = travelled; P[o + 8] = pens;
    P[o + 9] = 0;      // age
    P[o + 10] = 1;     // live
    this._pjW[i] = cfg;
    this._pjLive++;
  }

  _stepProjectiles(dt) {
    if (!this._pjLive) return;
    const P = this._pj;
    for (let i = 0; i < 32; i++) {
      const o = i * 11;
      if (P[o + 10] !== 1) continue;
      P[o + 9] += dt;
      // drag on a supersonic round is real; 0.0009 keeps a 780m/s round
      // subsonic by ~250m, which is well past anything the map can see
      const vx = P[o + 3], vy = P[o + 4], vz = P[o + 5];
      const sp = Math.hypot(vx, vy, vz) || 1;
      const drag = 1 - Math.min(0.5, 0.0009 * sp * dt);
      let nvx = vx * drag, nvy = vy * drag - GRAV * dt, nvz = vz * drag;
      const stepLen = Math.hypot(nvx, nvy, nvz) * dt;
      const dx = (nvx * dt) / (stepLen || 1), dy = (nvy * dt) / (stepLen || 1), dz = (nvz * dt) / (stepLen || 1);

      const cfg = this._pjW[i] || {};
      // Probe first, then retire the slot BEFORE resolving: a penetrating round
      // spawns a fresh projectile, and if the pool has wrapped that could be
      // this very slot — retiring afterwards would kill the new round instead.
      const gT = this._castWorld(P[o], P[o + 1], P[o + 2], dx, dy, dz, stepLen);
      const eh = this._castEnemies(P[o], P[o + 1], P[o + 2], dx, dy, dz, gT < 0 ? stepLen : gT);
      if (eh || gT >= 0) {
        const px = P[o], py = P[o + 1], pz = P[o + 2];
        const scale = P[o + 6], trav = P[o + 7], pens = P[o + 8];
        P[o + 10] = 0; this._pjW[i] = null; this._pjLive--;
        this._resolve(px, py, pz, dx, dy, dz, stepLen, cfg, scale, trav, pens, false, px, py, pz, sp);
        continue;
      }
      this._maybeCrack(P[o], P[o + 1], P[o + 2], dx, dy, dz, stepLen, sp);

      P[o] += nvx * dt; P[o + 1] += nvy * dt; P[o + 2] += nvz * dt;
      P[o + 3] = nvx; P[o + 4] = nvy; P[o + 5] = nvz;
      P[o + 7] += stepLen;
      const b = this.level?.bounds;
      const out = P[o + 9] > 3.0 ||
        (b && (P[o] < b.min.x - 8 || P[o] > b.max.x + 8 || P[o + 2] < b.min.z - 8 ||
          P[o + 2] > b.max.z + 8 || P[o + 1] < b.min.y - 4 || P[o + 1] > b.max.y + 30));
      if (out) { P[o + 10] = 0; this._pjW[i] = null; this._pjLive--; }
    }
  }

  /** Supersonic pass-by near the player. Audio owns the sound; we own the cue. */
  _maybeCrack(ox, oy, oz, dx, dy, dz, len, speed) {
    const p = this.world?.player?.pos;
    if (!p || speed < 340) return;
    const rx = p.x - ox, ry = p.y - oy, rz = p.z - oz;
    const t = rx * dx + ry * dy + rz * dz;
    if (t < 1.0 || t > len) return;
    const cx = rx - dx * t, cy = ry - dy * t, cz = rz - dz * t;
    const d2 = cx * cx + cy * cy + cz * cz;
    if (d2 > CRACK_RADIUS * CRACK_RADIUS) return;
    // the shooter's own muzzle is not a pass-by
    if (rx * rx + ry * ry + rz * rz < 36) return;
    const m = this._crackMsg;
    m.point.set(ox + dx * t, oy + dy * t, oz + dz * t);
    m.distance = Math.sqrt(d2);
    m.speed = speed;
    bus.emit(EV_CRACK, m);
  }

  // ------------------------------------------------------------------- FX
  _muzzleWorld(muzzle, origin, dx, dy, dz) {
    const v = this._v;
    if (muzzle && muzzle.isObject3D) {
      muzzle.updateWorldMatrix(true, false);
      v.setFromMatrixPosition(muzzle.matrixWorld);
    } else {
      v.set(origin.x + dx * 0.45, origin.y + dy * 0.45 - 0.06, origin.z + dz * 0.45);
    }
    return v;
  }

  _muzzleFlash(x, y, z, dx, dy, dz, cfg) {
    const r = this._rng, t = this.time;
    const seed = r();
    const scale = 0.9 + (cfg.damage ? Math.min(1.5, cfg.damage / 30) : 1) * 0.35;

    // two petal quads + a hot core; ~2 frames of life so it flickers, not glows
    this.fxB.emit(x, y, z, dx * 1.2, dy * 1.2, dz * 1.2,
      9.5, 5.2, 1.9, 0.30 * scale, 0.40 * scale, 0.038, 0, 6, 3, seed, t);
    this.fxB.emit(x + dx * 0.05, y + dy * 0.05, z + dz * 0.05, dx * 2.4, dy * 2.4, dz * 2.4,
      13.0, 7.5, 3.0, 0.16 * scale, 0.26 * scale, 0.028, 0, 6, 3, r(), t);
    this.fxB.emit(x, y, z, 0, 0.2, 0,
      6.0, 3.2, 1.1, 0.10 * scale, 0.05, 0.055, -0.4, 3, 0, r(), t);

    // a short spray of unburnt powder sparks
    for (let i = 0; i < 5; i++) {
      const sx = dx * (7 + r() * 9) + (r() - 0.5) * 3.2;
      const sy = dy * (7 + r() * 9) + (r() - 0.5) * 3.2;
      const sz = dz * (7 + r() * 9) + (r() - 0.5) * 3.2;
      this.fxB.emit(x, y, z, sx, sy, sz, 7.0, 2.6, 0.55,
        0.028, 0.006, 0.09 + r() * 0.06, 5.5, 5.0, 1, 0.05 + r() * 0.05, t);
    }

    // faint smoke, so sustained fire builds a haze at the muzzle
    this.fxA.emit(x + dx * 0.1, y + dy * 0.1, z + dz * 0.1, dx * 1.6, dy * 1.6 + 0.35, dz * 1.6,
      0.40, 0.38, 0.36, 0.06, 0.34, 0.42 + r() * 0.2, -0.5, 2.6, 0, r(), t);

    this.flash.position.set(x, y, z);
    this._flashPeak = 55 * scale;
    this._flashT = 0.05;
    this.flash.intensity = this._flashPeak;
  }

  _ejectShell(muzzle, mx, my, mz, dx, dy, dz) {
    let px = mx - dx * 0.36, py = my - dy * 0.36 + 0.02, pz = mz - dz * 0.36;
    let rx = 1, ry = 0, rz = 0;
    if (muzzle && muzzle.parent) {
      const port = this._port || (this._port = muzzle.parent.getObjectByName('ejectPort'));
      if (port) {
        port.updateWorldMatrix(true, false);
        this._v2.setFromMatrixPosition(port.matrixWorld);
        px = this._v2.x; py = this._v2.y; pz = this._v2.z;
        // right-hand ejection: the port's own +X in world space
        const e = port.matrixWorld.elements;
        const l = Math.hypot(e[0], e[1], e[2]) || 1;
        rx = e[0] / l; ry = e[1] / l; rz = e[2] / l;
      }
    }
    const r = this._rng;
    const floorY = this._floorUnder(px, py, pz);
    const vx = rx * (2.4 + r() * 1.1) + dx * 0.5 + (r() - 0.5) * 0.5;
    const vy = ry * (2.4 + r() * 1.1) + 1.35 + r() * 0.5;
    const vz = rz * (2.4 + r() * 1.1) + dz * 0.5 + (r() - 0.5) * 0.5;
    this.shells.spawn(px, py, pz, vx, vy, vz, floorY, this._lastFloorSurface);
  }

  _floorUnder(x, y, z) {
    const t = this._castWorld(x, y + 0.05, z, 0, -1, 0, 6);
    if (t < 0) { this._lastFloorSurface = 'sand'; return 0; }
    this._lastFloorSurface = this._hit.surface;
    return this._hit.py;
  }

  _tracer(x, y, z, tx, ty, tz, speed, clipped) {
    let dx = tx - x, dy = ty - y, dz = tz - z;
    const len = Math.hypot(dx, dy, dz) || 1;
    dx /= len; dy /= len; dz /= len;
    const life = Math.min(0.42, 0.055 + len / Math.max(120, speed));
    this.tracers.fire(x, y, z, dx, dy, dz, clipped ? len : 150,
      Math.max(190, speed * 0.72), 5.6, 2.4, 0.7, life, this.time);
  }

  /** Per-surface impact burst. Everything comes out of the two shared fields. */
  _impact(surface, px, py, pz, nx, ny, nz, dx, dy, dz, energy) {
    const S = SURF[surface] || SURF.concrete;
    const r = this._rng, t = this.time;
    const e = Math.min(2.2, Math.max(0.35, energy));

    // reflected direction — chips and sparks come off the ricochet, not the normal
    const dot = dx * nx + dy * ny + dz * nz;
    const rxv = dx - 2 * dot * nx, ryv = dy - 2 * dot * ny, rzv = dz - 2 * dot * nz;

    if (S.fx === 'spark') {
      const n = (S.sparks * e) | 0;
      for (let i = 0; i < n; i++) {
        const sp = 4.5 + r() * 11 * e;
        const jx = rxv + (r() - 0.5) * 1.35, jy = ryv + (r() - 0.5) * 1.35, jz = rzv + (r() - 0.5) * 1.35;
        const l = Math.hypot(jx, jy, jz) || 1;
        this.fxB.emit(px + nx * 0.02, py + ny * 0.02, pz + nz * 0.02,
          (jx / l) * sp, (jy / l) * sp, (jz / l) * sp,
          8.0, 2.9, 0.5, 0.030, 0.008, 0.16 + r() * 0.24, 9.0, 2.2, 1, 0.06 + r() * 0.06, t);
      }
      // hot flash at the point of contact
      this.fxB.emit(px + nx * 0.02, py + ny * 0.02, pz + nz * 0.02, 0, 0, 0,
        6.5, 3.0, 0.9, 0.10 * e, 0.02, 0.05, 0, 6, 0, r(), t);
      // thin grey smoke off the strike
      this.fxA.emit(px + nx * 0.05, py + ny * 0.05, pz + nz * 0.05, nx * 0.5, ny * 0.5 + 0.5, nz * 0.5,
        0.28, 0.27, 0.26, 0.05, 0.28 * e, 0.5, -0.6, 2.4, 0, r(), t);
    } else if (S.fx === 'blood') {
      const n = (7 * e) | 0;
      for (let i = 0; i < n; i++) {
        const sp = 1.2 + r() * 3.2 * e;
        const jx = dx * 0.6 + (r() - 0.5) * 1.5, jy = dy * 0.6 + (r() - 0.5) * 1.5 + 0.3, jz = dz * 0.6 + (r() - 0.5) * 1.5;
        const l = Math.hypot(jx, jy, jz) || 1;
        this.fxA.emit(px, py, pz, (jx / l) * sp, (jy / l) * sp, (jz / l) * sp,
          S.dust[0], S.dust[1], S.dust[2],
          0.035 + r() * 0.05, 0.12 + r() * 0.09 * e, 0.30 + r() * 0.16, 1.6, 3.4, 0, r(), t);
      }
      // one denser puff so the hit reads at range
      this.fxA.emit(px - dx * 0.05, py - dy * 0.05, pz - dz * 0.05, -dx * 0.8, -dy * 0.8 + 0.3, -dz * 0.8,
        0.34, 0.035, 0.03, 0.06 * e, 0.30 * e, 0.24, 0.8, 3.0, 0, r(), t);
    } else {
      // dust / poof / splinter / glass all share the puff+debris shape
      const puffs = S.fx === 'poof' ? 3 : 4;
      const rise = S.fx === 'poof' ? -0.15 : -0.9;   // negative gravity = it lifts
      const spd = S.fx === 'poof' ? 0.7 : 1.5;
      for (let i = 0; i < puffs; i++) {
        const jx = nx * (0.6 + r() * 0.9) + (r() - 0.5) * 0.85;
        const jy = ny * (0.6 + r() * 0.9) + (r() - 0.5) * 0.85;
        const jz = nz * (0.6 + r() * 0.9) + (r() - 0.5) * 0.85;
        const k = 0.85 + r() * 0.4;
        this.fxA.emit(px + nx * 0.03, py + ny * 0.03, pz + nz * 0.03,
          jx * spd * e, jy * spd * e, jz * spd * e,
          S.dust[0] * k, S.dust[1] * k, S.dust[2] * k,
          0.05 + r() * 0.05, (0.34 + r() * 0.30) * e, 0.55 + r() * 0.45,
          rise, 2.5, 0, r(), t);
      }
      const chips = (S.chips * e) | 0;
      for (let i = 0; i < chips; i++) {
        const sp = 2.2 + r() * 6.5 * e;
        const jx = rxv + (r() - 0.5) * 1.5, jy = ryv + (r() - 0.5) * 1.5 + 0.35, jz = rzv + (r() - 0.5) * 1.5;
        const l = Math.hypot(jx, jy, jz) || 1;
        const k = 0.8 + r() * 0.5;
        this.fxA.emit(px + nx * 0.02, py + ny * 0.02, pz + nz * 0.02,
          (jx / l) * sp, (jy / l) * sp, (jz / l) * sp,
          S.chip[0] * k, S.chip[1] * k, S.chip[2] * k,
          0.014 + r() * 0.016, 0.010 + r() * 0.012, 0.55 + r() * 0.6,
          GRAV, 0.35, 2, r(), t);
      }
      if (S.fx === 'glass') {
        // a couple of bright shards catching the key light
        for (let i = 0; i < 5; i++) {
          const sp = 2.5 + r() * 6;
          const jx = rxv + (r() - 0.5) * 1.6, jy = ryv + (r() - 0.5) * 1.6 + 0.4, jz = rzv + (r() - 0.5) * 1.6;
          const l = Math.hypot(jx, jy, jz) || 1;
          this.fxB.emit(px, py, pz, (jx / l) * sp, (jy / l) * sp, (jz / l) * sp,
            1.3, 1.6, 1.7, 0.014, 0.008, 0.5 + r() * 0.5, GRAV, 0.3, 2, r(), t);
        }
      }
    }
  }

  _decal(surface, px, py, pz, nx, ny, nz, energy) {
    const S = SURF[surface] || SURF.concrete;
    const r = this._rng;
    const cell = S.cell + (S.cellN > 1 ? (r() < 0.5 ? 0 : 1) : 0);
    const cx = cell % ATLAS_N, cy = (cell / ATLAS_N) | 0;

    // tangent: any axis not parallel to the normal, rolled by a seeded angle
    let tx, ty, tz;
    if (Math.abs(ny) < 0.9) { tx = -nz; ty = 0; tz = nx; }
    else { tx = 1; ty = 0; tz = 0; }
    let l = Math.hypot(tx, ty, tz) || 1;
    tx /= l; ty /= l; tz /= l;
    const bx = ny * tz - nz * ty, by = nz * tx - nx * tz, bz = nx * ty - ny * tx;
    const ang = r() * Math.PI * 2;
    const c = Math.cos(ang), s = Math.sin(ang);
    const rtx = tx * c + bx * s, rty = ty * c + by * s, rtz = tz * c + bz * s;

    // Quad size, not hole size: the hole core is ~30% of the quad and the crack
    // field fills the rest, so 0.34m of quad reads as a ~10cm hole with spall.
    const size = (surface === 'glass' ? 0.8 : 0.34) * (0.8 + energy * 0.35) * (0.85 + r() * 0.3);
    const T = S.tint;
    this.decals.place(px, py, pz, nx, ny, nz, rtx, rty, rtz, size, cx, cy,
      T[0], T[1], T[2], DECAL_LIFE, this.time);
  }

  // ------------------------------------------------------------------ frame
  update(dt) {
    this.time += dt;
    this._uTime.value = this.time;

    // the level can be rebuilt between waves; keep the proxy cache honest
    const rc = this.level?.raycastables;
    if (rc && rc.length !== this._proxyN) this._buildProxyCache();

    if (this._flashT > 0) {
      this._flashT -= dt;
      const u = Math.max(0, this._flashT / 0.05);
      this.flash.intensity = this._flashPeak * u * u;
      if (this._flashT <= 0) this.flash.intensity = 0;
    }

    this._stepProjectiles(dt);

    this.shells.update(dt, this._onShellLand || (this._onShellLand =
      (i, x, y, z, surface, energy) => {
        const m = this._shellMsg;
        m.point.set(x, y, z);
        m.surface = surface;
        m.energy = energy;
        bus.emit(EV_SHELL, m);
      }));

    this.fxA.flush();
    this.fxB.flush();
    this.decals.flush();
    this.tracers.flush();
    this.stats.particlesLive = this.fxA.used + this.fxB.used;
  }

  dispose() {
    this._off?.();
    this._offKill?.();
    for (const f of [this.fxA, this.fxB, this.decals, this.tracers]) {
      this.scene.remove(f.mesh);
      f.dispose();
    }
    this.scene.remove(this.shells.mesh);
    this.shells.dispose();
    this.scene.remove(this.flash);
    this.atlas.dispose();
    if (typeof window !== 'undefined' && window.__combat === this) window.__combat = null;
  }
}

export function falloff(table, dist) {
  if (!table || !table.length) return 1;
  for (let i = 1; i < table.length; i++) {
    if (dist <= table[i][0]) {
      const [d0, m0] = table[i - 1], [d1, m1] = table[i];
      const t = (dist - d0) / Math.max(1e-6, d1 - d0);
      return m0 + (m1 - m0) * t;
    }
  }
  return table[table.length - 1][1];
}
