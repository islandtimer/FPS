// OWNER: agent "tex" — procedural PBR texture synthesis. NO image files, ever.
// CONTRACT:
//   makeTextureSet(name, opts) -> { map, normalMap, roughnessMap, metalnessMap?, aoMap? }
//   TEXTURE_NAMES -> string[]  (names main/level/props may request)
//   textureBytes() -> approximate VRAM in bytes, for the perf budget
//
// HOW IT WORKS
// ------------
// Everything is synthesised on the GPU in two fragment passes per material.
//
//   pass A "fields"  -> RGBA16F   R = height, G/B = material masks, A = macro var
//   pass B "compose" -> RGBA8 x3  albedo (sRGB) / tangent normal / ORM packed
//
// The point of splitting it that way is correspondence: the normal, the ambient
// occlusion, the cavity grime and the albedo speckle are all read back out of the
// SAME height field, so a bump you can see in the normal map is a bump you can see
// in the colour. That correspondence is most of what reads as "real".
//
// Tiling is beaten three ways:
//   1. every noise primitive is periodic on an INTEGER lattice, and octave
//      frequencies are round(f * 2.17^i) — non-power-of-two ratios, so the
//      internal repeat of the pattern never lines up with itself inside the tile;
//   2. domain warping (which preserves the period, since the warp field is itself
//      periodic) destroys the axis-aligned feel of the lattice;
//   3. materials.js layers a low-frequency macro-variation map at 1/4.31 the
//      tile frequency over the top — an 8.6 m field on a 2 m tile, carrying
//      independent value and hue axes — which breaks the repeat ACROSS tiles
//      and supplies the 4-8 m band no tile can hold.
//
// Output is read back to DataTextures rather than kept as render-target textures:
// that makes them cheap to clone for different repeat values (clones share the
// GPU upload via Texture.source), and it makes the VRAM accounting exact.
//
// TEXEL DENSITY — read this before changing any frequency
// ------------------------------------------------------
// A frequency in here is meaningless without knowing how much WORLD the tile
// covers. level.js owns that (its UVS table, repeat 1 everywhere), so those
// numbers are mirrored below and every generator quotes its features in
// millimetres, not in cycles.
//
//   family    uv/m    tile span   texel @512   texel @1024
//   plaster   0.50    2.000 m     3.91 mm      1.95 mm  <- baked at 1024
//   concrete  0.50    2.000 m     3.91 mm      —
//   road      0.40    2.500 m     4.88 mm      —      (same bake as concrete)
//   brick     0.62    1.613 m     3.15 mm      —
//   ground    0.34    2.941 m     5.75 mm      —
//   wood      0.85    1.176 m     2.30 mm      1.15 mm  <- baked at 1024
//   metal     0.90    1.111 m     2.17 mm      —
//   fabric    1.10    0.909 m     1.78 mm      —
//
// Nothing narrower than about 1.5 texels survives the mip chain, so features
// are sized against the table above rather than picked by eye.
//
// HEIGHT UNITS. The compose pass turns the height field into a normal with
// slope = dh_over_two_texels * uBump, which means one unit of height equals
// (2 * texelSize * bump) of world. For plaster that is 2 * 3.91mm * 11 = 86 mm,
// so a 1 mm hairline crack is a height step of 0.012 — not 0.30. Getting this
// wrong is exactly how a hairline crack ends up reading as a metre-wide canyon.

import * as THREE from 'three';
import { hash32 } from '../core/rng.js';

export const TEXTURE_NAMES = [
  'concrete', 'plaster', 'sand', 'asphalt', 'rustMetal', 'paintedMetal',
  'wood', 'fabric', 'gunmetal', 'polymer', 'rubber', 'glass', 'tile', 'brick',
];

// id must match the `#if MAT ==` blocks in the shaders below.
const MAT_ID = {
  concrete: 0, plaster: 1, sand: 2, asphalt: 3, rustMetal: 4, paintedMetal: 5,
  gunmetal: 6, polymer: 7, wood: 8, fabric: 9, rubber: 10, glass: 11,
  tile: 12, brick: 13,
};

// Per-material bake settings.
//   size  albedo/normal resolution (ORM bakes at half — roughness/AO are
//         low-frequency and half res is free quality)
//   bump  height -> normal slope gain, at 512. Scaled with resolution so the
//         perceived relief does not change when the size changes.
//   ao    occlusion gain,  cav  cavity-grime gain,  det  fine-detail gain
const BAKE = {
  concrete:     { size: 512, bump: 13.0, ao: 3.4, cav: 26.0, det: 30.0 },
  // plaster is the hero surface — it is most of every wall in the level, and at
  // 512 over a 2 m tile a 6 mm hairline crack is 1.5 texels, which aliases into
  // a fat grey line before it ever reaches the mip chain. 1024 buys 1.95 mm
  // texels, which is what makes the craquelure read as craquelure.
  plaster:      { size: 1024, bump: 11.0, ao: 2.1, cav: 24.0, det: 40.0 },
  sand:         { size: 512, bump:  9.5, ao: 2.1, cav: 18.0, det: 46.0 },
  asphalt:      { size: 512, bump: 15.0, ao: 3.8, cav: 24.0, det: 24.0 },
  rustMetal:    { size: 512, bump: 14.0, ao: 3.6, cav: 22.0, det: 22.0 },
  paintedMetal: { size: 512, bump: 10.0, ao: 3.0, cav: 30.0, det: 40.0 },
  gunmetal:     { size: 512, bump:  9.0, ao: 2.4, cav: 34.0, det: 60.0 },
  polymer:      { size: 512, bump: 11.0, ao: 2.6, cav: 30.0, det: 46.0 },
  // timber at 512 could not hold a ring finer than ~8 mm (2.3 mm texels, and a
  // ring needs three or four of them or it mips straight to flat tone). Real
  // softwood grows at 3-8 mm, so the board direction — the single thing that
  // makes sawn timber read as sawn timber — was being lost. 1024 buys 1.15 mm
  // texels and with them a 4.3 mm ring at 3.7 texels.
  wood:         { size: 1024, bump: 12.0, ao: 3.0, cav: 26.0, det: 34.0 },
  fabric:       { size: 512, bump: 14.0, ao: 3.0, cav: 22.0, det: 30.0 },
  rubber:       { size: 512, bump: 11.0, ao: 2.6, cav: 30.0, det: 46.0 },
  glass:        { size: 512, bump:  6.0, ao: 1.4, cav: 40.0, det: 60.0 },
  tile:         { size: 512, bump: 13.0, ao: 3.4, cav: 24.0, det: 40.0 },
  brick:        { size: 512, bump: 13.0, ao: 2.4, cav: 16.0, det: 30.0 },
};

// ---------------------------------------------------------------- GLSL library

const GLSL_LIB = /* glsl */`
#define TAU 6.283185307179586

uint uhash(uvec2 c, uint s) {
  uint h = c.x * 1597334677u ^ c.y * 3812015801u ^ s * 2654435761u;
  h ^= h >> 15; h *= 2246822519u;
  h ^= h >> 13; h *= 3266489917u;
  h ^= h >> 16;
  return h;
}

// Cell hashes wrap the lattice index modulo the period, which is what makes
// every field below tile EXACTLY at uv period 1 for integer frequencies.
float hf(vec2 c, vec2 per, uint s) {
  uvec2 ic = uvec2(mod(c, per) + 0.5);
  return float(uhash(ic, s)) * (1.0 / 4294967296.0);
}
vec2 hf2(vec2 c, vec2 per, uint s) {
  uvec2 ic = uvec2(mod(c, per) + 0.5);
  uint h = uhash(ic, s);
  return vec2(float(h & 65535u), float((h >> 16) & 65535u)) * (1.0 / 65535.0);
}

float gnoise(vec2 p, vec2 per, uint s) {
  vec2 i = floor(p), f = p - i;
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float a0 = hf(i,              per, s) * TAU;
  float a1 = hf(i + vec2(1, 0), per, s) * TAU;
  float a2 = hf(i + vec2(0, 1), per, s) * TAU;
  float a3 = hf(i + vec2(1, 1), per, s) * TAU;
  float v0 = dot(vec2(cos(a0), sin(a0)), f);
  float v1 = dot(vec2(cos(a1), sin(a1)), f - vec2(1, 0));
  float v2 = dot(vec2(cos(a2), sin(a2)), f - vec2(0, 1));
  float v3 = dot(vec2(cos(a3), sin(a3)), f - vec2(1, 1));
  return mix(mix(v0, v1, u.x), mix(v2, v3, u.x), u.y) * 1.4;
}
float gn(vec2 uv, vec2 f, uint s) { return gnoise(uv * f, f, s); }

// Incommensurate octave ratio (2.17) rounded back to integers each step: keeps
// the lattice periodic while making the octaves refuse to line up.
float fbm(vec2 uv, vec2 f0, int oct, float gain, uint s) {
  float sum = 0.0, amp = 1.0, nrm = 0.0;
  vec2 f = f0;
  for (int i = 0; i < 7; i++) {
    if (i >= oct) break;
    sum += amp * gn(uv, f, s + uint(i) * 6151u);
    nrm += amp; amp *= gain;
    f = floor(f * 2.17 + 0.5);
  }
  return sum / max(nrm, 1e-5);
}
float fbm01(vec2 uv, vec2 f0, int oct, float gain, uint s) {
  return fbm(uv, f0, oct, gain, s) * 0.5 + 0.5;
}

float rfbm(vec2 uv, vec2 f0, int oct, uint s) {
  float sum = 0.0, amp = 1.0, nrm = 0.0;
  vec2 f = f0;
  for (int i = 0; i < 7; i++) {
    if (i >= oct) break;
    float n = 1.0 - abs(gn(uv, f, s + uint(i) * 7561u));
    sum += amp * n * n; nrm += amp; amp *= 0.55;
    f = floor(f * 2.17 + 0.5);
  }
  return sum / max(nrm, 1e-5);
}

vec2 dwarp(vec2 uv, vec2 f, float amt, uint s) {
  return uv + amt * vec2(fbm(uv, f, 3, 0.5, s), fbm(uv, f, 3, 0.5, s + 33331u));
}

// x = F1, y = F2, z = per-cell random. F2-F1 gives the crack/edge network.
vec3 worley(vec2 uv, vec2 f, float jitter, uint s) {
  vec2 p = uv * f;
  vec2 ip = floor(p), fp = p - ip;
  float f1 = 8.0, f2 = 8.0, id = 0.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 g = vec2(float(x), float(y));
      vec2 o = hf2(ip + g, f, s);
      vec2 r = g + 0.5 + (o - 0.5) * jitter - fp;
      float d = dot(r, r);
      if (d < f1) { f2 = f1; f1 = d; id = hf(ip + g, f, s + 991u); }
      else if (d < f2) { f2 = d; }
    }
  }
  return vec3(sqrt(f1), sqrt(f2), id);
}
`;

const VERT = /* glsl */`
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// ---------------------------------------------------------------- pass A: fields

const FIELDS_FRAG = /* glsl */`
precision highp float;
precision highp int;
in vec2 vUv;
out vec4 fragColor;
uniform float uSeed;
${GLSL_LIB}

void main() {
  uint S = uint(uSeed);
  vec2 uv = vUv;
  vec4 F = vec4(0.5, 0.0, 0.0, 0.5);

#if MAT == 0
  // ---- cast concrete. TILE 2.00 m; one height unit is ~102 mm, so every depth
  // below is written as millimetres/102.
  vec2 wq = dwarp(uv, vec2(3.0), 0.06, S + 11u);
  float macro  = smoothstep(0.33, 0.67, fbm01(wq, vec2(3.0), 4, 0.55, S + 2u));
  float cement = fbm(wq, vec2(23.0), 3, 0.5, S + 3u);
  float micro  = fbm(uv, vec2(97.0), 2, 0.5, S + 4u);
  vec3  agg    = worley(dwarp(uv, vec2(11.0), 0.02, S + 9u), vec2(58.0), 0.95, S + 7u);
  float stone  = 1.0 - smoothstep(0.02, 0.36, agg.x);

  // Formwork. 29 cm boards leave a shallow groove and a small step at every
  // joint, and no two boards sit at quite the same depth. On a big flat face
  // this is the single strongest "poured" cue there is.
  float bv    = uv.y * 7.0;
  float bJ    = abs(fract(bv) - 0.5);
  float seamF = 1.0 - smoothstep(0.012, 0.045, bJ);
  float bStep = (hf(vec2(0.5, floor(bv)), vec2(1.0, 7.0), S + 61u) - 0.5) * 0.013;

  float spallF = fbm01(dwarp(uv, vec2(5.0), 0.10, S + 21u), vec2(4.0), 4, 0.55, S + 5u);
  float spall  = smoothstep(0.595, 0.655, spallF);
  vec3  ph     = worley(uv, vec2(86.0), 1.0, S + 31u);
  float holes  = smoothstep(0.16, 0.02, ph.x) * step(0.87, ph.z);
  // Water staining. The noise is stretched ~9:1 vertically, then gated by a
  // periodic ramp that is zero at both ends of the tile: the run therefore has
  // a definite top (a sill) and dies out about a metre below it, without the
  // seam a real top-to-bottom gradient would leave at the tile edge.
  float streak = fbm01(uv, vec2(37.0, 4.0), 3, 0.6, S + 13u);
  float rPh    = fbm01(uv, vec2(3.0, 2.0), 2, 0.5, S + 27u);
  float rQ     = fract(uv.y + rPh * 0.8);
  float rEnv   = (1.0 - smoothstep(0.58, 0.66, rQ)) * smoothstep(0.02, 0.26, rQ)
               * (0.40 + 0.60 * smoothstep(0.04, 0.52, rQ));
  float src    = smoothstep(0.42, 0.74, fbm01(uv, vec2(7.0, 3.0), 3, 0.5, S + 17u));
  float run    = smoothstep(0.34, 0.78, streak) * src * rEnv;
  float grime  = fbm01(wq, vec2(9.0), 4, 0.5, S + 19u);
  float h = 0.5 + cement * 0.030 + micro * 0.008 + bStep
          - seamF * 0.015
          - spall * 0.055 + spall * stone * 0.040
          - holes * 0.045 + (macro - 0.5) * 0.024;
  F = vec4(h, spall * (0.35 + stone * 0.65),
           clamp(run * 0.90 + grime * 0.20 + seamF * 0.28, 0.0, 1.0), macro);

#elif MAT == 1
  // ---- lime/cement render on masonry. TILE 2.00 m at 1024 px => 1.95 mm/texel,
  // one height unit ~86 mm. Everything below is quoted in millimetres.
  vec2 wq = dwarp(uv, vec2(3.0), 0.05, S + 3u);

  // (a) MACRO TONE. 25 cm - 1 m patchiness: skim coats, old repairs, sun
  //     bleaching. materials.js lays a second ~8.6 m field over the top of this,
  //     so together they cover the whole 0.25-9 m band. Without both, a 6 m wall
  //     is one flat beige at 40 m no matter how good the 5 cm detail is.
  // contrast-stretched: raw fbm clusters hard around 0.5, so feeding it
  // straight into a colour ramp uses about a third of the range asked for and
  // the wall comes back flat however wide the ramp is
  float macro = smoothstep(0.34, 0.66, fbm01(wq, vec2(2.0), 4, 0.60, S + 2u));

  // (b) TROWEL WORK. 12 cm float sweeps at ~2 mm, 1 mm stipple on top.
  float roll    = fbm(dwarp(uv, vec2(7.0), 0.03, S + 41u), vec2(17.0), 3, 0.55, S + 6u);
  float stipple = fbm(uv, vec2(150.0), 2, 0.5, S + 5u);

  // (c) CRAQUELURE. 26 cells across a 2 m tile is 7.7 cm map cracking; the
  //     second net at 47 cells is 4.3 cm hairlines. Thresholds are in CELL
  //     units, so 0.085 of a 7.7 cm cell is a 6.5 mm line — three texels at
  //     1024, which is the narrowest thing that survives mipping.
  vec3 c1 = worley(dwarp(uv, vec2(11.0), 0.016, S + 11u), vec2(26.0), 1.0, S + 12u);
  vec3 c2 = worley(dwarp(uv, vec2(23.0), 0.008, S + 13u), vec2(47.0), 1.0, S + 14u);
  float crack = smoothstep(0.085, 0.006, c1.y - c1.x)
              + smoothstep(0.110, 0.010, c2.y - c2.x) * 0.42;
  crack *= smoothstep(0.30, 0.68, fbm01(uv, vec2(3.0), 3, 0.5, S + 15u));
  crack = clamp(crack, 0.0, 1.0);

  // (d) SPALL. The finish coat lets go in patches and the coarse base render
  //     shows through. The edge is hard because it breaks rather than fades,
  //     the loss is ~3 mm, and the intact coat stands proud around the hole so
  //     one side of every patch catches the key and the other side occludes.
  // Thresholds set by measured COVERAGE, not by eye: at 0.606 the coat had let
  // go over a quarter of the wall and the tile read as splashed white paint.
  // Around 8% is what a render coat that has been up for thirty years looks
  // like — enough to break the surface, not enough to become the surface.
  float spallF = fbm01(dwarp(uv, vec2(7.0), 0.07, S + 21u), vec2(9.0), 3, 0.50, S + 22u)
               + crack * 0.030;
  float spall  = smoothstep(0.666, 0.692, spallF);
  float lip    = smoothstep(0.640, 0.664, spallF) * (1.0 - smoothstep(0.664, 0.684, spallF));
  // the substrate has its own, much coarser grain — 2 cm aggregate, carried in
  // the height so the normal map and the albedo speckle agree about where it is.
  // Kept shallow: a deep grit field drives the baked AO to its floor across the
  // whole patch, which turns the exposed render into a dark blot instead of the
  // lighter, coarser thing it is.
  vec3  sg   = worley(dwarp(uv, vec2(23.0), 0.01, S + 33u), vec2(98.0), 1.0, S + 31u);
  float grit = smoothstep(0.30, 0.06, sg.x);

  // (e) RUNOFF. Same periodic-ramp trick as concrete: a hard top edge where a
  //     sill would be, then ~1.2 m of tail that fades out inside the tile.
  float rPh  = fbm01(uv, vec2(3.0, 2.0), 2, 0.5, S + 27u);
  float rQ   = fract(uv.y + rPh * 0.8);
  float rEnv = (1.0 - smoothstep(0.60, 0.67, rQ)) * smoothstep(0.02, 0.28, rQ)
             * (0.35 + 0.65 * smoothstep(0.05, 0.55, rQ));
  float rCol = smoothstep(0.38, 0.78, fbm01(uv, vec2(6.0, 2.0), 3, 0.5, S + 25u));
  float run  = smoothstep(0.44, 0.86, fbm01(uv, vec2(52.0, 5.0), 3, 0.6, S + 23u)) * rEnv * rCol;

  float dirt = smoothstep(0.50, 0.80, fbm01(wq, vec2(9.0), 4, 0.55, S + 19u));

  float h = 0.5
          + roll * 0.024              // 2.1 mm trowel undulation
          + stipple * 0.010           // 0.9 mm stipple
          - crack * 0.016             // 1.4 mm hairline, not a 26 mm trench
          - spall * 0.031             // 2.7 mm of finish coat gone
          + lip * 0.012               // the broken edge stands proud
          + spall * grit * 0.0065     // exposed aggregate inside the patch
          + (macro - 0.5) * 0.010;
  F = vec4(h, spall, clamp(crack * 0.80 + run * 1.00 + dirt * 0.45, 0.0, 1.0), macro);

#elif MAT == 2
  // ---- sand / dry earth: dune scale, wind ripples, pebble scatter, grain
  float dune = fbm(uv, vec2(2.0), 4, 0.55, S + 2u);
  // Two ripple trains at different bearings and wavelengths, blended by a
  // low-frequency field. One train alone reads as corduroy from 20m out.
  vec2  rw = dwarp(uv, vec2(4.0), 0.05, S + 4u);
  float jit = fbm(uv, vec2(6.0), 3, 0.5, S + 5u) * 3.0;
  float r1 = sin(TAU * (rw.x * 23.0 + rw.y * 9.0) + jit) * 0.5 + 0.5;
  float r2 = sin(TAU * (rw.x * 7.0 - rw.y * 17.0) - jit * 0.7) * 0.5 + 0.5;
  float bearing = smoothstep(0.36, 0.64, fbm01(uv, vec2(2.0), 3, 0.5, S + 21u));
  float ripple = mix(pow(r1, 1.7), pow(r2, 2.1), bearing);
  ripple *= smoothstep(0.20, 0.62, fbm01(uv, vec2(3.0), 3, 0.5, S + 6u) + 0.22);
  vec3  peb    = worley(uv, vec2(64.0), 1.0, S + 7u);
  float pebble = smoothstep(0.24, 0.11, peb.x) * step(0.84, peb.z);
  float grain  = fbm(uv, vec2(110.0), 2, 0.5, S + 8u);
  // Ripple relief is deliberately smaller than the dune term. A 3 m tile of
  // ground repeats fourteen times across the plaza, and the higher the ripple
  // train stands the more the eye locks onto that repeat as corduroy; the
  // metre-scale drift is what has to carry the ground.
  float h = 0.5 + dune * 0.100 + ripple * 0.029 + pebble * 0.028 + grain * 0.008;
  F = vec4(h, pebble, grain * 0.5 + 0.5, dune * 0.5 + 0.5);

#elif MAT == 3
  // ---- asphalt: patch repairs, coarse aggregate, crack network, tyre polish.
  // Polish is applied to the aggregate AMPLITUDE, not a mask channel: worn lanes
  // really are flatter, and the compose pass reads that back out of the height.
  vec3  rep   = worley(dwarp(uv, vec2(2.0), 0.12, S + 2u), vec2(3.0), 1.0, S + 3u);
  float seam  = smoothstep(0.035, 0.0, rep.y - rep.x);
  float polish = smoothstep(0.35, 0.78, fbm01(uv, vec2(2.0, 9.0), 3, 0.5, S + 9u));
  vec3  agg   = worley(dwarp(uv, vec2(9.0), 0.02, S + 5u), vec2(52.0), 1.0, S + 6u);
  float stone = smoothstep(0.34, 0.12, agg.x);
  float expo  = mix(0.30, 1.0, fbm01(uv, vec2(6.0), 3, 0.5, S + 7u));
  vec3  ck1 = worley(dwarp(uv, vec2(5.0),  0.09, S + 11u), vec2(7.0),  1.0, S + 12u);
  vec3  ck2 = worley(dwarp(uv, vec2(11.0), 0.06, S + 13u), vec2(17.0), 1.0, S + 14u);
  float crack = clamp(smoothstep(0.050, 0.0, ck1.y - ck1.x) * 0.9
                    + smoothstep(0.035, 0.0, ck2.y - ck2.x) * 0.55, 0.0, 1.0);
  crack *= smoothstep(0.30, 0.66, fbm01(uv, vec2(4.0), 3, 0.5, S + 15u));
  float grit = fbm(uv, vec2(100.0), 2, 0.5, S + 17u);
  float h = 0.5 + stone * expo * 0.075 * (1.0 - polish * 0.80) + grit * 0.010
          - crack * 0.30 - seam * 0.055;
  F = vec4(h, stone * expo, clamp(crack + seam * 0.6, 0.0, 1.0),
           mix(0.35, 0.72, rep.z));

#elif MAT == 4
  // ---- rusted metal: rust blooms outward from the fixings, then pits and flakes
  vec3  fx   = worley(uv, vec2(4.0), 0.55, S + 3u);
  float bolt = smoothstep(0.135, 0.085, fx.x);
  float rim  = smoothstep(0.185, 0.135, fx.x) - bolt;
  float bloomR = 0.13 + 0.30 * fbm01(uv, vec2(9.0), 3, 0.5, S + 5u);
  float bloom  = smoothstep(bloomR + 0.22, bloomR, fx.x);
  float patches = smoothstep(0.44, 0.74, fbm01(dwarp(uv, vec2(4.0), 0.10, S + 7u), vec2(5.0), 4, 0.55, S + 8u));
  float rust = clamp(bloom * 0.9 + patches * 0.8, 0.0, 1.0);
  rust *= smoothstep(0.10, 0.45, fbm01(uv, vec2(13.0), 3, 0.5, S + 9u) + 0.22);
  vec3  pt   = worley(uv, vec2(90.0), 1.0, S + 11u);
  float pit  = smoothstep(0.22, 0.05, pt.x) * step(0.55, pt.z) * rust;
  float flake = rfbm(dwarp(uv, vec2(21.0), 0.03, S + 13u), vec2(29.0), 3, S + 14u);
  float flakeEdge = smoothstep(0.70, 0.90, flake) * rust;
  float mill = fbm(uv, vec2(3.0, 90.0), 2, 0.5, S + 17u);
  float h = 0.5 + bolt * 0.10 + rim * 0.028 + mill * 0.006
          + flakeEdge * 0.035 - pit * 0.20 + (rust - 0.5) * 0.018;
  F = vec4(h, rust, clamp(pit + flakeEdge * 0.6, 0.0, 1.0), bolt);

#elif MAT == 5
  // ---- painted metal: coat over base, directional scratches, edge wear, chips
  float panel = fbm(uv, vec2(3.0), 3, 0.5, S + 2u);
  float peel  = fbm(uv, vec2(43.0), 3, 0.5, S + 3u);
  float scrA  = rfbm(uv, vec2(5.0, 150.0), 1, S + 5u);
  float scrB  = rfbm(uv, vec2(140.0, 7.0), 1, S + 6u);
  float scratch = clamp(smoothstep(0.87, 0.995, scrA)
                      + smoothstep(0.92, 0.998, scrB) * 0.5, 0.0, 1.0);
  float wearF = fbm01(dwarp(uv, vec2(4.0), 0.09, S + 7u), vec2(5.0), 4, 0.55, S + 8u);
  float wear  = smoothstep(0.60, 0.78, wearF);
  vec3  ch    = worley(dwarp(uv, vec2(17.0), 0.03, S + 9u), vec2(34.0), 1.0, S + 10u);
  float chip  = smoothstep(0.22, 0.10, ch.x) * step(0.74, ch.z);
  float bare  = clamp(wear * 0.8 + chip + scratch * 0.55, 0.0, 1.0);
  float h = 0.5 + peel * 0.012 + panel * 0.020 - chip * 0.045 - scratch * 0.006;
  F = vec4(h, bare, scratch, panel * 0.5 + 0.5);

#elif MAT == 6
  // ---- gunmetal: machining lay, phosphate crystal, edge wear, fine scratches
  float groove = fbm(uv, vec2(3.0, 110.0), 2, 0.55, S + 3u);
  float lay    = fbm(uv, vec2(150.0, 4.0), 1, 0.5, S + 4u);
  vec3  ph   = worley(uv, vec2(110.0), 1.0, S + 5u);
  float phos = smoothstep(0.42, 0.06, ph.x);
  float wearF = fbm01(dwarp(uv, vec2(6.0), 0.06, S + 7u), vec2(7.0), 4, 0.5, S + 8u);
  float wear  = smoothstep(0.66, 0.82, wearF);
  float scr   = rfbm(uv, vec2(5.0, 140.0), 1, S + 9u);
  float scratch = smoothstep(0.90, 0.998, scr);
  float h = 0.5 + groove * 0.010 + lay * 0.004 + phos * 0.014 - scratch * 0.004;
  F = vec4(h, clamp(wear + scratch * 0.7, 0.0, 1.0), phos, groove * 0.5 + 0.5);

#elif MAT == 7
  // ---- polymer: mould stipple, tool seams, subtle sheen variation
  vec3  md  = worley(uv, vec2(110.0), 1.0, S + 3u);
  float peb = smoothstep(0.42, 0.10, md.x);
  float fine = fbm(uv, vec2(130.0), 1, 0.5, S + 4u);
  float seamA = smoothstep(0.010, 0.0, abs(fract(uv.x + 0.25) - 0.5));
  float seamB = smoothstep(0.008, 0.0, abs(fract(uv.y * 2.0 + 0.13) - 0.5));
  float seam = clamp(seamA + seamB * 0.6, 0.0, 1.0);
  float macro = fbm01(uv, vec2(3.0), 3, 0.5, S + 5u);
  float h = 0.5 + peb * 0.022 + fine * 0.006 + seam * 0.030;
  F = vec4(h, seam, peb, macro);

#elif MAT == 8
  // ---- sawn timber. TILE 1.176 m (level.js UVS.wood 0.85) baked at 1024, so
  // one texel is 1.15 mm and one height unit is ~55 mm.
  //
  // DISCRETE BOARDS AT A REAL RING PITCH. Five 23.5 cm boards with a gap between
  // them and growth rings at 4.3-6.7 mm. The pitch is the whole point: rings at
  // 20 cm read as marbled paper and rings at 1 cm read as corduroy, and only
  // somewhere near 5 mm reads as timber. 4.3 mm is 3.7 texels at 1024 — the
  // finest that survives the mip chain, and below it the board loses the grain
  // DIRECTION that is most of what identifies the material at 3 m.
  //
  // Boards run the full width of the tile with ONE butt joint each at a random
  // position: a staggered grid of short boards reads as brickwork instead.
  const float NB = 5.0;
  float bv  = uv.y * NB;
  float bi  = floor(bv);
  float bf  = fract(bv);
  float bid = hf(vec2(0.0, bi), vec2(1.0, NB), S + 3u);

  // every board is cut from a different part of a different log: its own ring
  // pitch, its own distance from the pith, its own tone
  float pitch  = 175.0 + bid * 95.0;                       // 4.3 - 6.7 mm rings
  float centre = hf(vec2(1.0, bi), vec2(1.0, NB), S + 4u) * 2.4 - 0.7;
  float across = (bf - centre) / NB;

  // KNOTS. Two boards in five carry one. A knot is a branch stub sawn through:
  // the rings inside it are tight and concentric, the face grain SWEEPS AROUND
  // it rather than running past, and it stands slightly proud because it is
  // denser than the wood it is set in. Without knots a board is just a striped
  // rectangle, and striped rectangles are what procedural wood always looks like.
  float kOn = step(0.58, hf(vec2(5.0, bi), vec2(1.0, NB), S + 23u));
  float kx  = hf(vec2(4.0, bi), vec2(1.0, NB), S + 21u);
  float ky  = 0.28 + hf(vec2(6.0, bi), vec2(1.0, NB), S + 27u) * 0.44;
  vec2  kd  = vec2(fract(uv.x - kx + 0.5) - 0.5, (bf - ky) / NB);
  float kr  = length(vec2(kd.x, kd.y * 2.6));
  // continuous, so no seam where the deflection changes sign
  across += kd.y * kOn * smoothstep(0.085, 0.010, kr) * 0.85;
  float knotM = kOn * smoothstep(0.052, 0.030, kr);        // inside the stub
  float knotR = kOn * smoothstep(0.040, 0.024, kr);        // its dark core

  vec2  wq = dwarp(uv, vec2(3.0, 9.0), 0.006, S + 5u);
  float faceP = abs(across) * pitch + fbm(wq, vec2(3.0, 13.0), 3, 0.5, S + 6u) * 1.3;

  // one butt joint per board, at a position that has nothing to do with its
  // neighbours', plus a narrow band of cross-cut grain beside it so a board END
  // reads as an end and not as more face
  float lf   = fract(uv.x + hf(vec2(2.0, bi), vec2(1.0, NB), S + 13u));
  float lE   = min(lf, 1.0 - lf);
  // gap width varies per board, and about a third of them close up entirely —
  // five boards each with an identical visible butt joint reads as brickwork
  float bw   = max(0.0, hf(vec2(3.0, bi), vec2(1.0, NB), S + 17u) - 0.30) * 0.014;
  float butt = 1.0 - smoothstep(bw * 0.35, bw + 0.0015, lE);
  float endP = length(vec2(lE * 2.2, across * 1.1)) * pitch * 0.62;
  float endM = smoothstep(0.038, 0.013, lE);
  float ring = mix(sin(TAU * faceP), sin(TAU * endP), endM);
  ring = mix(ring, sin(TAU * kr * pitch * 2.4), knotM);
  ring = ring * 0.5 + 0.5;
  float late = pow(ring, 1.7);                             // band-limited: no comb alias
  late = max(late, knotR * 0.9);                           // the stub is dense throughout

  // gap between boards
  float seamY = 1.0 - smoothstep(0.005, 0.020, min(bf, 1.0 - bf));

  float grain = fbm(uv, vec2(7.0, 120.0), 2, 0.55, S + 7u); // fibre along the board
  float saw   = fbm(uv, vec2(170.0, 6.0), 1, 0.5, S + 8u);  // saw marks across it
  float sp = rfbm(uv, vec2(3.0, 90.0), 2, S + 9u);
  float split = smoothstep(0.93, 0.997, sp)
              * smoothstep(0.35, 0.72, fbm01(uv, vec2(4.0), 3, 0.5, S + 10u))
              * (1.0 - knotM);
  // Weathering greys along the grain, board by board — not in clouds. Cloudy
  // weathering over the top of board seams reads as spilt paint.
  float weather = clamp(bid * 0.55
                + fbm01(uv, vec2(4.0, 26.0), 3, 0.55, S + 11u) * 0.55
                + (across * across) * 1.2 - 0.20, 0.0, 1.0);

  float h = 0.5
          + late * 0.008 * (0.35 + weather)  // earlywood erodes, latewood stands proud
          + knotR * 0.007                    // 0.4 mm proud: the stub wears slower
          + grain * 0.006
          + saw * 0.003
          - split * 0.026                    // 1.4 mm check along the grain
          - butt * 0.034
          - seamY * 0.058
          - (bf - 0.5) * (bf - 0.5) * 0.026; // each board cups slightly
  F = vec4(h, late, clamp(split * 0.85 + seamY * 0.9 + butt * 0.9, 0.0, 1.0), weather);

#elif MAT == 9
  // ---- fabric: plain weave over/under, slub, sun fade, dirt at contact points
  const float K = 96.0;
  vec2 t = uv * K;
  vec2 c = floor(t), f = fract(t);
  float over = mod(c.x + c.y, 2.0);
  float su = sin(f.x * 3.14159265);
  float sv = sin(f.y * 3.14159265);
  float weave = mix(sv * 0.90 + su * 0.35, su * 0.90 + sv * 0.35, over);
  float twist = fbm(uv, vec2(K * 2.0, K * 0.5), 1, 0.5, S + 3u);
  float fuzz  = fbm(uv, vec2(150.0), 1, 0.5, S + 4u);
  float slub  = fbm(uv, vec2(6.0, 3.0), 3, 0.5, S + 5u);
  float fade  = fbm01(dwarp(uv, vec2(2.0), 0.07, S + 6u), vec2(2.0), 3, 0.5, S + 7u);
  float dirt  = smoothstep(0.54, 0.82, fbm01(dwarp(uv, vec2(4.0), 0.10, S + 8u), vec2(4.0), 4, 0.55, S + 9u));
  float h = 0.5 + weave * 0.050 + twist * 0.008 + fuzz * 0.004 + slub * 0.010;
  F = vec4(h, over, dirt, fade);

#elif MAT == 10
  // ---- rubber: fine matte mould pebble, parting line, scuffing
  vec3  md  = worley(uv, vec2(100.0), 1.0, S + 3u);
  float peb = smoothstep(0.40, 0.08, md.x);
  float fine = fbm(uv, vec2(140.0), 1, 0.5, S + 4u);
  float seam = smoothstep(0.007, 0.0, abs(fract(uv.y + 0.5) - 0.5));
  float scuff = smoothstep(0.55, 0.86, fbm01(uv, vec2(9.0, 4.0), 3, 0.5, S + 6u));
  float macro = fbm01(uv, vec2(3.0), 3, 0.5, S + 5u);
  float h = 0.5 + peb * 0.018 + fine * 0.005 + seam * 0.020;
  F = vec4(h, seam, scuff, macro);

#elif MAT == 11
  // ---- glass: rolled waviness, dust specks, edge grime
  float wave = fbm(uv, vec2(3.0), 3, 0.5, S + 3u);
  vec3  dst  = worley(uv, vec2(90.0), 1.0, S + 4u);
  float speck = smoothstep(0.14, 0.02, dst.x) * step(0.86, dst.z);
  float grime = smoothstep(0.55, 0.88, fbm01(uv, vec2(5.0, 2.0), 4, 0.55, S + 5u));
  float h = 0.5 + wave * 0.004 + speck * 0.05;
  F = vec4(h, speck, grime, wave * 0.5 + 0.5);

#elif MAT == 12
  // ---- tile: bevelled field tiles, recessed grout, per-tile tone, corner chips
  const vec2 N = vec2(4.0, 4.0);
  vec2 t = uv * N;
  vec2 c = floor(t), f = fract(t);
  float id = hf(c, N, S + 3u);
  vec2 e = min(f, 1.0 - f);
  float edge  = min(e.x, e.y);
  float grout = 1.0 - smoothstep(0.020, 0.048, edge);
  float bevel = smoothstep(0.048, 0.115, edge);
  float glaze = fbm(uv, vec2(60.0), 2, 0.5, S + 4u);
  float chipD = worley(uv, N * 3.0, 1.0, S + 5u).x;
  float chip  = smoothstep(0.18, 0.04, chipD) * step(0.86, id) * (1.0 - bevel);
  float h = 0.5 + bevel * 0.050 - grout * 0.060 + glaze * 0.004 - chip * 0.055;
  F = vec4(h, grout, id, chip);

#else
  // ---- brick, running bond. TILE 1.613 m (level.js UVS.brick 0.62), so 7 x 21
  // cells is a 230 x 77 mm module: a 215 x 65 mm brick with a 10 mm bed joint.
  // One height unit is ~82 mm, one texel 3.15 mm.
  const float ROWS = 21.0, COLS = 7.0;
  float ry  = uv.y * ROWS;
  float row = floor(ry);
  float rx  = uv.x * COLS + mod(row, 2.0) * 0.5;
  float col = floor(rx);
  vec2 f = vec2(fract(rx), fract(ry));
  float id = hf(vec2(col, row), vec2(COLS, ROWS), S + 3u);

  // The joint width is measured in UV, not in cell fractions. The cell is 3:1,
  // so the old cell-fraction test made the perp joints three times wider than
  // the bed joints — which is most of why the mortar read as a painted grid.
  vec2  e    = min(f, 1.0 - f) * vec2(1.0 / COLS, 1.0 / ROWS);
  float edge = min(e.x, e.y);
  float jw   = 0.0031 + (id - 0.5) * 0.0009;              // ~10 mm, struck by hand
  float mortar = 1.0 - smoothstep(jw * 0.55, jw * 1.55, edge);
  float arris  = 1.0 - smoothstep(jw * 1.5, jw * 3.6, edge);   // worn brick edge

  float erode = fbm(uv, vec2(90.0), 2, 0.5, S + 4u);
  float pit   = smoothstep(0.62, 0.90, fbm01(uv, vec2(150.0), 1, 0.5, S + 5u)) * (1.0 - mortar);
  float mgrit = fbm(uv, vec2(110.0), 1, 0.5, S + 6u);           // sand in the mortar
  float salt  = smoothstep(0.55, 0.88, fbm01(uv, vec2(5.0, 9.0), 3, 0.55, S + 7u));

  // The joint is RAKED — it sits ~4 mm behind the brick face, it does not stand
  // proud of it. Deeper than that and the baked AO turns every joint into a
  // black line, which is the painted-grid look this is trying to get away from.
  // Bricks are also laid a millimetre or two out of plane.
  float h = 0.5
          + erode * 0.006
          - pit * 0.012
          - mortar * 0.048
          - arris * 0.008
          + mortar * mgrit * 0.004
          + (id - 0.5) * 0.009;
  F = vec4(h, mortar, id, clamp(erode * 0.5 + 0.5 + salt * 0.30, 0.0, 1.0));
#endif

  fragColor = F;
}
`;

// ---------------------------------------------------------------- pass B: compose

const COMPOSE_FRAG = /* glsl */`
precision highp float;
precision highp int;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uFields;
uniform vec2  uTexel;   // 1 / fields resolution
uniform float uBump;
uniform float uAO;
uniform float uCav;
uniform float uDet;
uniform float uSeed;
uniform int   uOut;     // 0 albedo, 1 normal, 2 ORM
${GLSL_LIB}

float H(vec2 uv) { return texture(uFields, uv).r; }

void main() {
  uint S = uint(uSeed);
  vec4 f = texture(uFields, vUv);
  float h = f.r, m1 = f.g, m2 = f.b, ma = f.a;

  // Spatial roughness break. One sheen over every substance in the world is the
  // loudest material tell there is, so each family gets its own band AND its own
  // variation inside that band. These two fields cost nothing in the albedo and
  // normal bakes because only the ORM pass reads them.
  float rqA = 0.5, rqB = 0.5;
  if (uOut == 2) {
    rqA = fbm01(vUv, vec2(3.0), 3, 0.55, S + 6101u);   // region scale
    rqB = fbm01(vUv, vec2(11.0), 2, 0.50, S + 8221u);  // patch scale
  }

  // --- tangent-space normal straight off the height field
  float hl = H(vUv - vec2(uTexel.x, 0.0));
  float hr = H(vUv + vec2(uTexel.x, 0.0));
  float hd = H(vUv - vec2(0.0, uTexel.y));
  float hu = H(vUv + vec2(0.0, uTexel.y));
  vec3 nrm = normalize(vec3(-(hr - hl) * uBump, -(hu - hd) * uBump, 1.0));

  // --- occlusion + a wide blur of the same field, for cavity grime and for the
  //     fine-detail term that drives albedo speckle. One field, three products.
  float occ = 0.0, blur = h * 16.0;
  // The normal pass needs only the four cross taps, so the 16-tap gather is
  // skipped for it. That is a third of the whole compose cost, and at 1024 the
  // compose pass is the most expensive thing this file does.
  if (uOut != 1) {
    occ = 0.0; blur = 0.0;
    for (int i = 0; i < 8; i++) {
      float a = float(i) * 0.7853981634;
      vec2 dir = vec2(cos(a), sin(a));
      for (int k = 1; k <= 2; k++) {
        float r = float(k) * 3.0;
        float hs = H(vUv + dir * uTexel * r);
        occ  += max(0.0, hs - h) / (1.0 + r * 0.12);
        blur += hs;
      }
    }
  }
  blur *= 0.0625;
  float ao  = clamp(1.0 - occ * uAO, 0.30, 1.0);
  float cav = clamp((blur - h) * uCav, 0.0, 1.0);
  float dn  = clamp((h - blur) * uDet, -1.0, 1.0);

  vec3  alb   = vec3(0.5);
  float rough = 0.85;
  float metal = 0.0;

#if MAT == 0
  // formed faces are smooth (0.70-0.78); broken and spalled faces expose
  // aggregate and are much rougher (0.86-0.92). That split is the whole point.
  // Split near the MEDIAN of the field, not near its tail. At 0.55-0.86 the
  // "formed" case covered two thirds of the tile and the whole family pinned to
  // the bottom of its clamp — measured p50 0.71 against a declared 0.70-0.90
  // band, which is a split in the source that is not a split on screen.
  float broken = smoothstep(0.34, 0.72, rqA);
  alb = mix(vec3(0.415, 0.410, 0.390), vec3(0.545, 0.540, 0.516), ma);
  alb *= 1.0 + dn * 0.22;
  alb = mix(alb, vec3(0.655, 0.645, 0.610), m1 * 0.78);
  alb = mix(alb, vec3(0.225, 0.218, 0.202), m2 * 0.62);
  alb *= mix(1.0, 0.66, cav);
  // BAND 0.70-0.90, hard-clamped. Formed faces came off a steel shutter and are
  // smooth; broken, spalled and blown-out faces expose aggregate and are much
  // rougher. The clamp is what keeps concrete out of plaster's band no matter
  // where the noise lands — one family, one recognisable sheen.
  rough = clamp(0.745 + broken * 0.115 + m1 * 0.050 + m2 * 0.020
              + (rqB - 0.5) * 0.060 + abs(dn) * 0.040, 0.70, 0.90);

#elif MAT == 1
  float sub = m1, grime = m2;
  // Finish coat. Hue as well as value moves with the macro field — the low
  // patches go cooler and greyer, the high ones warmer. Value alone reads as a
  // dirty overlay; value plus hue reads as different material ages.
  vec3 coat = mix(vec3(0.706, 0.672, 0.606), vec3(0.862, 0.842, 0.790), ma);
  // hue rides with value but only a little, and biased warm: swing it hard and
  // the darker patches go blue, which on a sunlit render reads as damp concrete
  coat *= vec3(1.0 + (ma - 0.5) * 0.055, 1.0, 1.0 - (ma - 0.5) * 0.075);
  // Exposed base render. It must be lighter than the coat AROUND IT, not lighter
  // than some absolute value: an absolute colour matched the coat exactly
  // wherever the macro field ran bright, and the spall patches vanished on
  // precisely the sunlit walls where they were supposed to read. Derived from
  // the local coat instead, it is guaranteed +12% value and desaturated toward
  // grey everywhere. dn is the same fine relief the normal map carries, so the
  // grain you can see in the colour is the grain you can feel in the light.
  // It reads as EXPOSED SUBSTRATE because it is coarse, matte and slightly
  // greyer — NOT because it is bright. The value step is deliberately tiny, 4.5%,
  // and that number is the whole fix for this surface: at +11% the patches
  // multiplied through the level's warm tint and the macro brighten-side and
  // clipped to white, so a 60 cm spall read as a white blob stuck on the wall.
  // That is precisely the "unmapped placeholder decal" the frames were showing.
  // Contrast now comes from dn (coarse grain, three times the coat's), from
  // roughness (0.95 against 0.885) and from the lip's own occlusion.
  float coatY = dot(coat, vec3(0.30, 0.59, 0.11));
  vec3 base = mix(coat, vec3(coatY), 0.38) * 1.045;
  base *= 1.0 + dn * 0.44;
  alb = mix(coat, base, sub);
  alb *= 1.0 + dn * 0.06;
  // The stain in the MAP stays fairly light. Heavy darkening is left to the
  // cavity and base-of-wall grime in materials.js, which multiplies AFTER the
  // per-instance tint — bake it all in here and the level's warm tint clips the
  // clean plaster to white before the dirt ever gets a say.
  alb = mix(alb, vec3(0.462, 0.416, 0.352), grime * 0.58);
  alb *= mix(1.0, 0.84, cav);
  // BAND 0.85-0.95, plus hand-polished patches down to 0.60. Lime render is
  // matte everywhere except where a steel float was worked over it while it was
  // still green, and those burnished patches are a metre or two across — hence
  // the region-scale field rather than a per-texel one. Range and variation,
  // never one number.
  rough = clamp(mix(0.885, 0.950, sub) + grime * 0.028 + (rqB - 0.5) * 0.045,
                0.85, 0.95);
  rough -= smoothstep(0.50, 0.80, rqA) * (1.0 - sub) * 0.34;
  rough = clamp(rough, 0.60, 0.95);

#elif MAT == 2
  // Albedo comes off the SAME height field as the normal: crests are dry, warm
  // and bright, troughs hold damp, fine, grey material. Sand with a normal map
  // and a flat albedo is the classic giveaway.
  float hn = clamp((h - 0.5) * 6.0 + dn * 0.35, -1.0, 1.0) * 0.5 + 0.5;
  alb = mix(vec3(0.500, 0.416, 0.288), vec3(0.745, 0.664, 0.500), hn);
  alb *= vec3(1.0 + (hn - 0.5) * 0.10, 1.0, 1.0 - (hn - 0.5) * 0.14);
  alb *= 1.0 + dn * 0.20;
  alb = mix(alb, vec3(0.520, 0.472, 0.398), m1 * 0.75);
  alb *= mix(1.0, 0.84, cav);
  // BAND 0.90-0.99. Nothing in dry earth is anywhere near specular; the only
  // variation is that wind-polished pebbles and damp packed troughs sit at the
  // bottom of the band and loose dry crests at the top.
  rough = clamp(0.945 - m1 * 0.045 + (hn - 0.5) * 0.055 + (rqB - 0.5) * 0.050,
                0.90, 0.99);

#elif MAT == 3
  // flat local relief == worn smooth by traffic
  // A wheel path is a REGION metres wide that also happens to be flat. Driving
  // polish off local flatness alone fired over the entire road — measured p50
  // 0.67 on a family whose bulk should sit near 0.87 — because dn is fine grit
  // and fine grit is everywhere. Gating it on the region field puts the shine
  // in lanes, which is where tyres put it.
  float pol = smoothstep(0.48, 0.80, rqA)
            * (1.0 - smoothstep(0.10, 0.55, abs(dn)));
  alb = mix(vec3(0.082, 0.082, 0.088), vec3(0.138, 0.136, 0.134), ma);
  alb = mix(alb, vec3(0.300, 0.295, 0.285), m1 * 0.65);
  alb *= 1.0 + dn * 0.25;
  alb = mix(alb, vec3(0.045, 0.045, 0.048), m2 * 0.70);
  alb *= mix(1.0, 0.75, cav) * mix(1.0, 0.82, pol);
  // BAND 0.55-0.93. Fresh chip is near-matte; a wheel path polishes the binder
  // until it is almost wet-looking, and that split is the only thing that makes
  // a road read as a road rather than as dark gravel.
  rough = clamp(0.895 - m1 * 0.055 - pol * 0.330 + m2 * 0.045 + (rqB - 0.5) * 0.050,
                0.55, 0.93);

#elif MAT == 4
  float rust = m1, bolt = ma;
  vec3 steel = vec3(0.335, 0.345, 0.360);
  vec3 rustCol = mix(vec3(0.240, 0.105, 0.050), vec3(0.560, 0.295, 0.135),
                     clamp(dn * 0.5 + 0.5 + (1.0 - rust) * 0.35, 0.0, 1.0));
  alb = mix(steel, rustCol, smoothstep(0.05, 0.55, rust));
  alb = mix(alb, vec3(0.180, 0.090, 0.050), m2 * 0.60);
  alb *= 1.0 + bolt * (1.0 - rust) * 0.12;
  alb *= mix(1.0, 0.70, cav);
  // BAND 0.32-0.98 — deliberately the widest in the library, because the whole
  // subject is one substance turning into another. Sound mill scale is glossy,
  // scabbed rust is the roughest thing in the world.
  rough = clamp(mix(0.400, 0.955, smoothstep(0.05, 0.60, rust)) + m2 * 0.05
              + (rqB - 0.5) * 0.060, 0.32, 0.98);
  metal = mix(0.95, 0.08, smoothstep(0.10, 0.65, rust));

#elif MAT == 5
  float bare = m1;
  vec3 paint = mix(vec3(0.200, 0.215, 0.195), vec3(0.270, 0.285, 0.255), ma);
  alb = mix(paint, vec3(0.620, 0.630, 0.640), bare);
  alb *= 1.0 + dn * 0.12;
  alb = mix(alb, vec3(0.750, 0.760, 0.770), m2 * 0.45);
  alb *= mix(1.0, 0.80, cav);
  // BAND 0.26-0.65. Industrial enamel is a semi-gloss that chalks unevenly with
  // UV, so the sheen is regional; bared steel underneath is brighter and tighter.
  rough = clamp(mix(0.575, 0.300, bare) - m2 * 0.080 + (rqB - 0.5) * 0.055
              - smoothstep(0.46, 0.82, rqA) * 0.085 + dn * 0.05, 0.26, 0.65);
  metal = mix(0.04, 0.95, bare);

#elif MAT == 6
  float wear = m1, phos = m2;
  vec3 base = mix(vec3(0.112, 0.115, 0.120), vec3(0.155, 0.157, 0.162), ma);
  base *= 1.0 - phos * 0.12;
  alb = mix(base, vec3(0.520, 0.530, 0.545), wear * 0.85);
  alb *= 1.0 + dn * 0.10;
  alb *= mix(1.0, 0.85, cav);
  // the machining lay lives in the macro channel; modulating roughness along it
  // is a cheap stand-in for true anisotropy
  // BAND 0.25-0.45, hard-clamped. Phosphated steel sits at 0.43-0.45; the
  // machined flats read 0.31-0.35; the handled edges polish down to 0.25. Never
  // a mirror — 0.16 was reading as chrome, which is the wrong century for a
  // service rifle — and never above 0.45, which reads as cast iron.
  rough = clamp(0.375 + phos * 0.055 + (ma - 0.5) * 0.050
              - smoothstep(0.40, 0.78, rqA) * 0.090 - wear * 0.115, 0.25, 0.45);
  metal = mix(0.88, 1.0, wear);

#elif MAT == 7
  alb = mix(vec3(0.085, 0.085, 0.088), vec3(0.125, 0.122, 0.118), ma);
  alb *= 1.0 + dn * 0.10;
  alb = mix(alb, vec3(0.160, 0.158, 0.152), m1 * 0.50);
  alb *= mix(1.0, 0.88, cav);
  // BAND 0.52-0.76. Moulded polymer: 0.53 on the tool-polished faces, 0.74 on
  // the stippled ones. It has to stay clearly above gunmetal's ceiling or the
  // furniture and the receiver read as one injection moulding.
  rough = clamp(0.625 + m2 * 0.085 - m1 * 0.060 + (rqB - 0.5) * 0.065
              - smoothstep(0.44, 0.80, rqA) * 0.080 + dn * 0.04, 0.52, 0.76);

#elif MAT == 8
  float late = m1, seam = m2, tone = ma;
  // one axis from fresh heartwood to pale grey weathered sapwood, with the
  // latewood ring as a darker, denser line on top of whatever that lands on
  vec3 fresh = vec3(0.452, 0.302, 0.170);
  vec3 grey  = vec3(0.466, 0.428, 0.382);
  alb = mix(fresh, grey, smoothstep(0.34, 0.95, tone));
  alb = mix(alb, alb * vec3(0.58, 0.54, 0.50), late * 0.80);
  // Only the very densest wood — the core of a knot — reaches late^3, so this
  // reddens and darkens the branch stub without staining the ordinary latewood
  // bands the same colour.
  alb = mix(alb, alb * vec3(0.66, 0.46, 0.33), late * late * late * 0.60);
  alb *= 1.0 + dn * 0.20;
  alb = mix(alb, vec3(0.105, 0.080, 0.058), seam * 0.85);
  alb *= mix(1.0, 0.78, cav);
  // BAND 0.60-0.80, hard-clamped. Sawn timber, no varnish anywhere: 0.62 on a
  // planed, handled face at the very smoothest, 0.80 on saw-torn and weathered
  // grain. Anything glossier than 0.60 puts a coat of lacquer on a bombed street.
  rough = clamp(0.755 - smoothstep(0.42, 0.76, rqA) * 0.170 + late * 0.025
              + seam * 0.030 + (tone - 0.5) * 0.055 + (rqB - 0.5) * 0.040,
                0.60, 0.80);

#elif MAT == 9
  float over = m1, dirt = m2, fade = ma;
  alb = mix(vec3(0.235, 0.215, 0.155), vec3(0.345, 0.325, 0.245), fade);
  alb *= (1.0 + dn * 0.18) * mix(0.94, 1.06, over);
  alb = mix(alb, vec3(0.115, 0.100, 0.075), dirt * 0.70);
  alb *= mix(1.0, 0.70, cav);
  // BAND 0.84-0.98. Canvas is matte, but the parts that get sat on and rubbed
  // flatten and shine very slightly, which is the only specular cue cloth has.
  rough = clamp(0.915 - fade * 0.035 + dirt * 0.040
              - smoothstep(0.48, 0.84, rqA) * 0.075 + (rqB - 0.5) * 0.040,
                0.84, 0.98);

#elif MAT == 10
  alb = mix(vec3(0.042, 0.042, 0.044), vec3(0.062, 0.062, 0.064), ma);
  alb *= 1.0 + dn * 0.12;
  alb = mix(alb, vec3(0.100, 0.100, 0.100), m2 * 0.40);
  alb *= mix(1.0, 0.90, cav);
  // BAND 0.62-0.95. Mould-textured rubber is dead matte; the parting line is
  // tool-polished and the scuffed faces are burnished.
  rough = clamp(0.905 - m1 * 0.130 - m2 * 0.170 + (rqB - 0.5) * 0.060
              - smoothstep(0.44, 0.80, rqA) * 0.070, 0.62, 0.95);

#elif MAT == 11
  alb = vec3(0.860, 0.880, 0.900);
  alb = mix(alb, vec3(0.700, 0.700, 0.680), m2 * 0.50);
  alb = mix(alb, vec3(0.950, 0.950, 0.930), m1 * 0.80);
  // BAND 0.03-0.55. Only glass is allowed below 0.2 anywhere in the library.
  rough = clamp(0.045 + m2 * 0.200 + m1 * 0.320, 0.03, 0.55);

#elif MAT == 12
  float grout = m1, id = m2, chip = ma;
  alb = mix(mix(vec3(0.680, 0.665, 0.630), vec3(0.800, 0.790, 0.755), id),
            vec3(0.420, 0.410, 0.385), grout);
  alb *= 1.0 + dn * 0.08;
  alb = mix(alb, vec3(0.580, 0.565, 0.530), chip * 0.80);
  alb *= mix(1.0, 0.72, cav);
  // BAND 0.20-0.95. Glazed field, unglazed grout: the widest split on any single
  // surface in the level, and the reason a tiled floor reads as tiled at all.
  rough = clamp(mix(0.245, 0.885, grout) + chip * 0.350 + (rqB - 0.5) * 0.050
              - smoothstep(0.48, 0.84, rqA) * (1.0 - grout) * 0.075, 0.20, 0.95);

#else
  float mortar = m1, id = m2, tone = ma;
  // No two bricks out of the same kiln match. The id hash drives value, and a
  // sixth of them are overburnt headers that came out much darker and greyer.
  vec3 clay = mix(vec3(0.372, 0.222, 0.168), vec3(0.572, 0.382, 0.286), id);
  clay = mix(clay, vec3(0.300, 0.216, 0.190), smoothstep(0.86, 0.94, id) * 0.75);
  clay = mix(clay, vec3(0.470, 0.404, 0.344), smoothstep(0.45, 0.98, tone) * 0.45);
  clay *= 1.0 + dn * 0.22;
  // Mortar: DESATURATED WARM grey, DARKER than the brick it beds, and matte.
  // It has to be baked warm because the level tints this map toward a dusty
  // buff, and a tint that desaturates clay pushes anything neutral blue.
  vec3 joint = vec3(0.352, 0.320, 0.280) * (1.0 + dn * 0.22);
  alb = mix(clay, joint, mortar);
  // dirt collects in the rake of the joint, so the deep part is darker again
  alb *= mix(1.0, 0.80, cav);
  // BAND 0.80-0.98, split in two: fired clay 0.82-0.90, mortar 0.94-0.98. The
  // joint MUST be the matter of the two. Mortar baked glossier than the brick it
  // beds is what makes a wall read as a painted grid rather than as masonry.
  rough = clamp(mix(0.868, 0.962, mortar) + (rqB - 0.5) * 0.070 + abs(dn) * 0.030
              - smoothstep(0.50, 0.84, rqA) * (1.0 - mortar) * 0.075, 0.80, 0.98);
#endif

  if (uOut == 0)      fragColor = vec4(clamp(alb, 0.0, 1.0), 1.0);
  else if (uOut == 1) fragColor = vec4(nrm * 0.5 + 0.5, 1.0);
  else fragColor = vec4(ao, clamp(rough, 0.03, 1.0), clamp(metal, 0.0, 1.0),
                        // ALPHA = curvature. 0.5 flat, >0.5 convex (a ridge that
                        // should wear bright), <0.5 concave (a crevice that
                        // should collect grime). materials.js reads it back out
                        // of the same fetch it already makes for roughness, so
                        // every consumer gets cavity grime and edge wear for the
                        // cost of nothing.
                        clamp(0.5 + (h - blur) * uDet * 0.45, 0.0, 1.0));
}
`;

// ---------------------------------------------------------------- baker

let _renderer = null;
let _quadScene = null;
let _quadCam = null;
let _quad = null;
let _fieldsRT = new Map();   // size -> RGBA16F target
let _outRT = new Map();      // size -> RGBA8 target
let _readBuf = new Map();    // size -> Uint8Array
const _fieldsMat = new Map();
const _composeMat = new Map();

/** Wire the baker to the live renderer. Must be the renderer that draws the
 *  scene — the textures are uploaded into its context. materials.js calls this. */
export function setTextureRenderer(renderer) {
  _renderer = renderer || null;
}

function ensureQuad() {
  if (_quad) return;
  _quadScene = new THREE.Scene();
  _quadCam = new THREE.Camera();
  _quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), null);
  _quad.frustumCulled = false;
  _quadScene.add(_quad);
}

function fieldsRT(size) {
  let rt = _fieldsRT.get(size);
  if (rt) return rt;
  rt = new THREE.WebGLRenderTarget(size, size, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    wrapS: THREE.RepeatWrapping,   // neighbour taps must wrap, or the seam shows
    wrapT: THREE.RepeatWrapping,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
    colorSpace: THREE.NoColorSpace,
  });
  _fieldsRT.set(size, rt);
  bytes += size * size * 8;   // RGBA16F scratch, no mips
  return rt;
}

function outRT(size) {
  let rt = _outRT.get(size);
  if (rt) return rt;
  rt = new THREE.WebGLRenderTarget(size, size, {
    type: THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
    colorSpace: THREE.NoColorSpace,
  });
  _outRT.set(size, rt);
  bytes += size * size * 4;   // RGBA8 scratch, no mips
  return rt;
}

function fieldsMaterial(id, seed) {
  let m = _fieldsMat.get(id);
  if (!m) {
    // ShaderMaterial (not Raw): three's prefix declares position/uv/defines for
    // us. glslVersion GLSL3 suppresses the pc_fragColor define so we own the out.
    m = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: { uSeed: { value: 0 } },
      vertexShader: VERT,
      fragmentShader: FIELDS_FRAG,
      defines: { MAT: id },
      depthTest: false,
      depthWrite: false,
    });
    _fieldsMat.set(id, m);
  }
  m.uniforms.uSeed.value = seed;
  return m;
}

function composeMaterial(id, cfg, seed) {
  let m = _composeMat.get(id);
  if (!m) {
    m = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: {
        uFields: { value: null },
        uTexel: { value: new THREE.Vector2() },
        uBump: { value: 12 },
        uAO: { value: 3 },
        uCav: { value: 24 },
        uDet: { value: 30 },
        uSeed: { value: 0 },
        uOut: { value: 0 },
      },
      vertexShader: VERT,
      fragmentShader: COMPOSE_FRAG,
      defines: { MAT: id },
      depthTest: false,
      depthWrite: false,
    });
    _composeMat.set(id, m);
  }
  m.uniforms.uAO.value = cfg.ao;
  m.uniforms.uCav.value = cfg.cav;
  m.uniforms.uDet.value = cfg.det;
  m.uniforms.uSeed.value = seed;
  return m;
}

let bytes = 0;
const _made = [];

function readToTexture(rt, size, srgb) {
  let buf = _readBuf.get(size);
  if (!buf) { buf = new Uint8Array(size * size * 4); _readBuf.set(size, buf); }
  _renderer.readRenderTargetPixels(rt, 0, 0, size, size, buf);
  const tex = new THREE.DataTexture(
    buf.slice(), size, size, THREE.RGBAFormat, THREE.UnsignedByteType,
  );
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.anisotropy = maxAniso();
  tex.needsUpdate = true;
  bytes += size * size * 4 * (4 / 3); // + mip chain
  _made.push(tex);
  return tex;
}

function maxAniso() {
  try { return Math.min(8, _renderer.capabilities.getMaxAnisotropy()); }
  catch { return 4; }
}

/** Flat 1x1 stand-in so a missing renderer degrades instead of throwing. */
function fallbackSet() {
  const mk = (r, g, b, srgb, a = 255) => {
    const t = new THREE.DataTexture(new Uint8Array([r, g, b, a]), 1, 1);
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.needsUpdate = true;
    _made.push(t);
    return t;
  };
  // alpha 128 == flat curvature, so the fallback gets neither grime nor edge wear
  const orm = mk(255, 220, 0, false, 128);
  return {
    map: mk(140, 138, 132, true),
    normalMap: mk(128, 128, 255, false),
    roughnessMap: orm, metalnessMap: orm, aoMap: orm, ormMap: orm,
  };
}

const baked = new Map();   // "name@size" -> base texture set (repeat 1,1)

function bake(name, size) {
  const key = name + '@' + size;
  const hit = baked.get(key);
  if (hit) return hit;

  const id = MAT_ID[name] ?? MAT_ID.concrete;
  const cfg = BAKE[name] || BAKE.concrete;
  const seed = hash32('tex:' + name) % 8388593;  // < 2^23, exact as a float uniform

  ensureQuad();
  const prev = _renderer.getRenderTarget();
  const fRT = fieldsRT(size);

  const fm = fieldsMaterial(id, seed);
  _quad.material = fm;
  _renderer.setRenderTarget(fRT);
  _renderer.render(_quadScene, _quadCam);

  const cm = composeMaterial(id, cfg, seed);
  cm.uniforms.uFields.value = fRT.texture;
  cm.uniforms.uTexel.value.set(1 / size, 1 / size);
  cm.uniforms.uBump.value = cfg.bump * (size / 512);
  _quad.material = cm;

  const draw = (target, res, out) => {
    cm.uniforms.uOut.value = out;
    _renderer.setRenderTarget(target);
    _renderer.render(_quadScene, _quadCam);
    return res;
  };

  const full = outRT(size);
  draw(full, size, 0);
  const map = readToTexture(full, size, true);
  draw(full, size, 1);
  const normalMap = readToTexture(full, size, false);

  const half = Math.max(64, size >> 1);
  const halfRT = outRT(half);
  draw(halfRT, half, 2);
  const orm = readToTexture(halfRT, half, false);

  _renderer.setRenderTarget(prev);

  // Release the bake programs — they are never used again once the maps have been
  // read back, and compiled `programs` is a hard budget (<= 90 for the frame).
  _quad.material = null;
  _fieldsMat.delete(id); _composeMat.delete(id);
  fm.dispose(); cm.dispose();

  const set = { map, normalMap, roughnessMap: orm, metalnessMap: orm, aoMap: orm, ormMap: orm };
  baked.set(key, set);
  return set;
}

// ---------------------------------------------------------------- public API

const cache = new Map();

function stableKey(name, opts) {
  const k = Object.keys(opts).sort();
  let s = name;
  for (const key of k) s += '|' + key + '=' + opts[key];
  return s;
}

/**
 * Returns a coherent PBR set. Roughness / metalness / AO are one channel-packed
 * RGB texture (glTF ORM convention: R=AO, G=roughness, B=metalness), so a full
 * material costs 3 texture units, not 5.
 *
 * opts: { size, repeat | repeatX/repeatY, rotation }
 */
export function makeTextureSet(name, opts = {}) {
  const key = stableKey(name, opts);
  const hit = cache.get(key);
  if (hit) return hit;

  if (!_renderer) {
    const set = fallbackSet();
    cache.set(key, set);
    return set;
  }

  const cfg = BAKE[name] ? name : 'concrete';
  const size = clampPow2(opts.size || BAKE[cfg].size);
  const base = bake(cfg, size);

  const rx = opts.repeatX ?? opts.repeat ?? 1;
  const ry = opts.repeatY ?? opts.repeat ?? 1;
  const rot = opts.rotation || 0;

  let set = base;
  if (rx !== 1 || ry !== 1 || rot !== 0) {
    // clone() shares Texture.source, so the GPU upload is shared: repeat variants
    // are free in VRAM and still let each material batch on its own.
    const wrap = (t) => {
      const c = t.clone();
      c.repeat.set(rx, ry);
      c.rotation = rot;
      c.center.set(0.5, 0.5);
      c.needsUpdate = true;
      return c;
    };
    const orm = wrap(base.ormMap);
    set = {
      map: wrap(base.map),
      normalMap: wrap(base.normalMap),
      roughnessMap: orm, metalnessMap: orm, aoMap: orm, ormMap: orm,
    };
  }
  cache.set(key, set);
  return set;
}

function clampPow2(n) {
  const v = Math.max(64, Math.min(2048, n | 0));
  return 1 << Math.round(Math.log2(v));
}

// ---------------------------------------------------------------- macro variation
//
// One small tileable field sampled by every material at a period deliberately
// incommensurate with the tile. This is what stops a 512px texture repeated 24
// times across a 120m ground plane reading as a grid, and it is the ONLY layer
// that can carry the 4-8 m band: a 2 m tile physically cannot hold a feature
// bigger than 2 m, so without this a 6 m wall is one flat beige at 40 m however
// good its 5 cm detail is. Built on the CPU: 256x256, once.
//
// CHANNELS. These are four INDEPENDENT fields, not four views of one:
//   R  value       broad light/dark mottling, fundamental at half the map period
//   G  hue         warm <-> cool, uncorrelated with value. Value alone reads as
//                  a dirty overlay laid on top; value plus an independent hue
//                  reads as patches of render of different ages, which is what
//                  a wall that has been repaired four times actually looks like.
//   B  runoff      sampled with a stretched, world-anchored UV to make vertical
//                  weathering streaks that span a whole building
//   A  break       roughness / edge-wear regionality

let _macro = null;

export function macroTexture() {
  if (_macro) return _macro;
  const N = 256;
  const data = new Uint8Array(N * N * 4);
  const seed = hash32('tex:macro');

  const h2 = (x, y, per, s) => {
    const ix = ((x % per) + per) % per, iy = ((y % per) + per) % per;
    let v = Math.imul(ix, 1597334677) ^ Math.imul(iy, 3812015801) ^ Math.imul(s, 2654435761);
    v = (v ^ (v >>> 15)) >>> 0; v = Math.imul(v, 2246822519) >>> 0;
    v = (v ^ (v >>> 13)) >>> 0; v = Math.imul(v, 3266489917) >>> 0;
    return ((v ^ (v >>> 16)) >>> 0) / 4294967296;
  };
  const vnoise = (u, v, per, s) => {
    const x = u * per, y = v * per;
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const fx = x - x0, fy = y - y0;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const a = h2(x0, y0, per, s), b = h2(x0 + 1, y0, per, s);
    const c = h2(x0, y0 + 1, per, s), d = h2(x0 + 1, y0 + 1, per, s);
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy;
  };
  const fbmc = (u, v, p0, oct, s) => {
    let sum = 0, amp = 1, nrm = 0, p = p0;
    for (let i = 0; i < oct; i++) {
      sum += amp * vnoise(u, v, p, s + i * 6151);
      nrm += amp; amp *= 0.55; p = Math.round(p * 2.17);
    }
    return sum / nrm;
  };

  // Raw fbm piles up around 0.5 and uses about a third of the range it is given,
  // so every channel is contrast-stretched before it is written. Skipping this
  // is why a "40% macro variation" setting produces a 12% one on screen.
  const stretch = (t, k) => {
    const c = Math.min(1, Math.max(0, (t - 0.5) * k + 0.5));
    return c * c * (3 - 2 * c);
  };

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const u = x / N, v = y / N;
      // period 2 is the lowest a periodic lattice can carry (period 1 collapses
      // to a constant), so it sets the fundamental: half the macro period, which
      // at a 4.31-tile mapping over a 2 m tile is a 4.3 m feature. That is the
      // low end of the band this layer exists to supply.
      const val = fbmc(u, v, 2, 4, seed);
      const hue = fbmc(u, v, 3, 3, seed + 17);
      const run = fbmc(u, v, 5, 4, seed + 31);
      const brk = fbmc(u, v, 3, 3, seed + 53);
      const i = (y * N + x) * 4;
      data[i]     = (stretch(val, 1.9) * 255) | 0;
      data[i + 1] = (stretch(hue, 1.7) * 255) | 0;
      data[i + 2] = (stretch(run, 2.1) * 255) | 0;
      data[i + 3] = (stretch(brk, 1.8) * 255) | 0;
    }
  }

  _macro = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
  _macro.wrapS = _macro.wrapT = THREE.RepeatWrapping;
  _macro.minFilter = THREE.LinearMipmapLinearFilter;
  _macro.magFilter = THREE.LinearFilter;
  _macro.generateMipmaps = true;
  _macro.colorSpace = THREE.NoColorSpace;
  _macro.needsUpdate = true;
  bytes += N * N * 4 * (4 / 3);
  _made.push(_macro);
  return _macro;
}

// ---------------------------------------------------------------- housekeeping

export function textureBytes() { return bytes; }

export function textureStats() {
  return { bytes, sets: baked.size, variants: cache.size, textures: _made.length };
}

export function disposeTextures() {
  for (const t of _made) t.dispose?.();
  _made.length = 0;
  for (const rt of _fieldsRT.values()) rt.dispose();
  for (const rt of _outRT.values()) rt.dispose();
  for (const m of _fieldsMat.values()) m.dispose();
  for (const m of _composeMat.values()) m.dispose();
  _fieldsRT.clear(); _outRT.clear(); _readBuf.clear();
  _fieldsMat.clear(); _composeMat.clear();
  baked.clear(); cache.clear();
  if (_quad) { _quad.geometry.dispose(); _quad = null; _quadScene = null; _quadCam = null; }
  _macro = null;
  bytes = 0;
}
