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
//   3. materials.js layers a low-frequency macro-variation map at 1/8 the tile
//      frequency over the top, which breaks the repeat ACROSS tiles.
//
// Output is read back to DataTextures rather than kept as render-target textures:
// that makes them cheap to clone for different repeat values (clones share the
// GPU upload via Texture.source), and it makes the VRAM accounting exact.

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
  concrete:     { size: 512, bump: 13.0, ao: 3.4, cav: 26.0, det: 26.0 },
  plaster:      { size: 512, bump: 11.0, ao: 3.0, cav: 30.0, det: 34.0 },
  sand:         { size: 512, bump:  9.5, ao: 2.1, cav: 18.0, det: 46.0 },
  asphalt:      { size: 512, bump: 15.0, ao: 3.8, cav: 24.0, det: 24.0 },
  rustMetal:    { size: 512, bump: 14.0, ao: 3.6, cav: 22.0, det: 22.0 },
  paintedMetal: { size: 512, bump: 10.0, ao: 3.0, cav: 30.0, det: 40.0 },
  gunmetal:     { size: 512, bump:  9.0, ao: 2.4, cav: 34.0, det: 60.0 },
  polymer:      { size: 512, bump: 11.0, ao: 2.6, cav: 30.0, det: 46.0 },
  wood:         { size: 512, bump: 12.0, ao: 3.0, cav: 26.0, det: 34.0 },
  fabric:       { size: 512, bump: 14.0, ao: 3.0, cav: 22.0, det: 30.0 },
  rubber:       { size: 512, bump: 11.0, ao: 2.6, cav: 30.0, det: 46.0 },
  glass:        { size: 512, bump:  6.0, ao: 1.4, cav: 40.0, det: 60.0 },
  tile:         { size: 512, bump: 13.0, ao: 3.4, cav: 24.0, det: 40.0 },
  brick:        { size: 512, bump: 13.0, ao: 3.6, cav: 22.0, det: 28.0 },
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
  // ---- concrete: cast aggregate, spalled patches, water runs, air pockets
  vec2 wq = dwarp(uv, vec2(3.0), 0.06, S + 11u);
  float macro  = fbm01(wq, vec2(3.0), 4, 0.55, S + 2u);
  float cement = fbm(wq, vec2(23.0), 3, 0.5, S + 3u);
  float micro  = fbm(uv, vec2(97.0), 2, 0.5, S + 4u);
  vec3  agg    = worley(dwarp(uv, vec2(11.0), 0.02, S + 9u), vec2(58.0), 0.95, S + 7u);
  float stone  = 1.0 - smoothstep(0.02, 0.36, agg.x);
  float spallF = fbm01(dwarp(uv, vec2(5.0), 0.10, S + 21u), vec2(4.0), 4, 0.55, S + 5u);
  float spall  = smoothstep(0.58, 0.69, spallF);
  vec3  ph     = worley(uv, vec2(86.0), 1.0, S + 31u);
  float holes  = smoothstep(0.16, 0.02, ph.x) * step(0.87, ph.z);
  // Water staining: noise stretched ~10:1 vertically reads as runs without the
  // seam a real top-to-bottom gradient would leave at the tile edge.
  float streak = fbm01(uv, vec2(37.0, 4.0), 3, 0.6, S + 13u);
  float src    = smoothstep(0.50, 0.78, fbm01(uv, vec2(7.0, 3.0), 3, 0.5, S + 17u));
  float run    = smoothstep(0.34, 0.78, streak) * src;
  float grime  = fbm01(wq, vec2(9.0), 4, 0.5, S + 19u);
  float h = 0.5 + cement * 0.045 + micro * 0.012
          - spall * 0.085 + spall * stone * 0.055
          - holes * 0.16 + (macro - 0.5) * 0.03;
  F = vec4(h, spall * (0.35 + stone * 0.65), clamp(run * 0.85 + grime * 0.22, 0.0, 1.0), macro);

#elif MAT == 1
  // ---- plaster / painted wall: trowel roll, stipple, hairline cracks, chipping
  vec2 wq = dwarp(uv, vec2(3.0), 0.05, S + 3u);
  float macro   = fbm01(wq, vec2(2.0), 3, 0.5, S + 2u);
  float stipple = fbm(uv, vec2(110.0), 2, 0.5, S + 5u);
  float roll    = fbm(uv, vec2(17.0), 3, 0.5, S + 6u);
  vec3  c1 = worley(dwarp(uv, vec2(7.0),  0.05, S + 11u), vec2(9.0),  1.0, S + 12u);
  vec3  c2 = worley(dwarp(uv, vec2(13.0), 0.04, S + 13u), vec2(23.0), 1.0, S + 14u);
  float crack = smoothstep(0.045, 0.0, c1.y - c1.x) * 0.9
              + smoothstep(0.030, 0.0, c2.y - c2.x) * 0.5;
  crack *= smoothstep(0.35, 0.62, fbm01(uv, vec2(5.0), 3, 0.5, S + 15u));
  crack = clamp(crack, 0.0, 1.0);
  // Paint comes off in patches, not speckles: few octaves, hard threshold, and
  // it starts at the cracks.
  float chipF = fbm01(dwarp(uv, vec2(5.0), 0.10, S + 21u), vec2(4.0), 3, 0.5, S + 22u);
  float chip  = smoothstep(0.615, 0.665, chipF + crack * 0.14);
  float h = 0.5 + roll * 0.03 + stipple * 0.012 - crack * 0.30 - chip * 0.030;
  F = vec4(h, chip, crack, macro);

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
  float h = 0.5 + dune * 0.085 + ripple * 0.036 + pebble * 0.028 + grain * 0.008;
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
  // ---- wood: growth rings, fibre grain, splits along the grain, weathering
  vec2  wq = dwarp(uv, vec2(2.0, 5.0), 0.05, S + 3u);
  float ringPhase = wq.y * 9.0 + fbm(uv, vec2(2.0, 6.0), 3, 0.5, S + 4u) * 1.2;
  float ring = fract(ringPhase);
  float late = smoothstep(0.60, 0.80, ring) * (1.0 - smoothstep(0.86, 0.98, ring));
  float grain = fbm(uv, vec2(8.0, 110.0), 2, 0.55, S + 5u);
  float fibre = fbm(uv, vec2(4.0, 160.0), 1, 0.5, S + 6u);
  float sp = rfbm(uv, vec2(3.0, 90.0), 2, S + 7u);
  float split = smoothstep(0.93, 0.997, sp)
              * smoothstep(0.35, 0.72, fbm01(uv, vec2(4.0), 3, 0.5, S + 8u));
  float weather = smoothstep(0.40, 0.76, fbm01(dwarp(uv, vec2(3.0), 0.07, S + 9u), vec2(3.0), 4, 0.5, S + 10u));
  float h = 0.5 - late * 0.020 + grain * 0.012 + fibre * 0.005
          - split * 0.34 - weather * 0.008;
  F = vec4(h, late, split, weather);

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
  // ---- brick: running bond, recessed mortar, per-brick tone, face erosion
  const float ROWS = 8.0, COLS = 4.0;
  float ry  = uv.y * ROWS;
  float row = floor(ry);
  float rx  = uv.x * COLS + mod(row, 2.0) * 0.5;
  float col = floor(rx);
  vec2 f = vec2(fract(rx), fract(ry));
  float id = hf(vec2(col, row), vec2(COLS, ROWS), S + 3u);
  vec2 e = min(f, 1.0 - f);
  float edge   = min(e.x, e.y);
  float mortar = 1.0 - smoothstep(0.030, 0.070, edge);
  float erode = fbm(uv, vec2(70.0), 2, 0.5, S + 4u);
  float pit   = smoothstep(0.58, 0.88, fbm01(uv, vec2(110.0), 1, 0.5, S + 5u)) * (1.0 - mortar);
  float h = 0.5 + (1.0 - mortar) * 0.050 + erode * 0.012 - pit * 0.045 - mortar * 0.018;
  F = vec4(h, mortar, id, erode * 0.5 + 0.5);
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
uniform int   uOut;     // 0 albedo, 1 normal, 2 ORM

float H(vec2 uv) { return texture(uFields, uv).r; }

void main() {
  vec4 f = texture(uFields, vUv);
  float h = f.r, m1 = f.g, m2 = f.b, ma = f.a;

  // --- tangent-space normal straight off the height field
  float hl = H(vUv - vec2(uTexel.x, 0.0));
  float hr = H(vUv + vec2(uTexel.x, 0.0));
  float hd = H(vUv - vec2(0.0, uTexel.y));
  float hu = H(vUv + vec2(0.0, uTexel.y));
  vec3 nrm = normalize(vec3(-(hr - hl) * uBump, -(hu - hd) * uBump, 1.0));

  // --- occlusion + a wide blur of the same field, for cavity grime and for the
  //     fine-detail term that drives albedo speckle. One field, three products.
  float occ = 0.0, blur = 0.0;
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
  blur *= 0.0625;
  float ao  = clamp(1.0 - occ * uAO, 0.30, 1.0);
  float cav = clamp((blur - h) * uCav, 0.0, 1.0);
  float dn  = clamp((h - blur) * uDet, -1.0, 1.0);

  vec3  alb   = vec3(0.5);
  float rough = 0.85;
  float metal = 0.0;

#if MAT == 0
  alb = mix(vec3(0.415, 0.410, 0.390), vec3(0.545, 0.540, 0.516), ma);
  alb *= 1.0 + dn * 0.22;
  alb = mix(alb, vec3(0.655, 0.645, 0.610), m1 * 0.78);
  alb = mix(alb, vec3(0.225, 0.218, 0.202), m2 * 0.62);
  alb *= mix(1.0, 0.66, cav);
  rough = 0.93 - m1 * 0.05 - m2 * 0.10 + dn * 0.06;

#elif MAT == 1
  vec3 paint = mix(vec3(0.715, 0.705, 0.675), vec3(0.815, 0.805, 0.770), ma);
  alb = mix(paint, vec3(0.555, 0.512, 0.455), m1);
  alb *= 1.0 + dn * 0.10;
  alb = mix(alb, vec3(0.300, 0.290, 0.270), m2 * 0.70);
  alb *= mix(1.0, 0.78, cav);
  rough = mix(0.60, 0.90, m1) + m2 * 0.06 + dn * 0.03;

#elif MAT == 2
  alb = mix(vec3(0.585, 0.490, 0.335), vec3(0.725, 0.635, 0.455), ma);
  alb *= 1.0 + dn * 0.30;
  alb = mix(alb, vec3(0.500, 0.455, 0.385), m1 * 0.75);
  alb *= mix(1.0, 0.86, cav);
  rough = 0.96 - m1 * 0.10 - abs(dn) * 0.05;

#elif MAT == 3
  // flat local relief == worn smooth by traffic
  float pol = 1.0 - smoothstep(0.10, 0.55, abs(dn));
  alb = mix(vec3(0.082, 0.082, 0.088), vec3(0.138, 0.136, 0.134), ma);
  alb = mix(alb, vec3(0.300, 0.295, 0.285), m1 * 0.65);
  alb *= 1.0 + dn * 0.25;
  alb = mix(alb, vec3(0.045, 0.045, 0.048), m2 * 0.70);
  alb *= mix(1.0, 0.75, cav) * mix(1.0, 0.82, pol);
  rough = 0.90 - m1 * 0.06 - pol * 0.34 + m2 * 0.05;

#elif MAT == 4
  float rust = m1, bolt = ma;
  vec3 steel = vec3(0.335, 0.345, 0.360);
  vec3 rustCol = mix(vec3(0.240, 0.105, 0.050), vec3(0.560, 0.295, 0.135),
                     clamp(dn * 0.5 + 0.5 + (1.0 - rust) * 0.35, 0.0, 1.0));
  alb = mix(steel, rustCol, smoothstep(0.05, 0.55, rust));
  alb = mix(alb, vec3(0.180, 0.090, 0.050), m2 * 0.60);
  alb *= 1.0 + bolt * (1.0 - rust) * 0.12;
  alb *= mix(1.0, 0.70, cav);
  rough = mix(0.42, 0.96, smoothstep(0.05, 0.60, rust)) + m2 * 0.05;
  metal = mix(0.95, 0.08, smoothstep(0.10, 0.65, rust));

#elif MAT == 5
  float bare = m1;
  vec3 paint = mix(vec3(0.200, 0.215, 0.195), vec3(0.270, 0.285, 0.255), ma);
  alb = mix(paint, vec3(0.620, 0.630, 0.640), bare);
  alb *= 1.0 + dn * 0.12;
  alb = mix(alb, vec3(0.750, 0.760, 0.770), m2 * 0.45);
  alb *= mix(1.0, 0.80, cav);
  rough = mix(0.55, 0.30, bare) - m2 * 0.08 + dn * 0.05;
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
  rough = mix(0.44, 0.16, wear) + phos * 0.10 + (ma - 0.5) * 0.10 + dn * 0.04;
  metal = mix(0.88, 1.0, wear);

#elif MAT == 7
  alb = mix(vec3(0.085, 0.085, 0.088), vec3(0.125, 0.122, 0.118), ma);
  alb *= 1.0 + dn * 0.10;
  alb = mix(alb, vec3(0.160, 0.158, 0.152), m1 * 0.50);
  alb *= mix(1.0, 0.88, cav);
  rough = 0.58 + m2 * 0.14 - m1 * 0.12 + dn * 0.05;

#elif MAT == 8
  float late = m1, split = m2, weather = ma;
  alb = mix(vec3(0.420, 0.300, 0.175), vec3(0.245, 0.155, 0.085), late);
  alb *= 1.0 + dn * 0.22;
  alb = mix(alb, vec3(0.440, 0.420, 0.385), weather * 0.60);
  alb = mix(alb, vec3(0.100, 0.075, 0.050), split * 0.80);
  alb *= mix(1.0, 0.72, cav);
  rough = 0.72 + weather * 0.20 + late * 0.05 + dn * 0.05;

#elif MAT == 9
  float over = m1, dirt = m2, fade = ma;
  alb = mix(vec3(0.235, 0.215, 0.155), vec3(0.345, 0.325, 0.245), fade);
  alb *= (1.0 + dn * 0.18) * mix(0.94, 1.06, over);
  alb = mix(alb, vec3(0.115, 0.100, 0.075), dirt * 0.70);
  alb *= mix(1.0, 0.70, cav);
  rough = 0.92 - fade * 0.04 + dirt * 0.04;

#elif MAT == 10
  alb = mix(vec3(0.042, 0.042, 0.044), vec3(0.062, 0.062, 0.064), ma);
  alb *= 1.0 + dn * 0.12;
  alb = mix(alb, vec3(0.100, 0.100, 0.100), m2 * 0.40);
  alb *= mix(1.0, 0.90, cav);
  rough = 0.86 - m1 * 0.20 + m2 * 0.05;

#elif MAT == 11
  alb = vec3(0.860, 0.880, 0.900);
  alb = mix(alb, vec3(0.700, 0.700, 0.680), m2 * 0.50);
  alb = mix(alb, vec3(0.950, 0.950, 0.930), m1 * 0.80);
  rough = 0.04 + m2 * 0.18 + m1 * 0.30;

#elif MAT == 12
  float grout = m1, id = m2, chip = ma;
  alb = mix(mix(vec3(0.680, 0.665, 0.630), vec3(0.800, 0.790, 0.755), id),
            vec3(0.420, 0.410, 0.385), grout);
  alb *= 1.0 + dn * 0.08;
  alb = mix(alb, vec3(0.580, 0.565, 0.530), chip * 0.80);
  alb *= mix(1.0, 0.72, cav);
  rough = mix(0.22, 0.88, grout) + chip * 0.40;

#else
  float mortar = m1, id = m2, ero = ma;
  vec3 brick = mix(vec3(0.300, 0.145, 0.105), vec3(0.440, 0.235, 0.165), id);
  brick = mix(brick, vec3(0.360, 0.300, 0.255), ero * 0.25);
  alb = mix(brick, vec3(0.545, 0.535, 0.505), mortar);
  alb *= 1.0 + dn * 0.20;
  alb *= mix(1.0, 0.70, cav);
  rough = mix(0.88, 0.95, mortar) + dn * 0.05;
#endif

  if (uOut == 0)      fragColor = vec4(clamp(alb, 0.0, 1.0), 1.0);
  else if (uOut == 1) fragColor = vec4(nrm * 0.5 + 0.5, 1.0);
  else                fragColor = vec4(ao, clamp(rough, 0.03, 1.0), clamp(metal, 0.0, 1.0), 1.0);
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

function composeMaterial(id, cfg) {
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
  const mk = (r, g, b, srgb) => {
    const t = new THREE.DataTexture(new Uint8Array([r, g, b, 255]), 1, 1);
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.needsUpdate = true;
    _made.push(t);
    return t;
  };
  const orm = mk(255, 220, 0, false);
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

  const cm = composeMaterial(id, cfg);
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
// One small tileable field sampled by every material at ~1/8 the tile frequency.
// This is what stops a 512px texture repeated 24 times across a 120m ground plane
// reading as a grid. Built on the CPU: it is 256x256 and only happens once.

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

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const u = x / N, v = y / N;
      const a = fbmc(u, v, 3, 4, seed);
      const b = fbmc(u, v, 5, 4, seed + 17);
      const c = fbmc(u, v, 2, 3, seed + 31);
      const i = (y * N + x) * 4;
      // tint centred on 0.5 with a slight warm/cool split, alpha = roughness break
      data[i]     = Math.max(0, Math.min(255, ((a * 0.72 + c * 0.28) * 255) | 0));
      data[i + 1] = Math.max(0, Math.min(255, ((a * 0.62 + b * 0.38) * 255) | 0));
      data[i + 2] = Math.max(0, Math.min(255, ((b * 0.70 + c * 0.30) * 255) | 0));
      data[i + 3] = Math.max(0, Math.min(255, (c * 255) | 0));
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
