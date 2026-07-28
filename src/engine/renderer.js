// OWNER: agent "render" — post-processing & framebuffer pipeline.
// CONTRACT (do not change signatures; main.js depends on them):
//   new RenderPipeline(canvas)         -> pipeline
//   .setSize(w, h, renderScale)        -> void
//   .setQuality('low'|'medium'|'high'|'ultra')
//   .render(scene, camera, dt)         -> void
//   .reportCost(perf)                  -> fills perf.pipelineCost {fullscreenPasses, shadowTexels, overdraw}
//   .renderer                          -> THREE.WebGLRenderer (read-only for others)
//   .dispose()
//
// SHAPE OF THE FRAME
//   scene -> HDR RGBA16F target (+ 24bit depth texture)
//   half-res SSAO from depth (normals reconstructed, no extra geometry pass)
//   TAA resolve (Halton jitter, world-space reprojection, YCoCg variance clip)
//   64x36 -> 1x1 log-luminance chain for GPU-side eye adaptation (no readback)
//   dual-filter (Kawase) bloom pyramid off the resolved HDR image
//   one composite pass: CA, sharpen, bloom, exposure, filmic curve, grade,
//   vignette, grain, sRGB + dither
//
// Everything above is either full-res-cheap or half-res-or-smaller. Measured
// full-screen pass equivalents (see reportCost, which computes this from the
// passes that actually ran, not from a hardcoded guess):
//   low 1.0 | medium ~1.8 | high ~3.1 | ultra ~3.2   (budget is 8)
//
// Nothing here allocates per frame. Matrices, vectors and materials are made
// once; render targets only on an actual pixel-size change.

import * as THREE from 'three';
import { makeRng } from '../core/rng.js';

// ---------------------------------------------------------------------------
// shared GLSL
// ---------------------------------------------------------------------------

const VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

// Depth -> view-space position. uProjInv is the inverse of whatever projection
// actually produced the depth buffer (jittered when TAA is on), so the jitter
// cancels between reconstruct and re-project.
const RECONSTRUCT = /* glsl */ `
uniform mat4 uProjInv;
vec3 viewPosFromDepth(vec2 uv, float d) {
  vec4 ndc = vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
  vec4 v = uProjInv * ndc;
  return v.xyz / v.w;
}`;

const LUMA = /* glsl */ `
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }`;

// ---------------------------------------------------------------------------
// SSAO — half res, hemisphere kernel, normals from depth derivatives
// ---------------------------------------------------------------------------

const AO_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tDepth;
uniform mat4 uProj;
uniform vec2 uTexel;      // 1 / half-res size
uniform vec3 uKernel[16];
uniform float uRadius;
uniform float uBias;
uniform float uIntensity;
uniform float uFrame;
${RECONSTRUCT}

float depthAt(vec2 uv) { return texture2D(tDepth, uv).x; }

// Interleaved gradient noise: cheap, and its structure is the one TAA eats best.
float ign(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

void main() {
  float d = depthAt(vUv);
  if (d >= 0.9999) { gl_FragColor = vec4(1.0); return; }

  vec3 p = viewPosFromDepth(vUv, d);

  // Pick the closer neighbour on each axis so silhouettes don't smear a normal
  // across the depth discontinuity.
  vec2 dx = vec2(uTexel.x, 0.0), dy = vec2(0.0, uTexel.y);
  vec3 pl = viewPosFromDepth(vUv - dx, depthAt(vUv - dx));
  vec3 pr = viewPosFromDepth(vUv + dx, depthAt(vUv + dx));
  vec3 pd = viewPosFromDepth(vUv - dy, depthAt(vUv - dy));
  vec3 pu = viewPosFromDepth(vUv + dy, depthAt(vUv + dy));
  vec3 ddx = abs(pr.z - p.z) < abs(p.z - pl.z) ? pr - p : p - pl;
  vec3 ddy = abs(pu.z - p.z) < abs(p.z - pd.z) ? pu - p : p - pd;
  vec3 n = normalize(cross(ddx, ddy));

  float ang = ign(gl_FragCoord.xy + uFrame * 5.588238) * 6.2831853;
  vec3 rv = vec3(cos(ang), sin(ang), 0.0);
  vec3 t = normalize(rv - n * dot(rv, n));
  vec3 b = cross(n, t);
  mat3 tbn = mat3(t, b, n);

  // Constant world-space radius: contact darkening is a physical near-field
  // effect and must not change scale as the player walks toward a corner.
  float radius = uRadius;

  float occ = 0.0;
  for (int i = 0; i < SAMPLES; i++) {
    vec3 sp = p + (tbn * uKernel[i]) * radius;
    vec4 off = uProj * vec4(sp, 1.0);
    vec2 suv = (off.xy / off.w) * 0.5 + 0.5;
    if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) continue;
    float sd = depthAt(suv);
    vec3 sPos = viewPosFromDepth(suv, sd);
    float dz = sPos.z - sp.z;                        // view z is negative forward
    float range = smoothstep(0.0, 1.0, radius / max(1e-4, abs(p.z - sPos.z)));
    occ += step(uBias, dz) * range;
  }
  float ao = 1.0 - (occ / float(SAMPLES)) * uIntensity;
  gl_FragColor = vec4(clamp(ao, 0.0, 1.0), 0.0, 0.0, 1.0);
}`;

// Depth-aware cross blur, half res. 8 taps, separable-ish in one go — the AO is
// already low frequency so a wider kernel buys nothing.
const AO_BLUR_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tAO;
uniform sampler2D tDepth;
uniform vec2 uTexel;
uniform float uDepthSigma;

void main() {
  float dc = texture2D(tDepth, vUv).x;
  float sum = texture2D(tAO, vUv).r;
  float wsum = 1.0;
  for (int i = 0; i < 4; i++) {
    vec2 o = vec2(0.0);
    if (i == 0) o = vec2(uTexel.x, 0.0);
    else if (i == 1) o = vec2(-uTexel.x, 0.0);
    else if (i == 2) o = vec2(0.0, uTexel.y);
    else o = vec2(0.0, -uTexel.y);
    for (int k = 1; k <= 2; k++) {
      vec2 uv = vUv + o * float(k);
      float dd = texture2D(tDepth, uv).x;
      float w = exp(-abs(dd - dc) * uDepthSigma) / float(k);
      sum += texture2D(tAO, uv).r * w;
      wsum += w;
    }
  }
  gl_FragColor = vec4(sum / wsum, 0.0, 0.0, 1.0);
}`;

// ---------------------------------------------------------------------------
// TAA resolve
// ---------------------------------------------------------------------------

const TAA_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tCur;
uniform sampler2D tHist;
uniform sampler2D tDepth;
uniform sampler2D tAO;
uniform mat4 uInvViewProj;   // inverse of THIS frame's jittered view-projection
uniform mat4 uPrevViewProj;  // LAST frame's un-jittered view-projection
uniform vec2 uTexel;
uniform float uFeedback;
uniform float uReset;
uniform float uNearSplit;
uniform float uAoStrength;
${RECONSTRUCT}
${LUMA}

vec3 toYCoCg(vec3 c) {
  return vec3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
              0.5 * c.r - 0.5 * c.b,
             -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
vec3 toRGB(vec3 c) {
  float t = c.x - c.z;
  return vec3(t + c.y, c.x + c.z, t - c.y);
}

void main() {
  vec3 cur = texture2D(tCur, vUv).rgb;

  // 3x3 neighbourhood in YCoCg: mean/variance gives a much tighter clip volume
  // than min/max, which is the difference between "stable" and "smeary".
  vec3 m1 = vec3(0.0), m2 = vec3(0.0);
  vec3 nmin = vec3(1e9), nmax = vec3(-1e9);
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec3 s = toYCoCg(texture2D(tCur, vUv + vec2(float(x), float(y)) * uTexel).rgb);
      m1 += s; m2 += s * s;
      nmin = min(nmin, s); nmax = max(nmax, s);
    }
  }
  vec3 mean = m1 / 9.0;
  vec3 sigma = sqrt(max(vec3(0.0), m2 / 9.0 - mean * mean));
  vec3 lo = max(nmin, mean - 1.25 * sigma);
  vec3 hi = min(nmax, mean + 1.25 * sigma);

#ifdef USE_AO
  float ao = texture2D(tAO, vUv).r;
  cur *= mix(1.0, ao, uAoStrength);
#endif

  // Closest-of-5 depth so thin edges reproject with the foreground, not the
  // background sliding out from behind them.
  float d = texture2D(tDepth, vUv).x;
  vec2 duv = vUv;
  float d1 = texture2D(tDepth, vUv + vec2(uTexel.x, 0.0)).x;
  float d2 = texture2D(tDepth, vUv - vec2(uTexel.x, 0.0)).x;
  float d3 = texture2D(tDepth, vUv + vec2(0.0, uTexel.y)).x;
  float d4 = texture2D(tDepth, vUv - vec2(0.0, uTexel.y)).x;
  if (d1 < d) { d = d1; duv = vUv + vec2(uTexel.x, 0.0); }
  if (d2 < d) { d = d2; duv = vUv - vec2(uTexel.x, 0.0); }
  if (d3 < d) { d = d3; duv = vUv + vec2(0.0, uTexel.y); }
  if (d4 < d) { d = d4; duv = vUv - vec2(0.0, uTexel.y); }

  vec4 wp = uInvViewProj * vec4(duv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
  wp /= wp.w;
  vec4 pp = uPrevViewProj * wp;
  vec2 prevUv = (pp.xy / pp.w) * 0.5 + 0.5;

  vec3 histY = toYCoCg(texture2D(tHist, prevUv).rgb);

  // The viewmodel is parented to the camera: in un-jittered screen space it does
  // not move when the camera turns, but world reprojection says it flew across
  // the frame. For near pixels, also try the identity motion vector and keep
  // whichever history is closer to what we actually see this frame.
  float viewZ = -viewPosFromDepth(vUv, texture2D(tDepth, vUv).x).z;
  if (viewZ < uNearSplit) {
    vec3 altY = toYCoCg(texture2D(tHist, vUv).rgb);
    vec3 cY = toYCoCg(cur);
    float e1 = abs(histY.x - cY.x);
    float e2 = abs(altY.x - cY.x);
    if (e2 < e1) { histY = altY; prevUv = vUv; }
  }

  vec3 clipped = clamp(histY, lo, hi);
  vec3 hist = toRGB(clipped);

  float onScreen = (prevUv.x < 0.0 || prevUv.x > 1.0 || prevUv.y < 0.0 || prevUv.y > 1.0) ? 0.0 : 1.0;
  // Rejection is proportional to how far the history had to be dragged back in:
  // a hard clip means the sample was wrong, so trust it less.
  float clipDist = length(histY - clipped) / max(1e-3, sigma.x + 0.15);
  float fb = uFeedback * onScreen * (1.0 - uReset) / (1.0 + clipDist * 1.5);

  // Luminance-weighted blend (Karis): stops a single bright sample from
  // strobing through the accumulation as a firefly.
  float wc = (1.0 - fb) / (1.0 + luma(cur));
  float wh = fb / (1.0 + luma(hist));
  vec3 outc = (cur * wc + hist * wh) / max(1e-5, wc + wh);

  gl_FragColor = vec4(max(vec3(0.0), outc), 1.0);
}`;

// ---------------------------------------------------------------------------
// eye adaptation: 64x36 log-luma, then 1x1 with temporal hysteresis
// ---------------------------------------------------------------------------

const LUM_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tColor;
uniform vec2 uTexel;
${LUMA}
void main() {
  vec3 c = texture2D(tColor, vUv).rgb
         + texture2D(tColor, vUv + vec2( uTexel.x,  uTexel.y)).rgb
         + texture2D(tColor, vUv + vec2(-uTexel.x,  uTexel.y)).rgb
         + texture2D(tColor, vUv + vec2( uTexel.x, -uTexel.y)).rgb;
  gl_FragColor = vec4(log2(max(luma(c * 0.25), 1e-4)), 0.0, 0.0, 1.0);
}`;

const ADAPT_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tLum;
uniform sampler2D tPrev;
uniform float uRate;
void main() {
  float sum = 0.0;
  // Centre-weighted: what the player is looking at should drive the exposure,
  // not a bright sliver of sky in the corner.
  float wsum = 0.0;
  for (int y = 0; y < 6; y++) {
    for (int x = 0; x < 8; x++) {
      vec2 uv = (vec2(float(x), float(y)) + 0.5) / vec2(8.0, 6.0);
      float w = 1.0 - 0.6 * length(uv - 0.5) * 1.4142;
      sum += texture2D(tLum, uv).r * w;
      wsum += w;
    }
  }
  float cur = sum / wsum;
  float prev = texture2D(tPrev, vec2(0.5)).r;
  gl_FragColor = vec4(mix(prev, cur, uRate), 0.0, 0.0, 1.0);
}`;

// ---------------------------------------------------------------------------
// bloom: threshold + dual-filter pyramid
// ---------------------------------------------------------------------------

const PREFILTER_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tColor;
uniform sampler2D tAdapt;
uniform vec2 uTexel;
uniform float uThreshold;
uniform float uKnee;
uniform float uExposure;
uniform float uAutoKey;
uniform float uAutoMin;
uniform float uAutoMax;
${LUMA}

vec3 tap(vec2 uv) { return max(vec3(0.0), texture2D(tColor, uv).rgb); }

void main() {
  // Karis-weighted 5 tap box: kills the single-pixel fireflies that otherwise
  // pump through the pyramid as flicker.
  vec3 c0 = tap(vUv);
  vec3 c1 = tap(vUv + vec2( uTexel.x,  uTexel.y));
  vec3 c2 = tap(vUv + vec2(-uTexel.x,  uTexel.y));
  vec3 c3 = tap(vUv + vec2( uTexel.x, -uTexel.y));
  vec3 c4 = tap(vUv + vec2(-uTexel.x, -uTexel.y));
  float w0 = 1.0 / (1.0 + luma(c0));
  float w1 = 1.0 / (1.0 + luma(c1));
  float w2 = 1.0 / (1.0 + luma(c2));
  float w3 = 1.0 / (1.0 + luma(c3));
  float w4 = 1.0 / (1.0 + luma(c4));
  vec3 c = (c0 * w0 * 2.0 + c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4) /
           (w0 * 2.0 + w1 + w2 + w3 + w4);

  float ex = uExposure;
#ifdef USE_AUTOEXP
  ex *= clamp(uAutoKey / max(1e-4, exp2(texture2D(tAdapt, vec2(0.5)).r)), uAutoMin, uAutoMax);
#endif
  c *= ex;

  // Soft knee so a surface easing past the threshold doesn't pop.
  float br = max(c.r, max(c.g, c.b));
  float rq = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  rq = rq * rq / (4.0 * uKnee + 1e-5);
  float w = max(rq, br - uThreshold) / max(br, 1e-5);
  gl_FragColor = vec4(c * w, 1.0);
}`;

const DOWN_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tColor;
uniform vec2 uTexel;   // texel size of the SOURCE
void main() {
  vec2 o = uTexel;
  vec4 s = texture2D(tColor, vUv) * 4.0;
  s += texture2D(tColor, vUv + vec2( o.x,  o.y));
  s += texture2D(tColor, vUv + vec2(-o.x,  o.y));
  s += texture2D(tColor, vUv + vec2( o.x, -o.y));
  s += texture2D(tColor, vUv + vec2(-o.x, -o.y));
  gl_FragColor = s / 8.0;
}`;

const UP_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tColor;
uniform vec2 uTexel;   // texel size of the SOURCE (the smaller mip)
uniform float uScale;
void main() {
  vec2 o = uTexel;
  vec4 s = texture2D(tColor, vUv + vec2(-o.x,  o.y)) * 1.0
         + texture2D(tColor, vUv + vec2( 0.0,  o.y)) * 2.0
         + texture2D(tColor, vUv + vec2( o.x,  o.y)) * 1.0
         + texture2D(tColor, vUv + vec2(-o.x,  0.0)) * 2.0
         + texture2D(tColor, vUv)                    * 4.0
         + texture2D(tColor, vUv + vec2( o.x,  0.0)) * 2.0
         + texture2D(tColor, vUv + vec2(-o.x, -o.y)) * 1.0
         + texture2D(tColor, vUv + vec2( 0.0, -o.y)) * 2.0
         + texture2D(tColor, vUv + vec2( o.x, -o.y)) * 1.0;
  gl_FragColor = (s / 16.0) * uScale;
}`;

// ---------------------------------------------------------------------------
// composite: the whole finishing chain in one full-res pass
// ---------------------------------------------------------------------------

const COMPOSITE_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tColor;
uniform sampler2D tBloom;
uniform sampler2D tAO;
uniform sampler2D tAdapt;
uniform vec2 uTexel;
uniform float uExposure;
uniform float uAutoKey;
uniform float uAutoMin;
uniform float uAutoMax;
uniform float uBloom;
uniform float uAoStrength;
uniform float uVignette;
uniform float uCA;
uniform float uGrain;
uniform float uSharpen;
uniform float uFrame;
uniform vec3 uLift;
uniform vec3 uGain;
uniform float uGammaGrade;
uniform float uSaturation;
uniform float uContrast;
${LUMA}

vec3 fetch(vec2 uv) { return max(vec3(0.0), texture2D(tColor, uv).rgb); }

// Uchimura's generalised filmic curve. Chosen over ACES because the toe, the
// linear section and the shoulder are independently controllable, and the
// shoulder is an exponential asymptote to the peak — it cannot hard-clip.
float curve(float x) {
  const float P = 1.0;    // peak
  const float a = 1.06;   // linear slope
  const float m = 0.21;   // linear section start
  const float l = 0.36;   // linear section length
  const float c = 1.28;   // toe curvature
  const float b = 0.0;    // toe pedestal
  float l0 = ((P - m) * l) / a;
  float S0 = m + l0;
  float S1 = m + a * l0;
  float C2 = (a * P) / (P - S1);
  float CP = -C2 / P;
  float w0 = 1.0 - smoothstep(0.0, m, x);
  float w2 = step(m + l0, x);
  float w1 = 1.0 - w0 - w2;
  float T = m * pow(max(x / m, 1e-5), c) + b;
  float S = P - (P - S1) * exp(CP * (x - S0));
  float L = m + a * (x - m);
  return T * w0 + L * w1 + S * w2;
}

vec3 filmic(vec3 c) {
  // Split tone in scene-referred linear: cool the shadows, warm the highlights.
  // Doing it before the curve keeps it from reading as a colour filter.
  float l = luma(c);
  float t = l / (l + 0.55);
  c *= mix(vec3(0.945, 0.985, 1.085), vec3(1.075, 1.005, 0.918), t);

  // Highlight crosstalk: real film and real sensors desaturate as they clip.
  float peak = max(c.r, max(c.g, c.b));
  c = mix(c, vec3(peak), smoothstep(0.75, 4.0, peak) * 0.5);

  c = vec3(curve(c.r), curve(c.g), curve(c.b));

  // Overall slight desaturation — the reference look is never fully saturated.
  c = mix(vec3(luma(c)), c, uSaturation);
  return c;
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 srgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

void main() {
  vec2 d = vUv - 0.5;
  float r2 = dot(d, d);

  vec3 c;
#ifdef USE_CA
  // Lateral chromatic aberration only — zero in the centre third, quartic to the
  // corners, so it reads as glass and never as a filter.
  float ca = uCA * r2 * r2 * 4.0;
  c.r = fetch(vUv - d * ca).r;
  c.g = fetch(vUv).g;
  c.b = fetch(vUv + d * ca).b;
#else
  c = fetch(vUv);
#endif

#ifdef USE_FXAA
  // Edge-directional blend on tone-mapped luma. Cheap console-style FXAA: it
  // exists so the 'low' and 'medium' tiers still hold an edge without TAA.
  vec3 nw = fetch(vUv + vec2(-uTexel.x, -uTexel.y));
  vec3 ne = fetch(vUv + vec2( uTexel.x, -uTexel.y));
  vec3 sw = fetch(vUv + vec2(-uTexel.x,  uTexel.y));
  vec3 se = fetch(vUv + vec2( uTexel.x,  uTexel.y));
  float lnw = luma(nw / (1.0 + nw)), lne = luma(ne / (1.0 + ne));
  float lsw = luma(sw / (1.0 + sw)), lse = luma(se / (1.0 + se));
  float lm  = luma(c / (1.0 + c));
  float lmin = min(lm, min(min(lnw, lne), min(lsw, lse)));
  float lmax = max(lm, max(max(lnw, lne), max(lsw, lse)));
  if (lmax - lmin > max(0.035, lmax * 0.14)) {
    vec2 dir = vec2(-((lnw + lne) - (lsw + lse)), ((lnw + lsw) - (lne + lse)));
    float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + 0.03125);
    dir = clamp(dir * rcp, -8.0, 8.0) * uTexel;
    vec3 a = 0.5 * (fetch(vUv + dir * (1.0 / 3.0 - 0.5)) + fetch(vUv + dir * (2.0 / 3.0 - 0.5)));
    vec3 b = a * 0.5 + 0.25 * (fetch(vUv - dir * 0.5) + fetch(vUv + dir * 0.5));
    float lb = luma(b / (1.0 + b));
    c = (lb < lmin || lb > lmax) ? a : b;
  }
#endif

#ifdef USE_SHARPEN
  // TAA trades a little sharpness for stability; take it back with a clamped
  // unsharp mask rather than by lowering the feedback (which reintroduces crawl).
  vec3 blur = (fetch(vUv + vec2(uTexel.x, 0.0)) + fetch(vUv - vec2(uTexel.x, 0.0))
             + fetch(vUv + vec2(0.0, uTexel.y)) + fetch(vUv - vec2(0.0, uTexel.y))) * 0.25;
  vec3 delta = c - blur;
  c += clamp(delta, -0.35, 0.35) * uSharpen;
  c = max(c, vec3(0.0));
#endif

#ifdef USE_AO
  float ao = texture2D(tAO, vUv).r;
  c *= mix(1.0, ao, uAoStrength);
#endif

#ifdef USE_BLOOM
  c += texture2D(tBloom, vUv).rgb * uBloom;
#endif

  float ex = uExposure;
#ifdef USE_AUTOEXP
  ex *= clamp(uAutoKey / max(1e-4, exp2(texture2D(tAdapt, vec2(0.5)).r)), uAutoMin, uAutoMax);
#endif
  c *= ex;

  c = filmic(c);

  // Display-referred grade: lift/gamma/gain, then a soft S on the midtones.
  c = pow(max(c, vec3(0.0)), vec3(uGammaGrade));
  c = c * uGain + uLift * (1.0 - c);
  c = mix(c, c * c * (3.0 - 2.0 * c), uContrast);

  float vig = 1.0 - uVignette * smoothstep(0.12, 0.72, r2);
  c *= vig;

  c = srgb(c);

#ifdef USE_GRAIN
  // Grain goes on AFTER the transfer function. Added in linear it would be
  // perceptually enormous in the shadows — the encode curve has a slope of ~12
  // near black, so a 1% linear perturbation lands as a 13% display step.
  // Weighted toward the darks, where a real sensor's noise floor lives.
  float g = hash12(gl_FragCoord.xy + uFrame * 17.371) - 0.5;
  float shadowW = 1.0 - smoothstep(0.0, 0.62, luma(c));
  c += g * uGrain * (0.35 + shadowW * 0.65);
#endif

  // Triangular dither, one 8-bit LSB: removes gradient banding in the sky.
  float dth = (hash12(gl_FragCoord.xy + uFrame * 3.117) - hash12(gl_FragCoord.yx + 11.0 + uFrame * 3.117)) / 255.0;
  gl_FragColor = vec4(c + dth, 1.0);
}`;

// ---------------------------------------------------------------------------
// quality tiers
// ---------------------------------------------------------------------------

const TIERS = {
  low:    { taa: false, fxaa: true,  ao: false, aoBlur: false, aoSamples: 0,  bloom: false, mips: 0, ca: false, grain: false, sharpen: false, autoExp: false, shadow: 1024, bloomStrength: 0.0 },
  medium: { taa: false, fxaa: true,  ao: true,  aoBlur: true,  aoSamples: 6,  bloom: true,  mips: 3, ca: false, grain: true,  sharpen: false, autoExp: true,  shadow: 1536, bloomStrength: 0.055 },
  high:   { taa: true,  fxaa: false, ao: true,  aoBlur: true,  aoSamples: 10, bloom: true,  mips: 5, ca: true,  grain: true,  sharpen: true,  autoExp: true,  shadow: 2048, bloomStrength: 0.062 },
  ultra:  { taa: true,  fxaa: false, ao: true,  aoBlur: true,  aoSamples: 16, bloom: true,  mips: 6, ca: true,  grain: true,  sharpen: true,  autoExp: true,  shadow: 2048, bloomStrength: 0.065 },
};

// Halton(2,3) — the standard low-discrepancy jitter. 8 phases is enough to fill
// a pixel evenly without stretching the history so far that it ghosts.
function halton(i, base) {
  let f = 1, r = 0, n = i;
  while (n > 0) { f /= base; r += f * (n % base); n = Math.floor(n / base); }
  return r;
}

export class RenderPipeline {
  constructor(canvas) {
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      powerPreference: 'high-performance',
      stencil: false,
      depth: true,
    });
    renderer.setPixelRatio(1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    // Tone mapping is ours, in the composite pass, against the full HDR range.
    // (three skips its own tone mapping when the destination is a render target
    // anyway, so leaving it on would only affect the quad passes.)
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    // We issue ~10 draw calls after the scene render; with autoReset on, the perf
    // sampler would only ever see the last one. Reset once per frame ourselves so
    // renderer.info reports the whole frame honestly.
    renderer.info.autoReset = false;

    this.renderer = renderer;
    this.quality = 'high';
    this.tier = TIERS.high;
    this.width = 1920;
    this.height = 1080;
    this.renderScale = 1;
    this.rtWidth = 0;
    this.rtHeight = 0;

    // Controllable grade knobs (public: other tooling may poke these).
    this.exposure = 1.0;
    this.autoExposure = true;

    this._frame = 0;
    this._historyValid = false;
    this._histIndex = 0;
    this._adaptIndex = 0;
    this._shadowTexels = 0;
    this._shadowScanAt = -1e9;
    this._passCost = 1;

    this._jitter = [];
    for (let i = 1; i <= 8; i++) this._jitter.push([halton(i, 2) - 0.5, halton(i, 3) - 0.5]);

    // Scratch — nothing in render() allocates.
    this._mView = new THREE.Matrix4();
    this._mViewProj = new THREE.Matrix4();
    this._mPrevViewProj = new THREE.Matrix4();
    this._mJitProj = new THREE.Matrix4();
    this._mInvViewProj = new THREE.Matrix4();
    this._prevCamPos = new THREE.Vector3(1e9, 1e9, 1e9);

    this._targets = {};
    this._bloomMips = [];

    this._quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 2, 0, 0, 2]), 2));
    this._quadGeo = geo;
    this._quad = new THREE.Mesh(geo, null);
    this._quad.frustumCulled = false;
    this._quadScene = new THREE.Scene();
    this._quadScene.add(this._quad);

    this._buildMaterials();
    this.setQuality('high');
  }

  // -------------------------------------------------------------- materials
  _buildMaterials() {
    const kernel = [];
    const rng = makeRng('render:ssao:kernel');
    for (let i = 0; i < 16; i++) {
      const v = new THREE.Vector3(rng.range(-1, 1), rng.range(-1, 1), rng.range(0.15, 1)).normalize();
      // Cluster samples toward the origin: contact darkening is a near-field effect.
      const t = i / 16;
      v.multiplyScalar(0.25 + 0.75 * t * t);
      kernel.push(v);
    }

    const mk = (frag, uniforms, defines = {}, blending = THREE.NoBlending) => new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: frag,
      uniforms,
      defines,
      depthTest: false,
      depthWrite: false,
      blending,
      toneMapped: false,
    });

    this.mat = {};

    this.mat.ao = mk(AO_FRAG, {
      tDepth: { value: null },
      uProj: { value: new THREE.Matrix4() },
      uProjInv: { value: new THREE.Matrix4() },
      uTexel: { value: new THREE.Vector2() },
      uKernel: { value: kernel },
      uRadius: { value: 0.65 },
      uBias: { value: 0.022 },
      uIntensity: { value: 1.0 },
      uFrame: { value: 0 },
    }, { SAMPLES: 10 });

    this.mat.aoBlur = mk(AO_BLUR_FRAG, {
      tAO: { value: null },
      tDepth: { value: null },
      uTexel: { value: new THREE.Vector2() },
      uDepthSigma: { value: 900.0 },
    });

    this.mat.taa = mk(TAA_FRAG, {
      tCur: { value: null },
      tHist: { value: null },
      tDepth: { value: null },
      tAO: { value: null },
      uInvViewProj: { value: new THREE.Matrix4() },
      uPrevViewProj: { value: new THREE.Matrix4() },
      uProjInv: { value: new THREE.Matrix4() },
      uTexel: { value: new THREE.Vector2() },
      uFeedback: { value: 0.9 },
      uReset: { value: 0 },
      uNearSplit: { value: 0.8 },
      uAoStrength: { value: 0.85 },
    }, { USE_AO: '' });

    this.mat.lum = mk(LUM_FRAG, {
      tColor: { value: null },
      uTexel: { value: new THREE.Vector2() },
    });

    this.mat.adapt = mk(ADAPT_FRAG, {
      tLum: { value: null },
      tPrev: { value: null },
      uRate: { value: 1.0 },
    });

    this.mat.prefilter = mk(PREFILTER_FRAG, {
      tColor: { value: null },
      tAdapt: { value: null },
      uTexel: { value: new THREE.Vector2() },
      uThreshold: { value: 1.05 },
      uKnee: { value: 0.55 },
      uExposure: { value: 1.0 },
      uAutoKey: { value: 0.17 },
      uAutoMin: { value: 0.5 },
      uAutoMax: { value: 2.2 },
    }, { USE_AUTOEXP: '' });

    this.mat.down = mk(DOWN_FRAG, {
      tColor: { value: null },
      uTexel: { value: new THREE.Vector2() },
    });

    this.mat.up = mk(UP_FRAG, {
      tColor: { value: null },
      uTexel: { value: new THREE.Vector2() },
      uScale: { value: 1.0 },
    }, {}, THREE.AdditiveBlending);

    this.mat.composite = mk(COMPOSITE_FRAG, {
      tColor: { value: null },
      tBloom: { value: null },
      tAO: { value: null },
      tAdapt: { value: null },
      uTexel: { value: new THREE.Vector2() },
      uExposure: { value: 1.0 },
      uAutoKey: { value: 0.17 },
      uAutoMin: { value: 0.5 },
      uAutoMax: { value: 2.2 },
      uBloom: { value: 0.062 },
      uAoStrength: { value: 0.85 },
      uVignette: { value: 0.34 },
      uCA: { value: 0.0022 },
      uGrain: { value: 0.022 },
      uSharpen: { value: 0.38 },
      uFrame: { value: 0 },
      uLift: { value: new THREE.Vector3(0.006, 0.0085, 0.0135) },
      uGain: { value: new THREE.Vector3(1.005, 1.0, 0.992) },
      uGammaGrade: { value: 0.985 },
      uSaturation: { value: 0.9 },
      uContrast: { value: 0.16 },
    }, { USE_BLOOM: '', USE_SHARPEN: '' });
  }

  _applyDefines() {
    const t = this.tier;
    const cd = {};
    if (t.bloom) cd.USE_BLOOM = '';
    if (t.fxaa) cd.USE_FXAA = '';
    if (t.ca) cd.USE_CA = '';
    if (t.grain) cd.USE_GRAIN = '';
    if (t.sharpen) cd.USE_SHARPEN = '';
    if (t.autoExp && this.autoExposure) cd.USE_AUTOEXP = '';
    // With TAA on, AO is folded into the resolve so it accumulates temporally
    // instead of shimmering; without TAA the composite has to apply it.
    if (t.ao && !t.taa) cd.USE_AO = '';
    this.mat.composite.defines = cd;
    this.mat.composite.needsUpdate = true;

    this.mat.taa.defines = t.ao ? { USE_AO: '' } : {};
    this.mat.taa.needsUpdate = true;

    this.mat.ao.defines = { SAMPLES: Math.max(4, t.aoSamples) };
    this.mat.ao.needsUpdate = true;

    this.mat.prefilter.defines = (t.autoExp && this.autoExposure) ? { USE_AUTOEXP: '' } : {};
    this.mat.prefilter.needsUpdate = true;

    this.mat.composite.uniforms.uBloom.value = t.bloomStrength;
    this.mat.composite.uniforms.uVignette.value = this.quality === 'low' ? 0.24 : 0.34;
    this.mat.composite.uniforms.uAoStrength.value = this.quality === 'ultra' ? 0.9 : 0.85;
    this.mat.taa.uniforms.uAoStrength.value = this.mat.composite.uniforms.uAoStrength.value;
  }

  // ---------------------------------------------------------------- targets
  _makeColorRT(w, h, opts = {}) {
    const rt = new THREE.WebGLRenderTarget(Math.max(1, w), Math.max(1, h), {
      type: opts.type || THREE.HalfFloatType,
      format: opts.format || THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
      depthBuffer: !!opts.depth,
      stencilBuffer: false,
      generateMipmaps: false,
    });
    rt.texture.colorSpace = THREE.NoColorSpace;
    return rt;
  }

  _ensureTargets(w, h) {
    if (w === this.rtWidth && h === this.rtHeight && this._targets.scene) return;
    this._disposeTargets();
    this.rtWidth = w;
    this.rtHeight = h;

    const hw = Math.max(1, Math.floor(w / 2));
    const hh = Math.max(1, Math.floor(h / 2));

    const scene = this._makeColorRT(w, h, { depth: true });
    const depth = new THREE.DepthTexture(w, h, THREE.UnsignedIntType);
    depth.format = THREE.DepthFormat;
    depth.minFilter = THREE.NearestFilter;
    depth.magFilter = THREE.NearestFilter;
    scene.depthTexture = depth;

    this._targets = {
      scene,
      hist: [this._makeColorRT(w, h), this._makeColorRT(w, h)],
      ao: this._makeColorRT(hw, hh, { type: THREE.UnsignedByteType, format: THREE.RedFormat }),
      aoBlur: this._makeColorRT(hw, hh, { type: THREE.UnsignedByteType, format: THREE.RedFormat }),
      lum: this._makeColorRT(64, 36),
      adapt: [this._makeColorRT(1, 1), this._makeColorRT(1, 1)],
    };

    this._bloomMips.length = 0;
    let mw = hw, mh = hh;
    for (let i = 0; i < 7; i++) {
      if (mw < 8 || mh < 8) break;
      this._bloomMips.push(this._makeColorRT(mw, mh));
      mw = Math.max(1, Math.floor(mw / 2));
      mh = Math.max(1, Math.floor(mh / 2));
    }

    this._historyValid = false;
    this._adaptPrimed = false;
  }

  _disposeTargets() {
    const t = this._targets;
    if (t.scene) { t.scene.depthTexture?.dispose(); t.scene.dispose(); }
    if (t.hist) t.hist.forEach((r) => r.dispose());
    if (t.ao) t.ao.dispose();
    if (t.aoBlur) t.aoBlur.dispose();
    if (t.lum) t.lum.dispose();
    if (t.adapt) t.adapt.forEach((r) => r.dispose());
    this._bloomMips.forEach((r) => r.dispose());
    this._bloomMips.length = 0;
    this._targets = {};
  }

  // ------------------------------------------------------------------- api
  setSize(w, h, renderScale = 1) {
    // Quantise DRS so a 0.025 nudge from the adaptive controller doesn't
    // reallocate ~65MB of render targets and flush the TAA history.
    const s = Math.max(0.4, Math.min(1, Math.round(renderScale * 16) / 16));
    const rw = Math.max(2, Math.round(w * s));
    const rh = Math.max(2, Math.round(h * s));
    this.width = w; this.height = h; this.renderScale = s;
    if (rw !== this.rtWidth || rh !== this.rtHeight) {
      this.renderer.setSize(rw, rh, false);
      this._ensureTargets(rw, rh);
    }
    const c = this.renderer.domElement;
    if (c.style.width !== w + 'px') c.style.width = w + 'px';
    if (c.style.height !== h + 'px') c.style.height = h + 'px';
  }

  setQuality(q) {
    const tier = TIERS[q];
    if (!tier) return;
    if (this.quality === q && this._tierApplied) return;
    this.quality = q;
    this.tier = tier;
    this._tierApplied = true;
    this.renderer.shadowMap.type = q === 'low' ? THREE.PCFShadowMap : THREE.PCFSoftShadowMap;
    this._applyDefines();
    this._historyValid = false;
  }

  /** Manual exposure multiplier; auto-exposure (when on) scales around it. */
  setExposure(v) { this.exposure = v; }

  // ---------------------------------------------------------------- passes
  _pass(material, target, clear = true) {
    const r = this.renderer;
    r.autoClear = clear;
    r.setRenderTarget(target || null);
    this._quad.material = material;
    r.render(this._quadScene, this._quadCam);
    r.autoClear = true;
  }

  render(scene, camera, dt = 1 / 60) {
    const r = this.renderer;
    const t = this.tier;
    r.info.reset();

    if (!this._targets.scene) this._ensureTargets(this.rtWidth || 2, this.rtHeight || 2);
    const W = this.rtWidth, H = this.rtHeight;
    const T = this._targets;

    // --- camera matrices ---------------------------------------------------
    camera.updateMatrixWorld();
    this._mView.copy(camera.matrixWorld).invert();
    this._mViewProj.multiplyMatrices(camera.projectionMatrix, this._mView);

    let reset = 0;
    if (!this._historyValid) reset = 1;
    // A teleport (bench shot setup, respawn) invalidates every motion vector.
    if (this._prevCamPos.distanceToSquared(camera.position) > 4) reset = 1;
    this._prevCamPos.copy(camera.position);

    // --- jittered scene render into HDR ------------------------------------
    const useTaa = t.taa;
    let jx = 0, jy = 0;
    if (useTaa) {
      const j = this._jitter[this._frame % this._jitter.length];
      jx = (j[0] * 2) / W;
      jy = (j[1] * 2) / H;
      camera.projectionMatrix.elements[8] += jx;
      camera.projectionMatrix.elements[9] += jy;
    }
    this._mJitProj.copy(camera.projectionMatrix);

    r.autoClear = true;
    r.setRenderTarget(T.scene);
    r.render(scene, camera);

    if (useTaa) {
      camera.projectionMatrix.elements[8] -= jx;
      camera.projectionMatrix.elements[9] -= jy;
    }

    // Inverse of the matrices that actually produced this depth buffer.
    this._mInvViewProj.multiplyMatrices(this._mJitProj, this._mView).invert();

    // --- SSAO (half res) ---------------------------------------------------
    let aoTex = null;
    if (t.ao) {
      const u = this.mat.ao.uniforms;
      u.tDepth.value = T.scene.depthTexture;
      u.uProj.value.copy(this._mJitProj);
      u.uProjInv.value.copy(this._mJitProj).invert();
      u.uTexel.value.set(1 / T.ao.width, 1 / T.ao.height);
      u.uFrame.value = useTaa ? this._frame % 8 : 0;
      this._pass(this.mat.ao, T.ao);
      aoTex = T.ao.texture;
      if (t.aoBlur) {
        const b = this.mat.aoBlur.uniforms;
        b.tAO.value = T.ao.texture;
        b.tDepth.value = T.scene.depthTexture;
        b.uTexel.value.set(1 / T.ao.width, 1 / T.ao.height);
        this._pass(this.mat.aoBlur, T.aoBlur);
        aoTex = T.aoBlur.texture;
      }
    }

    // --- TAA resolve -------------------------------------------------------
    let colorTex = T.scene.texture;
    if (useTaa) {
      const write = T.hist[this._histIndex];
      const read = T.hist[1 - this._histIndex];
      const u = this.mat.taa.uniforms;
      u.tCur.value = T.scene.texture;
      u.tHist.value = read.texture;
      u.tDepth.value = T.scene.depthTexture;
      u.tAO.value = aoTex;
      u.uInvViewProj.value.copy(this._mInvViewProj);
      u.uPrevViewProj.value.copy(reset ? this._mViewProj : this._mPrevViewProj);
      u.uProjInv.value.copy(this._mJitProj).invert();
      u.uTexel.value.set(1 / W, 1 / H);
      u.uFeedback.value = this.quality === 'ultra' ? 0.92 : 0.9;
      u.uReset.value = reset;
      this._pass(this.mat.taa, write);
      colorTex = write.texture;
      this._histIndex = 1 - this._histIndex;
      this._historyValid = true;
    }

    // --- eye adaptation (64x36 -> 1x1, no readback, no CPU stall) ----------
    const useAuto = t.autoExp && this.autoExposure;
    if (useAuto) {
      const lu = this.mat.lum.uniforms;
      lu.tColor.value = colorTex;
      lu.uTexel.value.set(1 / W, 1 / H);
      this._pass(this.mat.lum, T.lum);

      const aw = T.adapt[this._adaptIndex];
      const ar = T.adapt[1 - this._adaptIndex];
      const au = this.mat.adapt.uniforms;
      au.tLum.value = T.lum.texture;
      au.tPrev.value = ar.texture;
      // Snap on the first frame after a resize/quality change so a static camera
      // is fully converged immediately — screenshots must not depend on dt.
      au.uRate.value = this._adaptPrimed ? 1 - Math.exp(-Math.min(0.1, dt) * 2.6) : 1.0;
      this._pass(this.mat.adapt, aw);
      this._adaptIndex = 1 - this._adaptIndex;
      this._adaptPrimed = true;
      this.mat.composite.uniforms.tAdapt.value = aw.texture;
      this.mat.prefilter.uniforms.tAdapt.value = aw.texture;
    }

    // --- bloom pyramid -----------------------------------------------------
    const mips = Math.min(t.mips, this._bloomMips.length);
    if (t.bloom && mips > 0) {
      const p = this.mat.prefilter.uniforms;
      p.tColor.value = colorTex;
      p.uTexel.value.set(1 / W, 1 / H);
      p.uExposure.value = this.exposure;
      this._pass(this.mat.prefilter, this._bloomMips[0]);

      for (let i = 1; i < mips; i++) {
        const src = this._bloomMips[i - 1];
        const d = this.mat.down.uniforms;
        d.tColor.value = src.texture;
        d.uTexel.value.set(1 / src.width, 1 / src.height);
        this._pass(this.mat.down, this._bloomMips[i]);
      }
      for (let i = mips - 1; i > 0; i--) {
        const src = this._bloomMips[i];
        const u = this.mat.up.uniforms;
        u.tColor.value = src.texture;
        u.uTexel.value.set(1 / src.width, 1 / src.height);
        u.uScale.value = 0.82;
        this._pass(this.mat.up, this._bloomMips[i - 1], false);
      }
      this.mat.composite.uniforms.tBloom.value = this._bloomMips[0].texture;
    }

    // --- composite to the back buffer --------------------------------------
    const c = this.mat.composite.uniforms;
    c.tColor.value = colorTex;
    c.tAO.value = aoTex;
    c.uTexel.value.set(1 / W, 1 / H);
    c.uExposure.value = this.exposure;
    c.uFrame.value = this._frame % 64;
    this._pass(this.mat.composite, null);

    // --- bookkeeping -------------------------------------------------------
    this._mPrevViewProj.copy(this._mViewProj);
    this._frame++;
    this._passCost = this._computePassCost(mips);
    if (this._frame - this._shadowScanAt > 60) this._scanShadows(scene);
  }

  /**
   * Full-screen pass equivalents at output resolution: every pass weighted by
   * the fraction of the frame's pixels it actually shades. The scene render
   * itself is not counted here — it is the `overdraw` term.
   */
  _computePassCost(mips) {
    const t = this.tier;
    const px = Math.max(1, this.rtWidth * this.rtHeight);
    let cost = 1; // composite, full res
    if (t.taa) cost += 1;
    if (t.ao) cost += 0.25;
    if (t.ao && t.aoBlur) cost += 0.25;
    if (t.bloom && mips > 0) {
      cost += 0.25;                                    // prefilter, half res
      for (let i = 1; i < mips; i++) {
        const m = this._bloomMips[i];
        cost += (m.width * m.height) / px;             // downsample
        cost += (this._bloomMips[i - 1].width * this._bloomMips[i - 1].height) / px; // upsample
      }
    }
    if (t.autoExp && this.autoExposure) cost += (64 * 36 + 1) / px;
    // Framebuffer scale: at DRS < 1 we genuinely shade fewer pixels, and the
    // budget is quoted at 1080p output.
    return +(cost * this.renderScale * this.renderScale).toFixed(3);
  }

  /**
   * Shadow cost is owned by whoever configured the lights, not by us — so read
   * it off the actual scene graph rather than asserting a number. Throttled:
   * light setups do not change every frame.
   */
  _scanShadows(scene) {
    this._shadowScanAt = this._frame;
    if (!this.renderer.shadowMap.enabled) { this._shadowTexels = 0; return; }
    let texels = 0;
    scene.traverse((o) => {
      if (o.isLight && o.castShadow && o.shadow) {
        const m = o.shadow.mapSize;
        texels += m.x * m.y * (o.isPointLight ? 6 : 1);
      }
    });
    this._shadowTexels = texels;
  }

  reportCost(perf) {
    perf.pipelineCost.fullscreenPasses = this._passCost;
    // Opaque scene + the transparent/FX layers on top. We do not author the
    // scene, so this stays the project's standing estimate.
    perf.pipelineCost.overdraw = 1.6;
    perf.pipelineCost.shadowTexels = this._shadowTexels;
  }

  dispose() {
    this._disposeTargets();
    if (this.mat) for (const m of Object.values(this.mat)) m.dispose();
    this._quadGeo.dispose();
    this.renderer.dispose();
  }
}
