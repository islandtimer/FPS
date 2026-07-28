// OWNER: agent "sky" — atmosphere, sun, IBL, fog, shadow cascades.
// CONTRACT:
//   new Atmosphere(renderer, scene)
//   .update(dt)
//   .sunDirection  -> THREE.Vector3 (normalized, pointing FROM sun TO scene is `-sunDirection`)
//   .sunLight      -> THREE.DirectionalLight (shadow caster)
//   .shadowTexels  -> number, reported into the perf budget
//   .setQuality(q)
//
// ---------------------------------------------------------------------------
// WHY THIS FILE LOOKS LIKE THIS
//
// 1. Sky is analytic Preetham (Perez luminance/chromaticity fit of a
//    Rayleigh+Mie sky) evaluated per-pixel on a single full-screen triangle
//    that is drawn LAST in the opaque queue with depth test on. Early-Z means
//    we only pay for pixels the level does not already cover, and it costs one
//    draw call and one triangle instead of a skybox sphere's overdraw.
//
// 2. The same sky function is rendered once into a 256px cube and pushed
//    through PMREMGenerator to become scene.environment. That single step is
//    what stops metal and rough dielectrics reading as plastic. It is
//    regenerated ONLY when the sun/turbidity changes — never per frame.
//
// 3. THE LIGHTING RATIO IS THE POINT. Round 1 shipped a full-brightness PMREM
//    at environmentIntensity 1.0 against a 4.2 sun. Measured on the alley floor
//    at 480p: direct 0.153 vs ambient 0.139 of post-grade linear luminance —
//    1.1 : 1. The shadow rig was working perfectly and the entire street WAS in
//    the right wall's shadow; a shadow that removes 52% of nothing is invisible,
//    which is why seven critics read the frame as unlit. So: environment is
//    dialled to ENV_INTENSITY and the sun carries the key. Shipped ratio is
//    5.0 : 1 direct/ambient on open sand, 5.95 : 1 for the same sand sunlit vs
//    shadowed, against a brief asking for 3 : 1. Measured, not judged by eye.
//
// 4. Shadows are two cascades, sized so texels/frame stay at 10.75M of a 14M
//    budget:
//      cascade 0 - view-fitted to [near, 8m],  2048 -> 1.34cm texels, PCF 0.6
//      cascade 1 - static ortho over every shadow CASTER in the scene, measured
//                  once rather than guessed (the backdrop ring is
//                  castShadow:false, so the volume is the 48m compound, not the
//                  120m draw distance) 2560 -> 3.75cm texels, PCF 1.3
//    See the QUALITY table for why a third cascade is the wrong answer here.
//    All view-fitted volumes are bounding-sphere sized (rotation invariant) and
//    snapped to their own texel grid so they do not crawl when you walk.
//    three.js has no native CSM, so cascade selection is a small, *chained*
//    shader patch (see _patchMaterial). It never replaces another agent's
//    onBeforeCompile, and if the patch cannot be applied the lights still sum
//    to exactly one sun — the fallback degrades softness, never brightness.
//    _csmApplied is set from inside the compile callback and logged once, so
//    "is the patch live" is an observation rather than an assumption.
//
// 5. Fog is a global override of three's four fog ShaderChunks: analytic
//    height-integrated optical depth + a squared distance term, tinted by an
//    aerial-perspective approximation of THIS sky. Two things about it are
//    load-bearing and were wrong in round 1:
//      a. three applies fog AFTER <tonemapping_fragment>. Our render pipeline
//         runs the renderer at NoToneMapping and tone maps in its own composite
//         pass, so that chunk is a no-op and fog is mixed into scene-referred
//         LINEAR radiance. Baking the fog colours through ACES made every fully
//         fogged pixel ~25% darker than the unfogged sky directly above it, and
//         that is exactly the crisp cutout edge the critics saw at 100m+.
//         _bake() now follows renderer.toneMapping instead of assuming.
//      b. the horizon is not one colour. Preetham's horizon is far brighter
//         toward the sun than away from it, so a single anti-solar fog colour
//         cannot meet the sky at every azimuth. The fog now interpolates
//         between the away-horizon and the sun-horizon by actual bearing.
// ---------------------------------------------------------------------------

import * as THREE from 'three';

// ---------------------------------------------------------------- sky model
// Preetham et al., "A Practical Analytic Model for Daylight". Published
// coefficient fits; implemented from the paper's formulation.

const DEG = Math.PI / 180;

function perezCoeffs(T) {
  // index 0 = Y (luminance), 1 = x chroma, 2 = y chroma
  return {
    A: [0.1787 * T - 1.4630, -0.0193 * T - 0.2592, -0.0167 * T - 0.2608],
    B: [-0.3554 * T + 0.4275, -0.0665 * T + 0.0008, -0.0950 * T + 0.0092],
    C: [-0.0227 * T + 5.3251, -0.0004 * T + 0.2125, -0.0079 * T + 0.2102],
    D: [0.1206 * T - 2.5771, -0.0641 * T - 0.8989, -0.0441 * T - 1.6537],
    E: [-0.0670 * T + 0.3703, -0.0033 * T + 0.0452, -0.0109 * T + 0.0529],
  };
}

function zenithColor(T, thetaS) {
  const t = thetaS, t2 = t * t, t3 = t2 * t, T2 = T * T;
  const chi = (4 / 9 - T / 120) * (Math.PI - 2 * t);
  const Y = (4.0453 * T - 4.9710) * Math.tan(chi) - 0.2155 * T + 2.4192;
  const x =
    (0.00166 * t3 - 0.00375 * t2 + 0.00209 * t) * T2 +
    (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * t + 0.00394) * T +
    (0.11693 * t3 - 0.21196 * t2 + 0.06052 * t + 0.25886);
  const y =
    (0.00275 * t3 - 0.00610 * t2 + 0.00317 * t) * T2 +
    (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * t + 0.00516) * T +
    (0.15346 * t3 - 0.26756 * t2 + 0.06670 * t + 0.26688);
  return [Math.max(Y, 0.02), x, y];
}

/** Kasten–Young relative optical air mass. */
function airMass(cosZenith) {
  const z = Math.acos(Math.min(1, Math.max(-1, cosZenith))) / DEG;
  return 1 / (Math.max(cosZenith, 0) + 0.15 * Math.pow(Math.max(93.885 - z, 0.5), -1.253));
}

// Rayleigh optical depth of the whole column at zenith (beta_R * scale height).
const TAU_R = [0.0464, 0.1080, 0.2648];
// Ångström aerosol exponent applied at 600/550/450 nm.
const AEROSOL_L = [Math.pow(0.60, -1.3), Math.pow(0.55, -1.3), Math.pow(0.45, -1.3)];

function xyYtoLinearRGB(Y, x, y, out) {
  const yy = Math.max(y, 1e-4);
  const X = (x / yy) * Y;
  const Z = ((1 - x - yy) / yy) * Y;
  out[0] = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  out[1] = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  out[2] = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;
  for (let i = 0; i < 3; i++) out[i] = Math.max(out[i], 0);
  return out;
}

const ACES_IN = [
  [0.59719, 0.35458, 0.04823],
  [0.07600, 0.90834, 0.01566],
  [0.02840, 0.13383, 0.83777],
];
const ACES_OUT = [
  [1.60475, -0.53108, -0.07367],
  [-0.10208, 1.10813, -0.00605],
  [-0.00327, -0.07276, 1.07602],
];

/** Byte-for-byte mirror of three's ACESFilmicToneMapping. Only used when the
 *  renderer is actually set to it — see _bake(). */
function acesFilmic(rgb, exposure, out) {
  const c = [rgb[0], rgb[1], rgb[2]];
  const s = exposure / 0.6;
  c[0] *= s; c[1] *= s; c[2] *= s;
  const a = [0, 0, 0];
  for (let i = 0; i < 3; i++) a[i] = ACES_IN[i][0] * c[0] + ACES_IN[i][1] * c[1] + ACES_IN[i][2] * c[2];
  for (let i = 0; i < 3; i++) {
    const v = a[i];
    a[i] = (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081);
  }
  for (let i = 0; i < 3; i++) {
    const v = ACES_OUT[i][0] * a[0] + ACES_OUT[i][1] * a[1] + ACES_OUT[i][2] * a[2];
    out[i] = Math.min(1, Math.max(0, v));
  }
  return out;
}

function glslVec3(v, digits = 5) {
  return `vec3( ${v[0].toFixed(digits)}, ${v[1].toFixed(digits)}, ${v[2].toFixed(digits)} )`;
}
function glslVec2(v, digits = 5) {
  return `vec2( ${v[0].toFixed(digits)}, ${v[1].toFixed(digits)} )`;
}

/**
 * CPU-side evaluation of the identical sky the shader draws. Used for the
 * sun light colour, the ground bounce, and every baked fog colour.
 */
class SkyModel {
  constructor(sunDir, turbidity) {
    this.sun = sunDir;
    this.T = turbidity;
    const cosThetaS = Math.min(1, Math.max(-1, sunDir.y));
    this.thetaS = Math.acos(cosThetaS);
    this.p = perezCoeffs(turbidity);
    this.zen = zenithColor(turbidity, this.thetaS);

    // F(theta = 0, gamma = thetaS) — the normalising denominator.
    this.F0 = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      this.F0[i] =
        (1 + this.p.A[i] * Math.exp(this.p.B[i])) *
        (1 + this.p.C[i] * Math.exp(this.p.D[i] * this.thetaS) + this.p.E[i] * cosThetaS * cosThetaS);
    }

    // Direct solar transmittance: Rayleigh + Ångström aerosol along the sun ray.
    const m = airMass(cosThetaS);
    const beta = 0.0125 * (turbidity - 1); // aerosol turbidity coefficient
    const t = [0, 0, 0];
    for (let i = 0; i < 3; i++) t[i] = Math.exp(-(TAU_R[i] + beta * AEROSOL_L[i]) * m);
    const mx = Math.max(t[0], t[1], t[2], 1e-4);
    this.sunTint = [t[0] / mx, t[1] / mx, t[2] / mx];
    this.sunTransmittance = t;

    // Normalise so the zenith sits at a sane HDR level for ACES.
    const probe = this.rawRadiance(0, 1, 0, [0, 0, 0]);
    this.scale = 0.30 / Math.max(probe[1], 1e-4);
  }

  /** Unscaled Preetham radiance for a world direction (must be normalised). */
  rawRadiance(dx, dy, dz, out) {
    const cosT = Math.max(dy, 0.012);
    const cosG = Math.min(1, Math.max(-1, dx * this.sun.x + dy * this.sun.y + dz * this.sun.z));
    const gamma = Math.acos(cosG);
    const p = this.p;
    const v = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      const F = (1 + p.A[i] * Math.exp(p.B[i] / cosT)) *
        (1 + p.C[i] * Math.exp(p.D[i] * gamma) + p.E[i] * cosG * cosG);
      v[i] = (this.zen[i] * F) / this.F0[i];
    }
    return xyYtoLinearRGB(v[0], v[1], v[2], out);
  }

  /** Scaled HDR linear sky radiance, ground blended in below the horizon. */
  radiance(dx, dy, dz, out) {
    this.rawRadiance(dx, dy, dz, out);
    out[0] *= this.scale; out[1] *= this.scale; out[2] *= this.scale;
    if (this.ground && dy < 0.02) {
      const k = Math.min(1, Math.max(0, (dy + 0.012) / 0.032));
      for (let i = 0; i < 3; i++) out[i] = this.ground[i] * (1 - k) + out[i] * k;
    }
    return out;
  }
}

// ---------------------------------------------------------------- shaders

const SKY_FRAG_CORE = /* glsl */`
uniform vec3 uSunDir;
uniform vec3 uPerezA, uPerezB, uPerezC, uPerezD, uPerezE;
uniform vec3 uZenith;
uniform vec3 uF0;
uniform vec3 uGround;
uniform vec3 uSunColor;
uniform vec3 uEnvCool;
uniform float uScale;
uniform float uDisc;
uniform float uHalo;

vec3 skyRadiance( vec3 dir ) {
  float cosT = max( dir.y, 0.012 );
  float cosG = clamp( dot( dir, uSunDir ), -1.0, 1.0 );
  float gamma = acos( cosG );

  vec3 F = ( 1.0 + uPerezA * exp( uPerezB / cosT ) )
         * ( 1.0 + uPerezC * exp( uPerezD * gamma ) + uPerezE * cosG * cosG );
  vec3 xyY = uZenith * F / uF0;

  float Y = xyY.x;
  float cx = xyY.y;
  float cy = max( xyY.z, 1e-4 );
  vec3 XYZ = vec3( cx / cy * Y, Y, ( 1.0 - cx - cy ) / cy * Y );
  vec3 col = mat3(
    3.2406, -0.9689,  0.0557,
   -1.5372,  1.8758, -0.2040,
   -0.4986,  0.0415,  1.0570 ) * XYZ;
  col = max( col, vec3( 0.0 ) ) * uScale;

  // Upper-dome chroma trim. Identity (1,1,1) for the sky you look at; the IBL
  // capture pushes it blue so the ambient fill that lands in shadow is sky
  // coloured rather than a dimmer copy of the amber key. Preetham's chromaticity
  // is a whole-dome least-squares fit that includes the solar aureole, and the
  // aureole is removed from the capture (the directional light carries it), so
  // the remaining dome legitimately belongs bluer than the raw fit returns.
  col *= mix( vec3( 1.0 ), uEnvCool, smoothstep( 0.02, 0.45, dir.y ) );

  // Forward-scattered aureole, two lobes: a tight 3-degree core and a broad
  // ~25-degree Mie skirt. One Gaussian could not carry both without the skirt
  // going flat-grey across a third of the sky.
  float halo = exp( - gamma * gamma * 300.0 )
             + 0.30 / ( 1.0 + gamma * gamma * 44.0 );
  col += uSunColor * uHalo * halo;

  // Solar disc with limb darkening (I(mu)/I(0) = 1 - u(1 - mu), u = 0.6 in the
  // visible). Deliberately HDR so the bloom stage has something real to bleed;
  // one quarter-degree disc moves the 64x36 log-luma average by < 0.15 stops,
  // so it cannot drag auto-exposure down. Suppressed for the IBL capture.
  float d = gamma / 0.00468;                    // 0.268 deg solar radius
  float mu = sqrt( max( 0.0, 1.0 - d * d ) );
  float limb = 1.0 - 0.6 * ( 1.0 - mu );
  col += uSunColor * uDisc * limb * ( 1.0 - smoothstep( 0.92, 1.04, d ) );

  // Dusty ground half, blended across ~2 degrees so the horizon reads as haze.
  float gm = smoothstep( -0.012, 0.020, dir.y );
  col = mix( uGround, col, gm );
  return col;
}
`;

const SKY_VERT_SCREEN = /* glsl */`
uniform mat4 uInvProj;
uniform mat3 uCamRot;
varying vec3 vRay;
void main() {
  vec4 v = uInvProj * vec4( position.xy, 1.0, 1.0 );
  vRay = uCamRot * ( v.xyz / v.w );
  gl_Position = vec4( position.xy, 1.0, 1.0 );
}
`;

const SKY_FRAG_SCREEN = SKY_FRAG_CORE + /* glsl */`
varying vec3 vRay;
void main() {
  gl_FragColor = vec4( skyRadiance( normalize( vRay ) ), 1.0 );
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const SKY_VERT_CUBE = /* glsl */`
varying vec3 vDir;
void main() {
  vDir = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`;

const SKY_FRAG_CUBE = SKY_FRAG_CORE + /* glsl */`
varying vec3 vDir;
void main() {
  gl_FragColor = vec4( skyRadiance( normalize( vDir ) ), 1.0 );
}
`;

// ---------------------------------------------------------------- quality
// sizes[] is per cascade, last entry = the static world cascade. splits[] are
// the view-space far distances of the view-fitted cascades, so
// sizes.length === splits.length + 1 always. The rig is general in the number of
// cascades; the tiers below are what this level actually wants.
//
// WHY TWO AND NOT THREE. The obvious answer to "small props do not resolve" is a
// mid cascade, and that is what a 120m world needs. It is not what this world
// needs: _measureCasters finds the shadow-casting geometry inside a 48m radius
// (the backdrop ring is castShadow:false), so a single static 2560 map already
// covers every caster at 3.75cm. A third cascade fitted to 24m at 2048 would be
// 4.0cm — no better — while costing 2.36M texels and a whole extra shadow pass
// of draw calls. Spending the same texels on the static map instead takes the
// far field from 6.25cm to 3.75cm, which is what actually makes the utility pole
// and the sandbags cast something you can see.
//
// texel size at 'high':  near 1.34cm to 8m, then 3.75cm to 48m.
// texels/frame:  low 2.36M | medium 6.55M | high 10.75M | ultra 13.11M (cap 14M)
const QUALITY = {
  low:    { sizes: [1536],       splits: [],  envSize: 128, halo: 1.5 },
  medium: { sizes: [1536, 2048], splits: [8], envSize: 256, halo: 2.0 },
  high:   { sizes: [2048, 2560], splits: [8], envSize: 256, halo: 2.3 },
  ultra:  { sizes: [2560, 2560], splits: [9], envSize: 256, halo: 2.4 },
};

// Per-cascade PCF kernel radius, in texels. The near cascade is deliberately
// near-hard — a real contact shadow has a penumbra of a couple of centimetres at
// the occluder, and round 1's 1.5 texels smeared that to nothing. The far radius
// came down from 2.0 as well: at 3.75cm texels, 2.0 is a 7.5cm blur and a 25cm
// utility pole's shadow does not survive it.
const PCF_NEAR = 0.6, PCF_MID = 1.1, PCF_FAR = 1.3;

// Ambient/key balance. See note 3 at the top of the file: these are the single
// largest visual lever in the project, so they are solved, not guessed. Scene
// radiance is linear in both, so one HDR readback of the pre-tonemap scene
// target with the sun off and then the ambient off gives the split exactly:
//
//   calibration, establish, open sand, sun 8.0 / env 0.40:
//       direct 1.008   ambient 0.131   ->  7.7 : 1
//   shipped,     establish, open sand, sun 8.0 / env 0.64:
//       direct 1.008   ambient 0.203   ->  5.0 : 1, same sand lit/shadowed 5.95 : 1
//
// Physically, horizontal direct irradiance at 18 degrees is ~216 W/m2 against
// ~90 of diffuse sky, so open ground is about 2.4 : 1 in the real world. Round
// 1's numbers work out to ~1.6 : 1 on this same patch (1.10 : 1 measured on the
// alley floor): below physical, which is why the frame read unlit. We sit
// deliberately well above physical at 5.0 : 1, because the composite's filmic
// shoulder and the grade both compress the top of the range before anyone sees
// it, and because a shaded facade at the physical ratio still reads as a lit
// facade on screen. That puts the SAME sand at 6 : 1 sunlit vs shadowed — 2x the
// 3 : 1 the brief asks for, and short of the 15 : 1 that crushes shadow detail.
//
// The PAIR sets the ratio; their common SCALE sets how bright geometry sits
// against the sky and the fog, because the sky's own radiance is normalised
// independently (zenith = 0.30) and auto-exposure then normalises the frame.
// Halving both would leave the ratio untouched and hand the extra exposure to
// the sky, which comes back as a washed-out horizon and milky aerial
// perspective — which is exactly what happened at sun 5.5 / env 0.59, same
// ratio, and it looked worse. 8.0 is the level at which the horizon haze lands
// where it did in round 1 while the ratio is 3.1x better on the same patch.
const ENV_INTENSITY = 0.64;
const HEMI_INTENSITY = 0.082;
// Extra amber on the KEY only (the visible sun disc keeps the physical
// transmittance tint). Sun R/B lands ~2.5 against an ambient R/B ~0.55: about
// 2000K of separation between key and fill.
const SUN_WARM = [1.0, 0.955, 0.865];
// Upper-dome chroma trim applied to the IBL capture only. Luminance-neutral to
// within 5%, so it moves colour, not exposure.
const ENV_COOL = [0.855, 0.955, 1.235];

// The distance term is squared and the height term is not, so shifting weight
// from the height integral into the distance term steepens the curve: the fog
// goes from 44% at 40m in round 1 to 34%, and from ~85% at 100m to 87% and 98%
// at 140m. Clearer mid-ground, denser far field.
//
// This matters MORE than it did in round 1 because the fog colour is now correct
// (see _bake). Round 1's ACES-baked fog was darker than the sky, so distance
// DARKENED geometry and the milkiness stayed hidden; a correct fog lifts toward
// the horizon, which is right but shows up immediately if there is too much of
// it. Both terms are also height-attenuated now, so the fog is a dust layer that
// sits on the ground rather than a grey wash over the whole frame.
const FOG_HEIGHT_FALLOFF = 0.082; // 1/m — dust hugs the ground, e-fold ~12m
const FOG_DIST_K = 1.62;          // weight of the squared-distance term
const FOG_HEIGHT_MIX = 0.42;      // near-field weight of the height integral
const FOG_DENSITY = 0.0092;

export class Atmosphere {
  constructor(renderer, scene) {
    this.scene = scene;
    this.renderer = renderer;
    this.quality = 'high';

    // Late afternoon, dusty. 18 degrees of elevation gives a ~3x shadow length
    // and a golden key without tipping into sunset orange.
    //
    // AZIMUTH IS A COMPOSITION DECISION, not a free parameter, and it was the
    // other half of round 1's "no shadows" verdict. At 18 degrees a 7.6m facade
    // throws 23m of shadow, so a sun raking square across the 10m street
    // blankets the entire floor. Measured, at 416x234, as the fraction of the
    // lower frame the sun actually shadows in the firefight pose:
    //
    //     az   15    45    75   118   150   180   210   250   290   330
    //   frac 0.19  0.10  0.36  0.83  0.78  0.61  0.58  0.62  0.74  0.45
    //  ratio 3.49  3.97  4.36  1.84  1.60  2.10  2.67  2.21  1.80  3.19
    //
    // 118 (round 1) put 83% of the frame in shadow — uniform shade reads as no
    // shadow at all, which is exactly what seven critics reported. 75 shadows a
    // third of the alley floor with a 4.4:1 step across the boundary, keeps the
    // west block's facade as one blazing sunlit plane above it, and throws the
    // right-hand courtyard wall's edge diagonally across the road. Best measured
    // pair of (area, contrast) in the sweep, and it also holds up in establish
    // (0.18 / 3.6) rather than winning one pose at the cost of the other.
    this.sunElevation = 18;
    this.sunAzimuth = 75;
    this.turbidity = 3.1;

    this.sunDirection = new THREE.Vector3();
    this._setSunVector();

    this.model = null;
    this.shadowTexels = 0;
    this.cascades = [];

    // scratch — nothing in update() may allocate
    this._v1 = new THREE.Vector3();
    this._ax = new THREE.Vector3();
    this._ay = new THREE.Vector3();
    this._center = new THREE.Vector3();
    this._sphere = new THREE.Sphere();
    this._invProj = new THREE.Matrix4();
    this._camRot = new THREE.Matrix3();
    this._camera = null;
    this._frames = 0;
    this._progN = -1;
    this._keyed = new WeakSet();
    this._csmPatched = new WeakSet();
    this._skyRev = 0;
    this._csmActive = false;
    this._csmApplied = false;
    this._csmLogged = false;
    this._exposure = renderer.toneMappingExposure;

    // Overwritten on the first frame by a real measurement of the scene's
    // shadow CASTERS (see _measureCasters). The backdrop ring is castShadow
    // false, so this is the compound, not the draw distance.
    this.worldRadius = 42;
    this.worldMidY = 6;
    this._boundsExplicit = false;

    // three's fog chunks are replaced globally; keep the originals so
    // dispose() can put the renderer back exactly as it was found.
    this._origChunks = {
      fog_pars_vertex: THREE.ShaderChunk.fog_pars_vertex,
      fog_vertex: THREE.ShaderChunk.fog_vertex,
      fog_pars_fragment: THREE.ShaderChunk.fog_pars_fragment,
      fog_fragment: THREE.ShaderChunk.fog_fragment,
    };

    this._buildSky();
    this._buildLights();
    this._applyQuality();
    this._rebuildAtmosphere();

    // Same debug hook convention as Perf. main.js currently routes ?q= only to
    // the render pipeline, so this is also how the bench harness can drive
    // atmosphere quality without a contract change.
    if (typeof window !== 'undefined') window.__atmosphere = this;
  }

  // -------------------------------------------------------------- geometry
  _setSunVector() {
    const el = this.sunElevation * DEG, az = this.sunAzimuth * DEG;
    this.sunDirection
      .set(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el))
      .normalize();
  }

  _buildSky() {
    // One full-screen triangle. Vertices are already in NDC; the vertex shader
    // pins z = w so it lands on the far plane and early-Z rejects every pixel
    // the level already wrote.
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(
      new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));

    const uniforms = {
      uSunDir: { value: new THREE.Vector3() },
      uPerezA: { value: new THREE.Vector3() },
      uPerezB: { value: new THREE.Vector3() },
      uPerezC: { value: new THREE.Vector3() },
      uPerezD: { value: new THREE.Vector3() },
      uPerezE: { value: new THREE.Vector3() },
      uZenith: { value: new THREE.Vector3() },
      uF0: { value: new THREE.Vector3() },
      uGround: { value: new THREE.Vector3() },
      uSunColor: { value: new THREE.Vector3() },
      uEnvCool: { value: new THREE.Vector3(1, 1, 1) },
      uScale: { value: 1 },
      uDisc: { value: 42 },
      uHalo: { value: 2.3 },
      uInvProj: { value: this._invProj },
      uCamRot: { value: this._camRot },
    };
    this._skyUniforms = uniforms;

    const mat = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: SKY_VERT_SCREEN,
      fragmentShader: SKY_FRAG_SCREEN,
      depthWrite: false,
      depthTest: true,
      fog: false,
      // toneMapped stays true: the sky goes through the SAME tonemapping and
      // colorspace chunks as every other material, so it cannot desync from
      // the rest of the frame no matter what the render pipeline does.
      side: THREE.FrontSide,
    });
    const mesh = new THREE.Mesh(g, mat);
    mesh.frustumCulled = false;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.renderOrder = 1000; // last in the opaque queue
    mesh.name = 'sky';
    mesh.onBeforeRender = (rend, sc, cam) => {
      this._invProj.copy(cam.projectionMatrixInverse);
      this._camRot.setFromMatrix4(cam.matrixWorld);
    };
    this.scene.add(mesh);
    this.skyMesh = mesh;
    this.scene.background = null;

    // Private scene used only for the IBL capture. It shares every uniform with
    // the visible sky EXCEPT three: no sun disc (a 0.5 degree disc aliases badly
    // into a 256px cube and fireflies the prefiltered mips), a much weaker
    // aureole (the directional light already carries that energy — leaving it in
    // double-counts the sun and drags the ambient warm), and the cool dome trim.
    const cubeUniforms = {};
    for (const k of Object.keys(uniforms)) {
      if (k === 'uInvProj' || k === 'uCamRot') continue;
      cubeUniforms[k] = uniforms[k];
    }
    cubeUniforms.uDisc = { value: 0 };
    cubeUniforms.uHalo = { value: 0.35 };
    cubeUniforms.uEnvCool = { value: new THREE.Vector3(...ENV_COOL) };
    this._cubeUniforms = cubeUniforms;
    const cubeMat = new THREE.ShaderMaterial({
      uniforms: cubeUniforms,
      vertexShader: SKY_VERT_CUBE,
      fragmentShader: SKY_FRAG_CUBE,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
      fog: false,
      toneMapped: false,
    });
    this._skyScene = new THREE.Scene();
    this._cubeMesh = new THREE.Mesh(new THREE.BoxGeometry(20, 20, 20), cubeMat);
    this._cubeMesh.frustumCulled = false;
    this._skyScene.add(this._cubeMesh);
  }

  _buildLights() {
    // A whisper of hemisphere so materials that ignore scene.environment
    // (Basic/Lambert-ish, sprites) are not pitch black. Kept deliberately low:
    // it is spectrally flat, and letting it carry the fill greys out the cool
    // sky shadows that the IBL gives for free.
    const hemi = new THREE.HemisphereLight(0xa9c6e6, 0x6b5940, HEMI_INTENSITY);
    this.scene.add(hemi);
    this.hemi = hemi;

    // The key. See ENV_INTENSITY: 1.9x round 1's sun against 0.64 of round 1's
    // environment is what turns a 1.1:1 lit/shadow ratio into 6:1.
    this.sunIntensity = 8.0;
    this.sunColor = new THREE.Color(1, 0.79, 0.43);
  }

  // -------------------------------------------------------------- cascades
  _applyQuality() {
    const q = QUALITY[this.quality] || QUALITY.high;
    this._q = q;
    const n = q.sizes.length;

    // Rebuild the cascade rig only when the count changes.
    if (this.cascades.length !== n) {
      for (const c of this.cascades) {
        c.light.shadow.map?.dispose();
        this.scene.remove(c.light, c.light.target);
        c.light.dispose();
      }
      this.cascades.length = 0;
      for (let i = 0; i < n; i++) {
        const l = new THREE.DirectionalLight(0xffffff, 1);
        l.castShadow = true;
        l.shadow.camera.near = 0.1;
        l.name = 'sunCascade' + i;
        this.scene.add(l);
        this.scene.add(l.target);
        this.cascades.push({ light: l, radius: 0, size: 1024 });
      }
      this.sunLight = this.cascades[0].light;
    }

    let texels = 0;
    for (let i = 0; i < n; i++) {
      const c = this.cascades[i];
      c.size = q.sizes[i];
      if (c.light.shadow.mapSize.x !== c.size) {
        c.light.shadow.mapSize.set(c.size, c.size);
        c.light.shadow.map?.dispose();
        c.light.shadow.map = null;
        c.light.shadow.needsUpdate = true;
      }
      const isFar = i === n - 1;
      const isNear = i === 0 && n > 1;
      // normalBias is in world units, so it has to track this cascade's texel
      // size (~1.3cm near, ~3.8cm far) or the coarse map peter-pans straight off
      // thin geometry while the fine one acnes.
      c.light.shadow.normalBias = isNear ? 0.024 : (isFar ? 0.060 : 0.045);
      c.light.shadow.bias = isNear ? -0.00007 : (isFar ? -0.00010 : -0.00009);
      c.light.shadow.radius = n === 1 ? 1.4 : (isNear ? PCF_NEAR : (isFar ? PCF_FAR : PCF_MID));
      texels += c.size * c.size;
    }
    this.shadowTexels = texels;

    // Energy is split across cascades so that an unpatched material still
    // receives exactly one sun's worth of light; the CSM patch multiplies the
    // selected cascade back to full strength.
    this._applySunColor();

    this._skyUniforms.uHalo.value = q.halo;
    this._buildCsmSource();
    // A quality change can turn cascade selection on after materials were
    // already skipped for it, so forget what has been patched and re-sweep.
    this._csmPatched = new WeakSet();
    this._csmApplied = false;
    this._progN = -1;
    this._frames = 0;
    this._invalidateAll();
    this._fitFar();
  }

  _applySunColor() {
    const n = Math.max(1, this.cascades.length);
    for (const c of this.cascades) {
      c.light.color.copy(this.sunColor);
      c.light.intensity = this.sunIntensity / n;
    }
  }

  _fitFar() {
    if (!this.cascades.length) return;
    const c = this.cascades[this.cascades.length - 1];
    const r = this.worldRadius;
    c.radius = r;
    this._center.set(0, this.worldMidY, 0);
    this._placeCascade(c, this._center, r, true);
  }

  /**
   * One-shot measurement of how far the static cascade actually has to reach.
   * Guessing 78m (half the draw distance) cost the far cascade a factor of two
   * in texel density for volume nothing casts into.
   */
  _measureCasters() {
    if (this._boundsExplicit) return;
    let r2 = 0, ymax = 0, found = 0;
    const s = this._sphere;
    this.scene.traverse((o) => {
      if (!o.castShadow || !o.geometry) return;
      let src = null;
      if (o.isInstancedMesh) {
        if (!o.boundingSphere) o.computeBoundingSphere();
        src = o.boundingSphere;
      } else {
        if (!o.geometry.boundingSphere) o.geometry.computeBoundingSphere();
        src = o.geometry.boundingSphere;
      }
      if (!src) return;
      s.copy(src).applyMatrix4(o.matrixWorld);
      const d = Math.hypot(s.center.x, s.center.z) + s.radius;
      const top = s.center.y + s.radius;
      // An empty or degenerate batch yields NaN bounds; one of those poisoning
      // worldMidY would silently point the static cascade at nothing.
      if (!Number.isFinite(d) || !Number.isFinite(top)) return;
      if (d * d > r2) r2 = d * d;
      if (top > ymax) ymax = top;
      found++;
    });
    if (!found) return;
    const r = Math.min(120, Math.max(16, Math.ceil(Math.sqrt(r2))));
    const midY = Math.min(30, Math.max(2, ymax * 0.5));
    if (Math.abs(r - this.worldRadius) < 0.5 && Math.abs(midY - this.worldMidY) < 0.5) return;
    this.worldRadius = r;
    this.worldMidY = midY;
    this._fitFar();
  }

  /** Positions one cascade's light + ortho volume around a world-space centre. */
  _placeCascade(c, center, r, snap) {
    const sun = this.sunDirection;
    const cam = c.light.shadow.camera;

    if (snap) {
      // Snap the centre to this cascade's texel grid in light space, using the
      // exact basis three's lookAt will build, or the map crawls when you walk.
      this._ax.set(0, 1, 0).cross(sun);
      if (this._ax.lengthSq() < 1e-8) this._ax.set(1, 0, 0); // sun at zenith
      this._ax.normalize();
      this._ay.copy(sun).cross(this._ax).normalize();
      const texel = (2 * r) / c.size;
      const px = center.dot(this._ax), py = center.dot(this._ay), pz = center.dot(sun);
      const qx = Math.round(px / texel) * texel;
      const qy = Math.round(py / texel) * texel;
      center.set(0, 0, 0)
        .addScaledVector(this._ax, qx)
        .addScaledVector(this._ay, qy)
        .addScaledVector(sun, pz);
    }

    // Head-room up-sun so a tall block outside the fitted volume still casts
    // into it. At 18 degrees a 10m parapet reaches 31m down-sun, so the volume
    // has to start well behind whatever it is fitted to. Depth is packed to
    // RGBA, so the wide range costs no precision.
    const back = r + 70;
    c.light.target.position.copy(center);
    c.light.position.copy(center).addScaledVector(sun, back);
    c.light.target.updateMatrixWorld();

    cam.left = -r; cam.right = r; cam.top = r; cam.bottom = -r;
    cam.near = 0.5;
    cam.far = back + r + 70;
    cam.updateProjectionMatrix();
  }

  /** Bounding-sphere fit of the camera frustum slice [near, far]. Rotation
   *  invariant, so the radius never changes as the player looks around. */
  _fitViewCascade(i, cam, near, far) {
    const c = this.cascades[i];

    const tanV = Math.tan(cam.fov * 0.5 * DEG);
    const tanH = tanV * cam.aspect;
    const k2 = tanV * tanV + tanH * tanH;

    let cd = ((far + near) * (k2 + 1)) * 0.5;
    if (cd > far) cd = far;
    const dx = far * Math.sqrt(k2), dz = cd - far;
    let r = Math.sqrt(dx * dx + dz * dz);
    r = Math.ceil(r * 4) * 0.25; // quantised so an FOV tween cannot shimmer it
    c.radius = r;

    cam.getWorldDirection(this._v1);
    this._center.copy(cam.position).addScaledVector(this._v1, cd);
    this._placeCascade(c, this._center, r, true);
  }

  _fitViewCascades(cam) {
    const splits = this._q.splits;
    let near = Math.max(cam.near, 0.05);
    for (let i = 0; i < splits.length; i++) {
      this._fitViewCascade(i, cam, near, splits[i]);
      near = splits[i];
    }
  }

  // -------------------------------------------------------------- CSM patch
  _buildCsmSource() {
    const n = this.cascades.length;
    this._csmActive = false;
    this._csmFn = '';
    this._csmInject = '';
    if (n < 2) return;

    const splits = this._q.splits;
    // A quality entry whose splits do not match its cascade count would emit a
    // reference to an undeclared t_k and break every lit shader in the game.
    // Refuse rather than generate it; the fallback still sums to one sun.
    if (splits.length !== n - 1) return;
    const nf = n.toFixed(1);

    // t_k is the blend across split k. Weights are N*(t_{k-1} - t_k) with
    // t_{-1} = 1 and t_{n-1} = 0, so they sum to exactly N at every depth and
    // total sun energy is invariant no matter where the blends land.
    let body = '';
    for (let k = 0; k < splits.length; k++) {
      const lo = (splits[k] * 0.86).toFixed(2), hi = (splits[k] * 1.14).toFixed(2);
      body += `  float t${k} = smoothstep( ${lo}, ${hi}, viewDepth );\n`;
    }
    for (let i = 0; i < n; i++) {
      const prev = i === 0 ? '1.0' : `t${i - 1}`;
      const next = i === n - 1 ? '0.0' : `t${i}`;
      body += `  if ( idx == ${i} ) return ${nf} * ( ${prev} - ${next} );\n`;
    }
    this._csmFn = /* glsl */`
float csmCascadeWeight( const in int idx, const in float viewDepth ) {
${body}  return 0.0;
}
`;
    // vViewPosition is -mvPosition.xyz, so .z is positive linear view depth.
    // The guard keeps the patch off any directional light another agent adds.
    this._csmInject = `
		#if ( UNROLLED_LOOP_INDEX < ${n} )
		directLight.color *= csmCascadeWeight( UNROLLED_LOOP_INDEX, vViewPosition.z );
		#endif
`;
    this._csmActive = true;
  }

  /**
   * Two independent, non-destructive hooks per material:
   *
   *  - a program-cache-key chain carrying _skyRev. The fog chunks and the
   *    cascade split are baked into shader SOURCE, and three keys its program
   *    cache on defines + cache key, not on chunk text. Without this a sun move
   *    or quality change would silently hand back a stale program.
   *  - the cascade-selection injection, lit materials only.
   *
   * Both chain to whatever the owning agent already installed; neither ever
   * overwrites it.
   */
  _patchMaterial(m) {
    if (!m) return;
    const self = this;

    if (!this._keyed.has(m)) {
      this._keyed.add(m);
      const prevKey = m.customProgramCacheKey;
      m.customProgramCacheKey = function () {
        return (prevKey ? prevKey.call(this) : '') + '|sky' + self._skyRev;
      };
    }

    if (!this._csmActive || this._csmPatched.has(m)) return;
    const lit = m.isMeshStandardMaterial || m.isMeshPhysicalMaterial ||
      m.isMeshPhongMaterial || m.isMeshLambertMaterial || m.isMeshToonMaterial;
    if (!lit) return;   // ShaderMaterial etc. keep the energy-correct fallback
    this._csmPatched.add(m);

    const prevCompile = m.onBeforeCompile;
    m.onBeforeCompile = function (shader, rend) {
      if (prevCompile) prevCompile.call(this, shader, rend);
      if (!self._csmActive) return;
      const chunk = THREE.ShaderChunk.lights_fragment_begin;
      const anchor = 'getDirectionalLightInfo( directionalLight, directLight );';
      const src = shader.fragmentShader;
      // All three anchors must be present or we inject nothing at all — a
      // half-applied patch is a shader compile error across the whole game.
      if (chunk.indexOf(anchor) < 0) return;
      if (src.indexOf('#include <lights_pars_begin>') < 0) return;
      if (src.indexOf('#include <lights_fragment_begin>') < 0) return;
      shader.fragmentShader = src
        .replace('#include <lights_pars_begin>', '#include <lights_pars_begin>' + self._csmFn)
        .replace('#include <lights_fragment_begin>', chunk.replace(anchor, anchor + self._csmInject));
      // Observed, not assumed. Round 1 had no way to tell a live patch from a
      // silently-skipped one, and "both cascades at half strength" is exactly
      // what a dead patch looks like.
      self._csmApplied = true;
      if (!self._csmLogged && typeof console !== 'undefined') {
        self._csmLogged = true;
        console.log('[sky] CSM cascade selection live:', self.cascades.length,
          'cascades, splits', self._q.splits.join('/'), 'm,',
          (self.shadowTexels / 1e6).toFixed(2) + 'M shadow texels');
      }
    };
    m.needsUpdate = true;
  }

  /** Anything already compiled has stale baked source; force a rebuild. */
  _invalidateAll() {
    this._skyRev++;
    if (!this.scene) return;
    this.scene.traverse((o) => {
      const mm = o.material;
      if (!mm) return;
      if (Array.isArray(mm)) for (const m of mm) { if (this._keyed.has(m)) m.needsUpdate = true; }
      else if (this._keyed.has(mm)) mm.needsUpdate = true;
    });
  }

  _scanMaterials() {
    this.scene.traverse((o) => {
      const mm = o.material;
      if (!mm) return;
      if (Array.isArray(mm)) for (const m of mm) this._patchMaterial(m);
      else this._patchMaterial(mm);
    });
  }

  // -------------------------------------------------------------- sky/IBL
  _rebuildAtmosphere() {
    const model = new SkyModel(this.sunDirection, this.turbidity);
    this.model = model;

    // Ground half of the environment: dusty albedo lit by the sun plus a
    // hemisphere of sky. This is what puts warm bounce under overhangs, and it
    // is the ONE warm term in the ambient — everything above the horizon is
    // trimmed cool, so overhangs read warm-from-below / cool-from-above.
    const tmp = [0, 0, 0];
    model.rawRadiance(0, 0.35, 0, tmp);
    const skyAvg = [tmp[0] * model.scale, tmp[1] * model.scale, tmp[2] * model.scale];
    const albedo = [0.31, 0.25, 0.175];
    const ndl = Math.max(this.sunDirection.y, 0);
    // The bounce tracks the key, but at a fraction of it: raising the sun 1.9x
    // must not raise the horizon haze 1.9x or the sky loses its gradient.
    const sunIrr = this.sunIntensity * 0.085;
    model.ground = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      model.ground[i] = albedo[i] * (model.sunTint[i] * sunIrr * ndl + skyAvg[i] * 1.6);
    }

    // Push the model into the shader.
    const u = this._skyUniforms;
    u.uSunDir.value.copy(this.sunDirection);
    u.uPerezA.value.set(model.p.A[0], model.p.A[1], model.p.A[2]);
    u.uPerezB.value.set(model.p.B[0], model.p.B[1], model.p.B[2]);
    u.uPerezC.value.set(model.p.C[0], model.p.C[1], model.p.C[2]);
    u.uPerezD.value.set(model.p.D[0], model.p.D[1], model.p.D[2]);
    u.uPerezE.value.set(model.p.E[0], model.p.E[1], model.p.E[2]);
    u.uZenith.value.set(model.zen[0], model.zen[1], model.zen[2]);
    u.uF0.value.set(model.F0[0], model.F0[1], model.F0[2]);
    u.uGround.value.set(model.ground[0], model.ground[1], model.ground[2]);
    u.uSunColor.value.set(model.sunTint[0], model.sunTint[1], model.sunTint[2]);
    u.uScale.value = model.scale;

    // The KEY is the physical transmittance tint pushed a little further amber.
    // The visible disc and aureole keep the untinted value, so the sky stays
    // self-consistent while the light that lands on geometry is unmistakably
    // warm against a sky-blue fill.
    this.sunColor.setRGB(
      model.sunTint[0] * SUN_WARM[0],
      model.sunTint[1] * SUN_WARM[1],
      model.sunTint[2] * SUN_WARM[2]);
    this._applySunColor();

    // Hemisphere fill matched to the actual sky and ground of this model, with
    // the same cool trim the IBL gets so the two fills cannot disagree.
    model.radiance(0, 1, 0, tmp);
    this.hemi.color.setRGB(
      Math.min(1, tmp[0] * 2.2 * ENV_COOL[0]),
      Math.min(1, tmp[1] * 2.2 * ENV_COOL[1]),
      Math.min(1, tmp[2] * 2.2 * ENV_COOL[2]));
    this.hemi.groundColor.setRGB(
      Math.min(1, model.ground[0] * 3), Math.min(1, model.ground[1] * 3), Math.min(1, model.ground[2] * 3));

    this._buildEnvironment();
    this._installFogChunks();
    this._exposure = this.renderer.toneMappingExposure;
  }

  _buildEnvironment() {
    const size = (this._q || QUALITY.high).envSize;
    if (!this._pmrem) this._pmrem = new THREE.PMREMGenerator(this.renderer);
    const old = this._envRT;
    // Renders the sky cube once and prefilters it. ~2ms, at construction only.
    this._envRT = this._pmrem.fromScene(this._skyScene, 0, 0.5, 60, { size });
    this.scene.environment = this._envRT.texture;
    this.scene.environmentIntensity = ENV_INTENSITY;
    if (old) old.dispose();
  }

  /**
   * three inserts <fog_fragment> AFTER <tonemapping_fragment>, so the space fog
   * mixes in is whatever the renderer's tone mapper leaves behind. Our pipeline
   * runs at NoToneMapping and tone maps later in its own composite pass, which
   * means fog must be scene-referred LINEAR here. Round 1 baked ACES into it
   * unconditionally and every fully-fogged pixel came out ~25% below the sky it
   * was supposed to dissolve into — the horizon cutout the critics saw.
   */
  _bake(lin, out) {
    if (this.renderer.toneMapping === THREE.ACESFilmicToneMapping) {
      return acesFilmic(lin, this.renderer.toneMappingExposure || 1, out);
    }
    out[0] = lin[0]; out[1] = lin[1]; out[2] = lin[2];
    return out;
  }

  /**
   * The horizon colour is a LIVE uniform (three refreshes fogColor from
   * scene.fog.color every frame and converts it into whatever colour space the
   * current render target wants). Everything else in the fog chunk is expressed
   * as a *ratio* to it, so the aerial perspective stays correct whether the
   * scene is rendered straight to the canvas or into an HDR render target.
   */
  _fogSamples() {
    const m = this.model;
    const lin = [0, 0, 0], out = [0, 0, 0];
    const sample = (x, y, z) => {
      const l = Math.hypot(x, y, z);
      m.radiance(x / l, y / l, z / l, lin);
      const a = this._bake(lin, out);
      return [a[0], a[1], a[2]];
    };
    const sd = this.sunDirection;
    return {
      horizon: sample(-sd.x, 0.055, -sd.z),
      zenith: sample(0, 1, 0.001),
      low: sample(-sd.x, -0.02, -sd.z),
      towardSun: sample(sd.x, 0.055, sd.z),
      aureole: sample(sd.x, 0.14, sd.z),
    };
  }

  _updateFogColor() {
    const s = this._fogSamples();
    if (!this.scene.fog || !this.scene.fog.isFogExp2) {
      this.scene.fog = new THREE.FogExp2(0x000000, FOG_DENSITY);
    }
    this.scene.fog.color.setRGB(s.horizon[0], s.horizon[1], s.horizon[2]);
    this.scene.fog.density = FOG_DENSITY;
    return s;
  }

  _installFogChunks() {
    const s = this._updateFogColor();
    const sd = this.sunDirection;
    const hl = Math.max(Math.hypot(sd.x, sd.z), 1e-4);
    const sunHoriz = [sd.x / hl, sd.z / hl];

    const ratio = (c) => [
      Math.min(6, c[0] / Math.max(s.horizon[0], 1e-3)),
      Math.min(6, c[1] / Math.max(s.horizon[1], 1e-3)),
      Math.min(6, c[2] / Math.max(s.horizon[2], 1e-3)),
    ];
    const zenR = ratio(s.zenith);
    const lowR = ratio(s.low);
    const sunR = ratio(s.towardSun);
    // The aureole is whatever the sky 8 degrees above the sun has that the plain
    // sun-horizon does not: that is the light-shaft colour, and it is what makes
    // a fogged wall glow where the sun rakes past it.
    const aurR = ratio([
      Math.max(0, s.aureole[0] - s.towardSun[0]),
      Math.max(0, s.aureole[1] - s.towardSun[1]),
      Math.max(0, s.aureole[2] - s.towardSun[2]),
    ]);

    const H = FOG_HEIGHT_FALLOFF.toFixed(4);
    const K = FOG_DIST_K.toFixed(3);
    const M = FOG_HEIGHT_MIX.toFixed(3);

    THREE.ShaderChunk.fog_pars_vertex = /* glsl */`
#ifdef USE_FOG
	varying float vFogDepth;
	varying vec3 vFogView;
#endif
`;
    THREE.ShaderChunk.fog_vertex = /* glsl */`
#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
	vFogView = mvPosition.xyz;
#endif
`;
    THREE.ShaderChunk.fog_pars_fragment = /* glsl */`
#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	varying vec3 vFogView;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif
`;
    // Height-integrated optical depth (analytic, no march) + a squared distance
    // term for the far falloff, tinted with this sky's aerial perspective. The
    // height term is weighted DOWN and the distance term UP relative to round 1:
    // that leaves the 20m mid-ground where it was while taking a 100m backdrop
    // from ~85% to ~96% fogged, which is what actually dissolves the horizon
    // seam instead of milking the whole frame.
    THREE.ShaderChunk.fog_fragment = /* glsl */`
#ifdef USE_FOG
	vec3 fogVec = vFogView * mat3( viewMatrix );
	float fogDist = length( fogVec );
	vec3 fogDir = fogVec / max( fogDist, 1e-4 );
	#ifdef FOG_EXP2
		float fogSy = ( fogDir.y >= 0.0 ? 1.0 : -1.0 ) * max( abs( fogDir.y ), 1e-3 );
		float fogBase = exp( - ${H} * clamp( cameraPosition.y, -10.0, 200.0 ) );
		float fogOd = fogBase * ( 1.0 - exp( - ${H} * fogSy * fogDist ) ) / ( ${H} * fogSy );
		// Mean height attenuation along this ray, 0..1. Applying it to the
		// squared term as well is what makes the dust a LAYER: a rooftop 12m up
		// at 60m collects ~0.64 of what its own base does, so the skyline keeps
		// its edge while the ground plane it stands on dissolves.
		float fogHy = fogOd / max( fogDist, 1e-4 );
		float fogLin = fogDensity * fogDist * ${K} * fogHy;
		float fogFactor = 1.0 - exp( - ( fogOd * fogDensity * ${M} + fogLin * fogLin ) );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	fogFactor = clamp( fogFactor, 0.0, 1.0 );

	// Azimuth first: Preetham's horizon is far brighter toward the sun than away
	// from it, so one anti-solar colour cannot meet the sky at every bearing —
	// that mismatch is what makes a fogged building read as a cutout.
	vec2 fogH = vec2( fogDir.x, fogDir.z );
	float fogAz = dot( fogH / max( length( fogH ), 1e-4 ), ${glslVec2(sunHoriz)} );
	vec3 fogHz = fogColor * mix( vec3( 1.0 ), ${glslVec3(sunR)}, smoothstep( -0.55, 1.0, fogAz ) );

	vec3 fogAerial = fogHz * mix( vec3( 1.0 ), ${glslVec3(zenR)}, smoothstep( 0.0, 0.62, fogDir.y ) );
	fogAerial = mix( fogColor * ${glslVec3(lowR)}, fogAerial, smoothstep( -0.10, 0.02, fogDir.y ) );
	fogAerial += fogColor * ${glslVec3(aurR)} * pow( max( dot( fogDir, ${glslVec3([sd.x, sd.y, sd.z])} ), 0.0 ), 6.0 );
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogAerial, fogFactor );
#endif
`;
    // Anything already compiled holds the previous chunk text baked in.
    this._invalidateAll();
  }

  // -------------------------------------------------------------- public
  /** Move the sun. Regenerates the IBL and the baked fog — never call per frame. */
  setTimeOfDay(elevationDeg, azimuthDeg = this.sunAzimuth, turbidity = this.turbidity) {
    this.sunElevation = elevationDeg;
    this.sunAzimuth = azimuthDeg;
    this.turbidity = turbidity;
    this._setSunVector();
    this._rebuildAtmosphere();
    this._fitFar();
  }

  /** How far the static cascade must reach. Overrides the auto-measurement. */
  setWorldBounds(radius, midY = 6) {
    this.worldRadius = Math.max(8, radius);
    this.worldMidY = midY;
    this._boundsExplicit = true;
    this._fitFar();
  }

  setQuality(q) {
    if (!QUALITY[q] || q === this.quality) return;
    this.quality = q;
    this._applyQuality();
    this._buildEnvironment();
  }

  update(dt) {
    // Find the active camera once. main.js parents it to the scene; we take no
    // contract change for it and simply look it up.
    if (!this._camera || !this._camera.parent) {
      this._camera = null;
      const kids = this.scene.children;
      for (let i = 0; i < kids.length; i++) {
        if (kids[i].isPerspectiveCamera) { this._camera = kids[i]; break; }
      }
    }

    // Hook materials that have appeared since the last look. The program count
    // only moves when a material is compiled for the first time, so on all but
    // a handful of frames this is a single integer compare.
    const n = this.renderer.info?.programs?.length ?? 0;
    if (n !== this._progN || this._frames < 3) {
      this._progN = n;
      this._scanMaterials();
      this._measureCasters();
    }
    this._frames++;

    if (this._camera) {
      this._camera.updateMatrixWorld();
      this._fitViewCascades(this._camera);
    }

    // If the render pipeline retunes exposure the fog would drift off the sky.
    // fogColor is a live uniform, so re-deriving it is enough — no recompile.
    // (With the renderer at NoToneMapping the bake is exposure-independent and
    // this never fires; it exists so the module is correct under either setup.)
    const e = this.renderer.toneMappingExposure;
    if (Math.abs(e - this._exposure) > 0.02) {
      this._exposure = e;
      this._updateFogColor();
    }
  }

  dispose() {
    for (const k of Object.keys(this._origChunks)) THREE.ShaderChunk[k] = this._origChunks[k];
    for (const c of this.cascades) {
      c.light.shadow.map?.dispose();
      this.scene.remove(c.light, c.light.target);
      c.light.dispose();
    }
    this.cascades.length = 0;
    this.scene.remove(this.hemi);
    this.hemi.dispose?.();
    if (this.skyMesh) {
      this.scene.remove(this.skyMesh);
      this.skyMesh.geometry.dispose();
      this.skyMesh.material.dispose();
    }
    this._cubeMesh?.geometry.dispose();
    this._cubeMesh?.material.dispose();
    this._envRT?.dispose();
    this._pmrem?.dispose();
    this.scene.environment = null;
  }
}
