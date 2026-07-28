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
// 3. Shadows are two cascades, not one map:
//      cascade 0 - view-fitted to the near slice of the camera frustum,
//                  bounding-sphere sized (rotation invariant) and snapped to
//                  the shadow texel grid so it does not crawl.
//      cascade 1 - a fixed ortho over the whole world. The map is 120m wide,
//                  so a view-fitted far cascade would be no cheaper and would
//                  shimmer; a static one is strictly better here.
//    three.js has no native CSM, so cascade selection is a small, *chained*
//    shader patch (see _patchMaterial). It never replaces another agent's
//    onBeforeCompile, and if the patch cannot be applied the lights still sum
//    to exactly one sun — the fallback degrades softness, never brightness.
//
// 4. Fog is a global override of three's four fog ShaderChunks: analytic
//    height-integrated optical depth + a squared distance term, tinted by an
//    aerial-perspective approximation of THIS sky (horizon / zenith / forward
//    scatter toward the sun). Because three applies fog after tone mapping,
//    every fog colour here is baked through ACES + sRGB on the CPU so the fog
//    and the sky agree pixel-for-pixel at the horizon line.
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

/** Byte-for-byte mirror of three's ACESFilmicToneMapping, so baked fog
 *  colours land on exactly the same pixel value as the shaded sky. */
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

  // Forward-scattered aureole. Preetham's Perez term already carries most of
  // it; this puts the last stop of punch back into the 5 degrees around the sun.
  col += uSunColor * uHalo * exp( -gamma * gamma * 260.0 );

  // Solar disc, deliberately HDR so the bloom stage has something real to
  // bleed when the sun rakes past an edge. Suppressed for the IBL capture.
  float disc = 1.0 - smoothstep( 0.0043, 0.0082, gamma );
  col += uSunColor * uDisc * disc;

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
const QUALITY = {
  low:    { cascades: 1, sizes: [1024], split: 45, envSize: 128, halo: 1.6 },
  medium: { cascades: 2, sizes: [1024, 1024], split: 12, envSize: 256, halo: 2.2 },
  high:   { cascades: 2, sizes: [2048, 2048], split: 14, envSize: 256, halo: 2.6 },
  ultra:  { cascades: 2, sizes: [2048, 2560], split: 18, envSize: 256, halo: 2.8 },
};

const FOG_HEIGHT_FALLOFF = 0.075;  // 1/m — dust hugs the ground, e-fold ~13m
const FOG_DIST_K = 1.10;           // weight of the squared-distance term
const FOG_HEIGHT_MIX = 1.00;

export class Atmosphere {
  constructor(renderer, scene) {
    this.scene = scene;
    this.renderer = renderer;
    this.quality = 'high';

    // Late afternoon, dusty. 18 degrees of elevation gives a ~3x shadow length
    // and a golden key without tipping into sunset orange.
    this.sunElevation = 18;
    this.sunAzimuth = 118;
    this.turbidity = 4.2;

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
    this._invProj = new THREE.Matrix4();
    this._camRot = new THREE.Matrix3();
    this._camera = null;
    this._frames = 0;
    this._progN = -1;
    this._keyed = new WeakSet();
    this._csmPatched = new WeakSet();
    this._skyRev = 0;
    this._csmActive = false;
    this._exposure = renderer.toneMappingExposure;

    this.worldRadius = 78;   // half-extent the far cascade must cover
    this.worldMidY = 8;

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
      uScale: { value: 1 },
      uDisc: { value: 26 },
      uHalo: { value: 2.6 },
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

    // Private scene used only for the IBL capture — same maths, no sun disc
    // (a 0.5 degree disc aliases badly into a 256px cube and fireflies the
    // prefiltered mips; the directional light carries that energy instead).
    const cubeUniforms = {};
    for (const k of Object.keys(uniforms)) {
      if (k === 'uInvProj' || k === 'uCamRot') continue;
      cubeUniforms[k] = uniforms[k];
    }
    cubeUniforms.uDisc = { value: 0 };
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
    const hemi = new THREE.HemisphereLight(0xa9c6e6, 0x6b5940, 0.09);
    this.scene.add(hemi);
    this.hemi = hemi;

    this.sunIntensity = 4.2;
    this.sunColor = new THREE.Color(1, 0.79, 0.43);
  }

  // -------------------------------------------------------------- cascades
  _applyQuality() {
    const q = QUALITY[this.quality] || QUALITY.high;
    this._q = q;

    // Rebuild the cascade rig only when the count changes.
    if (this.cascades.length !== q.cascades) {
      for (const c of this.cascades) {
        c.light.shadow.map?.dispose();
        this.scene.remove(c.light, c.light.target);
        c.light.dispose();
      }
      this.cascades.length = 0;
      for (let i = 0; i < q.cascades; i++) {
        const l = new THREE.DirectionalLight(0xffffff, 1);
        l.castShadow = true;
        l.shadow.camera.near = 0.1;
        l.shadow.bias = -0.0004;
        l.shadow.normalBias = 0.035;
        l.name = 'sunCascade' + i;
        this.scene.add(l);
        this.scene.add(l.target);
        this.cascades.push({ light: l, radius: 0, size: 1024 });
      }
      this.sunLight = this.cascades[0].light;
    }

    let texels = 0;
    for (let i = 0; i < this.cascades.length; i++) {
      const c = this.cascades[i];
      c.size = q.sizes[i];
      if (c.light.shadow.mapSize.x !== c.size) {
        c.light.shadow.mapSize.set(c.size, c.size);
        c.light.shadow.map?.dispose();
        c.light.shadow.map = null;
        c.light.shadow.needsUpdate = true;
      }
      // The far cascade is coarse; it needs a fatter normal bias or the
      // 8cm texels peter-pan straight through thin geometry.
      const far = i === this.cascades.length - 1 && this.cascades.length > 1;
      // normalBias is world units, so it must track the cascade's texel size
      // (~2.3cm near, ~7.6cm far) or the far map peter-pans off thin geometry.
      c.light.shadow.normalBias = far ? 0.12 : 0.03;
      c.light.shadow.bias = far ? -0.00015 : -0.00012;
      c.light.shadow.radius = far ? 2.0 : 1.5;
      texels += c.size * c.size;
    }
    this.shadowTexels = texels;

    // Energy is split across cascades so that an unpatched material still
    // receives exactly one sun's worth of light; the CSM patch multiplies the
    // selected cascade back to full strength.
    const n = this.cascades.length;
    for (const c of this.cascades) {
      c.light.color.copy(this.sunColor);
      c.light.intensity = this.sunIntensity / n;
    }

    this._skyUniforms.uHalo.value = q.halo;
    this._buildCsmSource();
    // A quality change can turn cascade selection on after materials were
    // already skipped for it, so forget what has been patched and re-sweep.
    this._csmPatched = new WeakSet();
    this._progN = -1;
    this._frames = 0;
    this._invalidateAll();
    this._fitFar();
  }

  _fitFar() {
    if (this.cascades.length < 2) return;
    const c = this.cascades[this.cascades.length - 1];
    const r = this.worldRadius;
    c.radius = r;
    this._center.set(0, this.worldMidY, 0);
    this._placeCascade(c, this._center, r, false);
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
    // into it. Depth is packed to RGBA so the wide range costs no precision.
    const back = r + 60;
    c.light.target.position.copy(center);
    c.light.position.copy(center).addScaledVector(sun, back);
    c.light.target.updateMatrixWorld();

    cam.left = -r; cam.right = r; cam.top = r; cam.bottom = -r;
    cam.near = 0.5;
    cam.far = back + r + 60;
    cam.updateProjectionMatrix();
  }

  /** Bounding-sphere fit of the camera frustum slice [near, far]. Rotation
   *  invariant, so the radius never changes as the player looks around. */
  _fitNear(cam) {
    const c = this.cascades[0];
    const far = this._q.split;
    const near = Math.max(cam.near, 0.05);

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

  // -------------------------------------------------------------- CSM patch
  _buildCsmSource() {
    const n = this.cascades.length;
    this._csmActive = false;
    this._csmFn = '';
    this._csmInject = '';
    if (n < 2) return;

    const s = this._q.split;
    const lo = (s * 0.86).toFixed(2), hi = (s * 1.14).toFixed(2);
    const nf = n.toFixed(1);

    // Weights sum to N at every depth, so total sun energy is invariant.
    this._csmFn = /* glsl */`
float csmCascadeWeight( const in int idx, const in float viewDepth ) {
  float t = smoothstep( ${lo}, ${hi}, viewDepth );
  return idx == 0 ? ${nf} * ( 1.0 - t ) : ${nf} * t;
}
`;
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
    // hemisphere of sky. This is what puts warm bounce under overhangs.
    const tmp = [0, 0, 0];
    model.rawRadiance(0, 0.35, 0, tmp);
    const skyAvg = [tmp[0] * model.scale, tmp[1] * model.scale, tmp[2] * model.scale];
    const albedo = [0.31, 0.25, 0.175];
    const ndl = Math.max(this.sunDirection.y, 0);
    const sunIrr = this.sunIntensity * 0.16;
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

    this.sunColor.setRGB(model.sunTint[0], model.sunTint[1], model.sunTint[2]);
    for (const c of this.cascades) c.light.color.copy(this.sunColor);

    // Hemisphere fill matched to the actual sky and ground of this model.
    model.radiance(0, 1, 0, tmp);
    this.hemi.color.setRGB(
      Math.min(1, tmp[0] * 2.2), Math.min(1, tmp[1] * 2.2), Math.min(1, tmp[2] * 2.2));
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
    this.scene.environmentIntensity = 1.0;
    if (old) old.dispose();
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
    const exp = this.renderer.toneMappingExposure || 1;
    const lin = [0, 0, 0], out = [0, 0, 0];
    const sample = (x, y, z) => {
      const l = Math.hypot(x, y, z);
      m.radiance(x / l, y / l, z / l, lin);
      const a = acesFilmic(lin, exp, out);
      return [a[0], a[1], a[2]];
    };
    const sd = this.sunDirection;
    return {
      horizon: sample(-sd.x, 0.055, -sd.z),
      zenith: sample(0, 1, 0.001),
      low: sample(-sd.x, -0.02, -sd.z),
      towardSun: sample(sd.x, 0.06, sd.z),
    };
  }

  _updateFogColor() {
    const s = this._fogSamples();
    if (!this.scene.fog || !this.scene.fog.isFogExp2) {
      this.scene.fog = new THREE.FogExp2(0x000000, 0.0105);
    }
    this.scene.fog.color.setRGB(s.horizon[0], s.horizon[1], s.horizon[2]);
    this.scene.fog.density = 0.0105;
    return s;
  }

  _installFogChunks() {
    const s = this._updateFogColor();
    const sd = this.sunDirection;

    const ratio = (c) => [
      Math.min(4, c[0] / Math.max(s.horizon[0], 1e-3)),
      Math.min(4, c[1] / Math.max(s.horizon[1], 1e-3)),
      Math.min(4, c[2] / Math.max(s.horizon[2], 1e-3)),
    ];
    const zenR = ratio(s.zenith);
    const lowR = ratio(s.low);
    // Forward scatter is whatever the sun-facing horizon has that the
    // away-facing horizon does not: that is the light-shaft colour.
    const sunR = ratio([
      Math.max(0, s.towardSun[0] - s.horizon[0]),
      Math.max(0, s.towardSun[1] - s.horizon[1]),
      Math.max(0, s.towardSun[2] - s.horizon[2]),
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
    // term for the far falloff, tinted with this sky's aerial perspective.
    // Runs post-tone-map, which is why every colour below is already ACES+sRGB.
    THREE.ShaderChunk.fog_fragment = /* glsl */`
#ifdef USE_FOG
	vec3 fogVec = vFogView * mat3( viewMatrix );
	float fogDist = length( fogVec );
	vec3 fogDir = fogVec / max( fogDist, 1e-4 );
	#ifdef FOG_EXP2
		float fogSy = ( fogDir.y >= 0.0 ? 1.0 : -1.0 ) * max( abs( fogDir.y ), 1e-3 );
		float fogBase = exp( - ${H} * clamp( cameraPosition.y, -10.0, 200.0 ) );
		float fogOd = fogBase * ( 1.0 - exp( - ${H} * fogSy * fogDist ) ) / ( ${H} * fogSy );
		float fogLin = fogDensity * fogDist * ${K};
		float fogFactor = 1.0 - exp( - ( fogOd * fogDensity * ${M} + fogLin * fogLin ) );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	fogFactor = clamp( fogFactor, 0.0, 1.0 );
	vec3 fogAerial = fogColor * mix( vec3( 1.0 ), ${glslVec3(zenR)}, smoothstep( 0.0, 0.62, fogDir.y ) );
	fogAerial = mix( fogColor * ${glslVec3(lowR)}, fogAerial, smoothstep( -0.10, 0.02, fogDir.y ) );
	fogAerial += fogColor * ${glslVec3(sunR)} * pow( max( dot( fogDir, ${glslVec3([sd.x, sd.y, sd.z])} ), 0.0 ), 5.0 );
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

  /** How far the far cascade must reach. Call once if the map is not ~120m. */
  setWorldBounds(radius, midY = 8) {
    this.worldRadius = Math.max(8, radius);
    this.worldMidY = midY;
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
    }
    this._frames++;

    if (this._camera) {
      this._camera.updateMatrixWorld();
      this._fitNear(this._camera);
    }

    // If the render pipeline retunes exposure the fog would drift off the sky.
    // fogColor is a live uniform, so re-deriving it is enough — no recompile.
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
