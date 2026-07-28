// OWNER: agent "weapon" — first-person weapon meshes, built in code.
// LEGAL: fictional designs and fictional names only. No real-world manufacturer
// names, no Activision/Infinity Ward/Treyarch weapon names, no trademarks.
// CONTRACT:
//   buildWeapon(id, materials) -> {
//     group:  THREE.Group      (origin at the eye-space rest position)
//     parts:  { receiver, barrel, bolt, mag, stock, foregrip, optic, muzzle, trigger, charging }
//     muzzleTip: THREE.Object3D   (world position used for tracers/flash)
//     ejectPort: THREE.Object3D
//     config: {
//       name, class, rpm, damage, headMult, magSize, reserve,
//       spreadHip, spreadAds, recoil:{vertical,horizontal,recovery},
//       adsTime, reloadTime, reloadEmptyTime, muzzleVelocity, falloff:[[m,mult],...]
//     }
//   }
//   WEAPON_IDS -> string[]
//
// ---------------------------------------------------------------------------
// HOW THIS IS BUILT, and why.
//
// The weapon is the only object on screen for 100% of frames, at 20-50cm, so it
// is modelled the way a real one is manufactured — an assembly of parts with
// their own wall thickness, chamfers, fasteners and parting lines — rather than
// as a silhouette. Every machined edge carries a chamfer, because the chamfer is
// what produces the bright specular sliver that reads as metal when the key
// light moves. Normals are flat everywhere: averaged normals on a 3cm part make
// it look inflated and soft.
//
// DRAW CALLS. Geometry accumulates into per-material buckets and is merged once,
// so the whole rifle is 9 draws instead of ~200: three for the static body
// (steel / polymer / rubber), two for the sight (tinted lens, emissive dot) and
// four for the parts weaponfx.js animates independently — bolt, magazine,
// trigger, charging handle. Transparency and emission cannot share a bucket with
// anything, and the four moving parts must not, so 9 is the floor for this
// design. The static named handles (barrel, stock, foregrip, optic, muzzle) are
// empty Object3D pivots sitting at the true position of that part; their
// geometry lives inside the merged body meshes.
//
// UV. Extrude and lathe UVs are useless when texel density is the whole point,
// so after merging, each bucket gets a box projection at a fixed tiles-per-metre
// rate. The phosphate grain is then the same physical size on the receiver flat
// as on the barrel, which is what stops procedural metal reading as plastic.
//
// SIGHT AXIS. weaponfx.js drives the rig to (0, -0.045, -0.16) at full ADS, so
// the optical axis of the sight sits at local (x = 0, y = +0.045) and the dot
// lands dead centre of frame. Do not move one without the other.

import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { makeRng } from '../core/rng.js';

export const WEAPON_IDS = ['ar_vector', 'smg_wasp', 'dmr_ridge'];

// ---------------------------------------------------------------- layout
// Every shared hard number lives here. Bore sits 53mm under the sight axis so
// the optic mount is a real part with a visible riser, not a shim.
const SIGHT_Y = 0.045;
const BORE_Y = -0.008;
const RAIL_BOT = 0.016;    // top face of the upper receiver and handguard
const RAIL_TOP = 0.0205;   // top of the rail base, under the teeth
const UP_F = -0.098, UP_R = 0.058;         // upper receiver extents in z
const UP_C = (UP_F + UP_R) / 2;
const PORT_Z = -0.055, PORT_Y = -0.008;    // ejection port centre
const HG_Z = -0.100;                       // handguard / receiver joint

// ---------------------------------------------------------------- geometry kit

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();

/** Position / rotate / scale a geometry in place. Normals come along correctly. */
function xf(g, o) {
  if (!o) return g;
  _e.set(o.rx || 0, o.ry || 0, o.rz || 0);
  _q.setFromEuler(_e);
  _p.set(o.x || 0, o.y || 0, o.z || 0);
  _s.set(o.sx || 1, o.sy || 1, o.sz || 1);
  g.applyMatrix4(_m.compose(_p, _q, _s));
  return g;
}

function put(bucket, g, o) { bucket.push(xf(g, o)); return g; }
function putAll(bucket, list, o) {
  for (const g of list) bucket.push(xf(g, o));
  return list;
}
/** Rotate a list of geometries about their shared local origin, in place. */
function turnY(list, a) { for (const g of list) g.rotateY(a); return list; }
function turnZ(list, a) { for (const g of list) g.rotateZ(a); return list; }

const EX = { steps: 1, bevelEnabled: true, bevelSegments: 1, curveSegments: 1 };

/**
 * Chamfered box — the workhorse, ~60 triangles. Every edge gets a 45 degree
 * land of `c`; that land is the difference between "a box" and "a milled part".
 */
function cbox(w, h, d, c = 0.0014) {
  c = Math.max(0.00015, Math.min(c, w * 0.24, h * 0.24, d * 0.24));
  const a = w / 2 - c, b = h / 2 - c;
  const s = new THREE.Shape();
  s.moveTo(-a + c, -b); s.lineTo(a - c, -b); s.lineTo(a, -b + c);
  s.lineTo(a, b - c); s.lineTo(a - c, b); s.lineTo(-a + c, b);
  s.lineTo(-a, b - c); s.lineTo(-a, -b + c); s.closePath();
  const g = new THREE.ExtrudeGeometry(s, {
    ...EX, depth: Math.max(0.0002, d - 2 * c), bevelThickness: c, bevelSize: c,
  });
  g.translate(0, 0, -(d / 2 - c));
  return g;
}

/** Six-sided box with independent top and bottom footprints. 12 flat triangles. */
function frustumBox(wt, dt, wb, db, h) {
  const xt = wt / 2, zt = dt / 2, xb = wb / 2, zb = db / 2, y = h / 2;
  const T = [[-xt, y, -zt], [xt, y, -zt], [xt, y, zt], [-xt, y, zt]];
  const M = [[-xb, -y, -zb], [xb, -y, -zb], [xb, -y, zb], [-xb, -y, zb]];
  const quads = [
    [T[3], T[2], T[1], T[0]],
    [M[0], M[1], M[2], M[3]],
    [M[3], M[2], T[2], T[3]],
    [M[1], M[0], T[0], T[1]],
    [M[2], M[1], T[1], T[2]],
    [M[0], M[3], T[3], T[0]],
  ];
  const pos = new Float32Array(quads.length * 18);
  let k = 0;
  for (const [a, b, c, d] of quads) {
    for (const v of [a, b, c, a, c, d]) { pos[k++] = v[0]; pos[k++] = v[1]; pos[k++] = v[2]; }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(pos.length / 3 * 2), 2));
  g.computeVertexNormals();   // non-indexed, so this gives flat faces
  return g;
}

/** Cylinder whose axis runs down -Z (muzzle-ward); rF is the forward radius. */
function tubeZ(rF, rB, len, seg, open) {
  const g = new THREE.CylinderGeometry(rF, rB, len, seg, 1, !!open);
  g.rotateX(-Math.PI / 2);
  return g;
}

/**
 * Lathe about the -Z axis. `pts` are [radius, distance forward] pairs. This is
 * how the barrel gets its steps and the bezels get their rings, at ~40 triangles
 * per profile segment.
 */
function latheZ(pts, seg) {
  const v = pts.map((p) => new THREE.Vector2(Math.max(1e-5, p[0]), p[1]));
  const g = new THREE.LatheGeometry(v, seg);
  g.rotateX(-Math.PI / 2);
  return g;
}

/**
 * Force flat shading. Lathes and cylinders come back smooth-normalled, which is
 * right for a barrel and completely wrong for a machined ring or a knurled cap —
 * those read as rubber o-rings until each facet gets its own normal.
 */
function flat(g) {
  const n = g.index ? g.toNonIndexed() : g;
  n.computeVertexNormals();
  return n;
}

/** Closed annulus — washer, spacer, baffle ring. Axis along -Z, flat shaded. */
function ringZ(rIn, rOut, len, seg) {
  return flat(latheZ([[rIn, 0], [rOut, 0], [rOut, len], [rIn, len], [rIn, 0]], seg));
}

function weld(list) {
  const nz = list.map((g) => (g.index ? g.toNonIndexed() : g));
  return nz.length === 1 ? nz[0] : mergeGeometries(nz, false);
}

const _protos = new Map();
/** Pan-head fastener with a driver slot, axis -Z. Cached, cloned per instance. */
function screw(r, h, spin) {
  const key = 'sc' + r + '_' + h;
  let g = _protos.get(key);
  if (!g) {
    g = weld([
      flat(latheZ([[0, 0], [r, 0], [r, h * 0.62], [r * 0.86, h], [0, h]], 8)),
      xf(cbox(r * 1.8, r * 0.32, h * 0.55, 0.0002), { z: -h * 0.78 }),
    ]);
    _protos.set(key, g);
  }
  const c = g.clone();
  if (spin) c.rotateZ(spin);
  return c;
}

/** Shallow blind socket — QD sling point. Axis -Z, mouth at the origin. */
function qdSocket(r) {
  return flat(latheZ([
    [r * 0.42, 0], [r, 0], [r, r * 0.62], [r * 0.62, r * 0.9], [r * 0.42, r * 0.9],
  ], 10));
}

/**
 * Flat plate with a row of through-slots: M-LOK on the handguard, witness slots
 * on the magazine. Decomposed into two side rails plus the bridges between
 * slots, so the openings are real holes with chamfered lips.
 * Local frame: length along Z, width along X, thickness along Y, outward +Y.
 */
function slotPanel(lenZ, wX, tY, n, pitch, slotLen, slotW, c) {
  const out = [];
  const railW = (wX - slotW) / 2;
  out.push(xf(cbox(railW, tY, lenZ, c), { x: -(slotW + railW) / 2 }));
  out.push(xf(cbox(railW, tY, lenZ, c), { x: (slotW + railW) / 2 }));
  const half = lenZ / 2;
  const z0 = -pitch * (n - 1) / 2;
  for (let i = 0; i <= n; i++) {
    const a = i === 0 ? -half : z0 + pitch * (i - 1) + slotLen / 2;
    const b = i === n ? half : z0 + pitch * i - slotLen / 2;
    if (b - a < 0.0008) continue;
    out.push(xf(cbox(slotW, tY, b - a, c), { z: (a + b) / 2 }));
  }
  return out;
}

/**
 * Rectangular plate with one rectangular hole — ejection port, sling loop.
 * Local frame: w along X, h along Y, thickness along Z.
 */
function framedPanel(w, h, t, hx, hy, hw, hh, c) {
  const out = [];
  const top = h / 2 - (hy + hh / 2), bot = (hy - hh / 2) + h / 2;
  if (top > 0.0008) out.push(xf(cbox(w, top, t, c), { y: h / 2 - top / 2 }));
  if (bot > 0.0008) out.push(xf(cbox(w, bot, t, c), { y: -h / 2 + bot / 2 }));
  const l = w / 2 + (hx - hw / 2), r = w / 2 - (hx + hw / 2);
  if (l > 0.0008) out.push(xf(cbox(l, hh, t, c), { x: -w / 2 + l / 2, y: hy }));
  if (r > 0.0008) out.push(xf(cbox(r, hh, t, c), { x: w / 2 - r / 2, y: hy }));
  return out;
}

/** Box-projected UVs at a fixed tiles-per-metre, applied after the merge. */
function boxUv(geo, scale) {
  const p = geo.getAttribute('position');
  const n = geo.getAttribute('normal');
  let uv = geo.getAttribute('uv');
  if (!uv || uv.count !== p.count) {
    uv = new THREE.BufferAttribute(new Float32Array(p.count * 2), 2);
    geo.setAttribute('uv', uv);
  }
  const a = uv.array, pa = p.array, na = n.array;
  for (let i = 0, j = 0; i < p.count; i++, j += 3) {
    const ax = Math.abs(na[j]), ay = Math.abs(na[j + 1]), az = Math.abs(na[j + 2]);
    let u, v;
    if (ax >= ay && ax >= az) { u = pa[j + 2]; v = pa[j + 1]; }
    else if (ay >= az) { u = pa[j]; v = pa[j + 2]; }
    else { u = pa[j]; v = pa[j + 1]; }
    a[i * 2] = u * scale; a[i * 2 + 1] = v * scale;
  }
  uv.needsUpdate = true;
  return geo;
}

// ---------------------------------------------------------------- variants
// One builder, three sets of proportions. ar_vector is the art-directed one;
// smg_wasp and dmr_ridge re-use the identical assembly with tuned lengths,
// slot counts and stats. That is a deliberate budget decision, not an oversight.
const SPECS = {
  ar_vector: {
    name: 'VK-7 CARBINE', cls: 'assault',
    hgLen: 0.230, hgR: 0.024, mlok: 4, barrel: 0.347, gasZ: -0.352,
    brakeLen: 0.063, brakeR: 0.0135, magH: 0.155, magCurve: 0.30,
    stockLen: 0.120, opticLen: 0.056, opticR: 0.0155,
    config: {
      rpm: 720, damage: 28, headMult: 1.9, magSize: 30, reserve: 210,
      spreadHip: 0.032, spreadAds: 0.0035,
      recoil: { vertical: 1.0, horizontal: 0.45, recovery: 7.5 },
      adsTime: 0.22, reloadTime: 1.9, reloadEmptyTime: 2.6,
      muzzleVelocity: 780, falloff: [[0, 1], [28, 1], [55, 0.72], [90, 0.55]],
    },
  },
  smg_wasp: {
    name: 'PDW-9 WASP', cls: 'smg',
    hgLen: 0.150, hgR: 0.023, mlok: 2, barrel: 0.232, gasZ: -0.272,
    brakeLen: 0.044, brakeR: 0.0128, magH: 0.170, magCurve: 0.16,
    stockLen: 0.100, opticLen: 0.050, opticR: 0.0150,
    config: {
      rpm: 900, damage: 22, headMult: 1.7, magSize: 36, reserve: 252,
      spreadHip: 0.038, spreadAds: 0.0052,
      recoil: { vertical: 0.72, horizontal: 0.52, recovery: 9.0 },
      adsTime: 0.16, reloadTime: 1.7, reloadEmptyTime: 2.3,
      muzzleVelocity: 400, falloff: [[0, 1], [14, 1], [30, 0.66], [55, 0.44]],
    },
  },
  dmr_ridge: {
    name: 'MR-4 RIDGELINE', cls: 'dmr',
    hgLen: 0.300, hgR: 0.025, mlok: 5, barrel: 0.430, gasZ: -0.436,
    brakeLen: 0.070, brakeR: 0.0142, magH: 0.140, magCurve: 0.22,
    stockLen: 0.132, opticLen: 0.082, opticR: 0.0180,
    config: {
      rpm: 380, damage: 52, headMult: 2.1, magSize: 20, reserve: 140,
      spreadHip: 0.046, spreadAds: 0.0016,
      recoil: { vertical: 1.9, horizontal: 0.40, recovery: 6.0 },
      adsTime: 0.30, reloadTime: 2.1, reloadEmptyTime: 2.9,
      muzzleVelocity: 900, falloff: [[0, 1], [45, 1], [80, 0.86], [140, 0.7]],
    },
  },
};

// ---------------------------------------------------------------- upper

function buildUpper(B, s, rng) {
  const L = UP_R - UP_F;

  // Top deck and left wall are solid; the right wall is framed around the
  // ejection port so the bolt carrier is genuinely visible through a hole.
  put(B.steel, cbox(0.041, 0.011, L, 0.0016), { y: 0.0105, z: UP_C });
  put(B.steel, cbox(0.005, 0.030, L, 0.0013), { x: -0.018, y: -0.010, z: UP_C });
  putAll(B.steel, turnY(framedPanel(
    L, 0.030, 0.005, PORT_Z * -1 + UP_C, PORT_Y + 0.010, 0.054, 0.016, 0.0011,
  ), Math.PI / 2), { x: 0.018, y: -0.010, z: UP_C });
  // the strip the lower closes against — leaves a 1mm parting line with two
  // chamfers in it, which is what a takedown seam actually looks like
  put(B.steel, cbox(0.041, 0.004, L, 0.0012), { y: -0.023, z: UP_C });

  // brass deflector and forward assist: the two lumps that make an upper read
  put(B.steel, frustumBox(0.008, 0.020, 0.014, 0.026, 0.016), { x: 0.024, y: -0.004, z: -0.014, rz: -0.35 });
  put(B.steel, tubeZ(0.0055, 0.0065, 0.020, 10), { x: 0.0225, y: 0.001, z: -0.006, ry: Math.PI / 2 });
  put(B.steel, tubeZ(0.0078, 0.0078, 0.004, 10), { x: 0.0285, y: 0.001, z: -0.006, ry: Math.PI / 2 });

  // dust cover, hanging open on its hinge pin just under the port
  putAll(B.steel, turnY([cbox(0.050, 0.015, 0.003, 0.0007), cbox(0.050, 0.004, 0.006, 0.0006)], Math.PI / 2),
    { x: 0.0228, y: -0.0245, z: PORT_Z, rz: 0.35 });
  put(B.steel, tubeZ(0.0021, 0.0021, 0.058, 6), { x: 0.0205, y: -0.0175, z: PORT_Z, ry: Math.PI / 2 });

  // takedown pins through the receiver, with a chamfered head each side
  for (const pz of [0.040, -0.086]) {
    put(B.steel, tubeZ(0.0034, 0.0034, 0.042, 8), { y: -0.018, z: pz, ry: Math.PI / 2 });
    for (const sx of [-1, 1]) {
      put(B.steel, latheZ([
        [0, 0], [0.0048, 0], [0.0048, 0.0022], [0.0036, 0.0032], [0, 0.0032],
      ], 10), { x: sx * 0.0208, y: -0.018, z: pz, ry: sx * Math.PI / 2 });
    }
  }

  // charging handle channel lip, rear
  put(B.steel, cbox(0.030, 0.005, 0.010, 0.0008), { y: 0.0125, z: UP_R - 0.002 });

  // QD sling socket, receiver rear left
  put(B.steel, xf(qdSocket(0.0062), {}), { x: -0.0208, y: -0.012, z: 0.044, ry: -Math.PI / 2 });

  // deck fasteners along the seam
  for (let i = 0; i < 3; i++) {
    put(B.steel, screw(0.0021, 0.0016, rng() * Math.PI),
      { x: -0.0162, y: 0.0157, z: 0.034 - i * 0.030, rx: -Math.PI / 2 });
  }
}

function buildRail(B, s) {
  const z0 = 0.050, z1 = HG_Z - s.hgLen + 0.005;
  const L = z0 - z1, zC = (z0 + z1) / 2;
  // picatinny section: widest at mid height with 45 degree flanks top and bottom
  put(B.steel, frustumBox(0.0210, L, 0.0180, L, 0.0025), { y: RAIL_BOT + 0.00125, z: zC });
  put(B.steel, frustumBox(0.0186, L, 0.0210, L, 0.0020), { y: RAIL_BOT + 0.0035, z: zC });
  // Recoil teeth, modelled individually. The row of specular slivers running the
  // length of the gun is most of what makes it read as manufactured, and at
  // 12 triangles each it is the cheapest detail on the whole model.
  const pitch = 0.0102, n = Math.floor(L / pitch);
  const off = (L - n * pitch) / 2;
  for (let i = 0; i < n; i++) {
    put(B.steel, frustumBox(0.0186, 0.0046, 0.0206, 0.0056, 0.0042),
      { y: RAIL_TOP + 0.0021, z: z0 - off - pitch * (i + 0.5) });
  }
}

// ---------------------------------------------------------------- barrel group

function buildBarrel(B, s, rng) {
  const z0 = -0.088, L = s.barrel, gd = z0 - s.gasZ;
  put(B.steel, latheZ([
    [0, 0], [0.017, 0], [0.017, 0.030], [0.0135, 0.036], [0.0135, 0.118],
    [0.0105, 0.126], [0.0105, gd - 0.016], [0.0125, gd - 0.012],
    [0.0125, gd + 0.012], [0.0100, gd + 0.016], [0.0100, L - 0.011],
    [0.0092, L - 0.006], [0.0092, L], [0, L],
  ], 20), { y: BORE_Y, z: z0 });

  // low-profile gas block with its set screws, gas tube running back inside the
  // handguard so it shows through the M-LOK slots
  put(B.steel, cbox(0.024, 0.026, 0.030, 0.0018), { y: BORE_Y + 0.001, z: s.gasZ });
  put(B.steel, screw(0.0022, 0.0015, rng() * Math.PI), { y: BORE_Y + 0.0145, z: s.gasZ - 0.008, rx: -Math.PI / 2 });
  put(B.steel, screw(0.0022, 0.0015, rng() * Math.PI), { y: BORE_Y + 0.0145, z: s.gasZ + 0.008, rx: -Math.PI / 2 });
  const gtL = Math.abs(s.gasZ) - 0.098;
  put(B.steel, tubeZ(0.0027, 0.0027, gtL, 8), { y: BORE_Y + 0.0105, z: s.gasZ + gtL / 2 });

  const mz = z0 - L;
  put(B.steel, ringZ(0.0092, 0.0126, 0.0022, 16), { y: BORE_Y, z: mz + 0.0022 });
  buildMuzzle(B, s, mz);
}

function buildMuzzle(B, s, mz) {
  const R = s.brakeR, L = s.brakeLen;
  // Collar, then four baffle rings bridged by three spines. The gaps between the
  // rings are the ports: real openings you can see through, not painted slots.
  put(B.steel, flat(latheZ([[0.0092, 0], [R, 0.002], [R, 0.011], [R * 0.86, 0.013]], 16)), { y: BORE_Y, z: mz });
  const n = 4, span = L - 0.021, pitch = span / n;
  for (let i = 0; i < n; i++) {
    put(B.steel, ringZ(0.0072, R * (i === n - 1 ? 0.96 : 1), pitch * 0.32, 16),
      { y: BORE_Y, z: mz - 0.013 - pitch * i - pitch * 0.24 });
  }
  for (let i = 0; i < 3; i++) {
    const a = Math.PI / 2 + (i * 2 * Math.PI) / 3;
    put(B.steel, cbox(0.0055, 0.0055, span + 0.004, 0.0009), {
      x: Math.cos(a) * (R - 0.0026),
      y: BORE_Y + Math.sin(a) * (R - 0.0026),
      z: mz - 0.013 - span / 2,
    });
  }
  // crowned exit face
  put(B.steel, flat(latheZ([
    [0.0072, 0], [R * 0.96, 0], [R * 0.96, 0.006], [0.0094, 0.008], [0.0072, 0.006],
  ], 16)), { y: BORE_Y, z: mz - L + 0.008 });
}

// ---------------------------------------------------------------- handguard

function buildHandguard(B, s, rng) {
  const zR = HG_Z, zF = HG_Z - s.hgLen, zC = (zF + zR) / 2, L = s.hgLen;
  const R = s.hgR, tY = 0.0045;
  const cy = RAIL_BOT - R;                                    // top facet flush with the rail
  const side = 2 * R * Math.tan(Math.PI / 8) + 0.0008;
  const pitch = 0.0415, slotLen = 0.032, slotW = 0.0075;

  // Seven facets; the eighth is the rail. Sides and lower diagonals carry M-LOK,
  // the upper diagonals stay plain so the silhouette does not get noisy.
  const near = Math.max(1, s.mlok - 1);
  const facets = [[0, s.mlok], [45, 0], [135, 0], [180, s.mlok], [225, near], [270, near], [315, near]];
  for (const [deg, slots] of facets) {
    const phi = (deg * Math.PI) / 180;
    const geos = slots > 0
      ? slotPanel(L - 0.010, side, tY, slots, pitch, slotLen, slotW, 0.0010)
      : [cbox(side, tY, L - 0.010, 0.0010)];
    putAll(B.poly, geos, {
      x: Math.cos(phi) * (R - tY / 2),
      y: cy + Math.sin(phi) * (R - tY / 2),
      z: zC, rz: phi - Math.PI / 2,
    });
  }
  // inner shroud, so the slots read as holes with something behind them
  put(B.poly, tubeZ(R - 0.0055, R - 0.0055, L - 0.014, 14, true), { y: cy, z: zC });
  put(B.poly, ringZ(R - 0.0075, R + 0.0005, 0.008, 14), { y: cy, z: zF + 0.008 });
  put(B.poly, ringZ(R - 0.0080, R + 0.0018, 0.014, 14), { y: cy, z: zR });

  // barrel nut the handguard clamps to, and the clamp screws
  put(B.steel, latheZ([
    [0.0135, 0], [0.019, 0], [0.019, 0.010], [0.0175, 0.012], [0.0175, 0.020], [0.0135, 0.020],
  ], 14), { y: BORE_Y, z: zR + 0.001 });
  for (let i = 0; i < 4; i++) {
    const a = (Math.PI / 4) + (i * Math.PI) / 2;
    const g = screw(0.0022, 0.0016, rng() * Math.PI);
    g.rotateY(Math.PI / 2);   // axis -Z -> -X, then rz swings it to point inward
    put(B.steel, g, { x: Math.cos(a) * (R + 0.0006), y: cy + Math.sin(a) * (R + 0.0006), z: zR - 0.007, rz: a });
  }
  // QD socket at 3 o'clock, forward
  put(B.steel, qdSocket(0.0062), { x: R - 0.001, y: cy, z: zF + 0.032, ry: Math.PI / 2 });
}

// ---------------------------------------------------------------- lower

function buildLower(B, s, rng) {
  // fire control housing, kept clear of the magwell so the parting lines read
  put(B.steel, cbox(0.039, 0.024, 0.062, 0.0016), { y: -0.038, z: 0.010 });
  put(B.steel, cbox(0.036, 0.014, 0.030, 0.0014), { y: -0.052, z: 0.022 });

  // magwell: four walls plus a flared lip, so it is a box with a hole in it
  put(B.steel, cbox(0.040, 0.030, 0.006, 0.0012), { y: -0.038, z: -0.080 });
  put(B.steel, cbox(0.040, 0.030, 0.006, 0.0012), { y: -0.038, z: -0.010 });
  put(B.steel, cbox(0.006, 0.030, 0.064, 0.0012), { x: -0.017, y: -0.038, z: -0.045 });
  put(B.steel, cbox(0.006, 0.030, 0.064, 0.0012), { x: 0.017, y: -0.038, z: -0.045 });
  put(B.steel, frustumBox(0.042, 0.074, 0.036, 0.066, 0.008), { y: -0.0555, z: -0.045 });
  // panel screws and a stamped-in relief on the right flat, so the lower is not
  // a plain slab from the one angle the player sees most
  put(B.steel, screw(0.0020, 0.0013, rng() * Math.PI), { x: 0.0202, y: -0.044, z: -0.004, ry: Math.PI / 2 });
  put(B.steel, screw(0.0020, 0.0013, rng() * Math.PI), { x: 0.0202, y: -0.044, z: 0.026, ry: Math.PI / 2 });
  put(B.steel, cbox(0.0035, 0.011, 0.030, 0.0008), { x: 0.0192, y: -0.0365, z: 0.014 });
  put(B.steel, cbox(0.0035, 0.011, 0.030, 0.0008), { x: -0.0192, y: -0.0365, z: 0.014 });

  // trigger guard: five chamfered members, angular in the modern way
  put(B.steel, cbox(0.009, 0.030, 0.0075, 0.0011), { y: -0.062, z: -0.004, rx: 0.18 });
  put(B.steel, cbox(0.009, 0.026, 0.0075, 0.0011), { y: -0.064, z: -0.046, rx: -0.30 });
  put(B.steel, cbox(0.009, 0.0075, 0.046, 0.0011), { y: -0.0775, z: -0.026 });
  put(B.steel, cbox(0.009, 0.0075, 0.013, 0.0011), { y: -0.0730, z: -0.048, rx: 0.7 });
  put(B.steel, screw(0.0020, 0.0014, rng() * Math.PI), { x: -0.0048, y: -0.0525, z: -0.052, ry: -Math.PI / 2 });

  // ambidextrous safety selector
  put(B.steel, tubeZ(0.0052, 0.0052, 0.044, 10), { y: -0.032, z: 0.014, ry: Math.PI / 2 });
  put(B.steel, cbox(0.0065, 0.009, 0.026, 0.0008), { x: -0.0235, y: -0.036, z: 0.008, rx: -0.5 });
  put(B.steel, cbox(0.0055, 0.008, 0.020, 0.0008), { x: 0.0230, y: -0.035, z: 0.010, rx: -0.5 });

  // magazine release inside its fence, bolt catch on the left
  put(B.steel, tubeZ(0.0048, 0.0048, 0.008, 10), { x: 0.0215, y: -0.030, z: -0.016, ry: Math.PI / 2 });
  put(B.steel, ringZ(0.0055, 0.0085, 0.004, 10), { x: 0.0195, y: -0.030, z: -0.016, ry: Math.PI / 2 });
  put(B.steel, cbox(0.005, 0.014, 0.030, 0.0009), { x: -0.0215, y: -0.028, z: -0.022 });
  put(B.steel, cbox(0.006, 0.010, 0.009, 0.0009), { x: -0.0225, y: -0.030, z: -0.036 });

  // buffer tube, castle nut, end plate with its own QD socket
  put(B.steel, tubeZ(0.0145, 0.0145, s.stockLen, 16), { y: -0.006, z: 0.058 + s.stockLen / 2 });
  put(B.steel, latheZ([[0.0145, 0], [0.019, 0], [0.019, 0.009], [0.0145, 0.009]], 12), { y: -0.006, z: 0.066 });
  put(B.steel, cbox(0.036, 0.040, 0.004, 0.0010), { y: -0.006, z: 0.060 });
  put(B.steel, qdSocket(0.0062), { x: -0.016, y: -0.012, z: 0.062, ry: -Math.PI / 2 });
  for (let i = 0; i < 6; i++) {
    put(B.steel, cbox(0.010, 0.004, 0.006, 0.0007), { y: -0.0195, z: 0.082 + i * 0.019 });
  }
}

function buildGrip(B) {
  // Built in a raked local frame; negative rx leans the butt of the grip back.
  const rake = -0.34, oy = -0.048, oz = 0.016;
  // Segments overlap by ~8mm so their chamfers hide inside each other; butt them
  // together instead and the grip reads as a stack of tins.
  const seg = [
    [0.0330, 0.030, 0.040, 0.000, 0.0000],
    [0.0336, 0.032, 0.041, -0.014, 0.0015],
    [0.0342, 0.032, 0.042, -0.038, 0.0028],
    [0.0332, 0.032, 0.040, -0.062, 0.0022],
    [0.0314, 0.030, 0.038, -0.086, 0.0000],
  ];
  const body = [];
  for (const [w, h, d, y, z] of seg) body.push(xf(cbox(w, h, d, 0.0022), { y, z }));
  body.push(xf(frustumBox(0.030, 0.030, 0.034, 0.044, 0.014), { y: 0.012, z: -0.004 }));  // beavertail
  body.push(xf(cbox(0.032, 0.008, 0.040, 0.0018), { y: -0.104 }));                        // grip cap
  putAll(B.poly, body, { y: oy, z: oz, rx: rake });

  // Moulded stipple: truncated pyramids on the front and back straps, coarse
  // ribs on the side panels. Cheap per stud and it is the one surface the player
  // sees at 15cm, so it earns its triangles.
  const studs = [];
  const rows = 9, cols = 6;
  for (const zf of [-1, 1]) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const bow = Math.cos((c - (cols - 1) / 2) * 0.42) * 0.0016;
        const g = new THREE.CylinderGeometry(0.0011, 0.0019, 0.0018, 4, 1);
        g.rotateY(Math.PI / 4);
        g.rotateX(zf * Math.PI / 2);
        studs.push(xf(g, {
          x: (c - (cols - 1) / 2) * 0.0050,
          y: -0.008 - r * 0.0100,
          z: zf * (0.0200 + bow),
        }));
      }
    }
  }
  // side panels: fine horizontal ribs rather than pads, which at this scale
  // would read as bolted-on plaques instead of a moulded grip surface
  for (let r = 0; r < 8; r++) {
    for (const sx of [-1, 1]) {
      studs.push(xf(cbox(0.0014, 0.0022, 0.019, 0.0004), { x: sx * 0.0168, y: -0.016 - r * 0.0095 }));
    }
  }
  putAll(B.rubber, studs, { y: oy, z: oz, rx: rake });
}

function buildStock(B, s, rng) {
  const z0 = 0.062, L = s.stockLen, zC = z0 + L * 0.52;
  // Deliberately the cheapest assembly on the gun: it is behind the camera's
  // focus and half of it is off the right edge of the frame.
  put(B.poly, cbox(0.034, 0.032, 0.042, 0.0025), { y: -0.009, z: z0 + 0.020 });
  put(B.poly, cbox(0.030, 0.016, L * 0.80, 0.0022), { y: 0.012, z: zC });
  put(B.poly, cbox(0.026, 0.022, L * 0.72, 0.0022), { y: -0.030, z: zC + 0.006 });
  put(B.poly, cbox(0.040, 0.078, 0.014, 0.0028), { y: -0.004, z: z0 + L - 0.004 });
  // cheek riser on two posts — the detail that says "adjustable"
  put(B.poly, cbox(0.028, 0.014, L * 0.62, 0.0024), { y: 0.028, z: zC + 0.008 });
  for (const pz of [zC - 0.026, zC + 0.030]) {
    put(B.poly, cbox(0.020, 0.014, 0.010, 0.0012), { y: 0.020, z: pz });
    put(B.steel, screw(0.0024, 0.0016, rng() * Math.PI), { x: 0.0142, y: 0.028, z: pz, ry: Math.PI / 2 });
  }
  put(B.poly, cbox(0.012, 0.016, 0.030, 0.0014), { y: -0.046, z: z0 + 0.040, rx: 0.25 });  // adjust lever
  // sling loop through the butt frame, and a QD socket opposite it
  putAll(B.poly, turnY(framedPanel(0.024, 0.030, 0.006, 0, 0, 0.012, 0.016, 0.0012), Math.PI / 2),
    { x: -0.019, y: -0.012, z: z0 + L - 0.016 });
  put(B.steel, qdSocket(0.0062), { x: 0.019, y: -0.012, z: z0 + L - 0.020, ry: Math.PI / 2 });

  // buttpad: four rubber pads with real grooves between them
  for (let i = 0; i < 4; i++) {
    put(B.rubber, cbox(0.040, 0.0165, 0.012, 0.0022), { y: 0.028 - i * 0.0195, z: z0 + L + 0.006 });
  }
  // The cheek pad wraps the whole top of the riser: at ADS the eye is 4cm above
  // this surface and a strip of bare polymer either side of the pad is the first
  // thing you notice.
  put(B.rubber, cbox(0.0296, 0.012, L * 0.60, 0.0018), { y: 0.036, z: zC + 0.008 });
}

// ---------------------------------------------------------------- optic

function buildOptic(B, s, rng) {
  const R = s.opticR, L = s.opticLen, zC = -0.018;
  const zR = zC + L / 2, zF = zC - L / 2;

  // Straight tube with proud bezel rings at each end. A waisted profile reads as
  // a beer keg the moment smooth normals get hold of it, so the body stays
  // cylindrical and the diameter changes happen at hard steps.
  put(B.steel, latheZ([
    [R * 0.80, 0], [R * 0.98, 0.003], [R, 0.007], [R, L - 0.007],
    [R * 0.98, L - 0.003], [R * 0.80, L],
  ], 26), { y: SIGHT_Y, z: zR });
  put(B.steel, ringZ(R * 0.76, R * 1.05, 0.005, 26), { y: SIGHT_Y, z: zR + 0.0005 });
  put(B.steel, ringZ(R * 0.76, R * 1.05, 0.005, 26), { y: SIGHT_Y, z: zF + 0.0055 });
  // internal hood: kills the flat black hole through the middle of the tube
  put(B.steel, tubeZ(R * 0.74, R * 0.74, L - 0.014, 20, true), { y: SIGHT_Y, z: zC });

  // turrets: short, stepped, flat shaded so the facets read as knurling
  const turrets = [[0, 1, -Math.PI / 2, 0], [1, 0, 0, Math.PI / 2]];
  for (const [dx, dy, rx, ry] of turrets) {
    put(B.steel, flat(tubeZ(0.0086, 0.0092, 0.005, 12)),
      { x: dx * (R + 0.002), y: SIGHT_Y + dy * (R + 0.002), z: zC + 0.004, rx, ry });
    put(B.steel, flat(tubeZ(0.0076, 0.0082, 0.007, 12)),
      { x: dx * (R + 0.008), y: SIGHT_Y + dy * (R + 0.008), z: zC + 0.004, rx, ry });
    put(B.steel, cbox(0.0090, 0.0016, 0.0030, 0.0003),
      { x: dx * (R + 0.0115), y: SIGHT_Y + dy * (R + 0.0115), z: zC + 0.004, rz: dx ? Math.PI / 2 : 0 });
  }
  put(B.steel, flat(tubeZ(0.0074, 0.0084, 0.008, 12)), { x: -(R + 0.002), y: SIGHT_Y, z: zC + 0.004, ry: -Math.PI / 2 });

  // mount: saddle, riser down to the rail, clamp jaw, cross bolt, throw lever
  put(B.steel, cbox(0.030, 0.014, 0.036, 0.0016), { y: SIGHT_Y - R - 0.004, z: zC });
  const riserH = Math.max(0.004, SIGHT_Y - R - RAIL_TOP + 0.003);
  put(B.steel, cbox(0.026, riserH, 0.034, 0.0016), { y: SIGHT_Y - R - 0.008 - riserH / 2 + 0.002, z: zC });
  put(B.steel, cbox(0.034, 0.011, 0.030, 0.0014), { y: RAIL_TOP - 0.0015, z: zC });
  put(B.steel, tubeZ(0.0030, 0.0030, 0.036, 8), { y: RAIL_TOP - 0.002, z: zC + 0.009, ry: Math.PI / 2 });
  // throw lever, folded flat against the mount the way it is carried
  put(B.steel, cbox(0.0045, 0.007, 0.020, 0.0007), { x: -0.0185, y: RAIL_TOP - 0.0015, z: zC + 0.004 });
  put(B.steel, flat(tubeZ(0.0042, 0.0042, 0.004, 8)), { x: -0.0185, y: RAIL_TOP - 0.0015, z: zC + 0.013, ry: -Math.PI / 2 });
  for (const bz of [zC - 0.011, zC + 0.011]) {
    put(B.steel, screw(0.0026, 0.0018, rng() * Math.PI), { y: SIGHT_Y - R + 0.004, z: bz, rx: -Math.PI / 2 });
  }
  // emitter housing at the bottom of the tube, angled up at the front lens
  put(B.steel, cbox(0.007, 0.005, 0.009, 0.0008), { y: SIGHT_Y - R * 0.60, z: zF + 0.015 });

  // lenses: shallow convex, subtly tinted, one at each end so the tube has depth
  const lens = (sign) => latheZ([
    [0, 0], [R * 0.30, sign * 0.0007], [R * 0.55, sign * 0.0013],
    [R * 0.72, sign * 0.0020], [R * 0.74, sign * 0.0020],
  ], 24);
  put(B.glass, lens(-1), { y: SIGHT_Y, z: zF + 0.006 });
  put(B.glass, lens(1), { y: SIGHT_Y, z: zR - 0.006 });

  // the dot, sitting on the front lens where the eye expects to focus, plus the
  // LED itself glowing in its housing
  put(B.emit, tubeZ(0.0011, 0.0011, 0.0018, 8), { y: SIGHT_Y - R * 0.56, z: zF + 0.014, rx: -0.6 });
  put(B.emit, new THREE.CircleGeometry(0.00072, 10), { y: SIGHT_Y, z: zF + 0.0078 });
}

// ---------------------------------------------------------------- moving parts
// All four are built in absolute rifle coordinates and re-rooted afterwards, so
// the pivot each one rotates or slides about is exactly where it should be.

function buildBolt(B) {
  // Sits directly under the ejection port with its face at the chamber, which is
  // the whole reason the port is a real hole: you watch this thing cycle.
  put(B.bolt, latheZ([
    [0, 0], [0.0125, 0], [0.0125, 0.012], [0.0112, 0.014], [0.0112, 0.020],
    [0.0125, 0.022], [0.0125, 0.048], [0.0112, 0.050], [0.0112, 0.056],
    [0.0125, 0.058], [0.0125, 0.078], [0.0100, 0.082], [0, 0.082],
  ], 16), { y: BORE_Y, z: -0.005 });
  put(B.bolt, cbox(0.010, 0.009, 0.022, 0.0010), { y: BORE_Y + 0.0165, z: -0.019 });  // gas key
  put(B.bolt, tubeZ(0.0032, 0.0032, 0.005, 8), { y: BORE_Y + 0.0140, z: -0.051, rx: -Math.PI / 2 });
  put(B.bolt, flat(latheZ([[0.0035, 0], [0.0095, 0], [0.0095, 0.006], [0.0035, 0.006]], 12)),
    { y: BORE_Y, z: -0.085 });
  put(B.bolt, cbox(0.004, 0.010, 0.016, 0.0006), { x: 0.0105, y: BORE_Y + 0.004, z: -0.075 });  // extractor
}

function buildMag(B, s) {
  const top = -0.020, zc = -0.045, n = 3, segH = s.magH / n;
  put(B.mag, cbox(0.028, 0.012, 0.058, 0.0016), { y: top - 0.002, z: zc });   // feed lips
  // Sections overlap generously: three boxes with gaps between them read as a
  // stack of boxes, three boxes that interpenetrate read as one curved body.
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) / n;
    const rx = -s.magCurve * t;
    const y = top - segH * (i + 0.5);
    const z = zc + s.magCurve * 0.155 * t * t;
    put(B.mag, cbox(0.026, segH + 0.010, 0.062, 0.0022), { y, z, rx });
    put(B.mag, cbox(0.028, segH * 0.40, 0.005, 0.0010), { y, z: z - 0.032, rx });   // spine rib
  }
  // witness slots down the right face
  putAll(B.mag, turnZ(turnY(slotPanel(
    s.magH * 0.62, 0.020, 0.0035, 4, s.magH * 0.155, 0.010, 0.0055, 0.0007,
  ), Math.PI / 2), -Math.PI / 2), {
    x: 0.0138, y: top - s.magH * 0.46, z: zc + 0.008, rx: -s.magCurve * 0.55,
  });
  // floorplate and its catch
  const fy = top - s.magH - 0.002, fz = zc + s.magCurve * 0.165;
  put(B.mag, frustumBox(0.034, 0.076, 0.030, 0.070, 0.011), { y: fy, z: fz, rx: -s.magCurve });
  put(B.mag, cbox(0.012, 0.006, 0.016, 0.0008), { y: fy - 0.008, z: fz - 0.026, rx: -s.magCurve });
}

function buildTrigger(B) {
  // Pivot at the top pin at y = -0.048; the shoe hangs down and forward.
  put(B.trig, cbox(0.0065, 0.016, 0.008, 0.0008), { y: -0.055, z: -0.009 });
  put(B.trig, cbox(0.0065, 0.014, 0.008, 0.0008), { y: -0.065, z: -0.014, rx: 0.5 });
  put(B.trig, cbox(0.0075, 0.009, 0.010, 0.0010), { y: -0.0725, z: -0.020, rx: 0.9 });
  for (let i = 0; i < 3; i++) {
    put(B.trig, cbox(0.0080, 0.0022, 0.0030, 0.0004), { y: -0.062 - i * 0.005, z: -0.0155 - i * 0.0035, rx: 0.6 });
  }
}

function buildCharging(B) {
  const y = 0.0130, z = 0.070;
  put(B.chg, cbox(0.044, 0.007, 0.011, 0.0013), { y, z });                     // T bar
  put(B.chg, cbox(0.014, 0.006, 0.034, 0.0010), { y, z: z - 0.022 });          // shaft
  put(B.chg, cbox(0.017, 0.0055, 0.014, 0.0009), { x: -0.021, y, z: z - 0.004, rz: 0.15 });   // latch
  put(B.chg, cbox(0.010, 0.0050, 0.010, 0.0008), { x: 0.021, y, z: z - 0.004 });
  for (let i = 0; i < 3; i++) {
    put(B.chg, cbox(0.040, 0.0016, 0.0022, 0.0004), { y: y + 0.0040, z: z - 0.0025 - i * 0.0035 });
  }
}

// ---------------------------------------------------------------- assembly

function meshFrom(list, mat, uvScale, name) {
  if (!list.length) return null;
  const g = boxUv(weld(list), uvScale);
  g.computeBoundingSphere();
  const m = new THREE.Mesh(g, mat);
  m.name = name;
  m.castShadow = false;
  m.receiveShadow = false;
  m.frustumCulled = false;     // always on screen; the cull test is pure cost
  m.matrixAutoUpdate = false;  // static under its pivot, never dirtied
  return m;
}

/** Re-root a bucket about `pivot` so a caller can rotate or slide it sanely. */
function pivotFrom(list, px, py, pz, mat, uvScale, name) {
  const o = new THREE.Object3D();
  o.name = name;
  o.position.set(px, py, pz);
  const m = meshFrom(list, mat, uvScale, name + ':geo');
  if (m) {
    m.geometry.translate(-px, -py, -pz);
    m.geometry.computeBoundingSphere();
    o.add(m);
  }
  return o;
}

export function buildWeapon(id, materials) {
  const key = SPECS[id] ? id : 'ar_vector';
  const s = SPECS[key];
  const rng = makeRng('weapon:' + key);   // only used for fastener clocking

  const B = {
    steel: [], poly: [], rubber: [], glass: [], emit: [],
    bolt: [], mag: [], trig: [], chg: [],
  };

  buildUpper(B, s, rng);
  buildRail(B, s);
  buildBarrel(B, s, rng);
  buildHandguard(B, s, rng);
  buildLower(B, s, rng);
  buildGrip(B);
  buildStock(B, s, rng);
  buildOptic(B, s, rng);
  buildBolt(B);
  buildMag(B, s);
  buildTrigger(B);
  buildCharging(B);

  // Four substances, four cache keys. Coated steel is dark and sharply
  // specular, polymer furniture is warmer and much rougher, rubber is nearly
  // matte, the lens is a tinted transmissive. They must not be confusable.
  const mSteel = materials.get('gunmetal', {
    color: 0x2e3238, roughness: 0.43, metalness: 0.95, envMapIntensity: 1.2,
  });
  // Furniture is deliberately a different family from the receiver: warmer, much
  // rougher, barely metallic. If polymer and steel are only two shades of black
  // apart the whole model reads as one injection moulding.
  const mPoly = materials.get('polymer', { color: 0x2c2e26, roughness: 0.66, metalness: 0.03 });
  const mRub = materials.get('rubber', { color: 0x121213, roughness: 0.95, metalness: 0 });
  const mGlass = materials.get('glass', {
    color: 0x8fb6cc, roughness: 0.05, opacity: 0.24, envMapIntensity: 2.2,
  });
  const mEmit = materials.get('rubber', {
    color: 0x300604, roughness: 0.9, emissive: 0xff1c06, emissiveIntensity: 1.8,
  });

  const group = new THREE.Group();
  group.name = 'weapon:' + key;

  // Body root: all static geometry hangs off this, so nudging parts.receiver
  // moves the whole rifle the way a shoulder would.
  const receiver = new THREE.Object3D();
  receiver.name = 'receiver';
  group.add(receiver);
  for (const m of [
    meshFrom(B.steel, mSteel, 15, 'body:steel'),
    meshFrom(B.poly, mPoly, 9, 'body:polymer'),
    meshFrom(B.rubber, mRub, 22, 'body:rubber'),
    meshFrom(B.emit, mEmit, 20, 'sight:emitter'),
    meshFrom(B.glass, mGlass, 4, 'sight:lens'),
  ]) if (m) receiver.add(m);

  // Independently animated parts, each on the pivot that matches its motion:
  // bolt and charging handle slide on +Z, the magazine drops and rotates about
  // its front lip, the trigger rotates about its top pin.
  const bolt = pivotFrom(B.bolt, 0, BORE_Y, -0.085, mSteel, 15, 'bolt');
  const mag = pivotFrom(B.mag, 0, -0.020, -0.076, mPoly, 9, 'mag');
  const trigger = pivotFrom(B.trig, 0, -0.048, -0.008, mSteel, 15, 'trigger');
  const charging = pivotFrom(B.chg, 0, 0.0130, 0.070, mSteel, 15, 'charging');
  group.add(bolt, mag, trigger, charging);

  // Static handles. Correct world transforms for anything wanting to hang an
  // effect off them; their geometry is merged into the body meshes.
  const mk = (name, x, y, z, parent) => {
    const o = new THREE.Object3D();
    o.name = name;
    o.position.set(x, y, z);
    parent.add(o);
    return o;
  };
  const barrelEnd = -0.088 - s.barrel;
  const barrel = mk('barrel', 0, BORE_Y, (-0.088 + barrelEnd) / 2, receiver);
  const muzzle = mk('muzzle', 0, BORE_Y, barrelEnd - s.brakeLen / 2, receiver);
  const foregrip = mk('foregrip', 0, BORE_Y, HG_Z - s.hgLen / 2, receiver);
  const stock = mk('stock', 0, -0.006, 0.062 + s.stockLen * 0.6, receiver);
  const optic = mk('optic', 0, SIGHT_Y, -0.018, receiver);

  const muzzleTip = mk('muzzleTip', 0, BORE_Y, barrelEnd - s.brakeLen - 0.002, group);
  const ejectPort = mk('ejectPort', 0.025, PORT_Y + 0.002, PORT_Z, group);

  // Two degrees of cant plus a little presence scale, both taken about the sight
  // axis so the reticle does not move a pixel. Roll is the only rotation that
  // leaves the optical axis pointing at the eye, so it is the only one baked in;
  // any yaw or pitch pose has to come from the rig in weaponfx.js, blended out
  // by adsT, or the sight picture goes off-centre.
  const cant = 0.036, k = 1.08;
  group.rotation.z = cant;
  group.scale.setScalar(k);
  group.position.x = k * SIGHT_Y * Math.sin(cant);
  group.position.y = SIGHT_Y * (1 - k * Math.cos(cant));
  // Attachment points are unscaled: a flash quad parented to muzzleTip should be
  // the size its owner asked for, not 8% bigger.
  muzzleTip.scale.setScalar(1 / k);
  ejectPort.scale.setScalar(1 / k);

  group.traverse((o) => { o.castShadow = false; o.receiveShadow = false; });

  let triangles = 0;
  group.traverse((o) => {
    if (o.isMesh && o.geometry) triangles += o.geometry.getAttribute('position').count / 3;
  });

  return {
    group,
    parts: { receiver, barrel, bolt, mag, stock, foregrip, optic, muzzle, trigger, charging },
    muzzleTip, ejectPort,
    triangles,
    config: { name: s.name, class: s.cls, ...s.config },
  };
}
