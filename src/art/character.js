// OWNER: agent "char" — enemy/ally characters: skinned mesh + skeleton + clips,
// all generated in code. Fictional faction, no real insignia or trademarks.
// CONTRACT:
//   buildCharacter(materials, variant, rng) -> {
//     group:  THREE.Group
//     skeleton: THREE.Skeleton | null
//     hitboxes: Array<{name:'head'|'chest'|'stomach'|'limb', mult:number, obj:THREE.Object3D, radius:number, halfHeight:number}>
//     setPose(state) -> void   state: {move:0..1, aim:0..1, crouch:0..1, dead:number, hitDir?:Vec3}
//     update(dt)
//     dispose()
//   }
//   CHARACTER_VARIANTS -> string[]
//
// ---------------------------------------------------------------------------
// HOW THIS IS BUILT, and why.
//
// SILHOUETTE FIRST. The previous version was a capsule and a sphere and scored
// 2/10 because at 25m an enemy has to read as an armed man from outline alone,
// before a single texel of the uniform is resolvable. Everything here is chosen
// for what it does to the outline at ~70px tall:
//
//   * A 1.83m figure. Sole 0, hip 0.905, chest 1.275, shoulder line 1.455,
//     neck base 1.505, helmet crown 1.83. Measured against a 1.8m reference so
//     the character is unambiguously taller than the waist-high crates.
//   * A NECK BREAK. Shoulder ring is 0.41m across, the neck is 0.11m, the
//     helmet 0.25m. Three distinct widths stacked in 0.35m of height is what
//     makes a head read as a head instead of a bump on a torso.
//   * SHOULDER MASS. A tapered loft (0.41 shoulders / 0.28 waist) plus separate
//     deltoid caps and a plate carrier shell that is squarer in section than
//     the body under it, so the armour reads as rigid against soft cloth.
//   * BREAK-UP ON THE OUTLINE. Pouches, a rear pack, an antenna, knee and
//     elbow pads, boots and a cargo pocket all push past the body outline.
//     A smooth outline reads as a mannequin; a lumpy one reads as loaded kit.
//   * THE RIFLE IS PART OF THE SILHOUETTE. Held across the chest at rest and
//     shouldered under `aim`, it puts a 0.7m horizontal bar across a vertical
//     figure. That single line is the strongest "this is a soldier" cue there
//     is at distance.
//
// ONE DRAW CALL. Every part above is merged into ONE indexed BufferGeometry
// with one material. Per-part value separation therefore has to live in vertex
// colours, not in extra materials — see PALETTE. ~1.9k triangles per character,
// 8 alive = 15k against a 950k budget, 1 draw + 2 shadow cascade re-draws each.
//
// GEOMETRY IS SHARED, SKELETONS ARE NOT. Three variants means three merged
// geometries for the whole game; every character instances one of them and gets
// its own Skeleton (19 bones). Per-character visual variety comes from a small
// set of tinted material clones plus a +-3% height scale, which costs nothing.
//
// SKINNING. Bones are authored with IDENTITY world rotation and their bind
// world position at the joint, so a bone's local rotation is a plain,
// readable rotation of that joint about the model axes and the mesh can be
// authored directly in character space. Weights are solved at build time by
// distance to the bone SEGMENT, with the radial component scaled down 4x — a
// plain 3D distance binds a hip vertex half to the spine, because every bone
// lies on the centre line and the surface is 0.16m away from all of them.
//
// POSE IS IK, NOT KEYFRAMES. Hands are solved onto two fixed points on the
// rifle every frame, so the hold never breaks no matter what the torso is
// doing; feet are solved onto gait targets, so knees always bend correctly and
// crouch plants the feet instead of sinking them. Two-bone analytic IK, four
// chains, no allocation. Hand-authoring the equivalent quaternions blind would
// have been guesswork.
//
// FACING. Object3D.lookAt() points +Z at the target for a non-camera, and
// ai.js drives characters with lookAt, so this model is authored facing +Z.

import * as THREE from 'three';
import { makeRng } from '../core/rng.js';

export const CHARACTER_VARIANTS = ['rifleman', 'shotgunner', 'sniper'];

// ---------------------------------------------------------------- skeleton

const BONES = [
  'root', 'hips', 'spine', 'chest', 'neck', 'head', 'weapon',
  'uArmL', 'fArmL', 'handL', 'uArmR', 'fArmR', 'handR',
  'thighL', 'shinL', 'footL', 'thighR', 'shinR', 'footR',
];
const PARENT = [-1, 0, 1, 2, 3, 4, 3, 3, 7, 8, 3, 10, 11, 1, 13, 14, 1, 16, 17];
const B = {};
for (let i = 0; i < BONES.length; i++) B[BONES[i]] = i;
const NB = BONES.length;

// Fixed joints. Arm elbows and knees are solved rather than typed, so the bind
// pose is exactly what the runtime IK reproduces at rest — otherwise frame one
// snaps the limbs into a different shape than the mesh was skinned against.
const JOINTS = {
  root: [0, 0, 0],
  hips: [0, 0.925, 0],
  spine: [0, 1.085, 0.005],
  chest: [0, 1.275, 0.010],
  neck: [0, 1.505, -0.010],
  head: [0, 1.590, -0.005],
  shL: [0.185, 1.455, 0.050],
  shR: [-0.185, 1.455, -0.035],
  hipL: [0.095, 0.905, 0.015],
  hipR: [-0.095, 0.905, -0.015],
  ankL: [0.106, 0.098, 0.140],
  ankR: [-0.126, 0.098, -0.112],
  toeL: [0.108, 0.030, 0.292],
  toeR: [-0.130, 0.030, 0.038],
};

const ARM_UP = 0.285, ARM_LO = 0.265;   // shoulder->elbow, elbow->wrist
const LEG_UP = 0.428, LEG_LO = 0.415;   // hip->knee, knee->ankle

// Elbow / knee pole hints, in character space. The right elbow rides down and
// back at rest and swings out as the rifle comes up; the knees always lead.
const POLE_ARM_L = new THREE.Vector3(0.50, -1.0, -0.15).normalize();
const POLE_ARM_R = new THREE.Vector3(-0.62, -1.0, -0.30).normalize();
const POLE_ARM_R_AIM = new THREE.Vector3(-0.78, -0.88, -0.30).normalize();
const POLE_ARM_L_AIM = new THREE.Vector3(0.62, -1.0, 0.05).normalize();
const POLE_KNEE = new THREE.Vector3(0, 0.05, 1).normalize();

// Rifle rest and shouldered placement, as a bore line. Everything about the
// weapon — its mesh, both hand targets, the aim transform — derives from these
// two points, so moving the rifle moves the hands with it for free.
const BORE_REST_ORIGIN = new THREE.Vector3(-0.205, 1.255, -0.045);
const BORE_REST_DIR = new THREE.Vector3(0.320, -0.100, 0.605).normalize();
const BORE_AIM_ORIGIN = new THREE.Vector3(-0.155, 1.468, 0.010);
const BORE_AIM_DIR = new THREE.Vector3(0.160, 0.026, 0.685).normalize();

const GUN = {
  rifleman: { len: 0.700, gripT: 0.30, gripDrop: 0.100, suppT: 0.56, suppDrop: 0.056, kind: 'carbine' },
  shotgunner: { len: 0.630, gripT: 0.32, gripDrop: 0.096, suppT: 0.62, suppDrop: 0.050, kind: 'scatter' },
  sniper: { len: 0.930, gripT: 0.235, gripDrop: 0.104, suppT: 0.44, suppDrop: 0.056, kind: 'marksman' },
};

const KIT = {
  rifleman: { head: 'combat', bulk: 1.00, pads: 0, antenna: true },
  shotgunner: { head: 'visor', bulk: 1.14, pads: 1, antenna: false },
  sniper: { head: 'boonie', bulk: 0.90, pads: 0, antenna: true, ghillie: true },
};

// ---------------------------------------------------------------- palette
//
// One material, so these are LINEAR multipliers written into the vertex colour
// attribute. The base material target is a mid-dark desaturated green (see
// KIT_TINTS); the level is pale warm sand at roughly 2.5x this luminance and
// the opposite side of neutral in hue, which is the separation the brief asks
// for. Helmet sits at 0.42 of the body so it stays the darkest large mass on
// the figure — a head that is the same value as the shoulders disappears.
const P = {
  cloth: [1.00, 1.00, 1.00],
  clothDark: [0.80, 0.83, 0.80],
  carrier: [0.60, 0.65, 0.59],
  pouch: [0.74, 0.77, 0.69],
  webbing: [0.44, 0.47, 0.43],
  helmet: [0.40, 0.44, 0.43],
  helmetTrim: [0.20, 0.21, 0.22],
  visor: [0.16, 0.19, 0.20],
  skin: [2.05, 1.48, 1.02],
  gaiter: [0.32, 0.35, 0.34],
  glove: [0.26, 0.27, 0.27],
  boot: [0.22, 0.23, 0.23],
  pad: [0.42, 0.45, 0.44],
  steel: [0.19, 0.20, 0.21],
  polymer: [0.27, 0.27, 0.26],
  optic: [0.13, 0.14, 0.15],
  lens: [0.34, 0.46, 0.42],
  ghillie: [0.62, 0.66, 0.50],
};

// Three squad tints. All mid-dark, all cool or green-cast against warm sand.
const KIT_TINTS = [0x555c4c, 0x4c5450, 0x5c5b47];

// ---------------------------------------------------------------- scratch
// Build-time allocation is free; these exist for the per-frame path.
const _v0 = new THREE.Vector3(), _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3(), _v4 = new THREE.Vector3(), _v5 = new THREE.Vector3();
const _q0 = new THREE.Quaternion(), _q1 = new THREE.Quaternion(), _q2 = new THREE.Quaternion();
const _q3 = new THREE.Quaternion();
const _e0 = new THREE.Euler();
const _up = new THREE.Vector3(0, 1, 0);
const _IDENT_Q = new THREE.Quaternion();
const _IDENT_M = new THREE.Matrix4();

// ---------------------------------------------------------------- geometry kit

const _m4 = new THREE.Matrix4();
const _tq = new THREE.Quaternion();
const _tv = new THREE.Vector3();
const _ts = new THREE.Vector3();

/** Position / rotate / scale a geometry in place (build time only). */
function xf(g, o) {
  if (!o) return g;
  _e0.set(o.rx || 0, o.ry || 0, o.rz || 0, 'YXZ');
  _tq.setFromEuler(_e0);
  _tv.set(o.x || 0, o.y || 0, o.z || 0);
  _ts.set(o.sx || 1, o.sy || 1, o.sz || 1);
  g.applyMatrix4(_m4.compose(_tv, _tq, _ts));
  return g;
}

/**
 * Lofted tube through a stack of superellipse rings. This is the workhorse for
 * the torso, the plate carrier and the belt: a ring can change width, depth,
 * centre, corner sharpness AND yaw independently, which is how the torso gets
 * a bladed stance and a hard-edged carrier over a soft body without a single
 * extra draw call.
 *
 * rings: [{ y, cx, cz, hw, hd, e, yaw }]  e = 2 ellipse, higher = squarer.
 */
function loft(rings, seg, capBottom, capTop) {
  const pos = [], idx = [];
  const n = rings.length;
  for (let r = 0; r < n; r++) {
    const R = rings[r];
    const e = R.e || 2, ca = Math.cos(R.yaw || 0), sa = Math.sin(R.yaw || 0);
    for (let i = 0; i < seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      const c = Math.cos(a), s = Math.sin(a);
      const px = Math.sign(c) * Math.pow(Math.abs(c), 2 / e) * R.hw;
      const pz = Math.sign(s) * Math.pow(Math.abs(s), 2 / e) * R.hd;
      pos.push((R.cx || 0) + px * ca + pz * sa, R.y, (R.cz || 0) - px * sa + pz * ca);
    }
  }
  for (let r = 0; r < n - 1; r++) {
    const a = r * seg, b = (r + 1) * seg;
    for (let i = 0; i < seg; i++) {
      const j = (i + 1) % seg;
      idx.push(a + i, b + i, b + j, a + i, b + j, a + j);
    }
  }
  // Caps get their own vertices so the silhouette edge stays hard.
  if (capBottom) capRing(pos, idx, rings[0], seg, false);
  if (capTop) capRing(pos, idx, rings[n - 1], seg, true);

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

function capRing(pos, idx, R, seg, top) {
  const base = pos.length / 3;
  const e = R.e || 2, ca = Math.cos(R.yaw || 0), sa = Math.sin(R.yaw || 0);
  for (let i = 0; i < seg; i++) {
    const a = (i / seg) * Math.PI * 2;
    const c = Math.cos(a), s = Math.sin(a);
    const px = Math.sign(c) * Math.pow(Math.abs(c), 2 / e) * R.hw;
    const pz = Math.sign(s) * Math.pow(Math.abs(s), 2 / e) * R.hd;
    pos.push((R.cx || 0) + px * ca + pz * sa, R.y, (R.cz || 0) - px * sa + pz * ca);
  }
  const cen = pos.length / 3;
  pos.push(R.cx || 0, R.y, R.cz || 0);
  for (let i = 0; i < seg; i++) {
    const j = (i + 1) % seg;
    if (top) idx.push(base + i, cen, base + j);
    else idx.push(base + j, cen, base + i);
  }
}

/** Tapered tube between two points. Limbs, barrels, the neck, the antenna. */
function tube(p0, p1, r0, r1, seg, hseg = 1, e = 2, capA = false, capB = false) {
  _tv.copy(p1).sub(p0);
  const len = _tv.length() || 1e-5;
  const rings = [];
  for (let i = 0; i <= hseg; i++) {
    const t = i / hseg;
    const r = r0 + (r1 - r0) * t;
    rings.push({ y: t * len, hw: r, hd: r, e });
  }
  const g = loft(rings, seg, capA, capB);
  _tq.setFromUnitVectors(_up, _tv.divideScalar(len));
  g.applyMatrix4(_m4.compose(p0, _tq, _ts.set(1, 1, 1)));
  return g;
}

/** Box with the corner rounding a real piece of kit has. 6-face, 12 tri. */
function box(w, h, d, o) { return xf(new THREE.BoxGeometry(w, h, d), o); }

/** Sphere segment — helmets, deltoids, knee caps. */
function dome(r, wseg, hseg, thetaLen, o) {
  return xf(new THREE.SphereGeometry(r, wseg, hseg, 0, Math.PI * 2, 0, thetaLen), o);
}

// ---------------------------------------------------------------- accumulator

function newBatch() {
  return { pos: [], nrm: [], uv: [], col: [], idx: [], parts: [], count: 0 };
}

// Low-frequency mottle so a flat-coloured uniform still breaks up at 25m where
// the fabric texture itself is below one texel per pixel. Deterministic, no rng
// state, so the same variant always bakes identically.
function mottle(x, y, z) {
  const a = Math.sin(x * 6.7 + y * 2.9 + z * 4.3);
  const b = Math.sin(y * 8.1 - z * 3.7 + 1.7);
  const c = Math.sin(x * 3.1 + z * 7.9 - 0.9);
  return a * b * 0.5 + c * 0.35;
}

const UV_PER_M = 3.4;

/**
 * Append a geometry to the batch. `col` is a linear RGB multiplier; `camo`
 * modulates it with the mottle field. `bones` restricts which bones may claim
 * these vertices — an unrestricted solve binds forearm vertices to the ribcage.
 */
function push(batch, geo, col, bones, camo) {
  const p = geo.attributes.position, nA = geo.attributes.normal;
  const base = batch.count, vc = p.count;
  for (let i = 0; i < vc; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    const nx = nA.getX(i), ny = nA.getY(i), nz = nA.getZ(i);
    batch.pos.push(x, y, z);
    batch.nrm.push(nx, ny, nz);
    // Triplanar-ish box projection: constant texel density everywhere, so the
    // weave is the same physical size on a sleeve as on a boot.
    const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
    if (ax >= ay && ax >= az) batch.uv.push(z * UV_PER_M, y * UV_PER_M);
    else if (ay >= az) batch.uv.push(x * UV_PER_M, z * UV_PER_M);
    else batch.uv.push(x * UV_PER_M, y * UV_PER_M);
    let k = 1;
    if (camo) k = 0.90 + 0.20 * Math.max(0, Math.min(1, mottle(x, y, z) * 0.5 + 0.5));
    batch.col.push(col[0] * k, col[1] * k, col[2] * k);
  }
  const ind = geo.index;
  if (ind) for (let i = 0; i < ind.count; i++) batch.idx.push(base + ind.getX(i));
  else for (let i = 0; i < vc; i++) batch.idx.push(base + i);
  batch.parts.push({ start: base, count: vc, bones });
  batch.count += vc;
  geo.dispose();
}

// ---------------------------------------------------------------- two-bone IK

/**
 * Classic analytic solve. Writes the joint position into `out` and returns the
 * effective end position (which is short of the target when over-extended, so
 * the limb straightens instead of snapping).
 */
function solve2(S, T, a, b, pole, outJoint, outEnd) {
  _v3.subVectors(T, S);
  let d = _v3.length();
  if (d < 1e-5) { _v3.set(0, -1, 0); d = 1e-5; }
  _v3.divideScalar(d);
  const dc = Math.min(a + b - 1e-4, Math.max(Math.abs(a - b) + 1e-4, d));
  const x = (a * a - b * b + dc * dc) / (2 * dc);
  const h = Math.sqrt(Math.max(0, a * a - x * x));
  _v4.copy(pole).addScaledVector(_v3, -pole.dot(_v3));
  if (_v4.lengthSq() < 1e-9) {
    _v4.set(0, 0, 1).addScaledVector(_v3, -_v3.z);
    if (_v4.lengthSq() < 1e-9) _v4.set(1, 0, 0);
  }
  _v4.normalize();
  outJoint.copy(S).addScaledVector(_v3, x).addScaledVector(_v4, h);
  if (outEnd) {
    _v5.subVectors(T, outJoint).normalize();
    outEnd.copy(outJoint).addScaledVector(_v5, b);
  }
  return outJoint;
}

// ---------------------------------------------------------------- rig layout

/** Bind-pose joint table for one variant, with elbows and knees solved. */
function buildRig(variant) {
  const g = GUN[variant] || GUN.rifleman;
  const dir = BORE_REST_DIR;
  const down = new THREE.Vector3(0, -1, 0).addScaledVector(dir, dir.y).normalize();
  const O = BORE_REST_ORIGIN;

  const grip = O.clone().addScaledVector(dir, g.len * g.gripT).addScaledVector(down, g.gripDrop);
  const supp = O.clone().addScaledVector(dir, g.len * g.suppT).addScaledVector(down, g.suppDrop);

  const bind = new Array(NB);
  for (const k of ['root', 'hips', 'spine', 'chest', 'neck', 'head']) {
    bind[B[k]] = new THREE.Vector3().fromArray(JOINTS[k]);
  }
  bind[B.weapon] = O.clone();
  bind[B.uArmL] = new THREE.Vector3().fromArray(JOINTS.shL);
  bind[B.uArmR] = new THREE.Vector3().fromArray(JOINTS.shR);
  bind[B.handL] = supp.clone();
  bind[B.handR] = grip.clone();
  bind[B.fArmL] = solve2(bind[B.uArmL], supp, ARM_UP, ARM_LO, POLE_ARM_L, new THREE.Vector3());
  bind[B.fArmR] = solve2(bind[B.uArmR], grip, ARM_UP, ARM_LO, POLE_ARM_R, new THREE.Vector3());
  bind[B.thighL] = new THREE.Vector3().fromArray(JOINTS.hipL);
  bind[B.thighR] = new THREE.Vector3().fromArray(JOINTS.hipR);
  bind[B.footL] = new THREE.Vector3().fromArray(JOINTS.ankL);
  bind[B.footR] = new THREE.Vector3().fromArray(JOINTS.ankR);
  bind[B.shinL] = solve2(bind[B.thighL], bind[B.footL], LEG_UP, LEG_LO, POLE_KNEE, new THREE.Vector3());
  bind[B.shinR] = solve2(bind[B.thighR], bind[B.footR], LEG_UP, LEG_LO, POLE_KNEE, new THREE.Vector3());

  // Bone tails, for the skin-weight solve and for limb geometry.
  const tail = new Array(NB);
  for (let i = 0; i < NB; i++) {
    // default: first child
    let child = -1;
    for (let j = 0; j < NB; j++) if (PARENT[j] === i) { child = j; break; }
    tail[i] = child >= 0 ? bind[child].clone() : bind[i].clone();
  }
  tail[B.root] = bind[B.hips].clone();
  tail[B.chest] = bind[B.neck].clone();
  tail[B.hips] = bind[B.spine].clone();
  tail[B.head] = bind[B.head].clone().add(new THREE.Vector3(0, 0.18, 0));
  tail[B.weapon] = O.clone().addScaledVector(dir, g.len);
  tail[B.handL] = supp.clone().addScaledVector(dir, 0.075);
  tail[B.handR] = grip.clone().addScaledVector(dir, 0.075);
  tail[B.footL] = new THREE.Vector3().fromArray(JOINTS.toeL);
  tail[B.footR] = new THREE.Vector3().fromArray(JOINTS.toeR);

  return { gun: g, bore: { O, dir, down, F: tail[B.weapon].clone(), grip, supp }, bind, tail };
}

// ---------------------------------------------------------------- the body

function buildBody(variant, rig, batch) {
  const kit = KIT[variant] || KIT.rifleman;
  const bind = rig.bind;
  const bulk = kit.bulk;

  const TORSO = ['hips', 'spine', 'chest'];
  const NECKB = ['chest', 'neck', 'head'];

  // -- torso. Shoulders 0.41 across, waist 0.28: the V is the whole read.
  const yaw = (v) => 0.02 + v * 0.19;
  const torsoRings = [
    { y: 0.855, hw: 0.134, hd: 0.098, e: 2.9, yaw: yaw(0.0), cz: 0.004 },
    { y: 0.930, hw: 0.160, hd: 0.113, e: 2.9, yaw: yaw(0.1), cz: 0.004 },
    { y: 1.012, hw: 0.150, hd: 0.108, e: 3.0, yaw: yaw(0.25), cz: 0.006 },
    { y: 1.085, hw: 0.139, hd: 0.102, e: 3.0, yaw: yaw(0.4), cz: 0.008 },
    { y: 1.172, hw: 0.152, hd: 0.113, e: 3.1, yaw: yaw(0.55), cz: 0.010 },
    { y: 1.275, hw: 0.176, hd: 0.126, e: 3.2, yaw: yaw(0.72), cz: 0.012 },
    { y: 1.382, hw: 0.197, hd: 0.124, e: 3.2, yaw: yaw(0.88), cz: 0.010 },
    { y: 1.455, hw: 0.206, hd: 0.113, e: 3.0, yaw: yaw(1.0), cz: 0.004 },
    { y: 1.512, hw: 0.150, hd: 0.092, e: 2.6, yaw: yaw(1.0), cz: 0.000 },
  ];
  push(batch, loft(torsoRings, 14, true, true), P.cloth, TORSO, true);

  // -- plate carrier. Squarer section (e=5) at a darker value, standing 3cm
  // proud of the body: at distance the step in the outline is what says armour.
  const cr = (y, hw, hd, w) => ({ y, hw: hw * bulk, hd: hd * bulk, e: 5.0, yaw: yaw(w), cz: 0.012 });
  const carrier = [
    cr(1.040, 0.152, 0.122, 0.38),
    cr(1.120, 0.163, 0.132, 0.52),
    cr(1.230, 0.184, 0.148, 0.68),
    cr(1.330, 0.203, 0.150, 0.82),
    cr(1.412, 0.211, 0.140, 0.95),
    cr(1.444, 0.186, 0.122, 1.0),
  ];
  push(batch, loft(carrier, 14, true, true), P.carrier, TORSO, true);

  // Shoulder straps over the traps — they close the carrier's outline and give
  // the top of the chest a horizontal line the eye can lock onto.
  for (const s of [-1, 1]) {
    push(batch, box(0.058, 0.036, 0.180, {
      x: s * 0.088, y: 1.452, z: 0.008, ry: yaw(1.0), rz: s * 0.10,
    }), P.webbing, TORSO);
  }

  // -- pouches, in the carrier's yawed frame.
  const ca = Math.cos(yaw(0.62)), sa = Math.sin(yaw(0.62));
  const onChest = (lx, y, lz, w, h, d, col) => {
    push(batch, box(w, h, d, { x: lx * ca + lz * sa, y, z: -lx * sa + lz * ca, ry: yaw(0.62) }),
      col, TORSO);
  };
  onChest(-0.085, 1.175, 0.170 * bulk, 0.078, 0.115, 0.058, P.pouch);
  onChest(0.000, 1.170, 0.174 * bulk, 0.078, 0.122, 0.060, P.pouch);
  onChest(0.085, 1.175, 0.170 * bulk, 0.078, 0.115, 0.058, P.pouch);
  onChest(-0.128, 1.300, 0.160 * bulk, 0.056, 0.062, 0.050, P.webbing);
  onChest(0.130, 1.302, 0.158 * bulk, 0.050, 0.058, 0.046, P.webbing);
  onChest(-0.196 * bulk, 1.150, 0.010, 0.052, 0.104, 0.092, P.pouch);
  onChest(0.198 * bulk, 1.158, 0.000, 0.050, 0.100, 0.088, P.pouch);
  // rear pack + dump pouch
  onChest(0.000, 1.290, -0.196 * bulk, 0.240, 0.210, 0.088, P.pouch);
  onChest(0.105, 1.075, -0.150 * bulk, 0.110, 0.100, 0.070, P.webbing);
  // belt
  push(batch, loft([
    { y: 1.000, hw: 0.152, hd: 0.110, e: 3.6, yaw: yaw(0.22), cz: 0.006 },
    { y: 1.044, hw: 0.156, hd: 0.114, e: 3.6, yaw: yaw(0.30), cz: 0.006 },
  ], 14, false, false), P.webbing, TORSO);

  if (kit.antenna) {
    // A 0.4m antenna off the pack. One of the cheapest silhouette reads there
    // is — a thin vertical line above a shoulder says "radio operator" instantly.
    push(batch, tube(new THREE.Vector3(0.10, 1.39, -0.20), new THREE.Vector3(0.155, 1.79, -0.30),
      0.008, 0.003, 5, 2), P.steel, TORSO);
  }
  if (kit.pads) {
    for (const s of [-1, 1]) {
      push(batch, dome(0.105, 9, 5, 1.35, { x: s * 0.198, y: 1.435, z: 0.005, rz: s * 0.55 }),
        P.carrier, ['chest', s > 0 ? 'uArmL' : 'uArmR']);
    }
  }
  if (kit.ghillie) {
    // Loose strips break the shoulder line up — a scoped figure reads as a
    // different threat than a rifleman only if its outline differs.
    for (let i = 0; i < 6; i++) {
      const a = -0.30 + i * 0.12;
      push(batch, box(0.052, 0.150, 0.020, {
        x: Math.sin(a * 3.1) * 0.19, y: 1.36 - (i % 3) * 0.035,
        z: Math.cos(a * 3.1) * 0.15 - 0.02, ry: a * 2.4, rz: a * 0.6,
      }), P.ghillie, TORSO, true);
    }
  }

  // -- neck and head. Traps ring is 0.30 across, the neck 0.11, the helmet
  // 0.25 — three widths inside 0.32m of height. That stack IS the neck break;
  // without it a head reads as a lump on the shoulders at any real distance.
  push(batch, tube(new THREE.Vector3(0, 1.455, -0.004), new THREE.Vector3(0, 1.625, -0.008),
    0.059, 0.052, 9, 1), P.gaiter, NECKB);
  // Skull: chin 1.585, crown 1.795, eyeline 1.690. Ellipsoid, not a ball — a
  // spherical head under a helmet reads as a bolt.
  const skull = dome(0.077, 10, 7, Math.PI, { y: 1.690, z: -0.005, sy: 1.364, sz: 1.208 });
  {
    const p = skull.attributes.position;
    const cols = [];
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i), z = p.getZ(i);
      // Only the strip below the helmet rim is ever seen; the lower half of it
      // is a face covering, so the visible skin is a narrow band at the brow.
      const face = z > 0.025 && y > 1.628;
      cols.push(face ? P.skin : P.gaiter);
    }
    // push() takes one colour; emit the two masks as two calls instead.
    const idx = skull.index;
    const faceIdx = [], gIdx = [];
    for (let t = 0; t < idx.count; t += 3) {
      const a = idx.getX(t), b2 = idx.getX(t + 1), c2 = idx.getX(t + 2);
      const isFace = cols[a] === P.skin && cols[b2] === P.skin && cols[c2] === P.skin;
      (isFace ? faceIdx : gIdx).push(a, b2, c2);
    }
    const mk = (list) => {
      const g2 = new THREE.BufferGeometry();
      g2.setAttribute('position', skull.attributes.position.clone());
      g2.setAttribute('normal', skull.attributes.normal.clone());
      g2.setIndex(list);
      return g2;
    };
    if (faceIdx.length) push(batch, mk(faceIdx), P.skin, ['head', 'neck']);
    if (gIdx.length) push(batch, mk(gIdx), P.gaiter, ['head', 'neck']);
    skull.dispose();
  }

  buildHeadgear(kit.head, batch);

  // -- arms.
  for (const s of [1, -1]) {
    const L = s > 0;
    const sh = bind[L ? B.uArmL : B.uArmR];
    const el = bind[L ? B.fArmL : B.fArmR];
    const ha = bind[L ? B.handL : B.handR];
    const bn = L ? ['chest', 'uArmL'] : ['chest', 'uArmR'];
    const bu = L ? ['uArmL', 'fArmL'] : ['uArmR', 'fArmR'];
    const bf = L ? ['fArmL', 'handL'] : ['fArmR', 'handR'];
    const bh = L ? ['handL', 'fArmL'] : ['handR', 'fArmR'];

    // deltoid: bound mostly to the chest so the shoulder never tears open when
    // the arm swings hard into an aim pose
    push(batch, dome(0.083 * (0.94 + 0.10 * bulk), 9, 6, 2.1,
      { x: sh.x * 0.98, y: sh.y + 0.014, z: sh.z, rz: s * 0.42, rx: -0.12 }), P.clothDark, bn, true);

    push(batch, tube(sh, el, 0.070, 0.052, 9, 2), P.cloth, bu, true);
    push(batch, tube(el, ha, 0.053, 0.041, 9, 2), P.clothDark, bf, true);
    // elbow pad
    _v0.copy(el).lerp(sh, 0.10);
    push(batch, dome(0.060, 8, 4, 1.6, { x: _v0.x, y: _v0.y, z: _v0.z, rx: -1.1, rz: s * 0.3 }),
      P.pad, bu);
    // glove: the hand is a small hard dark mass on the rifle, which is what
    // makes the weapon read as held rather than floating
    _v1.copy(ha).addScaledVector(rig.bore.dir, 0.018);
    push(batch, box(0.062, 0.070, 0.108, {
      x: _v1.x, y: _v1.y, z: _v1.z, ry: 0.30 * s, rx: -0.35,
    }), P.glove, bh);
  }

  // -- legs.
  for (const s of [1, -1]) {
    const L = s > 0;
    const hip = bind[L ? B.thighL : B.thighR];
    const kn = bind[L ? B.shinL : B.shinR];
    const an = bind[L ? B.footL : B.footR];
    const toe = rig.tail[L ? B.footL : B.footR];
    const bt = L ? ['hips', 'thighL', 'shinL'] : ['hips', 'thighR', 'shinR'];
    const bs = L ? ['shinL', 'thighL', 'footL'] : ['shinR', 'thighR', 'footR'];
    // The boot is rigid to the ankle. Letting the sole blend into the shin —
    // which auto-weighting does, because the toe is barely past the end of the
    // shin segment — drove the toe 5cm through the ground in a deep crouch.
    const bfo = L ? ['footL'] : ['footR'];

    push(batch, tube(_v0.copy(hip).setY(hip.y + 0.055), kn, 0.098, 0.070, 9, 2), P.cloth, bt, true);
    push(batch, tube(kn, an, 0.070, 0.050, 9, 2), P.cloth, bs, true);
    // cargo pocket on the outside of the thigh
    _v1.copy(hip).lerp(kn, 0.42);
    push(batch, box(0.048, 0.150, 0.104, {
      x: _v1.x + s * 0.082, y: _v1.y, z: _v1.z + 0.010, ry: -s * 0.10,
    }), P.pouch, bt);
    // knee pad
    _v2.copy(kn).lerp(hip, 0.06);
    push(batch, dome(0.078, 9, 5, 1.5, { x: _v2.x, y: _v2.y, z: _v2.z + 0.012, rx: 1.35 }),
      P.pad, bs);
    // boot: cuff (overlapping the foot block so there is no gap at the ankle),
    // foot block, toe
    push(batch, tube(_v3.copy(an).setY(0.034), _v4.copy(an).setY(an.y + 0.118),
      0.066, 0.058, 9, 1, 2.4, false, true), P.boot, bfo);
    _v0.copy(an).lerp(toe, 0.45);
    push(batch, box(0.088, 0.078, 0.215, {
      x: _v0.x, y: 0.040, z: _v0.z, ry: -s * 0.06,
    }), P.boot, bfo);
    push(batch, box(0.078, 0.052, 0.060, { x: toe.x, y: 0.028, z: toe.z + 0.012 }), P.boot, bfo);
  }

  // drop-leg pouch on the strong side
  {
    const hip = bind[B.thighR], kn = bind[B.shinR];
    _v0.copy(hip).lerp(kn, 0.60);
    push(batch, box(0.062, 0.135, 0.088, { x: _v0.x - 0.070, y: _v0.y, z: _v0.z - 0.005, ry: 0.10 }),
      P.webbing, ['thighR', 'shinR']);
  }
}

// Headgear is a loft rather than a sphere segment because a helmet's defining
// feature is that it FLARES at the rim — it is wider at the ear than at the
// brow. A sphere cut below the equator does the opposite and reads as a bald
// head. Crown sits at 1.828, which is the top of the figure.
function buildHeadgear(kind, batch) {
  const HB = ['head', 'neck'];

  if (kind === 'boonie') {
    push(batch, loft([
      { y: 1.826, hw: 0.040, hd: 0.042, e: 2.2, cz: -0.005 },
      { y: 1.806, hw: 0.082, hd: 0.086, e: 2.3, cz: -0.005 },
      { y: 1.772, hw: 0.104, hd: 0.108, e: 2.4, cz: -0.005 },
      { y: 1.730, hw: 0.113, hd: 0.117, e: 2.4, cz: -0.005 },
      { y: 1.700, hw: 0.115, hd: 0.119, e: 2.4, cz: -0.005 },
      { y: 1.688, hw: 0.198, hd: 0.202, e: 2.2, cz: -0.005 },   // brim
      { y: 1.676, hw: 0.192, hd: 0.196, e: 2.2, cz: -0.005 },
    ], 14, true, true), P.helmet, HB, true);
    for (let i = 0; i < 6; i++) {
      const a = i * 1.05 + 0.4;
      push(batch, box(0.038, 0.086, 0.016, {
        x: Math.sin(a) * 0.10, y: 1.712, z: Math.cos(a) * 0.095 - 0.005, ry: -a, rz: 0.2,
      }), P.ghillie, HB, true);
    }
    return;
  }

  // combat helmet
  push(batch, loft([
    { y: 1.828, hw: 0.024, hd: 0.026, e: 2.4, cz: -0.005 },
    { y: 1.818, hw: 0.054, hd: 0.058, e: 2.6, cz: -0.005 },
    { y: 1.795, hw: 0.085, hd: 0.091, e: 2.8, cz: -0.005 },
    { y: 1.762, hw: 0.104, hd: 0.112, e: 3.0, cz: -0.005 },
    { y: 1.726, hw: 0.114, hd: 0.123, e: 3.0, cz: -0.005 },
    { y: 1.700, hw: 0.119, hd: 0.128, e: 3.0, cz: -0.004 },
    { y: 1.676, hw: 0.122, hd: 0.131, e: 3.0, cz: -0.004 },
    { y: 1.662, hw: 0.127, hd: 0.136, e: 2.8, cz: -0.004 },   // flared rim
    { y: 1.650, hw: 0.121, hd: 0.129, e: 2.8, cz: -0.004 },
  ], 14, true, true), P.helmet, HB, true);

  for (const s of [-1, 1]) {
    // ear covers hang below the rim — the notch they cut in the outline is
    // what separates a helmet from a hood at distance
    push(batch, box(0.028, 0.078, 0.090, { x: s * 0.118, y: 1.632, z: -0.014, rz: s * 0.10 }),
      P.helmetTrim, HB);
    push(batch, box(0.013, 0.019, 0.108, { x: s * 0.120, y: 1.702, z: 0.008, rz: s * 0.26 }),
      P.helmetTrim, HB);
  }
  // front mount + rear counterweight: front-to-back asymmetry sells the shape
  push(batch, box(0.050, 0.046, 0.036, { y: 1.770, z: 0.104 }), P.helmetTrim, HB);
  push(batch, box(0.028, 0.028, 0.050, { y: 1.796, z: 0.120, rx: 0.30 }), P.helmetTrim, HB);
  push(batch, box(0.104, 0.058, 0.048, { y: 1.716, z: -0.126 }), P.pouch, HB);

  if (kind === 'visor') {
    push(batch, loft([
      { y: 1.716, hw: 0.106, hd: 0.108, e: 2.2, cz: 0.014 },
      { y: 1.668, hw: 0.116, hd: 0.120, e: 2.2, cz: 0.018 },
      { y: 1.616, hw: 0.108, hd: 0.113, e: 2.2, cz: 0.012 },
    ], 12, false, false), P.visor, HB);
  } else {
    // goggles pushed up on the shell
    push(batch, loft([
      { y: 1.752, hw: 0.117, hd: 0.126, e: 2.8, cz: -0.005 },
      { y: 1.724, hw: 0.121, hd: 0.130, e: 2.8, cz: -0.005 },
    ], 14, false, false), P.helmetTrim, HB);
    push(batch, box(0.128, 0.042, 0.028, { y: 1.740, z: 0.110, rx: -0.20 }), P.lens, HB);
  }
}

// ---------------------------------------------------------------- the weapon
//
// Fictional design. Built along the bore line so the two hand targets always
// land on the grip and the handguard whatever the variant's length is.

function buildGun(rig, batch) {
  const { O, dir, down } = rig.bore;
  const g = rig.gun;
  const W = ['weapon'];

  const side = new THREE.Vector3().crossVectors(dir, down).normalize();
  // Gun-space -> character-space: +Z bore, +Y up, +X side.
  const basis = new THREE.Matrix4().makeBasis(side, _v0.copy(down).negate(), dir).setPosition(O);
  const gun = (geo) => { geo.applyMatrix4(basis); return geo; };
  const gbox = (w, h, d, o) => gun(box(w, h, d, o));
  const gtube = (z0, y0, z1, y1, r0, r1, seg, x = 0) => gun(tube(
    new THREE.Vector3(x, y0, z0), new THREE.Vector3(x, y1, z1), r0, r1, seg, 1, 2, true, true));

  const L = g.len;
  const isShotgun = g.kind === 'scatter';
  const isDmr = g.kind === 'marksman';

  // buttstock + buffer
  push(batch, gbox(0.044, 0.086, 0.130, { z: 0.062, y: -0.004 }), P.polymer, W);
  push(batch, gbox(0.032, 0.030, 0.096, { z: 0.150, y: 0.010 }), P.steel, W);
  push(batch, gbox(0.040, 0.024, 0.086, { z: 0.108, y: 0.046, rx: -0.06 }), P.polymer, W);
  // receiver
  push(batch, gbox(0.048, 0.078, 0.215, { z: 0.288, y: 0.004 }), P.steel, W);
  push(batch, gbox(0.026, 0.014, 0.196, { z: 0.290, y: 0.048 }), P.steel, W);
  // grip + trigger guard
  push(batch, gbox(0.036, 0.112, 0.046, { z: L * g.gripT + 0.004, y: -0.078, rx: 0.34 }), P.polymer, W);
  push(batch, gbox(0.028, 0.010, 0.062, { z: L * g.gripT + 0.046, y: -0.036 }), P.steel, W);
  // magazine — the single biggest silhouette element on a rifle
  if (isShotgun) {
    push(batch, gtube(0.330, -0.036, L - 0.055, -0.036, 0.020, 0.020, 8), P.steel, W);
    push(batch, gbox(0.056, 0.052, 0.170, { z: 0.300, y: -0.044 }), P.polymer, W);
  } else {
    push(batch, gbox(0.030, isDmr ? 0.130 : 0.168, 0.074,
      { z: L * g.gripT + 0.096, y: isDmr ? -0.092 : -0.112, rx: -0.16 }), P.polymer, W);
  }
  // handguard
  const hgA = 0.400, hgB = Math.min(L - 0.075, hgA + (isDmr ? 0.320 : 0.215));
  push(batch, gun(tube(new THREE.Vector3(0, 0.002, hgA), new THREE.Vector3(0, 0.002, hgB),
    0.030, 0.027, 8, 1, 3.4, true, true)), P.steel, W);
  push(batch, gbox(0.024, 0.012, hgB - hgA - 0.02, { z: (hgA + hgB) / 2, y: 0.036 }), P.steel, W);
  // barrel + muzzle device
  push(batch, gtube(hgB, 0.002, L - 0.030, 0.002, 0.011, 0.010, 6), P.steel, W);
  push(batch, gtube(L - 0.030, 0.002, L, 0.002, isShotgun ? 0.023 : 0.018, 0.017, 8), P.steel, W);
  // optic
  if (isDmr) {
    push(batch, gbox(0.030, 0.036, 0.040, { z: 0.300, y: 0.070 }), P.optic, W);
    push(batch, gun(tube(new THREE.Vector3(0, 0.092, 0.230), new THREE.Vector3(0, 0.092, 0.400),
      0.024, 0.030, 9, 1, 2, true, true)), P.optic, W);
    push(batch, gbox(0.026, 0.030, 0.036, { z: 0.238, y: 0.086 }), P.lens, W);
    // folded bipod under the handguard
    for (const s of [-1, 1]) {
      push(batch, gun(tube(new THREE.Vector3(s * 0.012, -0.020, hgB - 0.030),
        new THREE.Vector3(s * 0.030, -0.052, hgB - 0.150), 0.008, 0.006, 5)), P.steel, W);
    }
  } else if (isShotgun) {
    push(batch, gbox(0.014, 0.026, 0.014, { z: L - 0.070, y: 0.030 }), P.steel, W);
    push(batch, gbox(0.020, 0.028, 0.016, { z: 0.190, y: 0.036 }), P.steel, W);
  } else {
    push(batch, gbox(0.036, 0.030, 0.028, { z: 0.268, y: 0.066 }), P.optic, W);
    push(batch, gun(tube(new THREE.Vector3(0, 0.076, 0.226), new THREE.Vector3(0, 0.076, 0.322),
      0.021, 0.023, 9, 1, 2, true, true)), P.optic, W);
    push(batch, gbox(0.022, 0.024, 0.024, { z: 0.230, y: 0.076 }), P.lens, W);
  }
  // support-hand furniture, right where the left glove lands
  push(batch, gbox(0.026, 0.058, 0.034, { z: L * g.suppT + 0.030, y: -0.046, rx: 0.22 }), P.polymer, W);
  // sling loop off the handguard
  push(batch, gbox(0.010, 0.034, 0.010, { z: hgA + 0.030, y: -0.036 }), P.webbing, W);
}

// ---------------------------------------------------------------- skinning

/**
 * Distance from p to bone i's segment, with the radial component scaled down.
 * Every bone lies on the centre line, so a plain euclidean distance makes a
 * surface vertex 0.16m from ALL of them and the solve turns to mush; measuring
 * mostly along the bone axis is what keeps a hip vertex on the hip bone.
 */
function boneDist(px, py, pz, head, tail) {
  const ax = tail.x - head.x, ay = tail.y - head.y, az = tail.z - head.z;
  const len2 = ax * ax + ay * ay + az * az || 1e-8;
  const dx = px - head.x, dy = py - head.y, dz = pz - head.z;
  let t = (dx * ax + dy * ay + dz * az) / len2;
  const outside = t < 0 ? -t : (t > 1 ? t - 1 : 0);
  if (t < 0) t = 0; else if (t > 1) t = 1;
  const rx = dx - ax * t, ry = dy - ay * t, rz = dz - az * t;
  const radial = Math.sqrt(rx * rx + ry * ry + rz * rz);
  return outside * Math.sqrt(len2) + radial * 0.25;
}

function skinBatch(batch, rig) {
  const n = batch.count;
  const si = new Uint16Array(n * 4);
  const sw = new Float32Array(n * 4);
  const wIdx = [0, 0, 0, 0], wVal = [0, 0, 0, 0];
  for (const part of batch.parts) {
    const list = part.bones;
    for (let v = part.start; v < part.start + part.count; v++) {
      const px = batch.pos[v * 3], py = batch.pos[v * 3 + 1], pz = batch.pos[v * 3 + 2];
      let nw = 0;
      for (let k = 0; k < 4; k++) { wIdx[k] = 0; wVal[k] = 0; }
      for (let bi = 0; bi < list.length; bi++) {
        const idx = B[list[bi]];
        const d = boneDist(px, py, pz, rig.bind[idx], rig.tail[idx]);
        const w = 1 / ((d + 0.008) * (d + 0.008));
        // insertion sort into the top-4
        let slot = -1;
        for (let k = 0; k < 4; k++) if (k >= nw || w > wVal[k]) { slot = k; break; }
        if (slot < 0) continue;
        for (let k = 3; k > slot; k--) { wVal[k] = wVal[k - 1]; wIdx[k] = wIdx[k - 1]; }
        wVal[slot] = w; wIdx[slot] = idx;
        if (nw < 4) nw++;
      }
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += wVal[k];
      if (sum <= 0) { wVal[0] = 1; wIdx[0] = B.hips; sum = 1; }
      for (let k = 0; k < 4; k++) {
        si[v * 4 + k] = wIdx[k];
        sw[v * 4 + k] = wVal[k] / sum;
      }
    }
  }
  return { si, sw };
}

// ---------------------------------------------------------------- assets

const _geoCache = new Map();       // variant -> { geo, rig, inverses, refs }
const _matCache = new WeakMap();   // materials -> THREE.Material[]
let _serial = 0;

function getGeometry(variant) {
  let c = _geoCache.get(variant);
  if (c) { c.refs++; return c; }

  const rig = buildRig(variant);
  const batch = newBatch();
  buildBody(variant, rig, batch);
  buildGun(rig, batch);
  const { si, sw } = skinBatch(batch, rig);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(batch.pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(batch.nrm, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(batch.uv, 2));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(batch.col, 3));
  geo.setAttribute('skinIndex', new THREE.BufferAttribute(si, 4));
  geo.setAttribute('skinWeight', new THREE.BufferAttribute(sw, 4));
  geo.setIndex(batch.idx);
  // aoMap reads uv1 in three's default channel layout; the character has no
  // baked AO worth a second set, so point it at the same projection.
  geo.setAttribute('uv1', geo.getAttribute('uv'));
  // Authored by hand: a skinned bounding sphere recomputed from the bind pose
  // pops when the pose changes, and the collapse animation reaches well past
  // the standing bounds.
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0.92, 0), 1.55);
  geo.boundingBox = new THREE.Box3(
    new THREE.Vector3(-0.9, -0.2, -1.1), new THREE.Vector3(0.9, 1.95, 1.3));

  // Bind matrices: bones have identity rotation, so the inverse bind is just a
  // translation by -jointPosition. No temporary Object3D graph needed.
  const inverses = [];
  for (let i = 0; i < NB; i++) {
    inverses.push(new THREE.Matrix4().makeTranslation(
      -rig.bind[i].x, -rig.bind[i].y, -rig.bind[i].z));
  }

  c = { geo, rig, inverses, refs: 1, tris: batch.idx.length / 3, verts: batch.count };
  _geoCache.set(variant, c);
  return c;
}

function getMaterials(materials) {
  let list = _matCache.get(materials);
  if (list) return list;
  list = KIT_TINTS.map((tint) => {
    // Cloned so vertexColors can be enabled without mutating the shared
    // library material; materials.js preserves the surface shader across
    // clone(), and sky.js chains its cascade patch on afterwards.
    const m = materials.get('fabric', {
      color: tint, roughness: 0.90, normalScale: 1.15, grimeStrength: 0.85,
    }).clone();
    m.vertexColors = true;
    m.name = 'mat:character';
    return m;
  });
  _matCache.set(materials, list);
  return list;
}

// ---------------------------------------------------------------- pose maths

function smoothstep(e0, e1, x) {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}
function damp(cur, target, rate, dt) {
  return target + (cur - target) * Math.exp(-rate * dt);
}

// ---------------------------------------------------------------- character

class Character {
  constructor(materials, variant, rng) {
    const asset = getGeometry(variant);
    this._asset = asset;
    this.variant = variant;
    const rig = asset.rig;
    this.rig = rig;

    this.group = new THREE.Group();
    this.group.name = 'char:' + variant;

    // ---- bones
    const bones = [];
    for (let i = 0; i < NB; i++) {
      const b = new THREE.Bone();
      b.name = BONES[i];
      bones.push(b);
    }
    this._bindLocal = [];
    for (let i = 0; i < NB; i++) {
      const p = PARENT[i];
      const v = rig.bind[i].clone();
      if (p >= 0) { v.sub(rig.bind[p]); bones[p].add(bones[i]); }
      this._bindLocal.push(v);
      bones[i].position.copy(v);
    }
    this.bones = bones;
    this.group.add(bones[0]);

    this.skeleton = new THREE.Skeleton(bones, asset.inverses);

    // ---- mesh
    const mats = getMaterials(materials);
    const r = rng || makeRng('char:' + variant + ':' + (_serial++));
    this._kit = Math.floor(r() * mats.length) % mats.length;
    const mesh = new THREE.SkinnedMesh(asset.geo, mats[this._kit]);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = true;
    mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0.92, 0), 1.55);
    this.group.add(mesh);
    mesh.bind(this.skeleton, _IDENT_M);
    this.mesh = mesh;

    // +-3% so a squad is not eight copies of one man.
    this.group.scale.setScalar(0.975 + 0.05 * r());
    this._idleBias = r() * Math.PI * 2;

    // ---- hitboxes. Spheres hung off bones; ai.js reads world position +
    // radius and picks the nearest, so they are sized to tile the body with
    // minimal overlap rather than to be individually generous.
    const node = (bone, y, z) => {
      const o = new THREE.Object3D();
      o.position.set(0, y, z || 0);
      bones[bone].add(o);
      return o;
    };
    this.hitboxes = [
      { name: 'head', mult: 2.0, obj: node(B.head, 0.100, -0.004), radius: 0.135, halfHeight: 0.05 },
      { name: 'chest', mult: 1.0, obj: node(B.chest, 0.105, 0.012), radius: 0.225, halfHeight: 0.12 },
      { name: 'stomach', mult: 0.85, obj: node(B.spine, -0.020, 0.008), radius: 0.185, halfHeight: 0.10 },
      { name: 'limb', mult: 0.70, obj: node(B.fArmL, 0.0, 0.0), radius: 0.140, halfHeight: 0.12 },
      { name: 'limb', mult: 0.70, obj: node(B.fArmR, 0.0, 0.0), radius: 0.140, halfHeight: 0.12 },
      { name: 'limb', mult: 0.65, obj: node(B.thighL, -0.190, 0.0), radius: 0.165, halfHeight: 0.16 },
      { name: 'limb', mult: 0.65, obj: node(B.thighR, -0.190, 0.0), radius: 0.165, halfHeight: 0.16 },
      { name: 'limb', mult: 0.55, obj: node(B.shinL, -0.180, 0.0), radius: 0.125, halfHeight: 0.16 },
      { name: 'limb', mult: 0.55, obj: node(B.shinR, -0.180, 0.0), radius: 0.125, halfHeight: 0.16 },
    ];

    // ---- weapon pose keys, in chest-local space.
    const chest = rig.bind[B.chest];
    this._wRest = BORE_REST_ORIGIN.clone().sub(chest);
    this._wAim = BORE_AIM_ORIGIN.clone().sub(chest);
    this._wAimQ = new THREE.Quaternion().setFromUnitVectors(BORE_REST_DIR, BORE_AIM_DIR);
    this._gripLocal = rig.bore.grip.clone().sub(BORE_REST_ORIGIN);
    this._suppLocal = rig.bore.supp.clone().sub(BORE_REST_ORIGIN);

    // ---- state
    this.state = { move: 0, aim: 0.6, crouch: 0, dead: 0 };
    this._t = { move: 0, aim: 0.6, crouch: 0, dead: 0 };
    this._explicit = 0;
    this._phase = r();
    this._speed = 0;
    this._prev = new THREE.Vector3().copy(this.group.position);
    this._hasPrev = false;
    this._fallYaw = 0;
    this._clock = 0;

    // ---- per-frame scratch owned by this instance (no shared mutation across
    // characters, and nothing allocated after construction)
    this._csP = []; this._csQ = [];
    for (let i = 0; i < NB; i++) { this._csP.push(new THREE.Vector3()); this._csQ.push(new THREE.Quaternion()); }
    this._joint = new THREE.Vector3();
    this._end = new THREE.Vector3();
    this._target = new THREE.Vector3();
    this._pole = new THREE.Vector3();
    this._footL = new THREE.Vector3();
    this._footR = new THREE.Vector3();

    this._pose();
  }

  setPose(s) {
    if (!s) return;
    if (s.move != null) { this._t.move = Math.min(1, Math.max(0, s.move)); this._explicit = 1; }
    if (s.aim != null) { this._t.aim = Math.min(1, Math.max(0, s.aim)); this._explicit |= 2; }
    if (s.crouch != null) this._t.crouch = Math.min(1, Math.max(0, s.crouch));
    if (s.dead != null) {
      if (s.dead > 0 && this._t.dead <= 0 && s.hitDir) {
        // Fall away from the shot. hitDir is world space; the character's own
        // facing is on the group, so bring it into local space once.
        _v0.copy(s.hitDir);
        this.group.getWorldQuaternion(_q0);
        _v0.applyQuaternion(_q0.invert());
        this._fallYaw = Math.atan2(-_v0.x, -_v0.z);
      }
      this._t.dead = Math.max(0, s.dead);
    }
  }

  update(dt) {
    if (!(dt > 0)) dt = 1 / 60;
    if (dt > 0.1) dt = 0.1;
    this._clock += dt;

    // Derive locomotion from actual displacement unless a caller has taken
    // over. ai.js drives position directly and never calls setPose, so without
    // this the walk cycle would never run in the shipped game.
    const p = this.group.position;
    if (this._hasPrev) {
      const dx = p.x - this._prev.x, dz = p.z - this._prev.z;
      const inst = Math.sqrt(dx * dx + dz * dz) / dt;
      this._speed = damp(this._speed, Math.min(6, inst), 9, dt);
    }
    this._prev.copy(p);
    this._hasPrev = true;

    if (!(this._explicit & 1)) this._t.move = Math.min(1, this._speed / 4.2);
    if (!(this._explicit & 2)) this._t.aim = 0.55 + 0.45 * (1 - this._t.move);

    const s = this.state, t = this._t;
    s.move = damp(s.move, t.move, 7, dt);
    s.aim = damp(s.aim, t.aim, 6, dt);
    s.crouch = damp(s.crouch, t.crouch, 8, dt);
    s.dead = damp(s.dead, t.dead, 5, dt);

    // Cadence rises with speed; stride is whatever that leaves, capped so the
    // leg never asks the IK for more reach than it has.
    const freq = 1.15 + 1.05 * s.move;
    this._phase = (this._phase + freq * dt) % 1;

    this._pose();
  }

  _pose() {
    const s = this.state;
    const bones = this.bones, bind = this._bindLocal;
    const move = s.move, aim = s.aim, crouch = s.crouch;
    const dead = Math.min(1, s.dead);
    const alive = 1 - dead;
    const mb = smoothstep(0.0, 0.10, move);          // gait blend-in
    const ph = this._phase, tau = Math.PI * 2;
    const sp = Math.sin(tau * ph);

    for (let i = 0; i < NB; i++) {
      bones[i].position.copy(bind[i]);
      bones[i].quaternion.identity();
    }

    // ---- root: crouch drop, gait bob, collapse
    const bob = (0.020 + 0.036 * move) * (0.5 - 0.5 * Math.cos(2 * tau * ph)) * mb;
    const crouchDrop = 0.40 * crouch;
    const idle = Math.sin(this._clock * 1.35 + this._idleBias);
    const root = bones[B.root];
    root.position.y = -(crouchDrop + bob) * alive + 0.30 * dead;
    root.position.z = -0.10 * dead;
    if (dead > 0.001) {
      _e0.set(-1.34 * dead, this._fallYaw * dead, 0.22 * dead, 'YXZ');
      root.quaternion.setFromEuler(_e0);
    }

    // ---- spine chain
    const lean = (0.055 + 0.19 * move) * alive + 0.34 * crouch * alive;
    const hips = bones[B.hips];
    _e0.set(
      (-0.05 * crouch) * alive + 0.26 * dead,
      (0.17 * sp * mb + 0.02 * idle) * alive,
      (0.055 * sp * mb) * alive,
      'YXZ');
    hips.quaternion.setFromEuler(_e0);

    _e0.set(
      lean * 0.55 + 0.30 * dead + 0.010 * idle * (1 - move),
      -0.07 * sp * mb * alive,
      -0.030 * sp * mb * alive,
      'YXZ');
    bones[B.spine].quaternion.setFromEuler(_e0);

    _e0.set(
      lean * 0.45 + 0.20 * dead - 0.010 * idle * (1 - move),
      -0.09 * sp * mb * alive,
      0.020 * sp * mb * alive,
      'YXZ');
    bones[B.chest].quaternion.setFromEuler(_e0);

    // Head counter-rotates the torso so the eyeline stays level while walking,
    // then drops to the stock as the rifle comes up.
    _e0.set(
      (-lean * 0.62 + 0.11 * aim) * alive + 0.34 * dead,
      (0.09 * sp * mb + 0.055 * aim) * alive,
      (0.085 * aim - 0.030 * sp * mb) * alive,
      'YXZ');
    bones[B.neck].quaternion.setFromEuler(_e0);
    _e0.set(
      (-lean * 0.30 + 0.06 * aim) * alive + 0.18 * dead,
      (-0.075 * aim) * alive,
      (0.075 * aim) * alive,
      'YXZ');
    bones[B.head].quaternion.setFromEuler(_e0);

    // ---- weapon: lerp the whole bore between the carry and the shoulder.
    const wa = aim * alive;
    const wp = bones[B.weapon];
    wp.position.copy(this._wRest).lerp(this._wAim, wa);
    wp.quaternion.copy(_IDENT_Q).slerp(this._wAimQ, wa);
    if (dead > 0.001) {
      // muzzle drops as the grip goes slack
      _e0.set(0.9 * dead, 0.3 * dead, 0, 'YXZ');
      _q0.setFromEuler(_e0);
      wp.quaternion.multiply(_q0);
      wp.position.y -= 0.10 * dead;
    }

    // ---- forward kinematics for the torso chain + weapon (parents first).
    const csP = this._csP, csQ = this._csQ;
    for (let i = 0; i <= B.weapon; i++) {
      const par = PARENT[i];
      if (par < 0) {
        csP[i].copy(bones[i].position);
        csQ[i].copy(bones[i].quaternion);
      } else {
        csP[i].copy(bones[i].position).applyQuaternion(csQ[par]).add(csP[par]);
        csQ[i].copy(csQ[par]).multiply(bones[i].quaternion);
      }
    }

    // ---- arms: both hands solved onto fixed points on the rifle.
    this._solveArm(true, aim, dead);
    this._solveArm(false, aim, dead);

    // ---- legs: gait targets, then plant.
    const stride = Math.min(0.66, 0.30 + 0.42 * move);
    const duty = 0.64 - 0.10 * move;
    const lift = (0.045 + 0.085 * move);
    this._footTarget(true, ph, stride, duty, lift, mb, this._footL);
    this._footTarget(false, ph, stride, duty, lift, mb, this._footR);
    this._solveLeg(true, this._footL, ph, duty, mb, dead);
    this._solveLeg(false, this._footR, ph, duty, mb, dead);
  }

  _solveArm(left, aim, dead) {
    const csP = this._csP, csQ = this._csQ;
    const iU = left ? B.uArmL : B.uArmR;
    const iF = left ? B.fArmL : B.fArmR;
    const iH = left ? B.handL : B.handR;
    const bones = this.bones;

    // shoulder in character space
    _v0.copy(this._bindLocal[iU]).applyQuaternion(csQ[B.chest]).add(csP[B.chest]);
    // hand target: a fixed point on the weapon bone
    this._target.copy(left ? this._suppLocal : this._gripLocal)
      .applyQuaternion(csQ[B.weapon]).add(csP[B.weapon]);
    if (dead > 0.001) {
      // arms release and fall to the sides
      _v1.copy(this._csP[B.hips]);
      _v1.x += left ? 0.26 : -0.26; _v1.y -= 0.16; _v1.z += 0.04;
      this._target.lerp(_v1, left ? dead : dead * 0.55);
    }

    this._pole.copy(left ? POLE_ARM_L : POLE_ARM_R)
      .lerp(left ? POLE_ARM_L_AIM : POLE_ARM_R_AIM, aim * (1 - dead)).normalize();

    solve2(_v0, this._target, ARM_UP, ARM_LO, this._pole, this._joint, this._end);

    // upper arm
    _v1.copy(this.rig.bind[iF]).sub(this.rig.bind[iU]).normalize();
    _v2.copy(this._joint).sub(_v0).normalize();
    _q0.setFromUnitVectors(_v1, _v2);
    csQ[iU].copy(_q0);
    bones[iU].quaternion.copy(csQ[B.chest]).invert().multiply(_q0);
    csP[iU].copy(_v0);

    // forearm
    _v1.copy(this.rig.bind[iH]).sub(this.rig.bind[iF]).normalize();
    _v2.copy(this._end).sub(this._joint).normalize();
    _q1.setFromUnitVectors(_v1, _v2);
    csQ[iF].copy(_q1);
    bones[iF].quaternion.copy(_q0).invert().multiply(_q1);
    csP[iF].copy(this._joint);

    // hand rides the weapon's roll so the glove never twists off the grip
    _q2.copy(csQ[B.weapon]).slerp(_IDENT_Q, dead * 0.7);
    bones[iH].quaternion.copy(_q1).invert().multiply(_q2);
    csQ[iH].copy(_q2);
    csP[iH].copy(this._end);
  }

  /** Where this foot wants to be, in character space, this frame. */
  _footTarget(left, ph, stride, duty, lift, mb, out) {
    const bindA = this.rig.bind[left ? B.footL : B.footR];
    const u = (ph + (left ? 0 : 0.5)) % 1;
    let z, y;
    if (u < duty) {
      const p = u / duty;
      z = stride * (0.5 - p);
      // The ankle rises through toe-off. Without it the boot pivots about a
      // planted ankle and drives its toe 4-5cm through the sand.
      const k = Math.max(0, (p - 0.50) / 0.50);
      y = 0.092 * k * k;
    } else {
      const p = (u - duty) / (1 - duty);
      z = stride * (-0.5 + p);
      y = lift * Math.sin(Math.PI * p) + 0.092 * Math.max(0, 1 - p * 4) * Math.max(0, 1 - p * 4);
    }
    // Narrow the track slightly at speed; a wide gait reads as a waddle.
    const x = bindA.x * (1 - 0.20 * mb);
    out.set(x, bindA.y + y * mb, (0.014 + z) * mb + bindA.z * (1 - mb));
  }

  _solveLeg(left, target, ph, duty, mb, dead) {
    const csP = this._csP, csQ = this._csQ, bones = this.bones;
    const iT = left ? B.thighL : B.thighR;
    const iS = left ? B.shinL : B.shinR;
    const iF = left ? B.footL : B.footR;

    _v0.copy(this._bindLocal[iT]).applyQuaternion(csQ[B.hips]).add(csP[B.hips]);

    this._target.copy(target);
    if (dead > 0.001) {
      // knees buckle and the feet slide in as the body goes down
      _v1.copy(_v0); _v1.y -= 0.30; _v1.z += left ? 0.30 : 0.20;
      this._target.lerp(_v1, dead);
    }

    this._pole.copy(POLE_KNEE);
    solve2(_v0, this._target, LEG_UP, LEG_LO, this._pole, this._joint, this._end);

    _v1.copy(this.rig.bind[iS]).sub(this.rig.bind[iT]).normalize();
    _v2.copy(this._joint).sub(_v0).normalize();
    _q0.setFromUnitVectors(_v1, _v2);
    bones[iT].quaternion.copy(csQ[B.hips]).invert().multiply(_q0);

    _v1.copy(this.rig.bind[iF]).sub(this.rig.bind[iS]).normalize();
    _v2.copy(this._end).sub(this._joint).normalize();
    _q1.setFromUnitVectors(_v1, _v2);
    bones[iS].quaternion.copy(_q0).invert().multiply(_q1);

    // Ankle: heel strike, roll flat, toe off. Without this the boot stays at a
    // fixed angle and the walk reads as a doll being slid along the ground.
    const u = (ph + (left ? 0 : 0.5)) % 1;
    let pitch;
    if (u < duty) { const p = u / duty; pitch = -0.20 + 0.78 * p * p; }
    else { const p = (u - duty) / (1 - duty); pitch = 0.55 * (1 - p) * (1 - p) - 0.24 * p; }
    _e0.set(pitch * mb * (1 - dead) - 0.35 * dead, 0, 0, 'YXZ');
    _q2.setFromEuler(_e0);
    _q3.copy(_q1).invert().multiply(_q2);
    bones[iF].quaternion.copy(_q3);
  }

  dispose() {
    if (this.group.parent) this.group.parent.remove(this.group);
    this.skeleton.dispose();
    const a = this._asset;
    if (a && --a.refs <= 0) {
      a.geo.dispose();
      _geoCache.delete(this.variant);
    }
    this._asset = null;
  }
}

// ---------------------------------------------------------------- public

export function buildCharacter(materials, variant = 'rifleman', rng) {
  const v = CHARACTER_VARIANTS.indexOf(variant) >= 0 ? variant : 'rifleman';
  const c = new Character(materials, v, rng);
  return {
    group: c.group,
    skeleton: c.skeleton,
    hitboxes: c.hitboxes,
    setPose: (s) => c.setPose(s),
    update: (dt) => c.update(dt),
    dispose: () => c.dispose(),
    // Debug/bench only. Not part of the contract other modules read.
    _stats: { triangles: c._asset ? c._asset.tris : 0, bones: NB },
  };
}
