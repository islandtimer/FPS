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
// THE SIGHT IS A HOLE, NOT A PICTURE. A non-magnifying reflex sight shows the
// world directly: the correct image through the glass is the scene behind the
// tube, unaltered. So the tube is genuinely open — an inward-facing bore wall,
// no lens discs blocking it — and all that sits in the light path is one nearly
// clear pane and an additive reticle. Zero extra passes, zero render targets,
// and it is also what the real device does. The pane carries its own eyebox
// vignette in vertex alpha, so the darkening of the outer rim costs no second
// mesh and no shader.
//
// THE SIGHT PICTURE IS SOLVED, NOT POSED. The optic sits high enough, and the
// top of the handguard low enough, that the forward half of the rifle falls
// OUTSIDE the cone the eye sees through the objective. `topOfForend()` computes
// that clearance from the real aperture and the real eye position rather than
// trusting eyeballed numbers to survive a change of optic. Nothing but sky and
// target is inside the tube.
//
// NOTHING LIVES ON THE NEAR PLANE. At full ADS the eye sits at rifle-local
// z = +0.113, so anything aft of z = +0.067 is closer to the camera than the
// 5cm near plane and gets sliced by it. A box sliced that way loses its front
// faces and renders as a shredded, zero-thickness plate stretching off past the
// silhouette. Everything aft of the receiver is therefore either wholly behind
// that line (the stock, which vanishes cleanly) or below y = +0.009 (the buffer
// tube), where it is under the bottom edge of the frustum. The charging handle
// is on the left flank for the same reason — a top-mounted one sits 5cm from
// the eyeball at ADS and can only ever be debris.
//
// FOUR MATERIALS, AND A VERTEX MASK THAT MAKES THEM MANY. Coated steel for the
// receiver and rail base, bare machined metal for rail teeth / muzzle / bolt /
// fasteners, matte polymer for the furniture, rubber for pads. On top of that
// every vertex carries a mask that does what no tiling texture can: part-scale
// edge wear on the chamfer lands (the material colour is the WORN value and the
// mask scales the field down from it), contact occlusion wherever two parts bolt
// together, and grime down in the slot valleys. That is where the metal wear
// comes from now — convexity, not a speckle baked into an albedo map at the same
// density on the sunlit face and the shadowed one.
//
// DRAW CALLS. Geometry accumulates into per-material buckets and is merged once,
// so the whole rifle plus both hands is 11 draws: five for the static body
// (coated / bare / polymer / rubber / glove), two for the sight (pane, additive
// reticle) and four for the parts weaponfx.js animates independently — bolt,
// magazine, trigger, charging handle. Transparency, additive blending and the
// four moving parts cannot share a bucket with anything, so 11 is the floor.
//
// UV. Extrude and lathe UVs are useless when texel density is the whole point,
// so after merging, each bucket gets a box projection at a fixed tiles-per-metre
// rate. The rate is set so the baked micro-grain lands BELOW a pixel at 30cm —
// at the old rate the phosphate cells were ~1px across, which is exactly the
// size that aliases into uniform white salt and pepper.
//
// SIGHT AXIS. weaponfx.js solves the ADS rest pose from parts.optic's actual
// transform, so moving SIGHT_Y moves the eye with it and the dot stays dead
// centre. Do not hard-code the number anywhere else.

import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { makeRng } from '../core/rng.js';

export const WEAPON_IDS = ['ar_vector', 'smg_wasp', 'dmr_ridge'];

// ---------------------------------------------------------------- layout
// Every shared hard number lives here. Bore sits 60mm under the sight axis: the
// riser is tall on purpose, because that is what lifts the rail teeth and the
// forend clear of the bottom of the sight picture.
const SIGHT_Y = 0.052;
const BORE_Y = -0.008;
const RAIL_BOT = 0.016;    // top face of the upper receiver
const RAIL_TOP = 0.0205;   // top of the rail base, under the teeth
const UP_F = -0.098, UP_R = 0.058;         // upper receiver extents in z
const UP_C = (UP_F + UP_R) / 2;
const PORT_Z = -0.055, PORT_Y = -0.008;    // ejection port centre
const HG_Z = -0.100;                       // handguard / receiver joint
const RAIL_F = -0.104;                     // rail teeth stop here: forward of
                                           // this the deck is shaded, not cut

// ADS geometry, in rifle-local units. weaponfx puts the optic on the eye, so the
// eye is ADS_EYE behind the tube centre; PRESENCE is the group scale, which
// divides out of every angle but not out of that fixed eye distance.
const PRESENCE = 1.08;
const ADS_EYE = 0.142;
const OPTIC_Z = -0.018;
const EYE_Z = (ADS_EYE + OPTIC_Z * PRESENCE * -1 * -1) / PRESENCE + OPTIC_Z * 0;
// = (ADS_EYE - PRESENCE*|OPTIC_Z|) / PRESENCE, written out so the sign is plain:
const EYE_ZR = (ADS_EYE + OPTIC_Z * PRESENCE) / PRESENCE;

// Aft of this, geometry straddles the 5cm near plane at ADS and gets sliced.
const NEAR_ZR = 0.067;

// ---------------------------------------------------------------- geometry kit

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c3 = new THREE.Vector3();
const _bx = new THREE.Vector3();
const _by = new THREE.Vector3();
const _bz = new THREE.Vector3();

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

/** Place a list of geometries with an explicit orthonormal basis + origin. */
function place(list, ex, ey, ez, ox, oy, oz) {
  _m.makeBasis(ex, ey, ez);
  _m.setPosition(ox, oy, oz);
  for (const g of list) g.applyMatrix4(_m);
  return list;
}

function put(bucket, g, o, w) {
  if (w !== undefined) paint(g, w);
  bucket.push(xf(g, o));
  return g;
}
function putAll(bucket, list, o, w) {
  for (const g of list) { if (w !== undefined) paint(g, w); bucket.push(xf(g, o)); }
  return list;
}
/** Rotate a list of geometries about their shared local origin, in place. */
function turnY(list, a) { for (const g of list) g.rotateY(a); return list; }
function turnZ(list, a) { for (const g of list) g.rotateZ(a); return list; }

const EX = { steps: 1, bevelEnabled: true, bevelSegments: 1, curveSegments: 1 };

// ---------------------------------------------------------------- vertex mask
//
// FIELD is the unworn value. The material's colour is the fully worn one, so
// every vertex multiplies DOWN from bare metal: a chamfer land reaches 1.0 and
// is the brightest thing on the part, a flat sits at FIELD, a slot valley or a
// bolted joint goes lower still. This is the only layer in the whole model that
// knows where the part's own edges are — the baked maps only know texel-scale
// curvature, which is why their wear reads as uniform speckle no matter which
// way the surface faces.
const FIELD = 0.60;
const DARK = 0.30;       // bores, slot valleys, anything down a hole
const GRIME = 0.42;

function colorAttr(g, n) {
  let c = g.getAttribute('color');
  const count = g.getAttribute('position').count;
  if (!c || c.itemSize !== n || c.count !== count) {
    c = new THREE.BufferAttribute(new Float32Array(count * n), n);
    g.setAttribute('color', c);
  }
  return c;
}

/**
 * Write the mask. `w` is a scalar field value, or one of two modes evaluated in
 * the piece's OWN local frame before it is transformed into the rifle:
 *   'edge'  chamfer lands — any face whose normal is not axis aligned. On a
 *           chamfered box that is exactly the set of milled corners.
 *   'rim'   the two ends of a lathe profile: muzzle crown, bezel lip, tube mouth.
 */
function paint(g, w) {
  const c = colorAttr(g, 3);
  const a = c.array;
  const n = g.getAttribute('normal');
  const p = g.getAttribute('position');
  if (w === 'edge' && n) {
    const na = n.array;
    for (let i = 0, j = 0; i < c.count; i++, j += 3) {
      const mx = Math.max(Math.abs(na[j]), Math.abs(na[j + 1]), Math.abs(na[j + 2]));
      // 1.0 on an axis face, 0.707 on a two-axis land, 0.577 on a corner
      const e = Math.min(1, Math.max(0, (0.985 - mx) / 0.26));
      const v = FIELD + (1 - FIELD) * e * e * (3 - 2 * e);
      a[j] = v; a[j + 1] = v; a[j + 2] = v;
    }
  } else if (w === 'rim' && p) {
    const pa = p.array;
    let z0 = Infinity, z1 = -Infinity;
    for (let j = 2; j < pa.length; j += 3) { if (pa[j] < z0) z0 = pa[j]; if (pa[j] > z1) z1 = pa[j]; }
    const half = Math.max(1e-5, (z1 - z0) * 0.5), mid = (z0 + z1) * 0.5;
    for (let i = 0, j = 0; i < c.count; i++, j += 3) {
      const t = Math.min(1, Math.abs(pa[j + 2] - mid) / half);
      const e = Math.min(1, Math.max(0, (t - 0.62) / 0.38));
      const v = FIELD + (1 - FIELD) * e;
      a[j] = v; a[j + 1] = v; a[j + 2] = v;
    }
  } else {
    const v = typeof w === 'number' ? w : FIELD;
    for (let i = 0; i < a.length; i++) a[i] = v;
  }
  c.needsUpdate = true;
  return g;
}

/**
 * Contact occlusion. A rifle is bolted together, and the tell is the dark line
 * where two parts meet — without it an optic looks stacked on a rail rather than
 * clamped to it. Each seam is an axis-aligned plane with a radius of influence
 * and an optional box, applied to the merged bucket in absolute rifle
 * coordinates, so one table covers every part that touches that joint.
 */
function applySeams(g, seams) {
  const p = g.getAttribute('position');
  const c = g.getAttribute('color');
  if (!p || !c || c.itemSize !== 3) return g;
  const pa = p.array, ca = c.array;
  for (let i = 0, j = 0; i < p.count; i++, j += 3) {
    const x = pa[j], y = pa[j + 1], z = pa[j + 2];
    let mul = 1;
    for (let k = 0; k < seams.length; k++) {
      const s = seams[k];
      if (x < s.x0 || x > s.x1 || y < s.y0 || y > s.y1 || z < s.z0 || z > s.z1) continue;
      const d = Math.abs((s.a === 0 ? x : s.a === 1 ? y : z) - s.v);
      if (d >= s.r) continue;
      const f = 1 - d / s.r;
      mul *= 1 - s.k * f * f;
    }
    if (mul < 1) { ca[j] *= mul; ca[j + 1] *= mul; ca[j + 2] *= mul; }
  }
  c.needsUpdate = true;
  return g;
}

function seam(a, v, r, k, box) {
  const b = box || {};
  return {
    a, v, r, k,
    x0: b.x0 ?? -1, x1: b.x1 ?? 1,
    y0: b.y0 ?? -1, y1: b.y1 ?? 1,
    z0: b.z0 ?? -1, z1: b.z1 ?? 1,
  };
}

// ---------------------------------------------------------------- primitives

/**
 * Chamfered box — the workhorse, ~60 triangles. Every edge gets a 45 degree
 * land of `c`; that land is the difference between "a box" and "a milled part",
 * and it is also where the edge wear goes.
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
  paint(g, 'edge');
  return g;
}

/** A chamfered box spanning A -> B, length along its own Z. Used for fingers. */
function boneBox(ax, ay, az, bx, by, bz, w, t, ux, uy, uz) {
  _a.set(ax, ay, az); _b.set(bx, by, bz);
  const len = _a.distanceTo(_b);
  const g = cbox(w, t, Math.max(0.004, len + t * 0.35), Math.min(0.0016, t * 0.22));
  _c3.addVectors(_a, _b).multiplyScalar(0.5);
  _bz.subVectors(_a, _b).normalize();
  _by.set(ux, uy, uz).normalize();
  _bx.crossVectors(_by, _bz).normalize();
  _by.crossVectors(_bz, _bx).normalize();
  place([g], _bx, _by, _bz, _c3.x, _c3.y, _c3.z);
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
  // Rail teeth are frustum boxes, and the top land of a rail tooth is the single
  // most rubbed surface on any rifle — every mount that was ever clamped there
  // wore it. So the mask peaks at both ends of the taper, not on the flanks.
  const c = colorAttr(g, 3).array;
  const half = h * 0.5;
  for (let i = 0, j = 0; i < pos.length / 3; i++, j += 3) {
    const t = Math.min(1, Math.abs(pos[j + 1]) / Math.max(1e-5, half));
    const e = Math.min(1, Math.max(0, (t - 0.55) / 0.45));
    const v = FIELD + (1 - FIELD) * e * e * (3 - 2 * e);
    c[j] = v; c[j + 1] = v; c[j + 2] = v;
  }
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

/**
 * Reverse the winding so a cylinder can be seen from the inside. The optic bore
 * is the reason this exists: a normally wound tube inside the sight is entirely
 * back-faced from the shooter's eye and therefore invisible, which is why the
 * old "internal hood" did nothing at all and the tube read as a flat disc.
 */
function flipFaces(g) {
  const n = g.index ? g.toNonIndexed() : g;
  for (const key of Object.keys(n.attributes)) {
    const at = n.getAttribute(key);
    const arr = at.array, is = at.itemSize;
    for (let i = 0; i < at.count; i += 3) {
      for (let k = 0; k < is; k++) {
        const p = (i + 1) * is + k, q = (i + 2) * is + k;
        const t = arr[p]; arr[p] = arr[q]; arr[q] = t;
      }
    }
    at.needsUpdate = true;
  }
  const nr = n.getAttribute('normal');
  if (nr) { const a = nr.array; for (let i = 0; i < a.length; i++) a[i] = -a[i]; nr.needsUpdate = true; }
  return n;
}

/** Closed annulus — washer, spacer, baffle ring. Axis along -Z, flat shaded. */
function ringZ(rIn, rOut, len, seg) {
  return flat(latheZ([[rIn, 0], [rOut, 0], [rOut, len], [rIn, len], [rIn, 0]], seg));
}

/**
 * Concentric-ring disc facing +Z with per-ring RGBA vertex colour.
 * `rings` are [radius, r, g, b, a, zBulge] from the centre out. One of these is
 * the entire lens — clear centre, cool anti-reflective tint, and the eyebox
 * vignette rolled into the outer band — and another is the entire reticle.
 */
function discZ(rings, seg) {
  const tri = seg * ((rings[0][0] <= 0 ? 1 : 2) + (rings.length - 2) * 2);
  const pos = new Float32Array(tri * 9);
  const nor = new Float32Array(tri * 9);
  const uv = new Float32Array(tri * 6);
  const col = new Float32Array(tri * 12);
  let k = 0, kc = 0, ku = 0;
  const emit = (ring, ang) => {
    const r = ring[0];
    pos[k] = Math.cos(ang) * r; pos[k + 1] = Math.sin(ang) * r; pos[k + 2] = ring[5] || 0;
    nor[k] = 0; nor[k + 1] = 0; nor[k + 2] = 1;
    k += 3;
    uv[ku] = 0.5 + Math.cos(ang) * 0.5; uv[ku + 1] = 0.5 + Math.sin(ang) * 0.5; ku += 2;
    col[kc] = ring[1]; col[kc + 1] = ring[2]; col[kc + 2] = ring[3]; col[kc + 3] = ring[4];
    kc += 4;
  };
  const TAU = Math.PI * 2;
  for (let s = 0; s < seg; s++) {
    const a0 = (s / seg) * TAU, a1 = ((s + 1) / seg) * TAU;
    for (let i = 0; i < rings.length - 1; i++) {
      const in0 = rings[i], out = rings[i + 1];
      if (in0[0] <= 0) {
        emit(in0, a0); emit(out, a0); emit(out, a1);
      } else {
        emit(in0, a0); emit(out, a0); emit(out, a1);
        emit(in0, a0); emit(out, a1); emit(in0, a1);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setAttribute('color', new THREE.BufferAttribute(col, 4));
  return g;
}

/**
 * Flat strip of alternating vertex-colour bands. Picatinny slots as SHADED
 * detail rather than silhouette: forward of the receiver the teeth are 1-2px
 * tall at a grazing angle, which no AA in this pipeline can hold, and they were
 * comfortably the worst crawling edges on the model. A painted slot at that
 * distance is indistinguishable from a cut one and cannot alias at all.
 * Plane is XZ, normal +Y, length along Z.
 */
function slotStrip(w, z0, z1, pitch, land, cLand, cSlot) {
  const bands = [];
  let z = z0;
  let dark = false;
  while (z < z1 - 1e-5) {
    const step = Math.min(dark ? pitch - land : land, z1 - z);
    bands.push([z, z + step, dark ? cSlot : cLand]);
    z += step;
    dark = !dark;
  }
  const n = bands.length;
  const pos = new Float32Array(n * 18);
  const nor = new Float32Array(n * 18);
  const uv = new Float32Array(n * 12);
  const col = new Float32Array(n * 18);
  let k = 0, ku = 0;
  const hw = w / 2;
  for (const [za, zb, cv] of bands) {
    const quad = [[-hw, za], [hw, za], [hw, zb], [-hw, za], [hw, zb], [-hw, zb]];
    for (const [x, zz] of quad) {
      pos[k] = x; pos[k + 1] = 0; pos[k + 2] = zz;
      nor[k] = 0; nor[k + 1] = 1; nor[k + 2] = 0;
      col[k] = cv; col[k + 1] = cv; col[k + 2] = cv;
      k += 3;
      uv[ku] = x; uv[ku + 1] = zz; ku += 2;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return g;
}

function weld(list) {
  const nz = list.map((g) => {
    const q = g.index ? g.toNonIndexed() : g;
    if (!q.getAttribute('color')) paint(q, FIELD);
    return q;
  });
  return nz.length === 1 ? nz[0] : mergeGeometries(nz, false);
}

const _protos = new Map();
/** Pan-head fastener with a driver slot, axis -Z. Cached, cloned per instance. */
function screw(r, h, spin) {
  const key = 'sc' + r + '_' + h;
  let g = _protos.get(key);
  if (!g) {
    g = weld([
      paint(flat(latheZ([[0, 0], [r, 0], [r, h * 0.62], [r * 0.86, h], [0, h]], 8)), 'rim'),
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
  return paint(flat(latheZ([
    [r * 0.42, 0], [r, 0], [r, r * 0.62], [r * 0.62, r * 0.9], [r * 0.42, r * 0.9],
  ], 10)), 'rim');
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
    hgLen: 0.230, hgR: 0.0225, mlok: 4, barrel: 0.347, gasZ: -0.352,
    brakeLen: 0.063, brakeR: 0.0130, magH: 0.155, magCurve: 0.30,
    stockLen: 0.118, opticLen: 0.056, opticR: 0.0155,
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
    hgLen: 0.150, hgR: 0.0215, mlok: 2, barrel: 0.232, gasZ: -0.272,
    brakeLen: 0.044, brakeR: 0.0124, magH: 0.170, magCurve: 0.16,
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
    hgLen: 0.250, hgR: 0.0230, mlok: 4, barrel: 0.430, gasZ: -0.436,
    brakeLen: 0.070, brakeR: 0.0138, magH: 0.140, magCurve: 0.22,
    stockLen: 0.130, opticLen: 0.082, opticR: 0.0180,
    config: {
      rpm: 380, damage: 52, headMult: 2.1, magSize: 20, reserve: 140,
      spreadHip: 0.046, spreadAds: 0.0016,
      recoil: { vertical: 1.9, horizontal: 0.40, recovery: 6.0 },
      adsTime: 0.30, reloadTime: 2.1, reloadEmptyTime: 2.9,
      muzzleVelocity: 900, falloff: [[0, 1], [45, 1], [80, 0.86], [140, 0.7]],
    },
  },
};

// ---------------------------------------------------------------- sight solve

/** Clear aperture of the objective. Deliberately generous: the tube wall is a
 *  wall, not a lens mount, so the hole is most of the outside diameter. */
function aperture(s) { return s.opticR * 0.84; }

/** Half-angle of the cone the eye sees through the objective, from the eye. */
function exitCone(s) {
  const frontRim = OPTIC_Z - s.opticLen / 2 + 0.002;
  return aperture(s) / (EYE_ZR - frontRim);
}

/**
 * Highest a part at rifle-z `zr` may sit and still fall outside the sight
 * picture, with 15% of margin. This is what stops the rifle occluding its own
 * optic — the single defect that undid the whole model last round, where the
 * forend and rail filled the bottom third of the lens interior.
 */
function clearTop(s, zr) {
  return SIGHT_Y - 1.15 * exitCone(s) * (EYE_ZR - zr);
}

// ---------------------------------------------------------------- upper

function buildUpper(B, s, rng) {
  const L = UP_R - UP_F;

  // Top deck and left wall are solid; the right wall is framed around the
  // ejection port so the bolt carrier is genuinely visible through a hole.
  put(B.coat, cbox(0.041, 0.011, L, 0.0016), { y: 0.0105, z: UP_C });
  put(B.coat, cbox(0.005, 0.030, L, 0.0013), { x: -0.018, y: -0.010, z: UP_C });
  putAll(B.coat, turnY(framedPanel(
    L, 0.030, 0.005, PORT_Z * -1 + UP_C, PORT_Y + 0.010, 0.054, 0.016, 0.0011,
  ), Math.PI / 2), { x: 0.018, y: -0.010, z: UP_C });
  // Proud lip standing off the flank all the way round the port. Without it the
  // port is a rectangle drawn on a slab; with it there is a shadowed step you
  // can read the wall thickness from.
  for (const [dy, h] of [[0.0095, 0.0030], [-0.0095, 0.0030]]) {
    put(B.bare, cbox(0.0030, h, 0.058, 0.0006), { x: 0.0212, y: PORT_Y + 0.010 + dy, z: PORT_Z });
  }
  for (const dz of [-0.0290, 0.0290]) {
    put(B.bare, cbox(0.0030, 0.019, 0.0030, 0.0006), { x: 0.0212, y: PORT_Y + 0.010, z: PORT_Z + dz });
  }
  // the strip the lower closes against — leaves a 1mm parting line with two
  // chamfers in it, which is what a takedown seam actually looks like
  put(B.coat, cbox(0.041, 0.004, L, 0.0012), { y: -0.023, z: UP_C });

  // brass deflector and forward assist: the two lumps that make an upper read
  put(B.coat, frustumBox(0.008, 0.020, 0.014, 0.026, 0.016), { x: 0.024, y: -0.004, z: -0.014, rz: -0.35 });
  put(B.bare, tubeZ(0.0055, 0.0065, 0.020, 10), { x: 0.0225, y: 0.001, z: -0.006, ry: Math.PI / 2 });
  put(B.bare, ringZ(0.0058, 0.0080, 0.004, 10), { x: 0.0285, y: 0.001, z: -0.006, ry: Math.PI / 2 }, 'rim');

  // dust cover, hanging open on its hinge pin just under the port
  putAll(B.coat, turnY([cbox(0.050, 0.015, 0.003, 0.0007), cbox(0.050, 0.004, 0.006, 0.0006)], Math.PI / 2),
    { x: 0.0228, y: -0.0245, z: PORT_Z, rz: 0.35 });
  put(B.bare, tubeZ(0.0021, 0.0021, 0.058, 6), { x: 0.0205, y: -0.0175, z: PORT_Z, ry: Math.PI / 2 });

  // takedown pins through the receiver, with a chamfered head each side
  for (const pz of [0.040, -0.086]) {
    put(B.bare, tubeZ(0.0034, 0.0034, 0.042, 8), { y: -0.018, z: pz, ry: Math.PI / 2 });
    for (const sx of [-1, 1]) {
      put(B.bare, latheZ([
        [0, 0], [0.0048, 0], [0.0048, 0.0022], [0.0036, 0.0032], [0, 0.0032],
      ], 10), { x: sx * 0.0208, y: -0.018, z: pz, ry: sx * Math.PI / 2 }, 'rim');
    }
  }

  // Left-flank charging handle track. The handle itself is a moving part; this
  // is the slot it runs in, and the polished streak either side of it is the one
  // piece of wear on a rifle that is unmistakably from a hand.
  put(B.coat, cbox(0.0035, 0.0055, 0.070, 0.0007), { x: -0.0196, y: 0.0088, z: -0.026 });
  put(B.coat, cbox(0.0035, 0.0055, 0.070, 0.0007), { x: -0.0196, y: -0.0022, z: -0.026 });
  put(B.bare, cbox(0.0022, 0.0044, 0.068, 0.0004), { x: -0.0188, y: 0.0033, z: -0.026 }, 0.92);

  // QD sling socket, receiver rear left
  put(B.bare, qdSocket(0.0062), { x: -0.0208, y: -0.012, z: 0.044, ry: -Math.PI / 2 });

  // deck fasteners along the seam
  for (let i = 0; i < 3; i++) {
    put(B.bare, screw(0.0021, 0.0016, rng() * Math.PI),
      { x: -0.0162, y: 0.0157, z: 0.034 - i * 0.030, rx: -Math.PI / 2 });
  }
}

function buildRail(B, s) {
  const z0 = 0.050, z1 = RAIL_F;
  const L = z0 - z1, zC = (z0 + z1) / 2;
  // Rail BASE is part of the receiver: coated, same family, same value.
  put(B.coat, frustumBox(0.0210, L, 0.0180, L, 0.0025), { y: RAIL_BOT + 0.00125, z: zC });
  put(B.coat, frustumBox(0.0186, L, 0.0210, L, 0.0020), { y: RAIL_BOT + 0.0035, z: zC });
  // The TEETH are bare machined aluminium — a different material from the thing
  // they stand on, which is both true of a real rail and the cheapest way to
  // stop the top of the gun reading as one extruded black mass. Modelled
  // individually only here, where they are 15-20px of screen and hold up.
  const pitch = 0.0102, n = Math.floor(L / pitch);
  const off = (L - n * pitch) / 2;
  for (let i = 0; i < n; i++) {
    put(B.bare, frustumBox(0.0186, 0.0046, 0.0206, 0.0056, 0.0042),
      { y: RAIL_TOP + 0.0021, z: z0 - off - pitch * (i + 0.5) });
  }
}

// ---------------------------------------------------------------- barrel group

function buildBarrel(B, s, rng) {
  const z0 = -0.088, L = s.barrel, gd = z0 - s.gasZ;
  put(B.bare, latheZ([
    [0, 0], [0.017, 0], [0.017, 0.030], [0.0135, 0.036], [0.0135, 0.118],
    [0.0105, 0.126], [0.0105, gd - 0.016], [0.0125, gd - 0.012],
    [0.0125, gd + 0.012], [0.0100, gd + 0.016], [0.0100, L - 0.011],
    [0.0092, L - 0.006], [0.0092, L], [0, L],
  ], 20), { y: BORE_Y, z: z0 }, 0.50);

  // low-profile gas block with its set screws, gas tube running back inside the
  // handguard so it shows through the M-LOK slots
  put(B.bare, cbox(0.023, 0.024, 0.030, 0.0018), { y: BORE_Y + 0.0005, z: s.gasZ }, 'edge');
  put(B.bare, screw(0.0022, 0.0015, rng() * Math.PI), { y: BORE_Y + 0.0135, z: s.gasZ - 0.008, rx: -Math.PI / 2 });
  put(B.bare, screw(0.0022, 0.0015, rng() * Math.PI), { y: BORE_Y + 0.0135, z: s.gasZ + 0.008, rx: -Math.PI / 2 });
  const gtL = Math.abs(s.gasZ) - 0.098;
  put(B.bare, tubeZ(0.0027, 0.0027, gtL, 8), { y: BORE_Y + 0.0098, z: s.gasZ + gtL / 2 }, 0.55);

  const mz = z0 - L;
  put(B.bare, ringZ(0.0092, 0.0126, 0.0022, 16), { y: BORE_Y, z: mz + 0.0022 }, 'rim');
  buildMuzzle(B, s, mz);
}

function buildMuzzle(B, s, mz) {
  const R = s.brakeR, L = s.brakeLen;
  // Collar, then four baffle rings bridged by three spines. The gaps between the
  // rings are the ports: real openings you can see through, not painted slots.
  put(B.bare, flat(latheZ([[0.0092, 0], [R, 0.002], [R, 0.011], [R * 0.86, 0.013]], 16)), { y: BORE_Y, z: mz }, 'rim');
  const n = 4, span = L - 0.021, pitch = span / n;
  for (let i = 0; i < n; i++) {
    put(B.bare, ringZ(0.0072, R * (i === n - 1 ? 0.96 : 1), pitch * 0.32, 16),
      { y: BORE_Y, z: mz - 0.013 - pitch * i - pitch * 0.24 }, 'rim');
  }
  for (let i = 0; i < 3; i++) {
    const a = Math.PI / 2 + (i * 2 * Math.PI) / 3;
    put(B.bare, cbox(0.0055, 0.0055, span + 0.004, 0.0009), {
      x: Math.cos(a) * (R - 0.0026),
      y: BORE_Y + Math.sin(a) * (R - 0.0026),
      z: mz - 0.013 - span / 2,
    });
  }
  // Crowned exit face. Bright right on the crown: gas cuts it clean every shot
  // and it is the one place on a rifle that is always polished.
  put(B.bare, flat(latheZ([
    [0.0072, 0], [R * 0.96, 0], [R * 0.96, 0.006], [0.0094, 0.008], [0.0072, 0.006],
  ], 16)), { y: BORE_Y, z: mz - L + 0.008 }, 1.0);
  put(B.bare, flipFaces(tubeZ(0.0072, 0.0072, 0.020, 12, true)), { y: BORE_Y, z: mz - L + 0.018 }, DARK);
}

// ---------------------------------------------------------------- handguard

/**
 * Rounded-octagon forend whose TOP facet height is solved against the sight
 * cone, not chosen. The section stays fat at the sides where the hand is and
 * flattens over the bore, which is both how a slim modern forend is actually
 * shaped and the only way a 23cm handguard stays out of a 13mm aperture.
 */
function buildHandguard(B, s, rng) {
  const zR = HG_Z, zF = HG_Z - s.hgLen, zC = (zF + zR) / 2, L = s.hgLen;
  const R = s.hgR, tY = 0.0045;
  const cy = BORE_Y;                       // concentric with the bore
  const pitch = 0.0415, slotLen = 0.032, slotW = 0.0075;

  // Solve the top of the section from the forward end, which is the worst case.
  const topY = Math.min(cy + R, Math.max(cy + 0.0115, clearTop(s, zF)));
  const rTop = topY - cy;
  const squash = 1 - rTop / R;
  const rad = (phi) => {
    const up = Math.max(0, Math.sin(phi));
    return R * (1 - squash * up * up);
  };

  // Seven facets around the section; slot counts by position so the silhouette
  // does not get noisy where it is closest to the eye.
  const near = Math.max(1, s.mlok - 1);
  const facets = [[0, s.mlok], [45, 0], [90, -1], [135, 0], [180, s.mlok], [225, near], [270, near], [315, near]];
  for (const [deg, slots] of facets) {
    const phi = (deg * Math.PI) / 180;
    const a0 = phi - Math.PI / 8, a1 = phi + Math.PI / 8;
    const r0 = rad(a0), r1 = rad(a1);
    const p0x = Math.cos(a0) * r0, p0y = Math.sin(a0) * r0;
    const p1x = Math.cos(a1) * r1, p1y = Math.sin(a1) * r1;
    const wide = Math.hypot(p1x - p0x, p1y - p0y) + 0.0008;
    const mx = (p0x + p1x) / 2, my = (p0y + p1y) / 2;
    const nAng = Math.atan2(p1x - p0x, -(p1y - p0y));   // outward normal angle
    const ox = mx - Math.cos(nAng) * tY / 2, oy = my - Math.sin(nAng) * tY / 2;

    if (slots < 0) {
      // Top deck: plain plank, with the rail slots painted onto it.
      put(B.poly, cbox(wide, tY, L - 0.010, 0.0010), { x: ox, y: cy + oy, z: zC, rz: nAng - Math.PI / 2 });
      put(B.poly, slotStrip(wide * 0.92, zF + 0.008, zR - 0.006, 0.0102, 0.0058, 0.86, 0.34),
        { x: ox + Math.cos(nAng) * (tY / 2 + 0.00018), y: cy + oy + Math.sin(nAng) * (tY / 2 + 0.00018) });
      continue;
    }
    const geos = slots > 0
      ? slotPanel(L - 0.010, wide, tY, slots, pitch, slotLen, slotW, 0.0010)
      : [cbox(wide, tY, L - 0.010, 0.0010)];
    putAll(B.poly, geos, { x: ox, y: cy + oy, z: zC, rz: nAng - Math.PI / 2 });
  }
  // Inner shroud, so the slots read as holes with something behind them, and it
  // is dark down there: that is where the grime is on a real forend.
  put(B.poly, tubeZ(R - 0.0075, R - 0.0075, L - 0.014, 14, true), { y: cy, z: zC }, GRIME * 0.7);
  put(B.poly, ringZ(R - 0.0095, rTop + 0.0005, 0.008, 14), { y: cy, z: zF + 0.008 }, 'rim');
  put(B.poly, ringZ(R - 0.0100, R + 0.0018, 0.014, 14), { y: cy, z: zR }, 'rim');

  // barrel nut the handguard clamps to, and the clamp screws
  put(B.coat, latheZ([
    [0.0135, 0], [0.019, 0], [0.019, 0.010], [0.0175, 0.012], [0.0175, 0.020], [0.0135, 0.020],
  ], 14), { y: BORE_Y, z: zR + 0.001 }, 'rim');
  for (let i = 0; i < 4; i++) {
    const a = (Math.PI / 4) + (i * Math.PI) / 2;
    const g = screw(0.0022, 0.0016, rng() * Math.PI);
    g.rotateY(Math.PI / 2);   // axis -Z -> -X, then rz swings it to point inward
    put(B.bare, g, { x: Math.cos(a) * (rad(a) + 0.0006), y: cy + Math.sin(a) * (rad(a) + 0.0006), z: zR - 0.007, rz: a });
  }
  // QD socket at 3 o'clock, forward
  put(B.bare, qdSocket(0.0062), { x: R - 0.001, y: cy, z: zF + 0.032, ry: Math.PI / 2 });
  return { topY, rad };
}

// ---------------------------------------------------------------- lower

function buildLower(B, s, rng) {
  // fire control housing, kept clear of the magwell so the parting lines read
  put(B.coat, cbox(0.039, 0.024, 0.062, 0.0016), { y: -0.038, z: 0.010 });
  put(B.coat, cbox(0.036, 0.014, 0.030, 0.0014), { y: -0.052, z: 0.022 });

  // magwell: four walls plus a flared lip, so it is a box with a hole in it
  put(B.coat, cbox(0.040, 0.030, 0.006, 0.0012), { y: -0.038, z: -0.080 });
  put(B.coat, cbox(0.040, 0.030, 0.006, 0.0012), { y: -0.038, z: -0.010 });
  put(B.coat, cbox(0.006, 0.030, 0.064, 0.0012), { x: -0.017, y: -0.038, z: -0.045 });
  put(B.coat, cbox(0.006, 0.030, 0.064, 0.0012), { x: 0.017, y: -0.038, z: -0.045 });
  // Flared mouth of the well. Bare metal on the lip: every magazine that was
  // ever slammed into it took the coating off, and it is a 4cm bright line the
  // eye reads as "this thing is used".
  put(B.bare, frustumBox(0.042, 0.074, 0.036, 0.066, 0.008), { y: -0.0555, z: -0.045 });
  // panel screws and a stamped-in relief on the right flat, so the lower is not
  // a plain slab from the one angle the player sees most
  put(B.bare, screw(0.0020, 0.0013, rng() * Math.PI), { x: 0.0202, y: -0.044, z: -0.004, ry: Math.PI / 2 });
  put(B.bare, screw(0.0020, 0.0013, rng() * Math.PI), { x: 0.0202, y: -0.044, z: 0.026, ry: Math.PI / 2 });
  put(B.coat, cbox(0.0035, 0.011, 0.030, 0.0008), { x: 0.0192, y: -0.0365, z: 0.014 });
  put(B.coat, cbox(0.0035, 0.011, 0.030, 0.0008), { x: -0.0192, y: -0.0365, z: 0.014 });

  // trigger guard: five chamfered members, angular in the modern way
  put(B.coat, cbox(0.009, 0.030, 0.0075, 0.0011), { y: -0.062, z: -0.004, rx: 0.18 });
  put(B.coat, cbox(0.009, 0.026, 0.0075, 0.0011), { y: -0.064, z: -0.046, rx: -0.30 });
  put(B.coat, cbox(0.009, 0.0075, 0.046, 0.0011), { y: -0.0775, z: -0.026 });
  put(B.coat, cbox(0.009, 0.0075, 0.013, 0.0011), { y: -0.0730, z: -0.048, rx: 0.7 });
  put(B.bare, screw(0.0020, 0.0014, rng() * Math.PI), { x: -0.0048, y: -0.0525, z: -0.052, ry: -Math.PI / 2 });

  // ambidextrous safety selector
  put(B.bare, tubeZ(0.0052, 0.0052, 0.044, 10), { y: -0.032, z: 0.014, ry: Math.PI / 2 }, 0.55);
  put(B.bare, cbox(0.0065, 0.009, 0.026, 0.0008), { x: -0.0235, y: -0.036, z: 0.008, rx: -0.5 });
  put(B.bare, cbox(0.0055, 0.008, 0.020, 0.0008), { x: 0.0230, y: -0.035, z: 0.010, rx: -0.5 });

  // magazine release inside its fence, bolt catch on the left
  put(B.bare, tubeZ(0.0048, 0.0048, 0.008, 10), { x: 0.0215, y: -0.030, z: -0.016, ry: Math.PI / 2 }, 0.85);
  put(B.coat, ringZ(0.0055, 0.0085, 0.004, 10), { x: 0.0195, y: -0.030, z: -0.016, ry: Math.PI / 2 }, 'rim');
  put(B.bare, cbox(0.005, 0.014, 0.030, 0.0009), { x: -0.0215, y: -0.028, z: -0.022 });
  put(B.bare, cbox(0.006, 0.010, 0.009, 0.0009), { x: -0.0225, y: -0.030, z: -0.036 });

  // Receiver extension. Coaxial with the bore, and its top is 45mm under the
  // sight axis so it passes below the bottom edge of the frustum at ADS instead
  // of being sliced open by the near plane.
  put(B.coat, tubeZ(0.0132, 0.0132, s.stockLen + 0.030, 16), { y: BORE_Y, z: 0.056 + (s.stockLen + 0.030) / 2 });
  put(B.bare, latheZ([[0.0132, 0], [0.0158, 0], [0.0158, 0.008], [0.0132, 0.008]], 12), { y: BORE_Y, z: 0.0640 }, 'rim');
  put(B.coat, cbox(0.034, 0.030, 0.004, 0.0010), { y: -0.011, z: 0.0565 });
  put(B.bare, qdSocket(0.0058), { x: -0.015, y: -0.016, z: 0.0585, ry: -Math.PI / 2 });
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

  // Moulded stipple, in the SAME polymer as the grip body — it is a texture cut
  // into the tool, not a bonded pad, and giving it its own darker material was
  // half of why the furniture read as one black mass.
  const studs = [];
  const rows = 9, cols = 6;
  for (const zf of [-1, 1]) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const bow = Math.cos((c - (cols - 1) / 2) * 0.42) * 0.0016;
        const g = new THREE.CylinderGeometry(0.0011, 0.0019, 0.0018, 4, 1);
        g.rotateY(Math.PI / 4);
        g.rotateX(zf * Math.PI / 2);
        paint(g, 0.86);
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
  putAll(B.poly, studs, { y: oy, z: oz, rx: rake });
}

/**
 * Stock. Its entire job at ADS is to not exist: the eye is at rifle-z +0.113 and
 * anything that straddles z = +0.067 is cut open by the near plane and renders
 * as a zero-thickness plate stretching past the silhouette and off into the
 * ground. So it starts at +0.090, wholly behind that line, and its top stays 50mm
 * under the sight axis, below the bottom edge of the frustum during recoil. From
 * the hip it is where it always was: bottom right, half out of frame.
 */
function buildStock(B, s, rng) {
  const z0 = 0.090, L = s.stockLen, zC = z0 + L * 0.50;
  const top = 0.0005;
  put(B.poly, cbox(0.036, 0.030, 0.040, 0.0025), { y: top - 0.016, z: z0 + 0.018 });
  put(B.poly, cbox(0.031, 0.013, L * 0.86, 0.0022), { y: top - 0.007, z: zC });
  put(B.poly, cbox(0.026, 0.024, L * 0.70, 0.0022), { y: -0.030, z: zC + 0.004 });
  put(B.poly, cbox(0.040, 0.062, 0.014, 0.0028), { y: -0.018, z: z0 + L - 0.004 });
  // Comb: a low integral cheek rest, not a riser on posts. The riser was one of
  // the two plates the near plane was shredding.
  put(B.poly, cbox(0.029, 0.008, L * 0.58, 0.0020), { y: top + 0.001, z: zC + 0.006 });
  for (const pz of [zC - 0.024, zC + 0.028]) {
    put(B.bare, screw(0.0024, 0.0016, rng() * Math.PI), { x: 0.0142, y: top - 0.006, z: pz, ry: Math.PI / 2 });
  }
  put(B.poly, cbox(0.012, 0.016, 0.030, 0.0014), { y: -0.044, z: z0 + 0.034, rx: 0.25 });  // adjust lever
  // sling loop through the butt frame, and a QD socket opposite it
  putAll(B.poly, turnY(framedPanel(0.024, 0.028, 0.006, 0, 0, 0.012, 0.015, 0.0012), Math.PI / 2),
    { x: -0.019, y: -0.020, z: z0 + L - 0.016 });
  put(B.bare, qdSocket(0.0062), { x: 0.019, y: -0.020, z: z0 + L - 0.020, ry: Math.PI / 2 });

  // buttpad: four rubber pads with real grooves between them
  for (let i = 0; i < 4; i++) {
    put(B.rub, cbox(0.040, 0.0140, 0.012, 0.0022), { y: 0.006 - i * 0.0165, z: z0 + L + 0.006 });
  }
  put(B.rub, cbox(0.0294, 0.007, L * 0.52, 0.0016), { y: top + 0.0055, z: zC + 0.006 });
}

// ---------------------------------------------------------------- optic
//
// A non-magnifying reflex sight. The tube is a hole: an inward-facing bore wall,
// nothing across the light path but a nearly clear pane and an additive dot. The
// crates downrange are visible THROUGH it, at the same brightness they are
// beside it, because they are the same pixels of the same scene.

function buildOptic(B, s, rng) {
  const R = s.opticR, L = s.opticLen, zC = OPTIC_Z;
  const zR = zC + L / 2, zF = zC - L / 2;
  const Ra = aperture(s);

  // Straight tube with proud bezel rings at each end. A waisted profile reads as
  // a beer keg the moment smooth normals get hold of it, so the body stays
  // cylindrical and the diameter changes happen at hard steps.
  put(B.coat, latheZ([
    [R * 0.82, 0], [R * 0.99, 0.003], [R, 0.007], [R, L - 0.007],
    [R * 0.99, L - 0.003], [R * 0.82, L],
  ], 28), { y: SIGHT_Y, z: zR });
  put(B.coat, ringZ(Ra, R * 1.07, 0.0055, 28), { y: SIGHT_Y, z: zR }, 'rim');
  put(B.coat, ringZ(Ra, R * 1.07, 0.0055, 28), { y: SIGHT_Y, z: zF + 0.0055 }, 'rim');
  // The bore, wound inside out so it is actually visible from the shooter's eye.
  // Matte black down the tube, which is what gives the sight picture its depth.
  put(B.coat, flipFaces(tubeZ(Ra, Ra, L - 0.012, 28, true)), { y: SIGHT_Y, z: zC }, DARK * 0.8);

  // turrets: short, stepped, flat shaded so the facets read as knurling
  const turrets = [[0, 1, -Math.PI / 2, 0], [1, 0, 0, Math.PI / 2]];
  for (const [dx, dy, rx, ry] of turrets) {
    put(B.coat, flat(tubeZ(0.0086, 0.0092, 0.005, 12)),
      { x: dx * (R + 0.002), y: SIGHT_Y + dy * (R + 0.002), z: zC + 0.004, rx, ry });
    put(B.coat, flat(tubeZ(0.0076, 0.0082, 0.007, 12)),
      { x: dx * (R + 0.008), y: SIGHT_Y + dy * (R + 0.008), z: zC + 0.004, rx, ry });
    put(B.bare, cbox(0.0090, 0.0016, 0.0030, 0.0003),
      { x: dx * (R + 0.0115), y: SIGHT_Y + dy * (R + 0.0115), z: zC + 0.004, rz: dx ? Math.PI / 2 : 0 });
  }
  put(B.coat, flat(tubeZ(0.0074, 0.0084, 0.008, 12)), { x: -(R + 0.002), y: SIGHT_Y, z: zC + 0.004, ry: -Math.PI / 2 });

  // mount: saddle, riser down to the rail, clamp jaw, cross bolt, throw lever
  put(B.coat, cbox(0.030, 0.014, 0.036, 0.0016), { y: SIGHT_Y - R - 0.004, z: zC });
  const riserH = Math.max(0.004, SIGHT_Y - R - RAIL_TOP + 0.003);
  put(B.coat, cbox(0.026, riserH, 0.034, 0.0016), { y: SIGHT_Y - R - 0.008 - riserH / 2 + 0.002, z: zC });
  put(B.coat, cbox(0.034, 0.011, 0.030, 0.0014), { y: RAIL_TOP - 0.0015, z: zC });
  put(B.bare, tubeZ(0.0030, 0.0030, 0.036, 8), { y: RAIL_TOP - 0.002, z: zC + 0.009, ry: Math.PI / 2 }, 0.80);
  // throw lever, folded flat against the mount the way it is carried
  put(B.bare, cbox(0.0045, 0.007, 0.020, 0.0007), { x: -0.0185, y: RAIL_TOP - 0.0015, z: zC + 0.004 });
  put(B.bare, flat(tubeZ(0.0042, 0.0042, 0.004, 8)), { x: -0.0185, y: RAIL_TOP - 0.0015, z: zC + 0.013, ry: -Math.PI / 2 }, 'rim');
  for (const bz of [zC - 0.011, zC + 0.011]) {
    put(B.bare, screw(0.0026, 0.0018, rng() * Math.PI), { y: SIGHT_Y - R + 0.004, z: bz, rx: -Math.PI / 2 });
  }
  // Emitter pod, in the wall UNDER the bore rather than inside it. Sitting in
  // the light path it clipped the bottom of the sight picture for no gain.
  put(B.coat, cbox(0.009, 0.006, 0.012, 0.0009), { y: SIGHT_Y - R - 0.0015, z: zF + 0.017 });
  put(B.coat, cbox(0.006, 0.004, 0.005, 0.0006), { y: SIGHT_Y - R - 0.005, z: zF + 0.020 });

  // ------------------------------------------------------------- the glass
  // One pane, and it carries three jobs in its vertex colour: a nearly clear
  // centre (an AR coating passes ~95%, so anything heavier than this is a
  // window you cannot see through), a faint cool cast, and the eyebox vignette
  // rolled into the outer 15% where the tube wall starts to cut the exit pupil.
  put(B.glass, discZ([
    [0.00, 0.42, 0.55, 0.66, 0.050, 0.0011],
    [Ra * 0.55, 0.40, 0.52, 0.64, 0.052, 0.0008],
    [Ra * 0.80, 0.34, 0.44, 0.55, 0.075, 0.0004],
    [Ra * 0.85, 0.22, 0.28, 0.36, 0.135, 0.0003],
    [Ra * 0.93, 0.06, 0.07, 0.09, 0.400, 0.0001],
    [Ra * 1.00, 0.02, 0.02, 0.03, 0.780, 0.0000],
  ], 30), { y: SIGHT_Y, z: zF + 0.009 });

  // ------------------------------------------------------------- the dot
  // Additive, unlit, with the material colour far above 1.0 so the core clips to
  // white through the filmic curve and the falloff lands in the bloom threshold
  // as a red bleed a few pixels wide. A lit disc can only ever be as bright as
  // the sun on it, which is exactly why the old reticle read as a salmon sticker
  // at the same value as the white HUD cross.
  put(B.glow, discZ([
    [0.00000, 1.00, 1.00, 1.00, 1.00, 0],
    [0.00085, 1.00, 0.98, 0.95, 0.98, 0],
    [0.00120, 0.90, 0.42, 0.30, 0.62, 0],
    [0.00175, 0.80, 0.16, 0.09, 0.24, 0],
    [0.00250, 0.72, 0.10, 0.05, 0.075, 0],
    [0.00340, 0.70, 0.08, 0.04, 0.000, 0],
  ], 22), { y: SIGHT_Y, z: zF + 0.0118 });
  // A hint of the emitter's own spill on the floor of the tube, so the dot has a
  // source instead of hanging in space.
  put(B.glow, discZ([
    [0.0000, 0.55, 0.10, 0.05, 0.28, 0],
    [0.0026, 0.50, 0.07, 0.03, 0.00, 0],
  ], 12), { y: SIGHT_Y - Ra + 0.0005, z: zF + 0.016, rx: -Math.PI / 2 });
}

// ---------------------------------------------------------------- hands
//
// Without hands the rifle floats and nothing in frame sets its scale — the hand
// is the only object on screen whose real size the player already knows. Both
// are built in a canonical frame (grip axis along +X, angle measured from +Y
// toward +Z) and placed with an explicit basis, because neither hand sits on a
// coordinate plane.

const PHAL = [[0.00, 0.95], [0.95, 1.85], [1.85, 2.62]];

function handPoint(out, x, ang, r) {
  out.set(x, Math.cos(ang) * r, Math.sin(ang) * r);
  return out;
}

/**
 * Gloved hand wrapped around a cylinder of radius `r`.
 * o: { r, a0, dir, n, span, fw, ft, thumbA, wristA, ox/oy/oz + basis }
 */
function buildHand(B, o) {
  const skin = [], pads = [];
  const n = o.n, span = o.span, dir = o.dir;
  const x0 = -(n - 1) / 2 * span;
  const A = new THREE.Vector3(), Bv = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    const x = x0 + i * span;
    const rel = Math.abs(i - (n - 1) * 0.42);
    const t = o.ft * (1 - rel * 0.055);
    const w = o.fw * (1 - rel * 0.07);
    const reach = 1 - rel * 0.055;
    const base = o.a0 + dir * (i - (n - 1) * 0.5) * 0.055;
    for (let j = 0; j < PHAL.length; j++) {
      const aa = base + dir * PHAL[j][0] * reach;
      const ab = base + dir * PHAL[j][1] * reach;
      const rc = o.r + t * 0.5 + (j === 2 ? -0.0008 : 0);
      handPoint(A, x, aa, rc);
      handPoint(Bv, x, ab, rc);
      const am = (aa + ab) * 0.5;
      skin.push(boneBox(A.x, A.y, A.z, Bv.x, Bv.y, Bv.z, w, t * (1 - j * 0.10),
        0, Math.cos(am), Math.sin(am)));
      if (j === 0) {
        // knuckle armour, the one hard element on a soft glove
        const rk = rc + t * 0.5 + 0.0016;
        skin.push(xf(cbox(w * 0.80, 0.0032, 0.0135, 0.0008),
          { x, y: Math.cos(am) * rk, z: Math.sin(am) * rk, rx: am }));
      }
      if (j === 1) {
        // panel seam across the middle joint: a proud welt, which is what makes
        // a glove read as a glove and not a mitten
        const rs = rc + t * 0.5 + 0.0004;
        pads.push(xf(cbox(w * 0.92, 0.0014, 0.0030, 0.0004),
          { x, y: Math.cos(am) * rs, z: Math.sin(am) * rs, rx: am }));
      }
    }
  }

  // back of the hand and the heel, behind the knuckle line
  const bw = (n - 1) * span + o.fw * 1.25;
  for (const [dA, rOff, th, len] of [[-0.55, 0.009, 0.016, 0.040], [-1.15, 0.010, 0.019, 0.032]]) {
    const am = o.a0 + dir * dA;
    const rc = o.r + rOff;
    skin.push(xf(cbox(bw, th, len, 0.0025), { x: 0, y: Math.cos(am) * rc, z: Math.sin(am) * rc, rx: am }));
  }
  // two longitudinal seams down the back of the hand
  for (const sx of [-0.26, 0.26]) {
    const am = o.a0 + dir * -0.55;
    const rc = o.r + 0.009 + 0.0085;
    pads.push(xf(cbox(0.0018, 0.0012, 0.034, 0.0003),
      { x: bw * sx, y: Math.cos(am) * rc, z: Math.sin(am) * rc, rx: am }));
  }

  // thumb: two segments, lying along the grip axis
  {
    const am = o.thumbA;
    const rc = o.r + 0.0085;
    skin.push(xf(cbox(0.026, 0.0145, 0.0165, 0.0018),
      { x: x0 - 0.004, y: Math.cos(am) * rc, z: Math.sin(am) * rc, rx: am }));
    const am2 = am + dir * 0.30;
    const rc2 = o.r + 0.0088;
    skin.push(xf(cbox(0.020, 0.0130, 0.0150, 0.0018),
      { x: x0 - 0.026, y: Math.cos(am2) * rc2, z: Math.sin(am2) * rc2, rx: am2 }));
  }

  // wrist and cuff. Short on purpose: the forearm is off frame or behind the
  // gun in every pose that matters, and a long one only finds ways to poke
  // through the receiver.
  {
    const am = o.wristA;
    const rc = o.r + 0.020;
    skin.push(xf(cbox(0.050, 0.030, 0.030, 0.0035),
      { x: x0 - 0.030, y: Math.cos(am) * rc, z: Math.sin(am) * rc, rx: am }));
    pads.push(xf(cbox(0.014, 0.033, 0.033, 0.0022),
      { x: x0 - 0.048, y: Math.cos(am) * rc, z: Math.sin(am) * rc, rx: am }));
  }

  place(skin, o.ex, o.ey, o.ez, o.ox, o.oy, o.oz);
  place(pads, o.ex, o.ey, o.ez, o.ox, o.oy, o.oz);
  for (const g of skin) B.glove.push(g);
  for (const g of pads) B.rub.push(g);
}

function buildHands(B, s, hg) {
  // ---- support hand, wrapped over the forend. Fingers come up from the left,
  // pass under and finish on the right; the thumb lies forward over the top.
  const zHand = HG_Z - s.hgLen * 0.60;
  buildHand(B, {
    r: s.hgR + 0.0035, a0: 4.712, dir: -1, n: 4, span: 0.0192, fw: 0.0176, ft: 0.0150,
    thumbA: 5.95, wristA: 4.97,
    ex: _bx.set(0, 0, -1), ey: _by.set(0, 1, 0), ez: _bz.set(1, 0, 0),
    ox: 0, oy: BORE_Y, oz: zHand,
  });

  // ---- firing hand on the pistol grip. Three fingers wrap; the index is built
  // separately because it is on the trigger, not on the grip.
  const rake = -0.34;
  const up = new THREE.Vector3(0, Math.cos(rake), -Math.sin(-rake)).normalize();
  up.set(0, Math.cos(rake), Math.sin(rake) * -1).normalize();
  const right = new THREE.Vector3(1, 0, 0);
  const fwd = new THREE.Vector3().crossVectors(up, right).normalize();
  buildHand(B, {
    r: 0.0215, a0: -0.50, dir: 1, n: 3, span: 0.0196, fw: 0.0178, ft: 0.0150,
    thumbA: 3.05, wristA: -1.55,
    ex: up, ey: right, ez: fwd,
    ox: 0, oy: -0.070, oz: 0.028,
  });

  // Trigger finger, laid in absolute coordinates from the knuckle to the shoe.
  const pts = [
    [0.0165, -0.0455, -0.0035],
    [0.0135, -0.0530, -0.0140],
    [0.0065, -0.0605, -0.0175],
    [0.0000, -0.0645, -0.0160],
  ];
  for (let i = 0; i < 3; i++) {
    const a = pts[i], b = pts[i + 1];
    B.glove.push(boneBox(a[0], a[1], a[2], b[0], b[1], b[2],
      0.0165 - i * 0.001, 0.0148 - i * 0.0014, 0.35, 0.62, -0.70));
  }
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
  ], 16), { y: BORE_Y, z: -0.005 }, 0.72);
  put(B.bolt, cbox(0.010, 0.009, 0.022, 0.0010), { y: BORE_Y + 0.0165, z: -0.019 });  // gas key
  put(B.bolt, tubeZ(0.0032, 0.0032, 0.005, 8), { y: BORE_Y + 0.0140, z: -0.051, rx: -Math.PI / 2 });
  put(B.bolt, flat(latheZ([[0.0035, 0], [0.0095, 0], [0.0095, 0.006], [0.0035, 0.006]], 12)),
    { y: BORE_Y, z: -0.085 }, 'rim');
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

/**
 * Non-reciprocating side charging handle. It used to be a T bar on the tail of
 * the receiver, which at ADS is five centimetres from the eyeball and inside the
 * near plane: it could only ever render as a wide, paper-thin plate sticking out
 * past both sides of the gun. On the left flank it is 25cm away, visible, and
 * unmistakably a charging handle.
 */
function buildCharging(B) {
  const x = -0.0250, y = 0.0033, z = -0.030;
  put(B.chg, cbox(0.011, 0.009, 0.058, 0.0012), { x: x + 0.002, y, z });          // shaft in the track
  put(B.chg, cbox(0.014, 0.020, 0.020, 0.0016), { x: x - 0.006, y, z: z - 0.020, rz: 0.10 });  // paddle
  put(B.chg, cbox(0.008, 0.026, 0.013, 0.0014), { x: x - 0.013, y: y + 0.002, z: z - 0.020, rz: 0.22 });
  for (let i = 0; i < 4; i++) {
    put(B.chg, cbox(0.0035, 0.0018, 0.0022, 0.0004), { x: x - 0.0175, y: y + 0.009 - i * 0.0055, z: z - 0.020, rz: 0.22 });
  }
  put(B.chg, cbox(0.010, 0.007, 0.012, 0.0010), { x: x + 0.002, y, z: z + 0.030 });  // rear stop
}

// ---------------------------------------------------------------- assembly

function meshFrom(list, mat, uvScale, name, seams) {
  if (!list.length) return null;
  let g = weld(list);
  if (seams) applySeams(g, seams);
  g = boxUv(g, uvScale);
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
function pivotFrom(list, px, py, pz, mat, uvScale, name, seams) {
  const o = new THREE.Object3D();
  o.name = name;
  o.position.set(px, py, pz);
  const m = meshFrom(list, mat, uvScale, name + ':geo', seams);
  if (m) {
    m.geometry.translate(-px, -py, -pz);
    m.geometry.computeBoundingSphere();
    o.add(m);
  }
  return o;
}

/**
 * Materials come from the shared library, then are cloned so the vertex mask can
 * be switched on without mutating anything another module might ask for. The
 * library's own clone() carries the surface-shader injection across, which is
 * the only reason cloning is safe here.
 */
function vc(m) {
  const c = m.clone();
  c.vertexColors = true;
  return c;
}

export function buildWeapon(id, materials) {
  const key = SPECS[id] ? id : 'ar_vector';
  const s = SPECS[key];
  const rng = makeRng('weapon:' + key);   // only used for fastener clocking

  const B = {
    coat: [], bare: [], poly: [], rub: [], glove: [], glass: [], glow: [],
    bolt: [], mag: [], trig: [], chg: [],
  };

  buildUpper(B, s, rng);
  buildRail(B, s);
  buildBarrel(B, s, rng);
  const hg = buildHandguard(B, s, rng);
  buildLower(B, s, rng);
  buildGrip(B);
  buildStock(B, s, rng);
  buildOptic(B, s, rng);
  buildHands(B, s, hg);
  buildBolt(B);
  buildMag(B, s);
  buildTrigger(B);
  buildCharging(B);

  // ------------------------------------------------------------- seams
  // Every place two parts bolt together. Applied to the merged buckets in
  // absolute rifle coordinates, so one entry darkens both sides of the joint.
  const SEAMS = [
    // optic clamp on the rail, and the rail base on the receiver deck
    seam(1, RAIL_TOP, 0.009, 0.58, { z0: -0.045, z1: 0.010, x0: -0.024, x1: 0.024 }),
    seam(1, RAIL_BOT, 0.0055, 0.40, { x0: -0.014, x1: 0.014 }),
    // handguard against the receiver, and its front cap
    seam(2, HG_Z, 0.011, 0.50, { y0: -0.040, y1: 0.030 }),
    seam(2, HG_Z - s.hgLen, 0.008, 0.35, { y0: -0.040, y1: 0.030 }),
    // stock and the receiver extension
    seam(2, 0.0585, 0.013, 0.45, { y0: -0.050, y1: 0.020 }),
    seam(2, 0.092, 0.012, 0.40, { y0: -0.050, y1: 0.020 }),
    // magazine in the well, grip on the lower, trigger guard roots
    seam(1, -0.0205, 0.010, 0.55, { z0: -0.086, z1: -0.006, x0: -0.024, x1: 0.024 }),
    seam(1, -0.049, 0.011, 0.45, { z0: -0.006, z1: 0.044 }),
    seam(1, -0.0525, 0.008, 0.30, { z0: -0.060, z1: 0.006, x0: -0.014, x1: 0.014 }),
    // muzzle device on the barrel shoulder, gas block on the barrel
    seam(2, -0.088 - s.barrel + 0.002, 0.006, 0.45, { y0: -0.030, y1: 0.014 }),
    seam(2, s.gasZ + 0.016, 0.006, 0.35, { y0: -0.030, y1: 0.014 }),
    // hands: contact shadow where the glove meets what it is holding
    seam(2, HG_Z - s.hgLen * 0.60, 0.055, 0.30, { y0: -0.045, y1: 0.020 }),
  ];

  // ------------------------------------------------------------- materials
  // Three responses that cannot be confused for each other at a glance, plus
  // rubber. The vertex mask multiplies DOWN from each of these, so what is
  // requested here is the fully worn value, not the average one.
  //
  // COATED STEEL — receiver, rail base, optic body, forend hardware. A chipped
  // semi-gloss coat over steel: the family carries chips and panel sheen that
  // phosphate does not, which is what separates it from the bare parts even
  // where the two are the same value.
  const mCoat = vc(materials.get('paintedMetal', {
    color: 0x3a4046, roughness: 0.44, metalness: 0.22, envMapIntensity: 1.15,
    wearStrength: 0.5, grimeStrength: 0.8,
  }));
  // BARE MACHINED METAL — rail teeth, muzzle, bolt, fasteners, charging handle.
  // Much lighter, much tighter specular, fully metallic. It is the only bright
  // value on the gun and every one of its instances is somewhere a hand or a
  // mount has actually rubbed.
  const mBare = vc(materials.get('gunmetal', {
    color: 0xb4bac2, roughness: 0.27, metalness: 0.95, envMapIntensity: 1.9,
    wearStrength: 0.35,
  }));
  // MATTE STIPPLED POLYMER — handguard, grip, stock, magazine. Warmer, lighter
  // than the receiver and rough enough to have no highlight at all, so the two
  // separate even in flat light.
  const mPoly = vc(materials.get('polymer', {
    color: 0x6e7060, roughness: 0.80, metalness: 0.0, envMapIntensity: 0.55,
    wearStrength: 0.6,
  }));
  const mRub = vc(materials.get('rubber', {
    color: 0x24241f, roughness: 0.95, metalness: 0, envMapIntensity: 0.4,
  }));
  // Gloves are deliberately light: they are the value reference that tells the
  // eye how dark the gun actually is, and how big it is.
  const mGlove = vc(materials.get('fabric', {
    color: 0x7d7360, roughness: 0.90, metalness: 0, envMapIntensity: 0.7,
  }));
  // The pane: transmissive, barely tinted, and NOT a mirror. envMapIntensity was
  // the other half of the blind optic — a polished disc at 2.2 reflects enough
  // sky to become opaque milk in daylight.
  const mGlass = vc(materials.get('glass', {
    color: 0xa8c4d4, roughness: 0.06, opacity: 1.0, envMapIntensity: 0.55,
    streakStrength: 0, grimeStrength: 0.25,
  }));
  // The reticle. Unlit and additive, because an emitter is not a surface: its
  // brightness must not depend on where the sun is. The colour is far above 1.0
  // so the core clips through the filmic curve and the falloff lands in bloom.
  const mGlow = new THREE.MeshBasicMaterial({
    color: new THREE.Color().setRGB(11.0, 1.35, 0.42, THREE.LinearSRGBColorSpace),
    vertexColors: true,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    fog: false,
  });
  mGlow.name = 'mat:reticle';

  const group = new THREE.Group();
  group.name = 'weapon:' + key;

  // Body root: all static geometry hangs off this, so nudging parts.receiver
  // moves the whole rifle the way a shoulder would.
  const receiver = new THREE.Object3D();
  receiver.name = 'receiver';
  group.add(receiver);
  for (const m of [
    meshFrom(B.coat, mCoat, 22, 'body:coated', SEAMS),
    meshFrom(B.bare, mBare, 30, 'body:bare', SEAMS),
    meshFrom(B.poly, mPoly, 11, 'body:polymer', SEAMS),
    meshFrom(B.rub, mRub, 20, 'body:rubber', SEAMS),
    meshFrom(B.glove, mGlove, 14, 'body:gloves', SEAMS),
    meshFrom(B.glass, mGlass, 4, 'sight:pane', null),
    meshFrom(B.glow, mGlow, 4, 'sight:reticle', null),
  ]) if (m) receiver.add(m);

  // Independently animated parts, each on the pivot that matches its motion:
  // bolt and charging handle slide on +Z, the magazine drops and rotates about
  // its front lip, the trigger rotates about its top pin.
  const bolt = pivotFrom(B.bolt, 0, BORE_Y, -0.085, mBare, 30, 'bolt', SEAMS);
  const mag = pivotFrom(B.mag, 0, -0.020, -0.076, mPoly, 11, 'mag', SEAMS);
  const trigger = pivotFrom(B.trig, 0, -0.048, -0.008, mCoat, 22, 'trigger', SEAMS);
  const charging = pivotFrom(B.chg, 0, 0.0033, -0.030, mBare, 30, 'charging', SEAMS);
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
  const stock = mk('stock', 0, -0.012, 0.090 + s.stockLen * 0.6, receiver);
  const optic = mk('optic', 0, SIGHT_Y, OPTIC_Z, receiver);

  const muzzleTip = mk('muzzleTip', 0, BORE_Y, barrelEnd - s.brakeLen - 0.002, group);
  const ejectPort = mk('ejectPort', 0.025, PORT_Y + 0.002, PORT_Z, group);

  // Two degrees of cant plus a little presence scale, both taken about the sight
  // axis so the reticle does not move a pixel. Roll is the only rotation that
  // leaves the optical axis pointing at the eye, so it is the only one baked in;
  // any yaw or pitch pose has to come from the rig in weaponfx.js, blended out
  // by adsT, or the sight picture goes off-centre.
  const cant = 0.036, k = PRESENCE;
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
