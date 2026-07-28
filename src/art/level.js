// OWNER: agent "level" — the map: geometry, blockout, modular kit, layout.
// CONTRACT:
//   buildLevel(materials, rng) -> {
//     group:        THREE.Group        (added to scene)
//     colliders:    Array<{min:Vec3,max:Vec3}>  AABBs for player/AI collision
//     raycastables: THREE.Object3D[]   meshes hitscan may test against
//     playerStart:  {pos:Vec3, yaw:number}
//     spawnPoints:  Array<{pos:Vec3, tag:string}>
//     bounds:       {min:Vec3, max:Vec3}
//     surfaceOf(mesh) -> 'concrete'|'sand'|'metal'|'wood'|'glass'|'fabric'  (for impact fx/audio)
//   }
//
// ---------------------------------------------------------------------------
// WHY THIS FILE LOOKS LIKE THIS
//
// 1. MODULAR KIT, NOT ONE-OFF GEOMETRY. Everything below is assembled from six
//    primitives — chamfered box, prism, wall-with-openings, slab-with-hole,
//    stair flight, parapet — driven by a seeded stream. The layout data is
//    hand-authored (a map has to be *designed*), the geometry is generated.
//
// 2. EVERY EDGE IS CHAMFERED. A perfectly sharp 90-degree edge is the loudest
//    tell of a blockout: it has exactly one shading value along its whole
//    length. A 4cm chamfer costs 32 extra triangles per box and gives every
//    silhouette edge a thin highlight. It matters more here than usual because
//    the sun sits at 18 degrees behind the map (azimuth 118), so almost every
//    surface facing the hero cameras is in shade and the chamfers are what
//    catch the key light and keep the forms readable.
//
// 3. DRAW CALLS, NOT INSTANCE COUNTS, ARE THE BUDGET. Two shadow cascades mean
//    every shadow-casting mesh is submitted three times a frame, so the unit of
//    cost is *meshes*, not triangles. Static geometry is therefore accumulated
//    straight into one interleaved buffer per material — merging beats
//    instancing whenever the geometry is unique anyway. InstancedMesh is used
//    for the two genuinely repeated elements (sandbags, rubble) where per-
//    instance colour and matrices are the cheaper way to get variation.
//
// 4. HITSCAN USES INVISIBLE PROXIES. Raycasting a 40k-triangle merged mesh is a
//    brute-force triangle loop on the main thread, once per bullet. Instead the
//    same registry that produces the movement AABBs also produces ~190 unit
//    boxes that are never added to the scene (zero draw calls) and are what
//    `raycastables` contains. Bounding-sphere rejection makes a shot ~20us
//    instead of ~1.5ms, and the impact normal comes out axis-aligned, which is
//    what a decal wants anyway.
//
// 5. UVs ARE WORLD-PROJECTED IN METRES. Texel density is then identical on every
//    surface in the map, no piece needs hand-unwrapping, and adjacent kit pieces
//    share a continuous texture flow. Per-building UV offsets stop two buildings
//    of the same size reading as copies.
// ---------------------------------------------------------------------------

import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

// ---------------------------------------------------------------- math scratch
const _e1 = [0, 0, 0], _e2 = [0, 0, 0], _nn = [0, 0, 0], _uv = [0, 0];

function sub3(o, a, b) { o[0] = a[0] - b[0]; o[1] = a[1] - b[1]; o[2] = a[2] - b[2]; }
function cross3(o, a, b) {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
}

/** World-space planar projection by dominant normal axis. Vertical faces keep
 *  world Y as V so textures never come out lying on their side. */
function projectUV(n, p, s, ox, oy, out) {
  const ax = Math.abs(n[0]), ay = Math.abs(n[1]), az = Math.abs(n[2]);
  if (ay >= ax && ay >= az) { out[0] = p[0]; out[1] = p[2]; }
  else if (ax >= az) { out[0] = -p[2] * Math.sign(n[0] || 1); out[1] = p[1]; }
  else { out[0] = p[0] * Math.sign(n[2] || 1); out[1] = p[1]; }
  out[0] = out[0] * s + ox;
  out[1] = out[1] * s + oy;
}

// ---------------------------------------------------------------- batch
/**
 * One material's worth of static geometry, accumulated as raw arrays. Winding
 * is fixed up per triangle against an outward reference so no kit function ever
 * has to think about vertex order — a bug that otherwise shows up as one black
 * facet halfway across the map.
 */
class Batch {
  constructor(name) {
    this.name = name;
    this.p = []; this.n = []; this.t = []; this.c = [];
    this.tris = 0;
  }

  tri(a, b, c, out, uvs, ox, oy, col) {
    sub3(_e1, b, a); sub3(_e2, c, a); cross3(_nn, _e1, _e2);
    const L = Math.hypot(_nn[0], _nn[1], _nn[2]);
    if (L < 1e-11) return;
    _nn[0] /= L; _nn[1] /= L; _nn[2] /= L;
    let v1 = b, v2 = c;
    if (_nn[0] * out[0] + _nn[1] * out[1] + _nn[2] * out[2] < 0) {
      v1 = c; v2 = b;
      _nn[0] = -_nn[0]; _nn[1] = -_nn[1]; _nn[2] = -_nn[2];
    }
    const P = this.p, N = this.n, T = this.t, C = this.c;
    const nx = _nn[0], ny = _nn[1], nz = _nn[2];
    const vs = [a, v1, v2];
    for (let i = 0; i < 3; i++) {
      const v = vs[i];
      P.push(v[0], v[1], v[2]);
      N.push(nx, ny, nz);
      projectUV(_nn, v, uvs, ox, oy, _uv);
      T.push(_uv[0], _uv[1]);
      C.push(col[0], col[1], col[2]);
    }
    this.tris++;
  }

  quad(a, b, c, d, out, uvs, ox, oy, col) {
    this.tri(a, b, c, out, uvs, ox, oy, col);
    this.tri(a, c, d, out, uvs, ox, oy, col);
  }

  geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.p), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(this.n), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(this.t), 2));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.c), 3));
    g.computeBoundingSphere();
    return g;
  }
}

// ---------------------------------------------------------------- primitives

const CORNER = (sx, sy, sz, f) =>
  (((sx > 0 ? 4 : 0) | (sy > 0 ? 2 : 0) | (sz > 0 ? 1 : 0)) * 3) + f;

/**
 * Chamfered box. The eight logical corners can be jittered independently
 * (`warp`), which turns a cuboid into a general hexahedron — walls stop being
 * mathematically flat and the courses of a wall stop being mathematically
 * parallel, which is most of what separates "built" from "extruded".
 */
function chamferBox(B, o) {
  const w = o.w, h = o.h, d = o.d;
  const hw = w * 0.5, hh = h * 0.5, hd = d * 0.5;
  const c = Math.min(o.c === undefined ? 0.045 : o.c, hw * 0.42, hh * 0.42, hd * 0.42);
  const ry = o.ry || 0;
  const cs = Math.cos(ry), sn = Math.sin(ry);
  const warp = o.warp || 0, warpY = o.warpY === undefined ? warp * 0.45 : o.warpY;
  const rng = o.rng, col = o.col, s = o.uvs, ox = o.ox || 0, oy = o.oy || 0;
  const cx = o.x, cy = o.y, cz = o.z;

  const V = new Array(24);
  for (let i = 0; i < 8; i++) {
    const sx = (i & 4) ? 1 : -1, sy = (i & 2) ? 1 : -1, sz = (i & 1) ? 1 : -1;
    const jx = warp ? (rng() - 0.5) * warp : 0;
    const jy = warpY ? (rng() - 0.5) * warpY : 0;
    const jz = warp ? (rng() - 0.5) * warp : 0;
    for (let f = 0; f < 3; f++) {
      let px = sx * hw + jx, py = sy * hh + jy, pz = sz * hd + jz;
      if (f === 0) { py -= sy * c; pz -= sz * c; }
      else if (f === 1) { px -= sx * c; pz -= sz * c; }
      else { px -= sx * c; py -= sy * c; }
      V[i * 3 + f] = [cx + px * cs + pz * sn, cy + py, cz - px * sn + pz * cs];
    }
  }

  const out = [0, 0, 0];
  const emitQuad = (i0, i1, i2, i3) => {
    const a = V[i0], b = V[i1], c2 = V[i2], d2 = V[i3];
    out[0] = (a[0] + b[0] + c2[0] + d2[0]) * 0.25 - cx;
    out[1] = (a[1] + b[1] + c2[1] + d2[1]) * 0.25 - cy;
    out[2] = (a[2] + b[2] + c2[2] + d2[2]) * 0.25 - cz;
    B.quad(a, b, c2, d2, out, s, ox, oy, col);
  };
  const emitTri = (i0, i1, i2) => {
    const a = V[i0], b = V[i1], c2 = V[i2];
    out[0] = (a[0] + b[0] + c2[0]) / 3 - cx;
    out[1] = (a[1] + b[1] + c2[1]) / 3 - cy;
    out[2] = (a[2] + b[2] + c2[2]) / 3 - cz;
    B.tri(a, b, c2, out, s, ox, oy, col);
  };

  // six inset faces
  for (const sx of [1, -1]) emitQuad(CORNER(sx, 1, 1, 0), CORNER(sx, 1, -1, 0), CORNER(sx, -1, -1, 0), CORNER(sx, -1, 1, 0));
  for (const sy of [1, -1]) emitQuad(CORNER(1, sy, 1, 1), CORNER(1, sy, -1, 1), CORNER(-1, sy, -1, 1), CORNER(-1, sy, 1, 1));
  for (const sz of [1, -1]) emitQuad(CORNER(1, 1, sz, 2), CORNER(1, -1, sz, 2), CORNER(-1, -1, sz, 2), CORNER(-1, 1, sz, 2));
  // twelve chamfer strips
  for (const sy of [1, -1]) for (const sz of [1, -1])
    emitQuad(CORNER(-1, sy, sz, 1), CORNER(1, sy, sz, 1), CORNER(1, sy, sz, 2), CORNER(-1, sy, sz, 2));
  for (const sx of [1, -1]) for (const sz of [1, -1])
    emitQuad(CORNER(sx, -1, sz, 0), CORNER(sx, 1, sz, 0), CORNER(sx, 1, sz, 2), CORNER(sx, -1, sz, 2));
  for (const sx of [1, -1]) for (const sy of [1, -1])
    emitQuad(CORNER(sx, sy, -1, 0), CORNER(sx, sy, 1, 0), CORNER(sx, sy, 1, 1), CORNER(sx, sy, -1, 1));
  // eight corner facets
  for (const sx of [1, -1]) for (const sy of [1, -1]) for (const sz of [1, -1])
    emitTri(CORNER(sx, sy, sz, 0), CORNER(sx, sy, sz, 1), CORNER(sx, sy, sz, 2));
}

/** n-gon prism, optionally tapered. Barrels, poles, pipes, rebar, tank legs. */
function prism(B, o) {
  const sides = o.sides || 8;
  const r0 = o.r0, r1 = o.r1 === undefined ? o.r0 : o.r1;
  const y0 = o.y, y1 = o.y + o.h;
  const rot = o.rot || 0;
  const tilt = o.tilt || 0, tiltAxis = o.tiltDir || 0;
  const col = o.col, s = o.uvs, ox = o.ox || 0, oy = o.oy || 0;
  const lo = [], hi = [];
  const shear = Math.tan(tilt) * o.h;
  for (let i = 0; i < sides; i++) {
    const a = rot + (i / sides) * Math.PI * 2;
    const ca = Math.cos(a), sa = Math.sin(a);
    lo.push([o.x + ca * r0, y0, o.z + sa * r0]);
    hi.push([o.x + ca * r1 + Math.cos(tiltAxis) * shear, y1, o.z + sa * r1 + Math.sin(tiltAxis) * shear]);
  }
  const out = [0, 0, 0];
  for (let i = 0; i < sides; i++) {
    const j = (i + 1) % sides;
    out[0] = (lo[i][0] + lo[j][0]) * 0.5 - o.x; out[1] = 0; out[2] = (lo[i][2] + lo[j][2]) * 0.5 - o.z;
    B.quad(lo[i], lo[j], hi[j], hi[i], out, s, ox, oy, col);
  }
  if (o.caps !== false) {
    for (let i = 1; i < sides - 1; i++) {
      B.tri(lo[0], lo[i], lo[i + 1], [0, -1, 0], s, ox, oy, col);
      B.tri(hi[0], hi[i], hi[i + 1], [0, 1, 0], s, ox, oy, col);
    }
  }
}

// ---------------------------------------------------------------- build ctx

const SURFACE_OF_BATCH = {
  plaster: 'concrete', concrete: 'concrete', brick: 'concrete', ground: 'sand',
  road: 'concrete', wood: 'wood', metal: 'metal', fabric: 'fabric', glass: 'glass',
  backdrop: 'concrete',
};

// Texels per metre is (uv scale) x (texture repeat); repeat stays 1 everywhere so
// density is controlled here, in one place, in world units.
const UVS = {
  plaster: 0.5, concrete: 0.5, brick: 0.62, ground: 0.34, road: 0.4,
  wood: 0.85, metal: 0.9, fabric: 1.1, glass: 0.5, backdrop: 0.3,
};

class Ctx {
  constructor(rng) {
    this.rng = rng;
    this.B = {};
    for (const k of Object.keys(SURFACE_OF_BATCH)) this.B[k] = new Batch(k);
    this.solids = [];      // {min,max,rot,surface,collide}
    this.tint = [1, 1, 1];
    this.ox = 0; this.oy = 0;
  }

  /** Per-building UV offset + albedo tint. Two identical footprints never read
   *  as the same building because their texture phase and bleach differ. */
  palette(tint, ox, oy) { this.tint = tint; this.ox = ox; this.oy = oy; }

  jitterTint(amount) {
    const r = this.rng, t = this.tint;
    const k = 1 + (r() - 0.5) * amount;
    return [t[0] * k, t[1] * k * (1 + (r() - 0.5) * amount * 0.35), t[2] * k * (1 + (r() - 0.5) * amount * 0.5)];
  }

  box(bn, o) {
    const B = this.B[bn];
    chamferBox(B, {
      rng: this.rng, uvs: o.uvs || UVS[bn],
      ox: o.ox === undefined ? this.ox : o.ox,
      oy: o.oy === undefined ? this.oy : o.oy,
      col: o.col || this.jitterTint(o.vary === undefined ? 0.07 : o.vary),
      ...o,
    });
  }

  /** Axis-aligned box given by extents rather than centre+size. */
  ext(bn, x0, x1, y0, y1, z0, z1, o = {}) {
    this.box(bn, {
      x: (x0 + x1) / 2, y: (y0 + y1) / 2, z: (z0 + z1) / 2,
      w: x1 - x0, h: y1 - y0, d: z1 - z0, ...o,
    });
  }

  prism(bn, o) {
    prism(this.B[bn], {
      uvs: o.uvs || UVS[bn],
      ox: o.ox === undefined ? this.ox : o.ox,
      oy: o.oy === undefined ? this.oy : o.oy,
      col: o.col || this.jitterTint(0.08),
      ...o,
    });
  }

  /** Register a solid: becomes a hitscan proxy and (optionally) a movement AABB. */
  solid(x0, x1, y0, y1, z0, z1, surface, collide = true) {
    this.solids.push({ x0, x1, y0, y1, z0, z1, surface, collide, ry: 0 });
  }

  solidRot(cx, cy, cz, w, h, d, ry, surface, collide = true) {
    this.solids.push({ cx, cy, cz, w, h, d, ry, surface, collide, rot: true });
  }
}

// ---------------------------------------------------------------- kit pieces

/**
 * A wall run with openings. Solid spans are stacked in courses so the surface
 * undulates; every opening gets a lintel above and, if it has a sill, a spandrel
 * below — which is what makes a doorway read as cut *through* mass rather than
 * printed on it. Reveals are inset a little so the opening has depth.
 */
function wall(ctx, o) {
  const { bn = 'plaster', axis, a, b, at, y0 = 0, h, t = 0.42, openings = [] } = o;
  const courseH = o.courseH || 1.05;
  const courses = Math.max(1, Math.round(h / courseH));
  const c = o.c === undefined ? 0.05 : o.c;
  const warp = o.warp === undefined ? 0.03 : o.warp;
  const half = t / 2;

  const spans = [];
  const sorted = openings.slice().sort((p, q) => p.at - q.at);
  let cur = a;
  for (const op of sorted) {
    const l = op.at - op.w / 2, r = op.at + op.w / 2;
    if (l > cur) spans.push([cur, l]);
    cur = Math.max(cur, r);
  }
  if (cur < b) spans.push([cur, b]);

  const put = (u0, u1, v0, v1, extra) => {
    if (u1 - u0 < 0.02 || v1 - v0 < 0.02) return;
    if (axis === 'x') ctx.ext(bn, u0, u1, v0, v1, at - half, at + half, extra);
    else ctx.ext(bn, at - half, at + half, v0, v1, u0, u1, extra);
  };

  for (const [u0, u1] of spans) {
    for (let i = 0; i < courses; i++) {
      const v0 = y0 + (h * i) / courses;
      const v1 = y0 + (h * (i + 1)) / courses;
      // 3cm course overlap: warped boxes must never open a seam you can see sky through
      put(u0 - 0.015, u1 + 0.015, v0 - (i ? 0.03 : 0), v1, { c, warp, warpY: 0 });
    }
    if (o.collide !== false) {
      if (axis === 'x') ctx.solid(u0, u1, y0, y0 + h, at - half, at + half, o.surface || 'concrete');
      else ctx.solid(at - half, at + half, y0, y0 + h, u0, u1, o.surface || 'concrete');
    }
  }

  for (const op of sorted) {
    const sill = op.sill || 0;
    const top = op.top === undefined ? Math.min(h, sill + 2.4) : op.top;
    if (sill > 0.02) {
      put(op.at - op.w / 2, op.at + op.w / 2, y0, y0 + sill, { c, warp, warpY: 0 });
      // projecting sill course — catches the key light as a hard bright line
      const sw = op.w + 0.34;
      if (axis === 'x') ctx.ext(bn, op.at - sw / 2, op.at + sw / 2, y0 + sill - 0.12, y0 + sill, at - half - 0.09, at + half + 0.09, { c: 0.035, warp: 0.008 });
      else ctx.ext(bn, at - half - 0.09, at + half + 0.09, y0 + sill - 0.12, y0 + sill, op.at - sw / 2, op.at + sw / 2, { c: 0.035, warp: 0.008 });
      if (o.collide !== false) {
        if (axis === 'x') ctx.solid(op.at - op.w / 2, op.at + op.w / 2, y0, y0 + sill, at - half, at + half, o.surface || 'concrete');
        else ctx.solid(at - half, at + half, y0, y0 + sill, op.at - op.w / 2, op.at + op.w / 2, o.surface || 'concrete');
      }
    }
    if (top < y0 + h - 0.02) {
      put(op.at - op.w / 2, op.at + op.w / 2, top, y0 + h, { c, warp, warpY: 0 });
      // lintel: a heavier band over the hole, slightly proud of the wall face
      const lw = op.w + 0.5;
      if (axis === 'x') ctx.ext(op.lintel || bn, op.at - lw / 2, op.at + lw / 2, top, top + 0.22, at - half - 0.06, at + half + 0.06, { c: 0.04, warp: 0.01 });
      else ctx.ext(op.lintel || bn, at - half - 0.06, at + half + 0.06, top, top + 0.22, op.at - lw / 2, op.at + lw / 2, { c: 0.04, warp: 0.01 });
      // above-opening mass is a collider too, but starts above head height so
      // the player walks straight through the doorway
      if (o.collide !== false && top > 2.0) {
        if (axis === 'x') ctx.solid(op.at - op.w / 2, op.at + op.w / 2, top, y0 + h, at - half, at + half, o.surface || 'concrete');
        else ctx.solid(at - half, at + half, top, y0 + h, op.at - op.w / 2, op.at + op.w / 2, o.surface || 'concrete');
      }
    }
  }
}

/** Roof slab, optionally with a blown-open hole. */
function slab(ctx, o) {
  const { bn = 'concrete', x0, x1, z0, z1, y0, y1, hole } = o;
  if (!hole) { ctx.ext(bn, x0, x1, y0, y1, z0, z1, { c: 0.05, warp: 0.015, warpY: 0 }); return; }
  const { hx0, hx1, hz0, hz1 } = hole;
  ctx.ext(bn, x0, x1, y0, y1, z0, hz0, { c: 0.05, warp: 0.02, warpY: 0 });
  ctx.ext(bn, x0, x1, y0, y1, hz1, z1, { c: 0.05, warp: 0.02, warpY: 0 });
  ctx.ext(bn, x0, hx0, y0, y1, hz0, hz1, { c: 0.05, warp: 0.02, warpY: 0 });
  ctx.ext(bn, hx1, x1, y0, y1, hz0, hz1, { c: 0.05, warp: 0.02, warpY: 0 });
}

/** Parapet ring with a coping course, optional gaps (stair heads, blown corners). */
function parapet(ctx, o) {
  const { x0, x1, z0, z1, y, h, t = 0.28, bn = 'plaster', gaps = [] } = o;
  const seg = (axis, a, b, at) => {
    let spans = [[a, b]];
    for (const g of gaps) {
      if (g.axis !== axis || Math.abs(g.at - at) > 0.7) continue;
      const next = [];
      for (const [s0, s1] of spans) {
        if (g.b <= s0 || g.a >= s1) { next.push([s0, s1]); continue; }
        if (g.a > s0) next.push([s0, g.a]);
        if (g.b < s1) next.push([g.b, s1]);
      }
      spans = next;
    }
    for (const [s0, s1] of spans) {
      if (s1 - s0 < 0.05) continue;
      if (axis === 'x') {
        ctx.ext(bn, s0, s1, y, y + h - 0.09, at - t / 2, at + t / 2, { c: 0.04, warp: 0.022, warpY: 0 });
        ctx.ext('concrete', s0, s1, y + h - 0.09, y + h, at - t / 2 - 0.06, at + t / 2 + 0.06, { c: 0.03, warp: 0.008 });
      } else {
        ctx.ext(bn, at - t / 2, at + t / 2, y, y + h - 0.09, s0, s1, { c: 0.04, warp: 0.022, warpY: 0 });
        ctx.ext('concrete', at - t / 2 - 0.06, at + t / 2 + 0.06, y + h - 0.09, y + h, s0, s1, { c: 0.03, warp: 0.008 });
      }
    }
  };
  seg('x', x0, x1, z0); seg('x', x0, x1, z1);
  seg('z', z0, z1, x0); seg('z', z0, z1, x1);
}

/** Straight stair flight. `dir` is +1/-1 along the run axis. */
function stairs(ctx, o) {
  const { axis, u0, u1, at, width, y0, y1, bn = 'concrete' } = o;
  const rise = y1 - y0;
  const n = Math.max(2, Math.round(rise / 0.185));
  const run = (u1 - u0) / n;
  for (let i = 0; i < n; i++) {
    const a = u0 + run * i, b = u0 + run * (i + 1) + 0.02;
    const top = y0 + (rise * (i + 1)) / n;
    const lo = Math.max(y0 - 0.15, top - 0.42);
    if (axis === 'x') ctx.ext(bn, Math.min(a, b), Math.max(a, b), lo, top, at - width / 2, at + width / 2, { c: 0.028, warp: 0.012 });
    else ctx.ext(bn, at - width / 2, at + width / 2, lo, top, Math.min(a, b), Math.max(a, b), { c: 0.028, warp: 0.012 });
  }
  // Proxy the flight as a short staircase of boxes rather than one wedge: a
  // single box would eat bullets aimed over the low end and stop the player a
  // metre short of the bottom step.
  const groups = 4;
  for (let g = 0; g < groups; g++) {
    const a0 = u0 + ((u1 - u0) * g) / groups, a1 = u0 + ((u1 - u0) * (g + 1)) / groups;
    const top = y0 + (rise * (g + 1)) / groups;
    if (axis === 'x') ctx.solid(Math.min(a0, a1), Math.max(a0, a1), y0 - 0.1, top, at - width / 2, at + width / 2, 'concrete');
    else ctx.solid(at - width / 2, at + width / 2, y0 - 0.1, top, Math.min(a0, a1), Math.max(a0, a1), 'concrete');
  }
}

/** Concrete traffic barrier — trapezoid profile, the real thing is never a box. */
function barrier(ctx, x, z, ry, len = 2.1) {
  const r = ctx.rng;
  ctx.box('concrete', { x, y: 0.36, z, w: len, h: 0.72, d: 0.56, ry, c: 0.05, warp: 0.02 });
  ctx.box('concrete', { x, y: 0.86, z, w: len - 0.1, h: 0.34, d: 0.3, ry, c: 0.045, warp: 0.02 });
  ctx.box('concrete', { x, y: 0.12, z, w: len + 0.08, h: 0.24, d: 0.7, ry, c: 0.04, warp: 0.015 });
  ctx.solidRot(x, 0.55, z, len + 0.1, 1.1, 0.72, ry, 'concrete');
  if (r() < 0.4) ctx.box('metal', { x: x + Math.sin(ry) * 0.1, y: 1.05, z: z + Math.cos(ry) * 0.1, w: 0.1, h: 0.12, d: 0.1, c: 0.02, warp: 0.01 });
}

/** Sandbag revetment: courses of bags, each rotated and squashed a little.
 *  Emitted as instances — this is the one element with a high enough repeat
 *  count that per-instance matrices beat unique geometry. */
function sandbagRun(ctx, out, x0, z0, x1, z1, courses) {
  const r = ctx.rng;
  const dx = x1 - x0, dz = z1 - z0;
  const len = Math.hypot(dx, dz);
  const ang = Math.atan2(dx, dz);
  const n = Math.max(1, Math.round(len / 0.44));
  for (let cI = 0; cI < courses; cI++) {
    const y = 0.115 + cI * 0.215;
    const inset = cI * 0.035;
    const stagger = (cI % 2) * 0.22;
    for (let i = 0; i < n; i++) {
      const tt = (i * 0.44 + stagger) / len;
      if (tt > 1) continue;
      out.push({
        x: x0 + dx * tt + Math.cos(ang) * (r() - 0.5) * 0.05,
        y: y - inset * 0.1,
        z: z0 + dz * tt + Math.sin(ang) * (r() - 0.5) * 0.05,
        ry: ang + (r() - 0.5) * 0.22,
        rz: (r() - 0.5) * 0.14,
        s: 0.92 + r() * 0.18,
        tint: 0.86 + r() * 0.26,
      });
    }
  }
  const pad = 0.28;
  ctx.solidRot((x0 + x1) / 2, courses * 0.11, (z0 + z1) / 2, len + pad, courses * 0.22, 0.56, ang, 'fabric');
}

/** Rubble field. Chunks are instanced; the big slabs are merged so they can be
 *  arbitrary shapes and still cost nothing extra. */
function rubblePile(ctx, out, x, z, radius, count, height) {
  const r = ctx.rng;
  for (let i = 0; i < count; i++) {
    const a = r() * Math.PI * 2;
    const d = Math.pow(r(), 0.6) * radius;
    const fall = 1 - d / radius;
    out.push({
      x: x + Math.cos(a) * d,
      y: 0.04 + r() * height * fall,
      z: z + Math.sin(a) * d,
      ry: r() * Math.PI * 2,
      rz: (r() - 0.5) * 0.9,
      s: 0.5 + r() * r() * 1.5,
      sy: 0.62 + r() * 0.85,
      tint: 0.8 + r() * 0.34,
    });
  }
  for (let i = 0; i < Math.max(1, count / 8) | 0; i++) {
    const a = r() * Math.PI * 2, d = r() * radius * 0.7;
    ctx.box('concrete', {
      x: x + Math.cos(a) * d, y: 0.1 + r() * 0.14, z: z + Math.sin(a) * d,
      w: 0.7 + r() * 1.3, h: 0.16 + r() * 0.12, d: 0.5 + r() * 1.1,
      ry: r() * 3.14, c: 0.05, warp: 0.16,
    });
  }
  ctx.solid(x - radius * 0.7, x + radius * 0.7, 0, Math.max(0.5, height * 0.8), z - radius * 0.7, z + radius * 0.7, 'concrete', height > 0.6);
}

/** Reinforcing bar bursting out of broken concrete. Bent, never straight. */
function rebar(ctx, x, y, z, len, dirX, dirZ, bend) {
  const r = ctx.rng;
  const seg = 3;
  let px = x, py = y, pz = z;
  let ax = dirX, az = dirZ, ay = 0.55 + r() * 0.5;
  for (let i = 0; i < seg; i++) {
    const l = len / seg;
    const nx = px + ax * l, ny = py + ay * l, nz = pz + az * l;
    // prism() shears a vertical prism, so its height must be the VERTICAL rise
    // and the tilt the angle off vertical — feeding it the 3D length instead
    // over-shears every near-horizontal bar into a ribbon.
    const dy = Math.max(0.04, ny - py);
    const horiz = Math.hypot(nx - px, nz - pz);
    ctx.prism('metal', {
      x: px, y: py, z: pz,
      r0: 0.016, r1: 0.014, h: dy, sides: 5, caps: false,
      tilt: Math.atan2(horiz, dy),
      tiltDir: Math.atan2(nz - pz, nx - px),
      col: [0.6, 0.44, 0.34],
    });
    px = nx; py = py + dy; pz = nz;
    ax += (r() - 0.5) * bend; az += (r() - 0.5) * bend; ay = Math.max(0.25, ay - 0.12 - r() * 0.2);
  }
}

/** A section of wall blown open: jagged courses, exposed rebar, spill of rubble. */
function brokenWall(ctx, o) {
  const { axis, a, b, at, h, t = 0.5, rubble } = o;
  const r = ctx.rng;
  const n = Math.max(3, Math.round((b - a) / 0.55));
  for (let i = 0; i < n; i++) {
    const u0 = a + ((b - a) * i) / n, u1 = a + ((b - a) * (i + 1)) / n + 0.03;
    const tt = i / (n - 1);
    // a ragged parabola: tall at the ends, chewed away in the middle
    const prof = Math.pow(Math.abs(tt * 2 - 1), 1.5);
    const hh = Math.max(0.24, h * (0.12 + prof * 0.88) + (r() - 0.5) * 0.4);
    const courses = Math.max(1, Math.round(hh / 0.5));
    for (let cI = 0; cI < courses; cI++) {
      const v0 = (hh * cI) / courses, v1 = (hh * (cI + 1)) / courses;
      const bite = cI === courses - 1 ? (r() * 0.16) : 0;
      if (axis === 'x') ctx.ext('brick', u0 + bite, u1 - bite, v0 - (cI ? 0.03 : 0), v1, at - t / 2, at + t / 2, { c: 0.05, warp: 0.07, warpY: 0.03 });
      else ctx.ext('brick', at - t / 2, at + t / 2, v0 - (cI ? 0.03 : 0), v1, u0 + bite, u1 - bite, { c: 0.05, warp: 0.07, warpY: 0.03 });
    }
    if (axis === 'x') ctx.solid(u0, u1, 0, hh, at - t / 2, at + t / 2, 'concrete', hh > 0.45);
    else ctx.solid(at - t / 2, at + t / 2, 0, hh, u0, u1, 'concrete', hh > 0.45);
    if (r() < 0.5 && hh > 0.5) {
      const ux = axis === 'x' ? (u0 + u1) / 2 : at;
      const uz = axis === 'x' ? at : (u0 + u1) / 2;
      rebar(ctx, ux + (r() - 0.5) * 0.2, hh - 0.1, uz + (r() - 0.5) * 0.2, 0.5 + r() * 0.7,
        (r() - 0.5) * 0.7, (r() - 0.5) * 0.7, 0.5);
    }
  }
  if (rubble) rubblePile(ctx, rubble, (a + b) / 2, at, 2.6, 26, 0.5);
}

/** Burnt-out utility truck. Generic fictional shape — no maker, no model. */
function vehicleShell(ctx, x, z, ry) {
  const r = ctx.rng;
  const burnt = [0.3, 0.28, 0.27];
  const rust = [0.58, 0.47, 0.4];
  const put = (o) => ctx.box('metal', { ry, ...o, col: o.col || burnt });
  const rot = (lx, lz) => [x + lx * Math.cos(ry) + lz * Math.sin(ry), z - lx * Math.sin(ry) + lz * Math.cos(ry)];

  // chassis rails + bed
  put({ x, y: 0.52, z, w: 2.05, h: 0.2, d: 5.0, c: 0.04, warp: 0.02 });
  put({ x: rot(0, 1.55)[0], y: 0.95, z: rot(0, 1.55)[1], w: 2.0, h: 0.7, d: 1.9, c: 0.06, warp: 0.05 });      // cab lower
  put({ x: rot(0, 1.62)[0], y: 1.62, z: rot(0, 1.62)[1], w: 1.86, h: 0.66, d: 1.5, c: 0.07, warp: 0.06 });    // cab upper, buckled
  put({ x: rot(0, 2.55)[0], y: 0.92, z: rot(0, 2.55)[1], w: 1.98, h: 0.62, d: 0.66, c: 0.06, warp: 0.04 });   // bonnet
  put({ x: rot(0, 2.92)[0], y: 0.78, z: rot(0, 2.92)[1], w: 2.06, h: 0.34, d: 0.22, c: 0.04, col: rust });    // bumper
  // load bed sides, one panel torn away
  put({ x: rot(-0.95, -0.9)[0], y: 1.0, z: rot(-0.95, -0.9)[1], w: 0.12, h: 0.72, d: 3.0, c: 0.03, warp: 0.05, col: rust });
  put({ x: rot(0.95, -1.6)[0], y: 0.98, z: rot(0.95, -1.6)[1], w: 0.12, h: 0.66, d: 1.5, c: 0.03, warp: 0.07, col: rust });
  put({ x: rot(0, -2.42)[0], y: 0.94, z: rot(0, -2.42)[1], w: 1.9, h: 0.6, d: 0.12, c: 0.03, warp: 0.05, col: rust });
  put({ x: rot(0, -0.9)[0], y: 0.66, z: rot(0, -0.9)[1], w: 1.9, h: 0.12, d: 3.1, c: 0.03, warp: 0.02 });
  // roll bar over the bed — reads instantly at silhouette distance
  for (const s of [-1, 1]) {
    const p = rot(s * 0.86, -0.55);
    ctx.prism('metal', { x: p[0], y: 1.0, z: p[1], r0: 0.055, h: 1.05, sides: 6, col: rust });
  }
  const pl = rot(-0.86, -0.55), pr = rot(0.86, -0.55);
  ctx.box('metal', {
    x: (pl[0] + pr[0]) / 2, y: 2.05, z: (pl[1] + pr[1]) / 2,
    w: 1.86, h: 0.1, d: 0.1, ry, c: 0.02, col: rust,
  });
  // wheels: two burnt to the rim, two flat
  const axles = [[-1.55], [1.55]];
  for (const [az] of axles) {
    for (const s of [-1, 1]) {
      const p = rot(s * 1.0, az);
      const flat = r() < 0.5;
      ctx.prism('metal', {
        x: p[0], y: flat ? 0.06 : 0.02, z: p[1], r0: flat ? 0.42 : 0.46, h: flat ? 0.42 : 0.5,
        sides: 9, rot: r() * 1.2, col: flat ? [0.3, 0.29, 0.29] : [0.34, 0.32, 0.31],
      });
    }
  }
  ctx.solidRot(x, 1.0, z, 2.3, 2.0, 5.2, ry, 'metal');
}

/** Bare dead tree. Cheap, and the best silhouette per triangle in the map. */
function deadTree(ctx, x, z, scale) {
  const r = ctx.rng;
  const col = [0.5, 0.42, 0.34];
  ctx.prism('wood', { x, y: 0, z, r0: 0.19 * scale, r1: 0.12 * scale, h: 1.9 * scale, sides: 7, col });
  const limbs = 5;
  for (let i = 0; i < limbs; i++) {
    const a = (i / limbs) * Math.PI * 2 + r() * 0.7;
    const y0 = (1.2 + r() * 0.7) * scale;
    const l = (1.1 + r() * 0.9) * scale;
    ctx.prism('wood', {
      x: x + Math.cos(a) * 0.1, y: y0, z: z + Math.sin(a) * 0.1,
      r0: 0.09 * scale, r1: 0.035 * scale, h: l, sides: 5, col,
      tilt: 0.5 + r() * 0.45, tiltDir: a,
    });
    if (r() < 0.7) {
      const bx = x + Math.cos(a) * l * 0.55, bz = z + Math.sin(a) * l * 0.55;
      ctx.prism('wood', {
        x: bx, y: y0 + l * 0.75, z: bz, r0: 0.04 * scale, r1: 0.015 * scale,
        h: 0.7 * scale, sides: 4, col, tilt: 0.35 + r() * 0.5, tiltDir: a + (r() - 0.5) * 1.6,
      });
    }
  }
  ctx.solid(x - 0.3, x + 0.3, 0, 2 * scale, z - 0.3, z + 0.3, 'wood');
}

/** Rooftop water tank on a welded frame — the map's tallest read against sky. */
function waterTank(ctx, x, y, z) {
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    ctx.box('metal', { x: x + sx * 0.72, y: y + 0.5, z: z + sz * 0.72, w: 0.11, h: 1.0, d: 0.11, c: 0.02, col: [0.7, 0.5, 0.38] });
  }
  ctx.box('metal', { x, y: y + 1.02, z, w: 1.75, h: 0.09, d: 1.75, c: 0.03, col: [0.6, 0.5, 0.44] });
  ctx.prism('metal', { x, y: y + 1.06, z, r0: 0.82, h: 1.35, sides: 12, col: [0.62, 0.6, 0.55] });
  ctx.prism('metal', { x, y: y + 2.41, z, r0: 0.42, h: 0.14, sides: 8, col: [0.62, 0.5, 0.42] });
  ctx.prism('metal', { x: x + 0.6, y: y - 0.1, z: z + 0.3, r0: 0.045, h: 1.2, sides: 5, col: [0.6, 0.48, 0.4] });
  ctx.solid(x - 0.9, x + 0.9, y, y + 2.5, z - 0.9, z + 0.9, 'metal');
}

/** Awning: fabric over a timber frame, sagging between posts. */
function awning(ctx, o) {
  const { x0, x1, z0, z1, y } = o;
  const r = ctx.rng;
  for (const [px, pz] of [[x0 + 0.1, z1 - 0.1], [x1 - 0.1, z1 - 0.1]]) {
    ctx.prism('wood', { x: px, y: 0, z: pz, r0: 0.075, r1: 0.065, h: y, sides: 6, col: [0.72, 0.6, 0.44] });
    ctx.solid(px - 0.12, px + 0.12, 0, y, pz - 0.12, pz + 0.12, 'wood');
  }
  ctx.ext('wood', x0, x1, y, y + 0.11, z1 - 0.16, z1, { c: 0.025, warp: 0.01 });
  ctx.ext('wood', x0, x1, y + 0.22, y + 0.33, z0, z0 + 0.16, { c: 0.025, warp: 0.01 });
  const n = 5;
  for (let i = 0; i < n; i++) {
    const u0 = x0 + ((x1 - x0) * i) / n, u1 = x0 + ((x1 - x0) * (i + 1)) / n + 0.02;
    const sag = 0.06 + r() * 0.09;
    ctx.ext('fabric', u0, u1, y + 0.2 - sag, y + 0.28 - sag, z0, z1, { c: 0.02, warp: 0.05, warpY: 0.05 });
  }
}

/** Timber crate. Frame members proud of the panels so the edges catch light. */
function crateGeom(ctx, x, y, z, s, ry) {
  const t = ctx.jitterTint(0.1);
  ctx.box('wood', { x, y: y + s / 2, z, w: s, h: s, d: s, ry, c: 0.02, warp: 0.012, col: t });
  const e = s / 2 + 0.012;
  for (const sy of [-1, 1]) for (const sz of [-1, 1])
    ctx.box('wood', { x, y: y + s / 2 + sy * e, z, w: s + 0.02, h: 0.055, d: 0.055, ry, c: 0.012, col: t, ox: ctx.ox + sz });
  for (const sx of [-1, 1]) for (const sy of [-1, 1])
    ctx.box('wood', { x: x + sx * e * Math.cos(ry), y: y + s / 2 + sy * e, z: z - sx * e * Math.sin(ry), w: 0.055, h: 0.055, d: s + 0.02, ry, c: 0.012, col: t });
  ctx.solidRot(x, y + s / 2, z, s + 0.06, s, s + 0.06, ry, 'wood');
}

function crateStack(ctx, x, z, ry, spec) {
  for (const [dx, dy, dz, s, dr] of spec) {
    const wx = x + dx * Math.cos(ry) + dz * Math.sin(ry);
    const wz = z - dx * Math.sin(ry) + dz * Math.cos(ry);
    crateGeom(ctx, wx, dy, wz, s, ry + dr);
  }
}

/** Steel drum: three rolling hoops, dented, some on their side. */
function barrel(ctx, x, z, ry, tipped) {
  const r = ctx.rng;
  const col = r() < 0.5 ? [0.6, 0.48, 0.4] : [0.5, 0.52, 0.5];
  if (!tipped) {
    ctx.prism('metal', { x, y: 0.02, z, r0: 0.29, h: 0.86, sides: 10, rot: ry, col });
    for (const yy of [0.06, 0.42, 0.8]) ctx.prism('metal', { x, y: yy, z, r0: 0.315, h: 0.06, sides: 10, rot: ry, col, caps: false });
    ctx.solid(x - 0.32, x + 0.32, 0, 0.9, z - 0.32, z + 0.32, 'metal');
  } else {
    ctx.box('metal', { x, y: 0.29, z, w: 0.86, h: 0.58, d: 0.58, ry, c: 0.14, warp: 0.05, col });
    ctx.solidRot(x, 0.29, z, 0.9, 0.6, 0.6, ry, 'metal');
  }
}

/** Leaning utility pole with a cross-arm. Reads as human infrastructure. */
function pole(ctx, x, z, h, lean, dir) {
  ctx.prism('wood', { x, y: 0, z, r0: 0.15, r1: 0.11, h, sides: 7, tilt: lean, tiltDir: dir, col: [0.6, 0.52, 0.44] });
  const tx = x + Math.cos(dir) * Math.tan(lean) * h * 0.86;
  const tz = z + Math.sin(dir) * Math.tan(lean) * h * 0.86;
  ctx.box('wood', { x: tx, y: h * 0.9, z: tz, w: 1.25, h: 0.1, d: 0.1, ry: 0.4, c: 0.02, col: [0.62, 0.54, 0.45] });
  ctx.box('wood', { x: tx, y: h * 0.79, z: tz, w: 0.8, h: 0.08, d: 0.08, ry: 0.36, c: 0.02, col: [0.62, 0.54, 0.45] });
  for (const sx of [-1, 1]) ctx.box('wood', { x: tx + sx * 0.5, y: h * 0.94, z: tz, w: 0.07, h: 0.22, d: 0.07, c: 0.015, col: [0.6, 0.52, 0.44] });
  ctx.solid(x - 0.2, x + 0.2, 0, h, z - 0.2, z + 0.2, 'wood');
}

// ---------------------------------------------------------------- ground

/** Value noise on a seeded lattice — deterministic, and cheap enough to
 *  evaluate per ground vertex at build time. */
function makeNoise(rng, n) {
  const g = new Float32Array(n * n);
  for (let i = 0; i < g.length; i++) g[i] = rng() * 2 - 1;
  return (u, v) => {
    const x = ((u % n) + n) % n, y = ((v % n) + n) % n;
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const x1 = (x0 + 1) % n, y1 = (y0 + 1) % n;
    const fx = x - x0, fy = y - y0;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const a = g[y0 * n + x0] * (1 - sx) + g[y0 * n + x1] * sx;
    const b = g[y1 * n + x0] * (1 - sx) + g[y1 * n + x1] * sx;
    return a * (1 - sy) + b * sy;
  };
}

function buildGround(ctx, o) {
  const B = ctx.B.ground;
  const S = o.size, N = o.segs;
  const noise = makeNoise(ctx.rng, 16);
  const h = (x, z) => {
    // dead flat inside the compound: the player walks on y=0 and a bump under a
    // wall footing is a light leak. Outside, let the desert breathe.
    const inside = Math.max(Math.abs(x) / 33, Math.abs(z) / 31);
    const k = THREE.MathUtils.smoothstep(inside, 0.94, 1.5);
    return (noise(x * 0.055, z * 0.055) * 0.55 + noise(x * 0.19, z * 0.19) * 0.16) * k
      + noise(x * 0.42, z * 0.42) * 0.018;
  };
  const col = [1, 1, 1];
  const step = S / N;
  const P = [];
  for (let j = 0; j <= N; j++) {
    const row = [];
    for (let i = 0; i <= N; i++) {
      const x = -S / 2 + i * step, z = -S / 2 + j * step;
      row.push([x, h(x, z), z]);
    }
    P.push(row);
  }
  const up = [0, 1, 0];
  const uvs = UVS.ground;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const a = P[j][i], b = P[j][i + 1], c = P[j + 1][i + 1], d = P[j + 1][i];
      // slight per-quad albedo drift: sand is never one value
      const t = 0.94 + ((i * 7 + j * 13) % 11) * 0.012;
      col[0] = t; col[1] = t * 0.995; col[2] = t * 0.985;
      B.tri(a, b, c, up, uvs, 0, 0, col);
      B.tri(a, c, d, up, uvs, 0, 0, col);
    }
  }
}

/** Flat surfacing quad — streets, thresholds, the courtyard floor. */
function surface(ctx, bn, x0, x1, z0, z1, y, tint) {
  const B = ctx.B[bn];
  const col = tint || ctx.jitterTint(0.05);
  const a = [x0, y, z0], b = [x1, y, z0], c = [x1, y, z1], d = [x0, y, z1];
  B.tri(a, b, c, [0, 1, 0], UVS[bn], ctx.ox, ctx.oy, col);
  B.tri(a, c, d, [0, 1, 0], UVS[bn], ctx.ox, ctx.oy, col);
}

function kerb(ctx, axis, a, b, at, side) {
  const t = [0.86, 0.85, 0.83];
  const n = Math.max(1, Math.round((b - a) / 1.6));
  for (let i = 0; i < n; i++) {
    const u0 = a + ((b - a) * i) / n, u1 = a + ((b - a) * (i + 1)) / n + 0.01;
    if (axis === 'x') ctx.ext('concrete', u0, u1, -0.05, 0.135, at - 0.11 * side, at + 0.11 * side, { c: 0.03, warp: 0.014, col: t });
    else ctx.ext('concrete', at - 0.11 * side, at + 0.11 * side, -0.05, 0.135, u0, u1, { c: 0.03, warp: 0.014, col: t });
  }
}

// ---------------------------------------------------------------- the map

export function buildLevel(materials, rng) {
  const ctx = new Ctx(rng);
  const r = rng;
  const sandbags = [];
  const rubble = [];

  // Palettes: sun-bleached renders over mud brick. Kept close together in hue —
  // a desert block is one material family with age and repair variation, not a
  // colour wheel.
  const PL = {
    bone: [1.06, 1.03, 0.96],
    ochre: [1.02, 0.94, 0.8],
    grey: [0.93, 0.93, 0.92],
    pink: [1.04, 0.95, 0.88],
    dust: [0.97, 0.93, 0.85],
  };

  // =============================================================== ground
  buildGround(ctx, { size: 180, segs: 44 });

  // road + plaza surfacing, 4cm proud of the sand with a kerb line
  ctx.palette(PL.grey, 0, 0);
  surface(ctx, 'road', -4.0, 6.0, -12.0, 7.4, 0.04);          // main street
  surface(ctx, 'road', -17.0, 12.0, -12.0, -6.0, 0.045);      // junction
  surface(ctx, 'road', -17.0, -6.0, -26.0, -12.0, 0.04);      // north street
  surface(ctx, 'road', -17.0, -13.0, -6.0, 7.4, 0.042);       // west alley
  surface(ctx, 'road', -20.0, 20.0, 7.4, 25.6, 0.038);        // plaza
  surface(ctx, 'road', 18.0, 28.0, -6.0, 7.4, 0.04);          // east yard
  kerb(ctx, 'z', -12, 7.4, -4.06, 1);
  kerb(ctx, 'z', -12, 7.4, 6.06, 1);
  kerb(ctx, 'x', -20, 20, 25.5, 1);

  // =============================================================== compound wall
  ctx.palette(PL.ochre, 3.1, 0.7);
  const WALL_H = 3.5, WALL_T = 0.5;
  const perim = (axis, a, b, at) => wall(ctx, {
    bn: 'plaster', axis, a, b, at, h: WALL_H, t: WALL_T, courseH: 1.15, c: 0.055, warp: 0.035,
  });
  perim('x', -30, -4.4, 26);
  perim('x', 4.4, 28, 26);
  perim('z', -26, 26, 28);
  perim('x', -30, -16, -26);
  perim('x', -9, -6, -26);
  perim('z', -26, 26, -30);
  // coping course, unbroken, so the wall reads as one built object
  const coping = (axis, a, b, at) => {
    const n = Math.max(1, Math.round((b - a) / 2.2));
    for (let i = 0; i < n; i++) {
      const u0 = a + ((b - a) * i) / n, u1 = a + ((b - a) * (i + 1)) / n + 0.02;
      if (axis === 'x') ctx.ext('concrete', u0, u1, WALL_H - 0.02, WALL_H + 0.16, at - WALL_T / 2 - 0.08, at + WALL_T / 2 + 0.08, { c: 0.035, warp: 0.02 });
      else ctx.ext('concrete', at - WALL_T / 2 - 0.08, at + WALL_T / 2 + 0.08, WALL_H - 0.02, WALL_H + 0.16, u0, u1, { c: 0.035, warp: 0.02 });
    }
  };
  coping('x', -30, -4.4, 26); coping('x', 4.4, 28, 26);
  coping('z', -26, 26, 28); coping('z', -26, 26, -30);
  coping('x', -30, -16, -26); coping('x', -9, -6, -26);

  // north wall breach — the way out, and the map's best close-up of structure
  brokenWall(ctx, { axis: 'x', a: -16, b: -9, at: -26, h: WALL_H, t: WALL_T, rubble });

  // =============================================================== south gate
  // Foreground frame for the establishing shot: at 4m the pylons sit at the
  // frame edges and the lintel kisses the top of the 80-degree frustum.
  ctx.palette(PL.dust, 11.3, 2.2);
  for (const sx of [-1, 1]) {
    const px = sx * 5.25;
    ctx.ext('plaster', px - 0.85, px + 0.85, 0, 4.2, 25.1, 26.9, { c: 0.07, warp: 0.03, warpY: 0 });
    ctx.ext('plaster', px - 0.78, px + 0.78, 4.2, 5.15, 25.2, 26.8, { c: 0.06, warp: 0.03, warpY: 0 });
    ctx.ext('concrete', px - 0.95, px + 0.95, 5.15, 5.36, 25.0, 27.0, { c: 0.04, warp: 0.012 });
    ctx.solid(px - 0.9, px + 0.9, 0, 5.4, 25.0, 27.0, 'concrete');
  }
  ctx.ext('plaster', -6.1, 6.1, 4.2, 5.05, 25.5, 26.5, { c: 0.06, warp: 0.02, warpY: 0 });
  ctx.ext('concrete', -6.3, 6.3, 5.05, 5.24, 25.4, 26.6, { c: 0.035, warp: 0.01 });
  ctx.solid(-6.1, 6.1, 4.2, 5.3, 25.5, 26.5, 'concrete', false);
  // corbels under the lintel: they break the hard soffit line without hanging
  // anything into the arch, which at 4m would read as a floating slab
  for (let i = 0; i < 4; i++) {
    const x = -4.5 + i * 3.0;
    ctx.ext('concrete', x - 0.28, x + 0.28, 3.94, 4.22, 25.36, 26.64, { c: 0.03, warp: 0.012 });
  }

  // =============================================================== B1 : the shop
  // The only fully enterable building, and the subject of two hero shots. Its
  // roof is holed so a single shaft of sun lands on the interior west wall just
  // above the doorway — that shaft is the whole reason the interior shot works.
  ctx.palette(PL.bone, 0.4, 0.15);
  {
    const x0 = 6, x1 = 18, z0 = -6, z1 = 4, t = 0.45;
    const H = 3.6;                     // interior clear height
    wall(ctx, {
      axis: 'z', a: z0, b: z1, at: x0, h: H + 0.3, t, courseH: 1.2, warp: 0.028,
      openings: [
        { at: -2.2, w: 1.7, sill: 0, top: 2.42 },
        { at: 1.6, w: 1.5, sill: 0.95, top: 2.3 },
      ],
    });
    wall(ctx, {
      axis: 'z', a: z0, b: z1, at: x1, h: H + 0.3, t, courseH: 1.2, warp: 0.028,
      openings: [
        { at: -4.2, w: 1.45, sill: 1.0, top: 2.5 },
        { at: 0.2, w: 1.45, sill: 1.0, top: 2.5 },
        { at: 2.9, w: 1.45, sill: 1.0, top: 2.5 },
      ],
    });
    wall(ctx, {
      axis: 'x', a: x0, b: x1, at: z1, h: H + 0.3, t, courseH: 1.2, warp: 0.028,
      openings: [{ at: 11.0, w: 3.6, sill: 0, top: 2.7, lintel: 'wood' }],
    });
    wall(ctx, {
      axis: 'x', a: x0, b: x1, at: z0, h: H + 0.3, t, courseH: 1.2, warp: 0.028,
      openings: [{ at: 9.0, w: 1.2, sill: 2.0, top: 3.1 }],
    });
    slab(ctx, {
      x0: x0 - 0.2, x1: x1 + 0.2, z0: z0 - 0.2, z1: z1 + 0.2, y0: H + 0.3, y1: H + 0.62,
      hole: { hx0: 7.5, hx1: 10.2, hz0: -5.4, hz1: -3.0 },
    });
    // torn slab lip + rebar around the hole
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * Math.PI * 2;
      rebar(ctx, 8.85 + Math.cos(a) * 1.3, H + 0.35, -4.2 + Math.sin(a) * 1.1, 0.42 + r() * 0.4, Math.cos(a) * 0.5, Math.sin(a) * 0.5, 0.6);
    }
    parapet(ctx, {
      x0: x0 - 0.2, x1: x1 + 0.2, z0: z0 - 0.2, z1: z1 + 0.2, y: H + 0.62, h: 0.86,
      gaps: [{ axis: 'x', at: z1 + 0.2, a: 13.4, b: 15.2 }],
    });
    // rubble on the floor under the hole, and the shaft's landing zone kept clear
    rubblePile(ctx, rubble, 8.9, -4.3, 1.9, 30, 0.42);
    // Interior column. The close material-read camera sits 1.4m off its west
    // face, so it carries that whole frame: chamfered arris down the middle,
    // a blown plaster patch showing the brick core, rebar, and the sunlit east
    // windows raking past it from behind.
    ctx.ext('plaster', 13.9, 14.75, 0, H + 0.32, 0.62, 1.52, { c: 0.06, warp: 0.022, warpY: 0 });
    ctx.solid(13.85, 14.75, 0, H, 0.62, 1.52, 'concrete');
    // Brick core standing proud of the render where the plaster spalled off. It
    // has to protrude, not sit flush: buried inside the plaster box it is simply
    // invisible, and 8cm of relief is what gives the patch a lit chamfer edge.
    ctx.palette(PL.pink, 5.2, 1.1);
    ctx.ext('brick', 13.82, 14.02, 0.72, 1.84, 0.74, 1.44, { c: 0.028, warp: 0.025, warpY: 0.02 });
    ctx.ext('brick', 13.84, 14.02, 1.84, 2.02, 0.86, 1.3, { c: 0.025, warp: 0.028, warpY: 0.02 });
    ctx.ext('brick', 13.85, 14.02, 0.5, 0.72, 0.9, 1.36, { c: 0.025, warp: 0.03, warpY: 0.02 });
    rebar(ctx, 13.9, 1.1, 0.86, 0.34, -0.9, -0.2, 0.4);
    rebar(ctx, 13.9, 1.62, 1.3, 0.28, -0.85, 0.3, 0.4);
    // broken masonry stub under the east windows, cut by the sun beams they throw
    brokenWall(ctx, { axis: 'z', a: -1.0, b: 0.9, at: 16.6, h: 1.35, t: 0.5 });
    ctx.ext('brick', 15.7, 17.5, 0, 0.2, -1.2, 1.1, { c: 0.05, warp: 0.03 });
    rubblePile(ctx, rubble, 16.4, 0.0, 1.3, 16, 0.3);
    ctx.palette(PL.bone, 0.4, 0.15);
    // counter along the north wall — interior needs furniture-scale mass or it
    // reads as a garage
    ctx.ext('wood', 11.4, 16.2, 0, 0.94, -5.5, -4.75, { c: 0.035, warp: 0.012 });
    ctx.ext('wood', 11.3, 16.3, 0.94, 1.02, -5.6, -4.65, { c: 0.025, warp: 0.008 });
    ctx.solid(11.3, 16.3, 0, 1.02, -5.6, -4.65, 'wood');
    // Furniture inside the doorway camera's wedge. An interior with nothing at
    // human scale in the near field gives the exposure balance nothing to read.
    crateStack(ctx, 7.45, 1.1, 0.5, [[0, 0, 0, 0.66, 0], [0.04, 0.66, 0.02, 0.56, 0.35], [0.72, 0, -0.06, 0.58, -0.25]]);
    barrel(ctx, 7.0, -5.0, 0.7, true);
    rubblePile(ctx, rubble, 7.7, -1.3, 1.1, 12, 0.22);
    ctx.ext('concrete', 6.08, 6.52, 0, 0.1, -3.15, -1.25, { c: 0.03, warp: 0.01 });   // door threshold
    // shopfront timber head + shutter box outside
    ctx.ext('wood', 9.0, 13.0, 2.7, 2.94, 3.6, 4.5, { c: 0.03, warp: 0.01 });
    awning(ctx, { x0: 8.6, x1: 13.4, z0: 4.4, z1: 6.5, y: 2.55 });
  }

  // =============================================================== B1b : north-east mass
  ctx.palette(PL.ochre, 6.7, 3.4);
  {
    const x0 = 12, x1 = 18, z0 = -14, z1 = -6, H = 10.4;
    wall(ctx, { axis: 'z', a: z0, b: z1, at: x0, h: H, t: 0.5, courseH: 1.25, warp: 0.032, openings: [
      { at: -8.4, w: 1.2, sill: 1.1, top: 2.4 }, { at: -11.2, w: 1.2, sill: 1.1, top: 2.4 },
      { at: -8.4, w: 1.2, sill: 4.6, top: 5.9 }, { at: -11.2, w: 1.2, sill: 4.6, top: 5.9 },
      { at: -8.4, w: 1.2, sill: 7.7, top: 9.0 }, { at: -11.2, w: 1.2, sill: 7.7, top: 9.0 },
    ] });
    wall(ctx, { axis: 'z', a: z0, b: z1, at: x1, h: H, t: 0.5, courseH: 1.25, warp: 0.032 });
    wall(ctx, { axis: 'x', a: x0, b: x1, at: z0, h: H, t: 0.5, courseH: 1.25, warp: 0.032, openings: [
      { at: 15.0, w: 1.3, sill: 4.6, top: 6.0 },
    ] });
    wall(ctx, { axis: 'x', a: x0, b: x1, at: z1, h: H, t: 0.5, courseH: 1.25, warp: 0.032, openings: [
      { at: 15.0, w: 1.3, sill: 1.2, top: 2.6 }, { at: 15.0, w: 1.3, sill: 4.6, top: 6.0 },
    ] });
    slab(ctx, { x0: x0 - 0.22, x1: x1 + 0.22, z0: z0 - 0.22, z1: z1 + 0.22, y0: H, y1: H + 0.34 });
    parapet(ctx, { x0: x0 - 0.22, x1: x1 + 0.22, z0: z0 - 0.22, z1: z1 + 0.22, y: H + 0.34, h: 1.0 });
    ctx.ext('metal', 13.2, 14.4, H + 0.34, H + 1.05, -13.0, -12.2, { c: 0.04, warp: 0.02, col: [0.62, 0.5, 0.42] });
  }

  // rear yard behind the shop — screened off the junction by a courtyard wall
  // with one gate, so it is a pocket you have to commit to entering
  ctx.palette(PL.dust, 2.2, 5.9);
  wall(ctx, { axis: 'z', a: -12, b: -6.2, at: 6.2, h: 2.65, t: 0.4, courseH: 1.0, warp: 0.04,
    openings: [{ at: -9.2, w: 1.6, sill: 0, top: 2.15 }] });
  coping('z', -12, -6.2, 6.2);

  // =============================================================== B2 : west block + roof
  // Tall enough that its upper storeys clear the shadow the shop throws across
  // the street, so there is one blazing sunlit plane above the dark canyon.
  ctx.palette(PL.pink, 8.9, 1.6);
  {
    const x0 = -13, x1 = -4, z0 = -6, z1 = 7, H = 7.6;
    wall(ctx, { axis: 'z', a: z0, b: 3.6, at: x1, h: H, t: 0.5, courseH: 1.22, warp: 0.03, openings: [
      { at: -4.4, w: 1.35, sill: 1.05, top: 2.45 },
      { at: -2.6, w: 3.0, sill: 0, top: 2.6 },
      { at: 2.0, w: 1.35, sill: 1.05, top: 2.45 },
      { at: -4.4, w: 1.35, sill: 4.3, top: 5.7 },
      { at: 0.5, w: 1.6, sill: 4.2, top: 5.8 },
      { at: 2.0, w: 1.35, sill: 4.3, top: 5.7 },
    ] });
    wall(ctx, { axis: 'z', a: z0, b: z1, at: x0, h: H, t: 0.5, courseH: 1.22, warp: 0.03, openings: [
      { at: -1.5, w: 1.3, sill: 1.1, top: 2.5 }, { at: 3.0, w: 1.3, sill: 1.1, top: 2.5 },
      { at: -1.5, w: 1.3, sill: 4.4, top: 5.8 }, { at: 3.0, w: 1.3, sill: 4.4, top: 5.8 },
    ] });
    wall(ctx, { axis: 'x', a: x0, b: x1, at: z0, h: H, t: 0.5, courseH: 1.22, warp: 0.03, openings: [
      { at: -8.5, w: 1.4, sill: 1.1, top: 2.5 }, { at: -8.5, w: 1.4, sill: 4.4, top: 5.8 },
    ] });
    wall(ctx, { axis: 'x', a: x0, b: -7.4, at: z1, h: H, t: 0.5, courseH: 1.22, warp: 0.03, openings: [
      { at: -10.4, w: 1.4, sill: 4.4, top: 5.8 }, { at: -8.0, w: 1.4, sill: 4.4, top: 5.8 },
    ] });
    // 45-degree splayed corner. Its face is one of the very few large planes in
    // the map that is both sunlit AND visible to the south-facing cameras — it
    // is the bright backdrop the silhouette shot puts the enemy against.
    const cxs = -7.4, czs = 7, cxe = -4, cze = 3.6;
    const mx = (cxs + cxe) / 2, mz = (czs + cze) / 2;
    const clen = Math.hypot(cxe - cxs, cze - czs);
    const cang = Math.atan2(cxe - cxs, cze - czs);
    for (let i = 0; i < 6; i++) {
      const v0 = (H * i) / 6, v1 = (H * (i + 1)) / 6;
      ctx.box('plaster', {
        x: mx, y: (v0 + v1) / 2 - (i ? 0.015 : 0), z: mz,
        w: 0.5, h: v1 - v0 + (i ? 0.03 : 0), d: clen + 0.02,
        ry: -cang, c: 0.06, warp: 0.03, warpY: 0,
      });
    }
    ctx.solidRot(mx, H / 2, mz, 0.6, H, clen, -cang, 'concrete');
    slab(ctx, { x0: x0 - 0.24, x1: x1 + 0.24, z0: z0 - 0.24, z1: z1 + 0.24, y0: H, y1: H + 0.36 });
    parapet(ctx, {
      x0: x0 - 0.24, x1: x1 + 0.24, z0: z0 - 0.24, z1: z1 + 0.24, y: H + 0.36, h: 1.02,
      gaps: [{ axis: 'x', at: z1 + 0.24, a: -8.4, b: -6.6 }],
    });

    // External switchback stairs on the south face — the map's verticality, and
    // a hard diagonal against the block's horizontals. Deliberately set at the
    // WEST end: put at the street corner it would sit dead centre in the
    // mid-distance approach shot and wall the frame off.
    ctx.palette(PL.grey, 4.4, 0.2);
    stairs(ctx, { axis: 'x', u0: -7.0, u1: -12.4, at: 8.9, width: 1.5, y0: 0.04, y1: 3.9 });
    stairs(ctx, { axis: 'x', u0: -12.4, u1: -7.2, at: 7.6, width: 1.4, y0: 3.94, y1: 7.98 });
    ctx.ext('plaster', -13.6, -12.4, 0, 3.94, 6.95, 9.65, { c: 0.055, warp: 0.025, warpY: 0 });
    ctx.solid(-13.6, -12.4, 0, 3.94, 6.95, 9.65, 'concrete');
    // solid masonry spines: these stairs are built, not bolted on, so the
    // underside is mass with a stepped soffit rather than a floating comb
    for (let i = 0; i < 5; i++) {
      const a = -7.0 - (5.4 * i) / 5, b = -7.0 - (5.4 * (i + 1)) / 5;
      const hh = 0.04 + 3.86 * ((i + 0.4) / 5) - 0.44;
      if (hh > 0.12) ctx.ext('plaster', b, a, 0, hh, 8.15, 9.65, { c: 0.05, warp: 0.028, warpY: 0 });
    }
    for (let i = 0; i < 6; i++) {
      const a = -12.4 + (5.2 * i) / 6, b = -12.4 + (5.2 * (i + 1)) / 6;
      const hh = 3.94 + 4.04 * ((i + 0.4) / 6) - 0.46;
      ctx.ext('plaster', a, b, 0, hh, 6.9, 8.3, { c: 0.05, warp: 0.028, warpY: 0 });
    }
    ctx.solid(-12.4, -7.0, 0, 7.5, 6.9, 9.65, 'concrete');
    // balustrade — chunky, so the diagonal still reads at 30m
    for (const [a, b, y0, y1, at] of [[-7.0, -12.4, 0.55, 4.4, 9.62], [-12.4, -7.2, 4.45, 8.5, 6.95]]) {
      const n = 7;
      for (let i = 0; i < n; i++) {
        const u0 = a + ((b - a) * i) / n, u1 = a + ((b - a) * (i + 1)) / n;
        const t0 = y0 + ((y1 - y0) * i) / n, t1 = y0 + ((y1 - y0) * (i + 1)) / n;
        ctx.ext('plaster', Math.min(u0, u1), Math.max(u0, u1) + 0.02, Math.min(t0, t1) - 0.55, Math.max(t0, t1), at - 0.11, at + 0.11, { c: 0.035, warp: 0.015 });
      }
    }

    // roof furniture: tank, vents, a sandbag firing position at the SE corner
    ctx.palette(PL.grey, 1.1, 4.0);
    waterTank(ctx, -10.4, H + 0.36, 0.5);
    ctx.ext('plaster', -7.6, -6.4, H + 0.36, H + 1.5, -3.4, -2.2, { c: 0.05, warp: 0.02 });   // stair head hut
    ctx.ext('concrete', -7.75, -6.25, H + 1.5, H + 1.66, -3.55, -2.05, { c: 0.035, warp: 0.01 });
    for (let i = 0; i < 3; i++) ctx.prism('metal', { x: -5.2 + i * 0.9, y: H + 0.36, z: -4.6, r0: 0.11, h: 0.6 + r() * 0.5, sides: 6, col: [0.62, 0.5, 0.42] });
    const bagsOnRoof = sandbags.length;
    sandbagRun(ctx, sandbags, -6.4, 6.1, -4.3, 6.1, 3);
    sandbagRun(ctx, sandbags, -6.4, 6.1, -6.4, 4.2, 3);
    for (let i = bagsOnRoof; i < sandbags.length; i++) sandbags[i].y += H + 0.36;
  }

  // =============================================================== B3a : courtyard house
  ctx.palette(PL.dust, 2.7, 7.1);
  {
    const x0 = -6, x1 = 8, z0 = -26, z1 = -12, H = 6.2;
    const cx0 = -2.5, cx1 = 4.5, cz0 = -22.5, cz1 = -16;
    wall(ctx, { axis: 'x', a: x0, b: x1, at: z1, h: H, t: 0.5, courseH: 1.24, warp: 0.032, openings: [
      { at: 1.0, w: 2.2, sill: 0, top: 2.65 },
      { at: -3.6, w: 1.35, sill: 1.05, top: 2.4 }, { at: 5.6, w: 1.35, sill: 1.05, top: 2.4 },
      { at: -3.6, w: 1.35, sill: 3.9, top: 5.2 }, { at: 1.0, w: 1.35, sill: 3.9, top: 5.2 }, { at: 5.6, w: 1.35, sill: 3.9, top: 5.2 },
    ] });
    wall(ctx, { axis: 'x', a: x0, b: x1, at: z0, h: H, t: 0.5, courseH: 1.24, warp: 0.032, openings: [
      { at: -2.0, w: 1.3, sill: 1.1, top: 2.5 }, { at: 4.0, w: 1.3, sill: 1.1, top: 2.5 },
    ] });
    wall(ctx, { axis: 'z', a: z0, b: z1, at: x0, h: H, t: 0.5, courseH: 1.24, warp: 0.032, openings: [
      { at: -15.0, w: 1.6, sill: 0, top: 2.35 },
      { at: -19.5, w: 1.3, sill: 1.05, top: 2.45 }, { at: -23.5, w: 1.3, sill: 1.05, top: 2.45 },
      { at: -19.5, w: 1.3, sill: 3.9, top: 5.2 },
    ] });
    wall(ctx, { axis: 'z', a: z0, b: z1, at: x1, h: H, t: 0.5, courseH: 1.24, warp: 0.032, openings: [
      { at: -18.0, w: 1.3, sill: 1.05, top: 2.45 }, { at: -22.5, w: 1.3, sill: 3.9, top: 5.2 },
    ] });
    // courtyard: inner walls, one storey lower so the void reads from outside
    const iH = 3.6;
    wall(ctx, { axis: 'x', a: cx0, b: cx1, at: cz1, h: iH, t: 0.36, courseH: 1.1, warp: 0.03, openings: [
      { at: 1.0, w: 1.7, sill: 0, top: 2.3 }, { at: 3.4, w: 1.0, sill: 1.2, top: 2.3 },
    ] });
    wall(ctx, { axis: 'x', a: cx0, b: cx1, at: cz0, h: iH, t: 0.36, courseH: 1.1, warp: 0.03, openings: [
      { at: 0.0, w: 1.1, sill: 1.2, top: 2.3 },
    ] });
    wall(ctx, { axis: 'z', a: cz0, b: cz1, at: cx0, h: iH, t: 0.36, courseH: 1.1, warp: 0.03, openings: [
      { at: -19.0, w: 1.5, sill: 0, top: 2.3 },
    ] });
    wall(ctx, { axis: 'z', a: cz0, b: cz1, at: cx1, h: iH, t: 0.36, courseH: 1.1, warp: 0.03, openings: [
      { at: -20.5, w: 1.1, sill: 1.2, top: 2.3 },
    ] });
    // roof ring: solid over the rooms, open over the court
    slab(ctx, { x0: x0 - 0.22, x1: x1 + 0.22, z0: z0 - 0.22, z1: cz0, y0: H, y1: H + 0.34 });
    slab(ctx, { x0: x0 - 0.22, x1: x1 + 0.22, z0: cz1, z1: z1 + 0.22, y0: H, y1: H + 0.34 });
    slab(ctx, { x0: x0 - 0.22, x1: cx0, z0: cz0, z1: cz1, y0: H, y1: H + 0.34 });
    slab(ctx, { x0: cx1, x1: x1 + 0.22, z0: cz0, z1: cz1, y0: H, y1: H + 0.34 });
    parapet(ctx, { x0: x0 - 0.22, x1: x1 + 0.22, z0: z0 - 0.22, z1: z1 + 0.22, y: H + 0.34, h: 0.92 });
    parapet(ctx, { x0: cx0, x1: cx1, z0: cz0, z1: cz1, y: iH, h: 0.5, t: 0.24 });
    // courtyard floor, well head, tree, market debris
    ctx.palette(PL.grey, 6.3, 2.4);
    surface(ctx, 'road', cx0, cx1, cz0, cz1, 0.05);
    ctx.prism('concrete', { x: 1.0, y: 0.02, z: -19.4, r0: 0.72, h: 0.62, sides: 10, col: [0.9, 0.88, 0.84] });
    ctx.prism('concrete', { x: 1.0, y: 0.62, z: -19.4, r0: 0.78, h: 0.14, sides: 10, col: [0.94, 0.92, 0.88] });
    ctx.solid(0.2, 1.8, 0, 0.8, -20.2, -18.6, 'concrete');
    deadTree(ctx, -1.0, -21.3, 1.5);
    crateStack(ctx, 3.3, -17.2, 0.5, [[0, 0, 0, 0.62, 0], [0.66, 0, 0.1, 0.58, 0.3], [0.05, 0.62, 0, 0.52, 0.22]]);
    barrel(ctx, 3.6, -21.4, 0.4, false);
    ctx.palette(PL.dust, 2.7, 7.1);
  }

  // =============================================================== B3b : tall north-east block
  ctx.palette(PL.ochre, 9.4, 5.5);
  {
    const x0 = 8, x1 = 20, z0 = -26, z1 = -14, H = 8.4;
    wall(ctx, { axis: 'x', a: x0, b: x1, at: z1, h: H, t: 0.52, courseH: 1.3, warp: 0.03, openings: [
      { at: 10.6, w: 1.4, sill: 1.1, top: 2.6 }, { at: 14.0, w: 1.4, sill: 1.1, top: 2.6 }, { at: 17.4, w: 1.4, sill: 1.1, top: 2.6 },
      { at: 10.6, w: 1.4, sill: 4.3, top: 5.8 }, { at: 14.0, w: 1.4, sill: 4.3, top: 5.8 }, { at: 17.4, w: 1.4, sill: 4.3, top: 5.8 },
    ] });
    wall(ctx, { axis: 'x', a: x0, b: x1, at: z0, h: H, t: 0.52, courseH: 1.3, warp: 0.03, openings: [
      { at: 12.0, w: 1.4, sill: 4.3, top: 5.8 }, { at: 16.0, w: 1.4, sill: 4.3, top: 5.8 },
    ] });
    wall(ctx, { axis: 'z', a: z0, b: z1, at: x0, h: H, t: 0.52, courseH: 1.3, warp: 0.03, openings: [
      { at: -17.0, w: 1.5, sill: 0, top: 2.4 }, { at: -22.0, w: 1.35, sill: 1.1, top: 2.5 },
      { at: -17.0, w: 1.35, sill: 4.3, top: 5.8 }, { at: -22.0, w: 1.35, sill: 4.3, top: 5.8 },
    ] });
    wall(ctx, { axis: 'z', a: z0, b: z1, at: x1, h: H, t: 0.52, courseH: 1.3, warp: 0.03 });
    slab(ctx, { x0: x0 - 0.24, x1: x1 + 0.24, z0: z0 - 0.24, z1: z1 + 0.24, y0: H, y1: H + 0.36 });
    parapet(ctx, { x0: x0 - 0.24, x1: x1 + 0.24, z0: z0 - 0.24, z1: z1 + 0.24, y: H + 0.36, h: 1.05 });
    // roof plant: the highest silhouette on the map, and it is not a box
    ctx.palette(PL.grey, 3.3, 6.2);
    ctx.ext('plaster', 9.4, 11.6, H + 0.36, H + 2.0, -17.4, -15.2, { c: 0.05, warp: 0.02 });
    ctx.ext('concrete', 9.2, 11.8, H + 2.0, H + 2.18, -17.6, -15.0, { c: 0.035, warp: 0.01 });
    // Square stair-head tower. From the gate it sits 12 degrees right of the
    // vista and is the only thing on the map that breaks 12m, so the skyline
    // stops being one flat band.
    ctx.palette(PL.ochre, 8.1, 2.9);
    for (let i = 0; i < 5; i++) {
      const v0 = H + 0.36 + (5.4 * i) / 5, v1 = H + 0.36 + (5.4 * (i + 1)) / 5;
      ctx.ext('plaster', 9.6, 14.2, v0 - (i ? 0.03 : 0), v1, -21.4, -16.8, { c: 0.06, warp: 0.03, warpY: 0 });
    }
    parapet(ctx, { x0: 9.45, x1: 14.35, z0: -21.55, z1: -16.65, y: H + 5.76, h: 1.0, t: 0.28 });
    ctx.ext('plaster', 10.4, 13.4, H + 2.4, H + 3.9, -21.55, -21.3, { c: 0.04, warp: 0.02 });
    ctx.solid(9.6, 14.2, H, H + 6.8, -21.4, -16.8, 'concrete', false);
    ctx.palette(PL.grey, 3.3, 6.2);
    ctx.prism('metal', { x: 16.4, y: H + 0.36, z: -22.6, r0: 0.5, h: 1.5, sides: 10, col: [0.6, 0.55, 0.5] });
    ctx.prism('metal', { x: 18.2, y: H + 0.36, z: -22.4, r0: 0.16, h: 2.4, sides: 6, col: [0.62, 0.5, 0.42] });
    ctx.ext('metal', 12.6, 14.4, H + 0.36, H + 1.0, -23.6, -22.0, { c: 0.04, warp: 0.02, col: [0.62, 0.5, 0.42] });
  }

  // =============================================================== B5 : west terrace, stepped
  {
    const steps = [
      { z0: -24, z1: -14, H: 5.6, pal: PL.dust, o: [3.0, 1.0] },
      { z0: -14, z1: -4, H: 11.0, pal: PL.pink, o: [7.5, 2.6] },
      { z0: -4, z1: 6, H: 4.3, pal: PL.ochre, o: [1.9, 5.4] },
    ];
    for (const s of steps) {
      ctx.palette(s.pal, s.o[0], s.o[1]);
      const x0 = -30, x1 = -17;
      const mid = (s.z0 + s.z1) / 2;
      const ops = [];
      for (let y = 1.1; y + 1.5 < s.H; y += 3.2) {
        ops.push({ at: mid - 3.0, w: 1.35, sill: y, top: y + 1.4 });
        ops.push({ at: mid + 3.0, w: 1.35, sill: y, top: y + 1.4 });
      }
      if (s.H > 5) ops.push({ at: mid, w: 2.4, sill: 0, top: 2.55 });
      wall(ctx, { axis: 'z', a: s.z0, b: s.z1, at: x1, h: s.H, t: 0.5, courseH: 1.25, warp: 0.032, openings: ops });
      wall(ctx, { axis: 'x', a: x0, b: x1, at: s.z0, h: s.H, t: 0.5, courseH: 1.25, warp: 0.032 });
      if (s === steps[steps.length - 1]) wall(ctx, { axis: 'x', a: x0, b: x1, at: s.z1, h: s.H, t: 0.5, courseH: 1.25, warp: 0.032, openings: [
        { at: -22.0, w: 1.35, sill: 1.1, top: 2.5 },
      ] });
      slab(ctx, { x0: x0 - 0.2, x1: x1 + 0.22, z0: s.z0 - 0.2, z1: s.z1 + 0.2, y0: s.H, y1: s.H + 0.34 });
      parapet(ctx, { x0: x0 - 0.2, x1: x1 + 0.22, z0: s.z0 - 0.2, z1: s.z1 + 0.2, y: s.H + 0.34, h: 0.88 });
      ctx.solid(x0, x1, 0, s.H, s.z0, s.z1, 'concrete');
    }
    // the alley bridge: a room slung across the west alley at first-floor level.
    // Kills the alley's sight line and gives it a ceiling to fight under.
    ctx.palette(PL.bone, 5.6, 3.3);
    const by0 = 2.9, by1 = 5.6;
    ctx.ext('plaster', -17.2, -12.8, by0, by0 + 0.3, -1.4, 2.2, { c: 0.05, warp: 0.02 });
    wall(ctx, { axis: 'x', a: -17.2, b: -12.8, at: -1.4, y0: by0 + 0.3, h: by1 - by0 - 0.3, t: 0.34, courseH: 1.1, warp: 0.025, collide: false, openings: [{ at: -15.0, w: 1.2, sill: 0.9, top: 2.0 }] });
    wall(ctx, { axis: 'x', a: -17.2, b: -12.8, at: 2.2, y0: by0 + 0.3, h: by1 - by0 - 0.3, t: 0.34, courseH: 1.1, warp: 0.025, collide: false, openings: [{ at: -15.0, w: 1.2, sill: 0.9, top: 2.0 }] });
    ctx.ext('concrete', -17.3, -12.7, by1, by1 + 0.3, -1.6, 2.4, { c: 0.05, warp: 0.015 });
    for (const px of [-16.6, -13.4]) {
      ctx.prism('wood', { x: px, y: 0, z: 2.05, r0: 0.11, h: by0, sides: 6, col: [0.66, 0.56, 0.45] });
      ctx.solid(px - 0.15, px + 0.15, 0, by0, 1.9, 2.2, 'wood');
    }
  }

  // =============================================================== B6 : east outbuilding
  ctx.palette(PL.grey, 10.2, 8.1);
  {
    const x0 = 21, x1 = 28, z0 = -4, z1 = 4, H = 3.3;
    wall(ctx, { axis: 'z', a: z0, b: z1, at: x0, h: H, t: 0.42, courseH: 1.1, warp: 0.03, openings: [
      { at: 1.0, w: 2.6, sill: 0, top: 2.6 },
    ] });
    wall(ctx, { axis: 'x', a: x0, b: x1, at: z0, h: H, t: 0.42, courseH: 1.1, warp: 0.03 });
    wall(ctx, { axis: 'x', a: x0, b: x1, at: z1, h: H, t: 0.42, courseH: 1.1, warp: 0.03 });
    // corrugated roof: alternating ribs, and it must not be flat like everything else
    const ribs = 16;
    for (let i = 0; i < ribs; i++) {
      const u0 = x0 - 0.3 + ((x1 - x0 + 0.6) * i) / ribs;
      const u1 = x0 - 0.3 + ((x1 - x0 + 0.6) * (i + 1)) / ribs + 0.01;
      ctx.ext('metal', u0, u1, H + (i % 2) * 0.05, H + 0.09 + (i % 2) * 0.05, z0 - 0.35, z1 + 0.35, { c: 0.02, warp: 0.012, col: [0.64, 0.53, 0.45] });
    }
    ctx.solid(x0, x1, 0, H, z0, z1, 'concrete');
  }

  // =============================================================== street furniture
  ctx.palette(PL.grey, 0, 0);

  // plaza: a firing lane broken by staggered hard cover at chest height
  barrier(ctx, -13.6, 17.0, 0.06);
  barrier(ctx, -5.6, 15.4, 0.32);
  barrier(ctx, -7.8, 9.2, 0.92);
  barrier(ctx, 2.4, 16.2, -0.12);
  barrier(ctx, 4.6, 16.6, 0.18, 2.4);
  barrier(ctx, 10.4, 19.2, 1.32);
  barrier(ctx, 12.0, 12.6, 1.51);
  barrier(ctx, -13.4, 12.0, 1.48);
  barrier(ctx, 0.3, 23.1, 1.55, 2.4);
  barrier(ctx, 6.2, 8.6, 0.02);
  barrier(ctx, -3.2, 10.9, 1.53);

  // sandbag emplacements: one covering the gate, one covering the street mouth
  sandbagRun(ctx, sandbags, -11.2, 12.6, -8.4, 12.6, 4);
  sandbagRun(ctx, sandbags, -11.2, 12.6, -11.2, 15.0, 3);
  sandbagRun(ctx, sandbags, 7.4, 20.4, 10.6, 20.4, 4);
  sandbagRun(ctx, sandbags, 10.6, 20.4, 10.6, 22.6, 3);
  sandbagRun(ctx, sandbags, -3.6, 7.0, -1.0, 7.0, 3);
  sandbagRun(ctx, sandbags, 3.4, -7.2, 5.6, -7.2, 3);
  sandbagRun(ctx, sandbags, -15.6, -8.6, -13.6, -8.6, 4);

  crateStack(ctx, 13.6, 14.2, 0.42, [
    [0, 0, 0, 0.78, 0], [0.84, 0, 0.06, 0.72, 0.24], [0.06, 0.78, 0.04, 0.66, 0.42],
    [-0.8, 0, -0.1, 0.6, -0.3],
  ]);
  crateStack(ctx, -16.4, 21.3, 1.1, [
    [0, 0, 0, 0.7, 0], [0, 0.7, 0.02, 0.62, 0.35], [0.76, 0, -0.05, 0.66, -0.2],
  ]);
  crateStack(ctx, -2.6, 2.0, 0.24, [[0, 0, 0, 0.74, 0], [0.06, 0.74, 0.03, 0.6, 0.4]]);
  crateStack(ctx, -8.6, -9.4, 1.9, [[0, 0, 0, 0.68, 0], [0.72, 0, 0.06, 0.62, 0.3]]);

  barrel(ctx, 5.2, 12.9, 0.3, false);
  barrel(ctx, 5.9, 13.6, 1.1, false);
  barrel(ctx, 5.4, 14.4, 0.8, true);
  barrel(ctx, -14.6, -3.0, 0.5, false);
  barrel(ctx, -14.0, -4.1, 0.2, true);
  barrel(ctx, 19.4, 2.2, 0.9, false);
  barrel(ctx, 19.6, 3.4, 0.1, false);
  barrel(ctx, -5.1, -10.6, 1.3, true);
  barrel(ctx, 9.2, -9.1, 0.7, false);

  pole(ctx, -12.6, 22.4, 6.4, 0.055, 2.3);
  pole(ctx, 17.2, 8.9, 5.8, 0.03, 0.7);
  pole(ctx, -15.4, -16.2, 6.0, 0.07, 1.2);
  pole(ctx, 6.4, 20.2, 5.2, 0.09, 3.4);

  // The street wreck is the midground subject of the hipfire/ADS pair and the
  // chokepoint that stops the main street being a straight run.
  vehicleShell(ctx, 2.5, -5.5, -0.12);
  vehicleShell(ctx, -9.5, 18.6, 0.5);

  // rubble: the map's connective tissue — every corner where a wall met a shell
  rubblePile(ctx, rubble, -13.9, 8.2, 2.4, 30, 0.5);
  rubblePile(ctx, rubble, 6.6, 5.4, 2.0, 24, 0.42);
  rubblePile(ctx, rubble, -4.8, -6.6, 2.6, 32, 0.6);
  rubblePile(ctx, rubble, -12.6, -22.4, 3.2, 38, 0.62);
  rubblePile(ctx, rubble, 20.6, -12.8, 2.8, 30, 0.55);
  rubblePile(ctx, rubble, 8.4, 22.4, 2.2, 26, 0.45);
  rubblePile(ctx, rubble, -20.4, 16.4, 2.6, 28, 0.5);
  rubblePile(ctx, rubble, 15.4, -16.6, 2.0, 22, 0.4);

  // free-standing broken walls: hard cover that is not a box, plus rebar reads
  ctx.palette(PL.pink, 4.1, 9.2);
  brokenWall(ctx, { axis: 'x', a: -2.6, b: 2.2, at: 11.4, h: 1.9, t: 0.44, rubble });
  brokenWall(ctx, { axis: 'z', a: -3.4, b: 1.0, at: -9.6, h: 1.7, t: 0.44, rubble });
  brokenWall(ctx, { axis: 'x', a: 14.6, b: 18.4, at: 18.2, h: 2.2, t: 0.46, rubble });
  ctx.palette(PL.grey, 0, 0);

  // low walls: the chest-height cover grid that actually decides firefights
  const lowWall = (axis, a, b, at, h) => {
    const n = Math.max(1, Math.round((b - a) / 1.4));
    for (let i = 0; i < n; i++) {
      const u0 = a + ((b - a) * i) / n, u1 = a + ((b - a) * (i + 1)) / n + 0.02;
      if (axis === 'x') {
        ctx.ext('plaster', u0, u1, 0, h, at - 0.19, at + 0.19, { c: 0.05, warp: 0.03, warpY: 0.01 });
        ctx.ext('concrete', u0, u1, h, h + 0.11, at - 0.25, at + 0.25, { c: 0.03, warp: 0.012 });
      } else {
        ctx.ext('plaster', at - 0.19, at + 0.19, 0, h, u0, u1, { c: 0.05, warp: 0.03, warpY: 0.01 });
        ctx.ext('concrete', at - 0.25, at + 0.25, h, h + 0.11, u0, u1, { c: 0.03, warp: 0.012 });
      }
    }
    if (axis === 'x') ctx.solid(a, b, 0, h + 0.11, at - 0.25, at + 0.25, 'concrete');
    else ctx.solid(at - 0.25, at + 0.25, 0, h + 0.11, a, b, 'concrete');
  };
  lowWall('x', -20.4, -14.2, 7.2, 1.05);
  lowWall('x', 8.2, 15.6, 7.2, 1.1);
  lowWall('z', 14.0, 21.0, 18.6, 1.0);
  lowWall('z', -22.0, -15.4, -6.4, 1.15);
  lowWall('x', -11.0, -6.6, -12.6, 1.05);
  lowWall('z', 9.4, 15.6, 24.4, 0.95);
  lowWall('x', -26.0, -21.0, 20.4, 1.1);

  // =============================================================== ground dressing
  // Open ground with nothing on it is the second loudest tell of a blockout
  // after sharp edges: a real street has a continuous litter of spall, and a
  // desert one has sand piled against every windward face. Both are close to
  // free — the debris rides the existing rubble instancer, the drifts go into
  // the ground batch, which does not cast.
  const OPEN = [
    [-24, 24, 8.5, 25], [-3.4, 5.4, -11, 7], [-16, 11, -11.5, -6.5],
    [-16.4, -6.6, -25, -12], [-16.4, -13.6, -5.5, 6.5], [18.6, 27.4, -5.5, 6.5],
    [6.9, 17.4, -5.2, 3.4], [-2.4, 4.4, -22.4, -16.1], [-28, 28, 27.5, 38],
  ];
  for (const [x0, x1, z0, z1] of OPEN) {
    const area = (x1 - x0) * (z1 - z0);
    const n = Math.round(area * 0.06);
    for (let i = 0; i < n; i++) {
      rubble.push({
        x: x0 + r() * (x1 - x0), y: 0.015 + r() * 0.03, z: z0 + r() * (z1 - z0),
        ry: r() * Math.PI * 2, rz: (r() - 0.5) * 0.5,
        s: 0.16 + r() * r() * 0.42, sy: 0.35 + r() * 0.5,
        tint: 0.82 + r() * 0.3,
      });
    }
  }

  // wind-blown sand banked against wall feet
  const drift = (axis, a, b, at, side, depth) => {
    const n = Math.max(2, Math.round((b - a) / 2.1));
    for (let i = 0; i < n; i++) {
      const u0 = a + ((b - a) * i) / n, u1 = a + ((b - a) * (i + 1)) / n + 0.05;
      const d = depth * (0.45 + r() * 0.85);
      const hgt = 0.09 + r() * 0.17;
      const col = [1.02 + (r() - 0.5) * 0.06, 1.0, 0.96];
      if (axis === 'x') ctx.ext('ground', u0, u1, -0.05, hgt, at, at + d * side, { c: 0.06, warp: 0.12, warpY: 0.05, col });
      else ctx.ext('ground', at, at + d * side, -0.05, hgt, u0, u1, { c: 0.06, warp: 0.12, warpY: 0.05, col });
    }
  };
  drift('x', -29, -5.0, 25.7, 1, 1.4); drift('x', 5.0, 27, 25.7, 1, 1.4);
  drift('x', -29, -5.0, 26.3, -1, 1.2); drift('x', 5.0, 27, 26.3, -1, 1.2);
  drift('z', -25, 25, 27.7, 1, 1.5); drift('z', -25, 25, -29.7, -1, 1.5);
  drift('x', -29, -16.5, -25.7, -1, 1.3); drift('z', -23, 5, -17.3, -1, 1.1);
  drift('z', -6, 3, 6.3, 1, 0.9); drift('z', -13, 3, -4.3, -1, 0.9);
  drift('x', -12.8, -4.3, 7.3, 1, 1.0); drift('x', 6.3, 17.8, 4.3, 1, 0.9);
  drift('z', -25, -14.3, 8.3, 1, 1.1); drift('z', -25, -14.3, 20.3, 1, 1.1);

  // =============================================================== backdrop
  // A ring of distant blocks outside the compound. No shadows, no collision —
  // they exist so the horizon has structure to desaturate into.
  ctx.palette(PL.dust, 0.5, 0.5);
  for (let i = 0; i < 26; i++) {
    const a = (i / 26) * Math.PI * 2 + r() * 0.12;
    const rad = 46 + r() * 30;
    const h = 3.4 + r() * r() * 9.5;
    const x = Math.cos(a) * rad, z = Math.sin(a) * rad;
    if (Math.abs(x) < 34 && Math.abs(z) < 33) continue;
    const w = 5 + r() * 9, d = 5 + r() * 9;
    ctx.box('backdrop', { x, y: h / 2, z, w, h, d, ry: r() * 1.5, c: 0.09, warp: 0.12, warpY: 0.05 });
    if (r() < 0.45) ctx.box('backdrop', { x: x + (r() - 0.5) * w, y: h + 1.1, z: z + (r() - 0.5) * d, w: w * 0.55, h: 2.4, d: d * 0.55, ry: r() * 1.5, c: 0.08, warp: 0.1 });
  }

  // =============================================================== assemble
  const group = new THREE.Group();
  group.name = 'level';
  const raycastables = [];
  const colliders = [];
  const surfaces = new WeakMap();
  const disposables = [];

  const MAT_SPEC = {
    ground: ['sand', { repeat: 1, roughness: 0.97, color: 0xc9b189 }, false, true],
    road: ['concrete', { repeat: 1, roughness: 0.96, color: 0xd8c9a8 }, false, true],
    plaster: ['plaster', { repeat: 1, roughness: 0.9, color: 0xd8c8a9 }, true, true],
    concrete: ['concrete', { repeat: 1, roughness: 0.88, color: 0xa8a49a }, true, true],
    brick: ['brick', { repeat: 1, roughness: 0.93, color: 0x9c7f6b }, true, true],
    wood: ['wood', { repeat: 1, roughness: 0.86, color: 0x9c7a4f }, true, true],
    metal: ['rustMetal', { repeat: 1, roughness: 0.62, metalness: 0.55, color: 0x9a8574 }, true, true],
    fabric: ['fabric', { repeat: 1, roughness: 0.95, color: 0xb3a486 }, true, true],
    glass: ['glass', { repeat: 1, roughness: 0.2, metalness: 0.1, color: 0x5a6a6a }, false, true],
    backdrop: ['plaster', { repeat: 1, roughness: 0.95, color: 0xc7b596 }, false, false],
  };

  const mats = {};
  for (const [key, [name, opts, cast, recv]] of Object.entries(MAT_SPEC)) {
    // Cloned so vertexColors can be enabled without mutating the shared library
    // material another agent may also be holding.
    const m = materials.get(name, opts).clone();
    m.vertexColors = true;
    m.name = 'level_' + key;
    mats[key] = { mat: m, cast, recv };
    disposables.push(m);
  }

  let triangles = 0;
  for (const key of Object.keys(ctx.B)) {
    const b = ctx.B[key];
    if (!b.tris) continue;
    const mesh = new THREE.Mesh(b.geometry(), mats[key].mat);
    mesh.name = 'level_' + key;
    mesh.castShadow = mats[key].cast;
    mesh.receiveShadow = mats[key].recv;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    group.add(mesh);
    triangles += b.tris;
  }

  // ---- instanced props ------------------------------------------------------
  const instanced = [];
  const _m4 = new THREE.Matrix4();
  const _q = new THREE.Quaternion();
  const _e = new THREE.Euler();
  const _v = new THREE.Vector3();
  const _s = new THREE.Vector3();
  const _c = new THREE.Color();

  const makeInstanced = (protoBatch, list, matKey, cast) => {
    if (!list.length) return 0;
    const geo = protoBatch.geometry();
    const mesh = new THREE.InstancedMesh(geo, mats[matKey].mat, list.length);
    mesh.castShadow = cast;
    mesh.receiveShadow = true;
    mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    for (let i = 0; i < list.length; i++) {
      const o = list[i];
      _e.set(o.rz || 0, o.ry || 0, o.rx || 0, 'YXZ');
      _q.setFromEuler(_e);
      _v.set(o.x, o.y, o.z);
      _s.set(o.s, o.s * (o.sy || 1), o.s);
      _m4.compose(_v, _q, _s);
      mesh.setMatrixAt(i, _m4);
      const t = o.tint;
      _c.setRGB(t, t * 0.99, t * 0.97);
      mesh.setColorAt(i, _c);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.frustumCulled = false;   // one AABB over the whole map; culling it is a lie
    group.add(mesh);
    instanced.push(mesh);
    return protoBatch.tris * list.length;
  };

  // sandbag prototype: heavily chamfered, warped, so it reads as a filled bag
  const bagBatch = new Batch('bag');
  chamferBox(bagBatch, {
    x: 0, y: 0, z: 0, w: 0.5, h: 0.21, d: 0.31, c: 0.085, warp: 0.05, warpY: 0.03,
    rng, uvs: UVS.fabric, ox: 0, oy: 0, col: [1, 1, 1],
  });
  triangles += makeInstanced(bagBatch, sandbags, 'fabric', true);

  // Rubble prototype. 380 instances at ~0.3m: one heavily warped chamfered box
  // is the whole silhouette anyone will ever resolve, and a second merged chunk
  // would have cost 51k triangles a frame across the shadow cascades for detail
  // that is sub-pixel past 4m. Variety comes from per-instance rotation, non-
  // uniform scale and colour instead.
  const rubBatch = new Batch('rub');
  {
    const parts = [];
    const b = new Batch('p');
    chamferBox(b, {
      x: 0, y: 0, z: 0, w: 0.34, h: 0.23, d: 0.3,
      c: 0.06, warp: 0.16, warpY: 0.1, ry: 0.4,
      rng, uvs: UVS.concrete, ox: 0, oy: 0, col: [1, 1, 1],
    });
    parts.push(b.geometry());
    const merged = mergeGeometries(parts, false);
    for (const g of parts) g.dispose();
    const p = merged.getAttribute('position'), n = merged.getAttribute('normal');
    const uv = merged.getAttribute('uv'), c = merged.getAttribute('color');
    rubBatch.p = Array.from(p.array); rubBatch.n = Array.from(n.array);
    rubBatch.t = Array.from(uv.array); rubBatch.c = Array.from(c.array);
    rubBatch.tris = p.count / 3;
    merged.dispose();
  }
  triangles += makeInstanced(rubBatch, rubble, 'concrete', true);

  // ---- hitscan proxies + movement AABBs -------------------------------------
  const proxyGeo = new THREE.BoxGeometry(1, 1, 1);
  const proxyMat = new THREE.MeshBasicMaterial();
  disposables.push(proxyGeo, proxyMat);
  const proxyRoot = new THREE.Group();   // never added to the scene: zero draw cost

  const addProxy = (cx, cy, cz, w, h, d, ry, surface) => {
    const m = new THREE.Mesh(proxyGeo, proxyMat);
    m.position.set(cx, cy, cz);
    m.scale.set(Math.max(0.02, w), Math.max(0.02, h), Math.max(0.02, d));
    m.rotation.y = ry || 0;
    proxyRoot.add(m);
    raycastables.push(m);
    surfaces.set(m, surface);
    return m;
  };

  for (const s of ctx.solids) {
    let cx, cy, cz, w, h, d;
    if (s.rot) { cx = s.cx; cy = s.cy; cz = s.cz; w = s.w; h = s.h; d = s.d; }
    else {
      cx = (s.x0 + s.x1) / 2; cy = (s.y0 + s.y1) / 2; cz = (s.z0 + s.z1) / 2;
      w = s.x1 - s.x0; h = s.y1 - s.y0; d = s.z1 - s.z0;
    }
    addProxy(cx, cy, cz, w, h, d, s.ry, s.surface);
    if (!s.collide) continue;
    const ca = Math.abs(Math.cos(s.ry || 0)), sa = Math.abs(Math.sin(s.ry || 0));
    const ew = (w * ca + d * sa) / 2, ed = (w * sa + d * ca) / 2;
    colliders.push({
      min: new THREE.Vector3(cx - ew, cy - h / 2, cz - ed),
      max: new THREE.Vector3(cx + ew, cy + h / 2, cz + ed),
    });
  }

  // ground + surfacing proxies, so a shot that hits nothing structural still
  // reports the right surface for footstep/impact FX
  addProxy(0, -1.0, 0, 180, 2.0, 180, 0, 'sand');
  addProxy(1.0, 0.0, -2.3, 10.0, 0.14, 19.4, 0, 'concrete');
  addProxy(-2.5, 0.0, -9.0, 29.0, 0.14, 6.0, 0, 'concrete');
  addProxy(-11.5, 0.0, -19.0, 11.0, 0.14, 14.0, 0, 'concrete');
  addProxy(-15.0, 0.0, 0.7, 4.0, 0.14, 13.4, 0, 'concrete');
  addProxy(0.0, 0.0, 16.5, 40.0, 0.14, 18.2, 0, 'concrete');
  addProxy(23.0, 0.0, 0.7, 10.0, 0.14, 13.4, 0, 'concrete');

  proxyRoot.updateMatrixWorld(true);

  // ---- spawns ---------------------------------------------------------------
  // Order matters: bench shots read [0] for the silhouette pose and [0..4] for
  // the firefight pose, so the first five are chosen against those frustums.
  const spawnPoints = [
    { pos: new THREE.Vector3(-4.84, 0, 20.0), tag: 'gate' },
    { pos: new THREE.Vector3(-3.0, 0, 0.4), tag: 'street' },
    { pos: new THREE.Vector3(4.6, 0, -4.2), tag: 'street' },
    { pos: new THREE.Vector3(-1.4, 0, -8.4), tag: 'junction' },
    { pos: new THREE.Vector3(2.6, 0, -11.0), tag: 'junction' },
    { pos: new THREE.Vector3(-15.0, 0, 0.0), tag: 'west_alley' },
    { pos: new THREE.Vector3(-11.5, 0, -19.5), tag: 'north_street' },
    { pos: new THREE.Vector3(-9.0, 0, -24.4), tag: 'breach' },
    { pos: new THREE.Vector3(8.2, 0, -11.0), tag: 'rear_yard' },
    { pos: new THREE.Vector3(23.0, 0, -9.0), tag: 'east_yard' },
    { pos: new THREE.Vector3(23.4, 0, 6.4), tag: 'east_yard' },
    { pos: new THREE.Vector3(16.0, 0, 13.6), tag: 'plaza' },
    { pos: new THREE.Vector3(-17.6, 0, 13.0), tag: 'plaza' },
    { pos: new THREE.Vector3(2.6, 0, 23.6), tag: 'gate' },
    { pos: new THREE.Vector3(11.0, 0, 1.0), tag: 'shop' },
    { pos: new THREE.Vector3(0.0, 0, -21.0), tag: 'courtyard' },
  ];

  // Walkable tops the player module can adopt once it grows floor support.
  // Purely additive — nothing in the contract depends on it.
  const floors = [
    { min: new THREE.Vector3(-13.2, 7.96, -6.2), max: new THREE.Vector3(-4.2, 7.96, 7.2), tag: 'b2_roof' },
    { min: new THREE.Vector3(-12.3, 3.94, 7.6), max: new THREE.Vector3(-11.0, 3.94, 9.1), tag: 'b2_landing' },
  ];

  const surfaceOf = (mesh) => surfaces.get(mesh) || 'concrete';

  if (typeof window !== 'undefined') {
    window.__level = {
      triangles, meshes: group.children.length, proxies: raycastables.length,
      colliders: colliders.length, instanced: instanced.length,
    };
  }

  return {
    group,
    colliders,
    raycastables,
    playerStart: { pos: new THREE.Vector3(0, 0, 21.5), yaw: 0 },
    spawnPoints,
    bounds: { min: new THREE.Vector3(-36, 0, -34), max: new THREE.Vector3(36, 26, 36) },
    surfaceOf,
    floors,
    triangles,
    dispose() {
      group.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
      for (const d of disposables) d.dispose();
    },
  };
}

export function aabbOf(mesh) {
  mesh.updateMatrixWorld(true);
  const b = new THREE.Box3().setFromObject(mesh);
  return { min: b.min.clone(), max: b.max.clone() };
}
