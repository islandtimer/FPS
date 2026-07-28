// OWNER: agent "combat" — ballistics, hit resolution, impact FX, tracers, decals, shells.
// CONTRACT:
//   new Combat(scene, level, world)   world = { enemies: [] , player }
//   .update(dt)
//   .dispose()
// Subscribes EV.SHOT, emits EV.HIT / EV.KILL. All FX must be pooled + instanced;
// zero per-shot allocation in steady state.

import * as THREE from 'three';
import { bus, EV } from '../core/bus.js';

export class Combat {
  constructor(scene, level, world) {
    this.scene = scene;
    this.level = level;
    this.world = world;
    this.ray = new THREE.Raycaster();
    this.ray.far = 300;
    this._off = bus.on(EV.SHOT, (e) => this.onShot(e));
  }

  onShot({ origin, dir, spread, weapon }) {
    const d = dir.clone();
    d.x += (Math.random() * 2 - 1) * spread;
    d.y += (Math.random() * 2 - 1) * spread;
    d.normalize();
    this.ray.set(origin, d);

    let best = null;
    for (const e of this.world.enemies) {
      if (!e.alive) continue;
      const hit = e.raycast?.(this.ray);
      if (hit && (!best || hit.distance < best.distance)) best = { ...hit, enemy: e };
    }
    const hits = this.ray.intersectObjects(this.level.raycastables, false);
    const geo = hits[0];
    if (geo && (!best || geo.distance < best.distance)) {
      bus.emit(EV.HIT, {
        point: geo.point, normal: geo.face?.normal, distance: geo.distance,
        surface: this.level.surfaceOf(geo.object), victim: null,
      });
      return;
    }
    if (best) {
      const dmg = weapon.damage * best.mult * falloff(weapon.falloff, best.distance);
      best.enemy.damage(dmg, best.name === 'head');
      bus.emit(EV.HIT, {
        point: best.point, normal: null, distance: best.distance,
        surface: 'flesh', victim: best.enemy, headshot: best.name === 'head', damage: dmg,
      });
    }
  }

  update(dt) {}
  dispose() { this._off?.(); }
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
