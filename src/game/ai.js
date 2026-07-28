// OWNER: agent "ai" — enemy behaviour, pathing, cover, engagement, death.
// CONTRACT:
//   new Enemies(scene, level, materials, player)
//   .spawn(spawnPoint, variant) -> enemy
//   .update(dt)
//   .list -> enemy[]     each: { alive, pos, damage(n,head), raycast(ray)->{point,distance,mult,name}|null }
//   .clear()

import * as THREE from 'three';
import { bus, EV } from '../core/bus.js';
import { buildCharacter } from '../art/character.js';

export class Enemies {
  constructor(scene, level, materials, player) {
    this.scene = scene; this.level = level; this.materials = materials; this.player = player;
    this.list = [];
    this._sphere = new THREE.Sphere();
  }

  spawn(sp, variant = 'rifleman') {
    const ch = buildCharacter(this.materials, variant);
    ch.group.position.copy(sp.pos);
    this.scene.add(ch.group);
    const self = this;
    const e = {
      alive: true, health: 100, ch, variant,
      get pos() { return ch.group.position; },
      damage(n, head) {
        if (!this.alive) return;
        this.health -= n;
        if (this.health <= 0) {
          this.alive = false;
          bus.emit(EV.KILL, { victim: this, headshot: !!head, distance: ch.group.position.distanceTo(self.player.pos) });
          self.scene.remove(ch.group);
        }
      },
      raycast(ray) {
        let best = null;
        for (const hb of ch.hitboxes) {
          hb.obj.getWorldPosition(self._sphere.center);
          self._sphere.radius = hb.radius;
          const p = ray.ray.intersectSphere(self._sphere, new THREE.Vector3());
          if (p) {
            const d = ray.ray.origin.distanceTo(p);
            if (!best || d < best.distance) best = { point: p, distance: d, mult: hb.mult, name: hb.name };
          }
        }
        return best;
      },
    };
    this.list.push(e);
    bus.emit(EV.SPAWN, { enemy: e });
    return e;
  }

  update(dt) {
    for (const e of this.list) {
      if (!e.alive) continue;
      const to = e.ch.group.position.clone().sub(this.player.pos);
      to.y = 0;
      const dist = to.length();
      if (dist > 8) e.ch.group.position.addScaledVector(to.normalize(), -dt * 2.2);
      e.ch.group.lookAt(this.player.pos.x, e.ch.group.position.y, this.player.pos.z);
      e.ch.update(dt);
    }
    this.list = this.list.filter((e) => e.alive);
  }

  clear() {
    for (const e of this.list) this.scene.remove(e.ch.group);
    this.list.length = 0;
  }
}
