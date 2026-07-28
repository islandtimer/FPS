// OWNER: agent "player" — movement, collision, camera. The "feel" of walking.
// CONTRACT:
//   new Player(camera, level, input)
//   .update(dt) -> void       (moves camera, emits FOOTSTEP/LAND on the bus)
//   .state -> { pos, vel, yaw, pitch, speed, grounded, crouch, sprint, ads, stance }
//   .setAds(bool), .damage(n, fromDir), .health, .alive
//   .applyRecoil(pitchRad, yawRad)
//   .teleport(pos, yaw)

import * as THREE from 'three';
import { bus, EV } from '../core/bus.js';

const TMP = new THREE.Vector3();

export class Player {
  constructor(camera, level, input) {
    this.camera = camera;
    this.level = level;
    this.input = input;
    this.pos = level.playerStart.pos.clone();
    this.vel = new THREE.Vector3();
    this.yaw = level.playerStart.yaw;
    this.pitch = 0;
    this.health = 100;
    this.alive = true;
    this.grounded = true;
    this.crouch = 0;
    this.sprint = false;
    this.ads = false;
    this.eyeHeight = 1.68;
    this._bob = 0;
    this._stepAccum = 0;
    this.state = this;
  }

  get speed() { return Math.hypot(this.vel.x, this.vel.z); }

  applyRecoil(p, y) { this.pitch += p; this.yaw += y; }

  setAds(v) { if (this.ads !== v) { this.ads = v; bus.emit(EV.ADS, { ads: v }); } }

  teleport(pos, yaw = this.yaw) { this.pos.copy(pos); this.yaw = yaw; this.vel.set(0, 0, 0); }

  damage(n, fromDir) {
    if (!this.alive) return;
    this.health -= n;
    bus.emit(EV.DAMAGE, { amount: n, fromDir, health: this.health });
    if (this.health <= 0) { this.alive = false; bus.emit(EV.DEATH, { who: 'player' }); }
  }

  update(dt) {
    const inp = this.input;
    this.yaw -= inp.lookX;
    this.pitch = THREE.MathUtils.clamp(this.pitch - inp.lookY, -1.5, 1.5);
    inp.lookX = 0; inp.lookY = 0;

    const wish = TMP.set(inp.moveX, 0, inp.moveZ);
    if (wish.lengthSq() > 1) wish.normalize();
    wish.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw);

    this.sprint = inp.sprint && inp.moveZ < -0.2 && !this.ads;
    const target = this.sprint ? 7.2 : this.ads ? 2.6 : 4.6;
    const accel = this.grounded ? 55 : 8;
    this.vel.x += (wish.x * target - this.vel.x) * Math.min(1, accel * dt);
    this.vel.z += (wish.z * target - this.vel.z) * Math.min(1, accel * dt);

    this.vel.y -= 22 * dt;
    if (this.grounded && inp.jump) { this.vel.y = 6.6; this.grounded = false; }

    this.pos.addScaledVector(this.vel, dt);
    if (this.pos.y <= 0) {
      if (!this.grounded) bus.emit(EV.LAND, { speed: -this.vel.y });
      this.pos.y = 0; this.vel.y = 0; this.grounded = true;
    }
    this._collide();

    if (this.grounded && this.speed > 0.6) {
      this._stepAccum += this.speed * dt;
      const stride = this.sprint ? 2.05 : 1.55;
      if (this._stepAccum > stride) {
        this._stepAccum = 0;
        bus.emit(EV.FOOTSTEP, { surface: 'sand', speed: this.speed, sprint: this.sprint });
      }
      this._bob += dt * this.speed * 1.9;
    }

    const bobAmt = this.ads ? 0.006 : 0.022;
    this.camera.position.set(
      this.pos.x + Math.cos(this._bob) * bobAmt * 0.5,
      this.pos.y + this.eyeHeight + Math.sin(this._bob * 2) * bobAmt,
      this.pos.z
    );
    this.camera.rotation.set(this.pitch, this.yaw, 0, 'YXZ');
  }

  _collide() {
    const r = 0.34;
    for (const c of this.level.colliders) {
      if (this.pos.x + r < c.min.x || this.pos.x - r > c.max.x) continue;
      if (this.pos.z + r < c.min.z || this.pos.z - r > c.max.z) continue;
      if (this.pos.y > c.max.y || this.pos.y + 1.7 < c.min.y) continue;
      const dxa = c.max.x - (this.pos.x - r), dxb = (this.pos.x + r) - c.min.x;
      const dza = c.max.z - (this.pos.z - r), dzb = (this.pos.z + r) - c.min.z;
      const px = Math.min(dxa, dxb), pz = Math.min(dza, dzb);
      if (px < pz) { this.pos.x += dxa < dxb ? px : -px; this.vel.x = 0; }
      else { this.pos.z += dza < dzb ? pz : -pz; this.vel.z = 0; }
    }
  }
}
