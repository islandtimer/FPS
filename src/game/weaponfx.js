// OWNER: agent "feel" — viewmodel animation, recoil, ADS, reload, weapon state machine.
// This module owns "how the gun feels", not "what the gun looks like" (art/weapon.js)
// and not "what the bullet does" (game/combat.js).
// CONTRACT:
//   new ViewModel(camera, weapon, player)
//   .update(dt) -> void
//   .triggerDown(bool), .reload(), .setAds(bool), .switchFire()
//   .state -> { ammo, reserve, reloading, ads, adsT, firing, fireMode }
// Emits: EV.SHOT {origin,dir,spread,weapon}, EV.RELOAD, EV.WEAPON_STATE

import * as THREE from 'three';
import { bus, EV } from '../core/bus.js';

export class ViewModel {
  constructor(camera, weapon, player) {
    this.camera = camera;
    this.weapon = weapon;
    this.player = player;
    this.cfg = weapon.config;

    this.rig = new THREE.Group();
    this.rig.add(weapon.group);
    camera.add(this.rig);

    this.ammo = this.cfg.magSize;
    this.reserve = this.cfg.reserve;
    this.reloading = false;
    this.ads = false;
    this.adsT = 0;
    this.firing = false;
    this._cool = 0;
    this._reloadT = 0;
    this._recoil = new THREE.Vector2();
    this.state = this;
  }

  triggerDown(v) { this.firing = v; }
  setAds(v) { this.ads = v; this.player.setAds(v); }

  reload() {
    if (this.reloading || this.ammo >= this.cfg.magSize || this.reserve <= 0) return;
    this.reloading = true;
    this._reloadT = this.ammo === 0 ? this.cfg.reloadEmptyTime : this.cfg.reloadTime;
    bus.emit(EV.RELOAD, { phase: 'start', empty: this.ammo === 0, weapon: this.cfg });
  }

  update(dt) {
    this.adsT += ((this.ads ? 1 : 0) - this.adsT) * Math.min(1, dt / Math.max(0.01, this.cfg.adsTime) * 0.6);
    this._cool -= dt;

    if (this.reloading) {
      this._reloadT -= dt;
      if (this._reloadT <= 0) {
        const need = this.cfg.magSize - this.ammo;
        const take = Math.min(need, this.reserve);
        this.ammo += take; this.reserve -= take;
        this.reloading = false;
        bus.emit(EV.RELOAD, { phase: 'end', weapon: this.cfg });
      }
    } else if (this.firing && this._cool <= 0 && this.ammo > 0) {
      this._fire();
    }

    // recoil decay
    this._recoil.multiplyScalar(Math.max(0, 1 - this.cfg.recoil.recovery * dt));

    const t = this.adsT;
    this.rig.position.set(
      THREE.MathUtils.lerp(0.13, 0, t),
      THREE.MathUtils.lerp(-0.12, -0.045, t),
      THREE.MathUtils.lerp(-0.24, -0.16, t) - this._recoil.y * 0.04
    );
    this.rig.rotation.set(this._recoil.y * 0.08, this._recoil.x * 0.05, 0);
    bus.emit(EV.WEAPON_STATE, this.snapshot());
  }

  snapshot() {
    return {
      ammo: this.ammo, reserve: this.reserve, reloading: this.reloading,
      ads: this.ads, adsT: this.adsT, name: this.cfg.name,
    };
  }

  _fire() {
    this.ammo--;
    this._cool = 60 / this.cfg.rpm;
    const spread = this.ads ? this.cfg.spreadAds : this.cfg.spreadHip;
    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const origin = new THREE.Vector3().setFromMatrixPosition(this.camera.matrixWorld);
    const kickV = this.cfg.recoil.vertical * (0.9 + Math.random() * 0.2);
    const kickH = this.cfg.recoil.horizontal * (Math.random() * 2 - 1);
    this._recoil.y += kickV;
    this._recoil.x += kickH;
    this.player.applyRecoil(kickV * 0.006, kickH * 0.003);
    bus.emit(EV.SHOT, { origin, dir, spread, weapon: this.cfg, muzzle: this.weapon.muzzleTip });
  }
}
