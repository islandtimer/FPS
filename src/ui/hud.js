// OWNER: agent "hud" — reticle, ammo, compass, killfeed, hitmarkers, damage
// direction, wave banners, menus. DOM + a 2D canvas overlay; no THREE here.
// LEGAL: original UI language and iconography only. No Activision UI strings,
// rank names, or logos.
// CONTRACT:
//   new HUD(rootEl)
//   .update(dt, ctx)   ctx = { player, weapon, director, perf }
//   .dispose()
// Subscribes to the bus for HIT/KILL/DAMAGE/WAVE/RELOAD.

import { bus, EV } from '../core/bus.js';

export class HUD {
  constructor(root) {
    this.root = root;
    this.el = document.createElement('div');
    this.el.style.cssText = 'position:absolute;inset:0;color:#e8eef5;';
    this.el.innerHTML = `
      <div id="reticle" style="position:absolute;left:50%;top:50%;width:3px;height:3px;
        margin:-1.5px 0 0 -1.5px;background:#fff;opacity:.9;border-radius:50%"></div>
      <div id="ammo" style="position:absolute;right:42px;bottom:34px;font:600 34px/1 ui-monospace,monospace;
        letter-spacing:1px;text-shadow:0 2px 6px #000a"></div>
      <div id="wave" style="position:absolute;left:50%;top:44px;transform:translateX(-50%);
        font:600 15px/1 ui-monospace,monospace;opacity:.85;text-shadow:0 2px 6px #000a"></div>`;
    root.appendChild(this.el);
    this.$ammo = this.el.querySelector('#ammo');
    this.$wave = this.el.querySelector('#wave');
    this._offs = [bus.on(EV.HIT, () => {})];
  }

  update(dt, ctx) {
    const w = ctx.weapon;
    this.$ammo.textContent = `${w.ammo} / ${w.reserve}`;
    this.$wave.textContent = `WAVE ${ctx.director.wave}  ·  ${ctx.director.remaining} HOSTILE`;
  }

  dispose() { this._offs.forEach((f) => f()); this.el.remove(); }
}
