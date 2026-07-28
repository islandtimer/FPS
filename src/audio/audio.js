// OWNER: agent "audio" — 100% procedural sound via WebAudio. No sample files.
// Gunshots, mech/foley, impacts, footsteps, bullet-crack, tails/reverb, ambience,
// mixing + ducking. LEGAL: no recorded/licensed audio of any kind.
// CONTRACT:
//   new AudioEngine(camera)
//   .resume()                 (must be called from a user gesture)
//   .update(dt, ctx)
//   .setMasterVolume(v)
//   .dispose()
// Subscribes to the bus; never called directly by gameplay modules.

import { bus, EV } from '../core/bus.js';

export class AudioEngine {
  constructor(camera) {
    this.camera = camera;
    this.ctx = null;
    this.enabled = false;
    this.master = 0.8;
    this._offs = [
      bus.on(EV.SHOT, () => this.gunshot()),
      bus.on(EV.FOOTSTEP, () => {}),
    ];
  }

  resume() {
    if (this.ctx) { this.ctx.resume?.(); return; }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    this.ctx = new AC({ latencyHint: 'interactive' });
    this.out = this.ctx.createGain();
    this.out.gain.value = this.master;
    this.out.connect(this.ctx.destination);
    this.enabled = true;
  }

  gunshot() {
    if (!this.enabled) return;
    const c = this.ctx, t = c.currentTime;
    const n = c.createBufferSource();
    const len = (c.sampleRate * 0.25) | 0;
    const buf = c.createBuffer(1, len, c.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 6);
    n.buffer = buf;
    const g = c.createGain();
    g.gain.setValueAtTime(0.35, t);
    const f = c.createBiquadFilter();
    f.type = 'lowpass'; f.frequency.setValueAtTime(5200, t);
    f.frequency.exponentialRampToValueAtTime(420, t + 0.2);
    n.connect(f); f.connect(g); g.connect(this.out);
    n.start(t); n.stop(t + 0.26);
  }

  setMasterVolume(v) { this.master = v; if (this.out) this.out.gain.value = v; }
  update() {}
  dispose() { this._offs.forEach((f) => f()); this.ctx?.close?.(); }
}
