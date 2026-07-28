// OWNER: agent "loop" — round structure, wave pacing, spawn budget, scoring, match flow.
// CONTRACT:
//   new Director(enemies, level, player)
//   .update(dt)
//   .state -> { wave, alive, remaining, phase:'intermission'|'active'|'over', score, streak }
//   .start(), .reset()

import { bus, EV } from '../core/bus.js';

export class Director {
  constructor(enemies, level, player) {
    this.enemies = enemies; this.level = level; this.player = player;
    this.wave = 0; this.phase = 'intermission'; this.timer = 3;
    this.score = 0; this.streak = 0; this.toSpawn = 0;
    this.state = this;
    bus.on(EV.KILL, ({ headshot, distance }) => {
      this.score += headshot ? 150 : 100;
      this.streak++;
    });
  }

  get alive() { return this.enemies.list.length; }
  get remaining() { return this.alive + this.toSpawn; }

  start() { this.phase = 'intermission'; this.timer = 3; }
  reset() { this.wave = 0; this.score = 0; this.streak = 0; this.enemies.clear(); this.start(); }

  update(dt) {
    if (this.phase === 'intermission') {
      this.timer -= dt;
      if (this.timer <= 0) {
        this.wave++;
        this.toSpawn = 4 + this.wave * 2;
        this.phase = 'active';
        this._spawnT = 0;
        bus.emit(EV.WAVE, { index: this.wave, phase: 'start', remaining: this.toSpawn });
      }
    } else if (this.phase === 'active') {
      this._spawnT -= dt;
      if (this.toSpawn > 0 && this.alive < 8 && this._spawnT <= 0) {
        const sp = this.level.spawnPoints[(Math.random() * this.level.spawnPoints.length) | 0];
        this.enemies.spawn(sp);
        this.toSpawn--;
        this._spawnT = 0.6;
      }
      if (this.toSpawn === 0 && this.alive === 0) {
        this.phase = 'intermission';
        this.timer = 6;
        bus.emit(EV.WAVE, { index: this.wave, phase: 'clear', remaining: 0 });
      }
    }
  }
}
