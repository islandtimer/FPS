// Bootstrap + wiring. OWNED BY THE ORCHESTRATOR — module agents must not edit
// this file; they edit only the module they own and keep its contract stable.

import * as THREE from 'three';
import { rand, makeRng, WORLD_SEED } from './core/rng.js';
import { bus, EV } from './core/bus.js';
import { Perf } from './engine/perf.js';
import { RenderPipeline } from './engine/renderer.js';
import { Atmosphere } from './engine/sky.js';
import { buildMaterials } from './art/materials.js';
import { buildLevel } from './art/level.js';
import { buildWeapon } from './art/weapon.js';
import { Player } from './game/player.js';
import { ViewModel } from './game/weaponfx.js';
import { Combat } from './game/combat.js';
import { Enemies } from './game/ai.js';
import { Director } from './game/director.js';
import { HUD } from './ui/hud.js';
import { AudioEngine } from './audio/audio.js';
import { CAMERA_SHOTS } from './bench/shots.js';

const params = new URLSearchParams(location.search);
const BENCH = params.has('bench');
const SHOT = params.get('shot');
const QUALITY = params.get('q') || 'high';

const canvas = document.getElementById('view');
const uiRoot = document.getElementById('ui');

const pipeline = new RenderPipeline(canvas);
pipeline.setQuality(QUALITY);
const perf = new Perf(pipeline.renderer, { adaptive: !BENCH });

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(80, 16 / 9, 0.05, 400);
scene.add(camera);

const atmosphere = new Atmosphere(pipeline.renderer, scene);
const materials = buildMaterials(pipeline.renderer);
const level = buildLevel(materials, makeRng('level:' + WORLD_SEED));
scene.add(level.group);

const input = {
  moveX: 0, moveZ: 0, lookX: 0, lookY: 0,
  jump: false, sprint: false, crouch: false, fire: false, ads: false, reload: false,
};

const player = new Player(camera, level, input);
const weapon = buildWeapon('ar_vector', materials);
const viewmodel = new ViewModel(camera, weapon, player);
const enemies = new Enemies(scene, level, materials, player);
const combat = new Combat(scene, level, { enemies: enemies.list, player });
const director = new Director(enemies, level, player);
const hud = new HUD(uiRoot);
const audio = new AudioEngine(camera);

// combat needs the live array reference, not a snapshot
combat.world.enemies = enemies.list;
director.start();

// ---------------------------------------------------------------- input
const KEY = {
  KeyW: 'fwd', KeyS: 'back', KeyA: 'left', KeyD: 'right',
  Space: 'jump', ShiftLeft: 'sprint', ControlLeft: 'crouch', KeyC: 'crouch', KeyR: 'reload',
};
const held = new Set();
if (!BENCH) {
  addEventListener('keydown', (e) => { if (KEY[e.code]) { held.add(KEY[e.code]); e.preventDefault(); } });
  addEventListener('keyup', (e) => { if (KEY[e.code]) held.delete(KEY[e.code]); });
  canvas.addEventListener('click', () => { canvas.requestPointerLock(); audio.resume(); });
  addEventListener('mousemove', (e) => {
    if (document.pointerLockElement !== canvas) return;
    const s = 0.0022 * (viewmodel.ads ? 0.55 : 1);
    input.lookX += e.movementX * s;
    input.lookY += e.movementY * s;
  });
  addEventListener('mousedown', (e) => {
    if (document.pointerLockElement !== canvas) return;
    if (e.button === 0) viewmodel.triggerDown(true);
    if (e.button === 2) viewmodel.setAds(true);
  });
  addEventListener('mouseup', (e) => {
    if (e.button === 0) viewmodel.triggerDown(false);
    if (e.button === 2) viewmodel.setAds(false);
  });
  addEventListener('contextmenu', (e) => e.preventDefault());
}

function pumpInput() {
  input.moveZ = (held.has('fwd') ? -1 : 0) + (held.has('back') ? 1 : 0);
  input.moveX = (held.has('right') ? 1 : 0) + (held.has('left') ? -1 : 0);
  input.jump = held.has('jump');
  input.sprint = held.has('sprint');
  input.crouch = held.has('crouch');
  if (held.has('reload')) { viewmodel.reload(); held.delete('reload'); }
}

// ---------------------------------------------------------------- resize
function resize() {
  // ?w/?h let the harness shrink the framebuffer to isolate CPU cost from
  // SwiftShader's fragment cost while keeping the 16:9 frustum (identical culling).
  const w = BENCH ? +(params.get('w') || 1920) : innerWidth;
  const h = BENCH ? +(params.get('h') || 1080) : innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  pipeline.setSize(w, h, BENCH ? 1 : perf.renderScale);
}
addEventListener('resize', resize);
resize();

// ---------------------------------------------------------------- bench hooks
const benchState = { paused: false, fixedDt: 1 / 60, frames: 0, halted: false, raf: 0 };
window.__game = {
  perf, scene, camera, player, viewmodel, director, enemies, level, pipeline,
  shots: CAMERA_SHOTS,
  ready: false,
  applyShot(name) {
    const s = CAMERA_SHOTS[name];
    if (!s) throw new Error('no shot ' + name);
    player.teleport(new THREE.Vector3(...s.pos), s.yaw);
    player.pitch = s.pitch;
    viewmodel.ads = !!s.ads; viewmodel.adsT = s.ads ? 1 : 0;
    if (s.setup) s.setup({ enemies, director, level, viewmodel, THREE });
    for (let i = 0; i < 3; i++) step(benchState.fixedDt);
  },
  step(n = 1, dt = 1 / 60) { for (let i = 0; i < n; i++) step(dt); },
  snapshot: () => perf.snapshot(1920, 1080),
  setPaused(v) { benchState.paused = v; },

  // Screenshot support. Capturing while the canvas re-renders every frame means
  // racing the compositor, which under a software rasteriser never settles. So the
  // harness drives N frames to let temporal accumulation converge, halts the loop,
  // captures a completely static surface, then resumes.
  halt() {
    benchState.halted = true;
    if (benchState.raf) cancelAnimationFrame(benchState.raf);
    benchState.raf = 0;
  },
  resume() {
    if (!benchState.halted) return;
    benchState.halted = false;
    benchState.raf = requestAnimationFrame(frame);
  },
  /** Advance exactly n rendered frames at a fixed dt, then halt. */
  settle(n = 40, dt = 1 / 60) {
    this.halt();
    for (let i = 0; i < n; i++) {
      if (!benchState.paused) step(dt);
      pipeline.render(scene, camera, dt);
    }
    pipeline.reportCost(perf);
  },
};

// ---------------------------------------------------------------- self-benchmark
// The build machine has no GPU, so the only trustworthy 1080p/60 measurement is the
// one the player takes on their own laptop. This makes that a single keypress.
const selfBench = {
  running: false, t0: 0, frames: 0, samples: [], result: null,
  start() {
    if (this.running) return;
    this.running = true; this.t0 = performance.now(); this.frames = 0; this.samples = [];
    this.result = null;
    bus.emit(EV.UI, { kind: 'benchStart' });
  },
  tick(frameMs) {
    if (!this.running) return;
    this.frames++;
    if (this.frames > 30) this.samples.push(frameMs); // discard warm-up
    if (performance.now() - this.t0 > 20000) this.finish();
  },
  finish() {
    this.running = false;
    const s = this.samples.slice().sort((a, b) => a - b);
    if (!s.length) return;
    const at = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
    this.result = {
      avgFps: +(1000 / (s.reduce((a, b) => a + b, 0) / s.length)).toFixed(1),
      onePercentLowFps: +(1000 / at(0.99)).toFixed(1),
      medianMs: +at(0.5).toFixed(2),
      width: pipeline.width, height: pipeline.height,
      renderScale: +perf.renderScale.toFixed(3),
      quality: perf.quality,
      drawCalls: perf.stats.calls, triangles: perf.stats.triangles,
      gpu: (() => {
        try {
          const gl = pipeline.renderer.getContext();
          const d = gl.getExtension('WEBGL_debug_renderer_info');
          return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'unknown';
        } catch { return 'unknown'; }
      })(),
    };
    bus.emit(EV.UI, { kind: 'benchResult', result: this.result });
    console.log('[benchmark]', this.result);
  },
};
window.__selfBench = selfBench;

if (!BENCH) {
  addEventListener('keydown', (e) => {
    if (e.code === 'KeyB' && e.shiftKey) selfBench.start();
    if (e.code === 'KeyF' && e.shiftKey) bus.emit(EV.UI, { kind: 'togglePerf' });
  });
}

// ---------------------------------------------------------------- loop
let last = performance.now();

function step(dt) {
  pumpInput();
  player.update(dt);
  viewmodel.update(dt);
  enemies.update(dt);
  combat.update(dt);
  director.update(dt);
  atmosphere.update(dt);
  audio.update(dt, { player, camera });
  hud.update(dt, { player, weapon: viewmodel, director, perf });
}

function frame(now) {
  benchState.raf = requestAnimationFrame(frame);
  const raw = perf.beginFrame(now);
  const dt = Math.min(0.05, raw / 1000) || 1 / 60;
  const t0 = performance.now();
  if (!benchState.paused) step(dt);
  const t1 = performance.now();
  perf.setLogicMs(t1 - t0);
  perf.markCpuDone(t1);
  if (!BENCH) pipeline.setSize(innerWidth, innerHeight, perf.renderScale);
  pipeline.render(scene, camera, dt);
  pipeline.reportCost(perf);
  perf.endFrame(performance.now());
  selfBench.tick(perf.frameMs);
  benchState.frames++;
}

if (SHOT) window.__game.applyShot(SHOT);
window.__game.ready = true;
requestAnimationFrame((t) => { last = t; frame(t); });
