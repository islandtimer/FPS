// OWNER: agent "feel" — viewmodel animation, recoil, ADS, reload, weapon state machine.
// This module owns "how the gun feels", not "what the gun looks like" (art/weapon.js)
// and not "what the bullet does" (game/combat.js).
// CONTRACT:
//   new ViewModel(camera, weapon, player)
//   .update(dt) -> void
//   .triggerDown(bool), .reload(), .setAds(bool), .switchFire()
//   .forceShot() -> bool          (screenshot harness: fire one round NOW)
//   .state -> { ammo, reserve, reloading, ads, adsT, firing, fireMode }
// Emits: EV.SHOT {origin,dir,spread,weapon}, EV.RELOAD, EV.WEAPON_STATE
// Writes: camera.fov (ADS compression — see ADS below). Restored on dispose().
//
// ---------------------------------------------------------------------------
// HOW THIS IS BUILT, and why.
//
// FIXED SUB-STEP. Every continuous quantity that has inertia (look lag, recoil
// kick, ADS settle, landing jolt) is a spring-damper integrated at a fixed
// 1/240s sub-step, never at the frame's dt. Semi-implicit Euler on a variable
// dt changes both the frequency and the damping ratio of a spring, so a gun
// tuned at 144fps feels mushy at 30fps and rings at 240. Sub-stepping costs
// ~200 float ops a frame and makes the feel identical everywhere. All springs
// live in one flat Float32Array bank so the integrator is a single tight loop.
//
// LOOK LAG is the single biggest "AAA" tell. The rig's rotation and position
// springs are driven by *camera angular velocity*, not by camera angle: whipping
// right sets a target offset proportional to the turn rate, the spring trails
// into it, and when the turn stops the target snaps to zero and the underdamped
// spring carries the gun a little past centre before settling. That overshoot is
// the whole effect. Recoil the module applies to the camera itself is excluded
// from the measurement, otherwise every shot would double-kick the viewmodel.
//
// PIVOT BLENDING. The rig is posed as: rotate about a pivot point, then place
// that pivot. Hipfire pivots near the shoulder, so rotation swings the muzzle
// across frame. ADS pivots exactly on the optic, so sway rotates the rifle
// *around the dot* and the sight picture stays nailed to screen centre. The ADS
// rest position is not a hand-tuned constant — it is solved at construction from
// the optic's actual transform, so the reticle centres itself if the art moves.
//
// REST POSE. The rig sits 11cm from the eye at the rear of the receiver and
// 71cm at the muzzle, so the near end is cropped by the right and bottom edges
// of frame and a 1.4mm chamfer on it is seven pixels wide at 1080p. The crop is
// the point: a viewmodel that fits entirely inside the frame reads as a prop on
// a table, because nothing in the shot is nearer than arm's length. Before this
// the body was neither in the quadrant nor out of it — the magazine and grip
// fell below the bottom edge while the barrel ran across the middle of frame,
// which is the worst of both. Three angles do the rest — 4.9 degrees of muzzle
// rise, 4.3 degrees of inward yaw and 8.9 degrees of cant on top of the 2 the
// art bakes in — so the receiver presents its top deck and its right flank at
// once instead of the head-on sliver a rig parallel to the view axis gives you.
// The mass of the gun is ~1.4x nearer than it was, so every positional offset in
// the module subtends ~1.4x the screen angle it was tuned at; NEAR_COMP scales
// the whole offset sum back, in one place, so sway, bob, lag and kick keep the
// angular amplitude they were tuned to have.
//
// ADS is a fast eased transition plus real FOV compression, not a lerp. The
// blend parameter is integrated with a rate that is a power of the remaining
// distance, so it leaves the rest pose at full speed and decelerates into the
// sight — the same curve as 1-(1-t)^2.6 but as an ODE, which means reversing
// mid-transition is continuous instead of snapping between two curves. On top of
// that, the camera's vertical FOV compresses by ADS_ZOOM. Sliding a gun forward
// magnifies nothing; the reason a real sight picture reads as aiming is that the
// world got bigger.
//
// RECOIL is two systems. The viewmodel kick is large, fast and fully recovers.
// The camera displacement is smaller, recovers slowly, and only ~65% of it comes
// back — the residue is the climb the player has to pull down. The per-shot
// values come from a fixed per-weapon pattern table: the first shots go nearly
// straight up, then the pattern bends into a repeatable horizontal shape that
// can be learned and countered. The random part is small and grows with heat, so
// the first burst is precise and a held trigger is not.

import * as THREE from 'three';
import { bus, EV } from '../core/bus.js';
import { makeRng } from '../core/rng.js';

const _tmp = new THREE.Vector3();

const SUB = 1 / 240;          // spring integration step
const MAX_SUB = 16;           // dt is clamped to 50ms upstream; this is headroom
const TAU = Math.PI * 2;
const DEG = Math.PI / 180;

// Hip rest pose. Solved against the real part positions in art/weapon.js so the
// magwell lands mid lower-right quadrant, the ejection port sits at (0.53,-0.55)
// in NDC and the rear of the receiver leaves frame past the right edge.
const HIP_POS = [0.176, -0.115, -0.178];
const HIP_RX = 0.085;   // muzzle up
const HIP_RY = 0.075;   // muzzle inboard, toward the centre line
const HIP_RZ = 0.155;   // cant: top deck rolled toward the middle of frame
// Positional offsets were tuned when the body sat ~1.4x further from the eye.
const NEAR_COMP = 0.72;
// Linear magnification at full ADS. The base frustum is 80 degrees vertical,
// which is very wide, so a red dot has to take a real bite out of it to read as
// aimed at all.
const ADS_ZOOM = 1.50;
// The ADS blend leaves rest at full speed and decelerates in (see header).
const ADS_IN_P = 2.6;
const ADS_OUT_P = 1.8;
// A dry magazine rolls into a reload after this long — six frames, enough for
// the last flash and the bolt locking back to read as the reason the gun
// stopped, short enough that it never feels like a hang.
const AUTO_RELOAD_DELAY = 0.10;

// Spring slots. One flat bank, indices instead of objects, so `step()` is a
// single loop with no property lookups per element.
const SP = {
  LAG_X: 0, LAG_Y: 1, LAG_RX: 2, LAG_RY: 3, LAG_RZ: 4,
  KICK_Z: 5, KICK_Y: 6, KICK_RX: 7, KICK_RY: 8, KICK_RZ: 9,
  ADS: 10, JOLT: 11, HEAVE: 12,
  N: 13,
};

class SpringBank {
  constructor(n) {
    this.x = new Float32Array(n);
    this.v = new Float32Array(n);
    this.t = new Float32Array(n);
    this.k = new Float32Array(n);
    this.c = new Float32Array(n);
  }
  /** freq in Hz, zeta < 1 overshoots (that is usually the point). */
  tune(i, freq, zeta) {
    const w = TAU * freq;
    this.k[i] = w * w;
    this.c[i] = 2 * zeta * w;
  }
  step(h) {
    const { x, v, t, k, c } = this;
    for (let i = 0, n = x.length; i < n; i++) {
      v[i] += (-k[i] * (x[i] - t[i]) - c[i] * v[i]) * h;
      x[i] += v[i] * h;
    }
  }
}

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const sat = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (x) => { const t = sat(x); return t * t * (3 - 2 * t); };
const smoother = (x) => { const t = sat(x); return t * t * t * (t * (t * 6 - 15) + 10); };
/** 0 at both ends, 1 across the middle — the envelope every pose blend uses. */
const bell = (u, rise, fall) => smooth(u / rise) * smooth((1 - u) / fall);
/** Sub-range remap: 0 before a, 1 after b. */
const seg = (u, a, b) => sat((u - a) / (b - a));

// Reload choreography, in normalised time so tactical and empty share one path.
// The empty sequence spends its extra time on the charging handle at the end.
const RELOAD_TAC = [
  ['start', 0.00], ['magout', 0.19], ['magin', 0.60], ['end', 1.00],
];
const RELOAD_EMPTY = [
  ['start', 0.00], ['magout', 0.15], ['magin', 0.52], ['charge', 0.78], ['end', 1.00],
];

// Fire modes offered per weapon class. Fictional hardware, so these are a design
// choice, not a spec sheet.
const MODES = {
  assault: ['auto', 'semi'],
  smg: ['auto', 'burst'],
  dmr: ['semi', 'burst'],
};

/**
 * Per-weapon recoil pattern. Deterministic — drawn once from a seeded stream
 * keyed on the weapon name, so the shape is identical every run and a player can
 * actually memorise it. Index is the shot number since the last trigger release.
 */
function buildPattern(cfg, rng) {
  const n = Math.max(36, cfg.magSize);
  const p = new Float32Array(n * 2);
  const lock = 3 + Math.floor(rng() * 3);     // shots that stay near-vertical
  const w1 = rng.range(0.22, 0.40);           // primary horizontal wavelength
  const w2 = rng.range(0.07, 0.13);           // slow secondary bend
  const ph = rng.range(0, TAU);
  const side = rng.sign();                    // which way the pattern hooks first
  const drift = rng.range(0.25, 0.6) * rng.sign();
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    // Vertical: hardest for the first few, then a lower plateau that sags as the
    // shooter's arms absorb more of it.
    const climb = 0.58 + 0.85 * Math.exp(-i / 3.0) - 0.14 * u;
    p[i * 2 + 1] = cfg.recoil.vertical * climb;
    // Horizontal: silent until `lock`, then a fixed two-harmonic hook plus a
    // one-way drift, which is what makes a long spray shaped instead of noisy.
    const gate = seg(i, lock, lock + 3);
    const shape = Math.sin(i * w1 + ph) * 0.78 + Math.sin(i * w2 + ph * 0.5) * 0.5;
    p[i * 2] = cfg.recoil.horizontal * gate * side * (shape + drift * u * 1.5);
  }
  return p;
}

export class ViewModel {
  constructor(camera, weapon, player) {
    this.camera = camera;
    this.weapon = weapon;
    this.player = player;
    this.cfg = weapon.config;
    this.parts = weapon.parts;

    this.rig = new THREE.Group();
    this.rig.name = 'viewmodel:rig';
    this.rig.add(weapon.group);
    camera.add(this.rig);

    this.ammo = this.cfg.magSize;
    this.reserve = this.cfg.reserve;
    this.reloading = false;
    this.ads = false;
    this.adsT = 0;
    this.firing = false;
    this.ready = true;
    this.spread = this.cfg.spreadHip;
    this.fireMode = (MODES[this.cfg.class] || MODES.assault)[0];
    this.state = this;

    this._rng = makeRng('feel:' + this.cfg.name);
    this._pattern = buildPattern(this.cfg, makeRng('pattern:' + this.cfg.name));
    this._patN = this._pattern.length / 2;

    // ------------------------------------------------------------ timing
    this._cool = 0;
    this._shotIndex = 0;      // position in the recoil pattern
    this._heat = 0;           // 0..1, sustained-fire penalty
    this._bloom = 0;          // hipfire spread growth
    this._sinceShot = 99;
    this._burst = 0;
    this._triggerEdge = false;
    this._t = 0;              // module clock, drives idle sway
    this._acc = 0;            // spring sub-step accumulator
    this._adsRaw = 0;         // pre-ease ADS parameter
    this._idle = 0;
    this._raise = 0;          // sprint-exit lockout
    this._boltCycle = 1;      // 0..1, 1 = at rest
    this._trigPull = 0;
    this._muzzleFlash = 0;
    this._autoReload = -1;    // <0 = disarmed, else seconds until the dry reload

    // ------------------------------------------------------------ optics
    // The camera is not ours, so its rest FOV is read once and always restored.
    this._fovBase = camera.isPerspectiveCamera ? camera.fov : 0;
    this._fovAds = this._fovBase > 0
      ? 2 * Math.atan(Math.tan(this._fovBase * 0.5 * DEG) / ADS_ZOOM) / DEG
      : 0;
    this._fovWritten = this._fovBase;

    // ------------------------------------------------------------ reload
    this._reloadT = 0;
    this._reloadLen = 0;
    this._reloadPhase = 0;
    this._reloadSeq = RELOAD_TAC;
    this._reloadEmpty = false;
    this._chargeT = 0;

    // ------------------------------------------------------------ blends
    this._sprintT = 0;
    this._lowReady = 0;
    this._inspectT = -1;      // <0 = not playing
    this._bobPhase = 0;
    this._bobAmp = 0;

    // ------------------------------------------------------------ camera recoil
    // _camRec is the offset the camera currently carries because of recoil;
    // _camPerm is the part of it that will never be handed back. Both are
    // rebased to zero once the gun settles, which is what makes recovery
    // partial without the camera ever snapping.
    this._camRecP = 0; this._camRecY = 0;
    this._camPermP = 0; this._camPermY = 0;
    this._camAppP = 0; this._camAppY = 0;
    this._shakeP = 0; this._shakeY = 0; this._shakeAmp = 0; this._shakePh = 0;
    this._prevYaw = player.yaw || 0;
    this._prevPitch = player.pitch || 0;
    this._yawVel = 0; this._pitchVel = 0;
    // Teleport detection (see _rePose). Both start where the player does, so the
    // first frame is never mistaken for a jump.
    this._lastPos = new THREE.Vector3().copy(this._playerPos());
    this._trigAt = new THREE.Vector3().copy(this._lastPos);

    // ------------------------------------------------------------ springs
    const s = new SpringBank(SP.N);
    // Look lag is deliberately loose and underdamped — the trail and the small
    // return overshoot are the effect, not artefacts of it.
    s.tune(SP.LAG_X, 3.4, 0.52); s.tune(SP.LAG_Y, 3.6, 0.55);
    s.tune(SP.LAG_RX, 3.2, 0.50); s.tune(SP.LAG_RY, 3.0, 0.46); s.tune(SP.LAG_RZ, 2.6, 0.55);
    // Kick is stiff: the viewmodel must be home again before the next round.
    s.tune(SP.KICK_Z, 8.5, 0.42); s.tune(SP.KICK_Y, 9.0, 0.40);
    s.tune(SP.KICK_RX, 8.0, 0.38); s.tune(SP.KICK_RY, 9.5, 0.50); s.tune(SP.KICK_RZ, 7.5, 0.45);
    s.tune(SP.ADS, 11.0, 0.80);   // slight settle at the end of the raise
    s.tune(SP.JOLT, 5.0, 0.35);   // landing
    s.tune(SP.HEAVE, 2.2, 0.65);  // sprint / reload weight shift
    this._s = s;

    // ------------------------------------------------------------ geometry
    // Solve the ADS rest position from where the optic actually is, rather than
    // trusting a magic number to stay in sync with art/weapon.js.
    this._opticLocal = new THREE.Vector3(0, 0.045, -0.018);
    const optic = this.parts && this.parts.optic;
    if (optic) {
      this.rig.updateMatrixWorld(true);
      const inv = new THREE.Matrix4().copy(this.rig.matrixWorld).invert();
      this._opticLocal.setFromMatrixPosition(optic.matrixWorld).applyMatrix4(inv);
    }
    const ADS_EYE = 0.142;   // optic tube centre to eye; front lens lands ~0.18m
    this._hipBase = new THREE.Vector3(HIP_POS[0], HIP_POS[1], HIP_POS[2]);
    this._adsBase = new THREE.Vector3(
      -this._opticLocal.x, -this._opticLocal.y, -ADS_EYE - this._opticLocal.z,
    );
    this._hipPivot = new THREE.Vector3(0.02, -0.05, 0.13);   // roughly the shoulder
    this._pivot = new THREE.Vector3();
    this._pos = new THREE.Vector3();
    this._euler = new THREE.Euler(0, 0, 0, 'YXZ');
    this._quat = new THREE.Quaternion();

    // Rest transforms of the four animated parts, so offsets are relative.
    const P = this.parts || {};
    this._rest = {
      boltZ: P.bolt ? P.bolt.position.z : 0,
      magY: P.mag ? P.mag.position.y : 0,
      magZ: P.mag ? P.mag.position.z : 0,
      magX: P.mag ? P.mag.position.x : 0,
      chgZ: P.charging ? P.charging.position.z : 0,
    };

    // Pooled event payloads — SHOT can fire 15x a second and must not allocate.
    this._shotOrigin = new THREE.Vector3();
    this._shotDir = new THREE.Vector3();
    this._shotMsg = {
      origin: this._shotOrigin, dir: this._shotDir, spread: 0,
      weapon: this.cfg, muzzle: weapon.muzzleTip, index: 0,
    };
    this._snap = {
      ammo: 0, reserve: 0, reloading: false, ads: false, adsT: 0, firing: false,
      fireMode: this.fireMode, name: this.cfg.name, ready: true, spread: 0,
      bloom: 0, sprint: 0, reloadPhase: '', flash: 0,
    };
    this._adsWritten = 0;

    this._offs = [
      // Stride sync: the player module owns the gait, so the bob phase is
      // corrected toward a foot plant whenever a real footstep lands rather
      // than being re-derived from speed and drifting out of step.
      bus.on(EV.FOOTSTEP, () => this._syncStride()),
      bus.on(EV.LAND, (e) => this._land(e && e.speed)),
    ];
  }

  // ------------------------------------------------------------------ input
  triggerDown(v) {
    v = !!v;
    if (v && !this.firing) { this._triggerEdge = true; this._burst = 0; }
    if (!v) { this._shotIndex = 0; }   // pattern resets on trigger release
    this.firing = v;
    // Where the trigger was pressed, so a discontinuity in the player's position
    // can tell a trigger held across it from one pressed after it. See _rePose.
    if (v) this._trigAt.copy(this._playerPos());
    this._idle = 0;
    this._inspectT = -1;
  }

  _playerPos() {
    const p = this.player.pos;
    return p && p.isVector3 ? p : this.camera.position;
  }

  /**
   * The player has moved discontinuously: a respawn, or the screenshot harness
   * re-posing for the next shot. Nothing mid-flight survives the cut. A reload
   * belonging to the place we just left, recoil the camera is still carrying, a
   * magazine drained on the other side of the jump and a trigger that was held
   * down before it are all state about a moment that no longer exists — carried
   * across, they show up as a gun that is aiming somewhere the camera is not.
   *
   * The trigger is the delicate one: the harness sets a pose and *then* holds the
   * trigger for its combat shot, so a press that happened at the destination is
   * kept and one that happened before the jump is let go.
   */
  _rePose() {
    const pl = this.player;
    const pos = this._playerPos();
    if (this.reloading) this._cancelReload();
    if (this.firing && this._trigAt.distanceToSquared(pos) > 1) this.triggerDown(false);

    this.ammo = this.cfg.magSize;
    this.reserve = this.cfg.reserve;
    this._autoReload = -1;
    this._cool = 0; this._shotIndex = 0; this._heat = 0; this._bloom = 0;
    this._sinceShot = 99; this._burst = 0; this._triggerEdge = false;
    this._boltCycle = 1; this._chargeT = 0; this._muzzleFlash = 0; this._trigPull = 0;
    this._idle = 0; this._inspectT = -1; this._lowReady = 0; this._raise = 0;
    this._bobAmp = 0; this._bobPhase = 0; this._sprintT = 0;

    // Recoil bookkeeping is dropped rather than handed back: the camera has just
    // been re-aimed, so there is nothing left to recover toward.
    this._camRecP = 0; this._camRecY = 0; this._camPermP = 0; this._camPermY = 0;
    this._camAppP = 0; this._camAppY = 0;
    this._shakeAmp = 0; this._shakeP = 0; this._shakeY = 0;

    // A re-aim is not a look input. Latch the new angles before the angular
    // velocity is measured, or the whole turn arrives as one 14rad/s spike and
    // the gun swings off frame for half a second.
    this._prevYaw = pl.yaw || 0; this._prevPitch = pl.pitch || 0;
    this._yawVel = 0; this._pitchVel = 0;

    // Springs to rest, so the first frame after the cut is a settled pose.
    const s = this._s;
    for (let i = 0; i < SP.N; i++) { s.x[i] = 0; s.v[i] = 0; }
    s.x[SP.ADS] = this._adsRaw;

    // Restart the idle clock and the sub-step accumulator. Every idle term is a
    // bare sine of _t, so a pose re-entered at t=0 and advanced a fixed number of
    // frames lands in exactly the same sway phase every time — which is what
    // makes a screenshot comparable with the same screenshot from last round.
    // Left running, the phase depends on however long the page happened to be
    // rendering beforehand.
    this._t = 0; this._acc = 0; this._shakePh = 0;
  }

  setAds(v) {
    v = !!v;
    this.ads = v;
    this.player.setAds(v);
    this._idle = 0;
    this._inspectT = -1;
  }

  switchFire() {
    const list = MODES[this.cfg.class] || MODES.assault;
    this.fireMode = list[(list.indexOf(this.fireMode) + 1) % list.length];
    this._burst = 0;
    this._idle = 0;
    bus.emit(EV.WEAPON_STATE, this.snapshot());
  }

  reload() {
    if (this.reloading || this.ammo >= this.cfg.magSize || this.reserve <= 0) return;
    this._reloadEmpty = this.ammo === 0;
    this._reloadSeq = this._reloadEmpty ? RELOAD_EMPTY : RELOAD_TAC;
    this._reloadLen = Math.max(0.4, this._reloadEmpty ? this.cfg.reloadEmptyTime : this.cfg.reloadTime);
    this.reloading = true;
    this._reloadT = 0;
    this._reloadPhase = 1;    // 'start' is emitted here, so the next hook is [1]
    this._idle = 0;
    this._inspectT = -1;
    this._s.v[SP.HEAVE] -= 0.9;
    bus.emit(EV.RELOAD, { phase: 'start', empty: this._reloadEmpty, weapon: this.cfg });
  }

  dispose() {
    for (const off of this._offs) off();
    this._offs.length = 0;
    if (this.rig.parent) this.rig.parent.remove(this.rig);
    // The camera is borrowed, not owned: give its frustum back exactly as found.
    if (this._fovBase > 0 && this.camera.isPerspectiveCamera) {
      this.camera.fov = this._fovBase;
      this.camera.updateProjectionMatrix();
      this._fovWritten = this._fovBase;
    }
  }

  // ------------------------------------------------------------------ update
  update(dt) {
    if (!(dt > 0)) dt = 1 / 60;
    dt = Math.min(dt, 0.1);
    this._t += dt;

    const pl = this.player;
    const cfg = this.cfg;

    // --- discontinuity check first, before anything reads a delta. 1.5m in one
    // frame is half again what a sprint plus a long fall can produce at the
    // slowest dt the loop allows, so this only ever fires on a real teleport:
    // a respawn, or the harness re-posing between shots.
    const ppos = this._playerPos();
    if (this._lastPos.distanceToSquared(ppos) > 2.25) this._rePose();
    this._lastPos.copy(ppos);

    // --- stance, read defensively: player.js may express crouch as 0..1 or bool
    const crouch = typeof pl.crouch === 'number' ? sat(pl.crouch) : (pl.crouch ? 1 : 0);
    const grounded = pl.grounded !== false;
    const speed = pl.speed || 0;
    const sprinting = !!pl.sprint && speed > 2.0;

    // --- camera angular velocity, excluding recoil this module applied itself.
    // _prevYaw is latched *after* recoil is pushed, so what is measured here is
    // purely the player's look input.
    const rawYaw = (pl.yaw || 0) - this._prevYaw;
    const rawPitch = (pl.pitch || 0) - this._prevPitch;
    const inv = 1 / dt;
    // Low-pass: a single 8ms mouse spike must not snap the gun across frame.
    const lp = 1 - Math.exp(-dt * 22);
    this._yawVel += (clamp(rawYaw * inv, -14, 14) - this._yawVel) * lp;
    this._pitchVel += (clamp(rawPitch * inv, -14, 14) - this._pitchVel) * lp;

    // --- sprint / raise blends. Coming out of sprint the gun has to be brought
    // back on target before it can fire; that lockout is what stops sprint
    // cancelling from being a free instant-shot exploit. It only arms once the
    // weapon is actually stowed, so brushing the sprint key costs nothing.
    const sprintTarget = sprinting && !this.reloading ? 1 : 0;
    const rate = sprintTarget ? dt / 0.17 : dt / 0.13;
    this._sprintT = clamp(this._sprintT + (sprintTarget ? rate : -rate), 0, 1);
    if (sprintTarget && this._sprintT > 0.5) this._raise = 0.20;
    this._raise = Math.max(0, this._raise - dt);
    if (sprinting && this.reloading) this._cancelReload();

    // --- ADS. adsT is the eased public value; the spring adds the settle.
    if (Math.abs(this.adsT - this._adsWritten) > 1e-6) {
      // Someone (the bench harness) assigned adsT directly. Adopt it and snap
      // the settle spring, so a captured ADS frame is a settled sight picture
      // rather than whatever the transition happened to be doing.
      this._adsRaw = sat(this.adsT);
      this._s.x[SP.ADS] = this._adsRaw;
      this._s.v[SP.ADS] = 0;
    }
    const adsAllowed = !sprinting;
    const adsGoal = this.ads && adsAllowed ? 1 : 0;
    // Eased as an ODE, not as a curve stretched over a linear ramp: the rate is a
    // power of the distance left to travel, which integrates to exactly
    // 1-(1-t/T)^p going in and (1-t/T)^q coming out. Both leave their rest value
    // at full speed and land soft, both finish in finite time, and — the reason
    // it is done this way — a mind changed halfway through reverses continuously
    // instead of snapping between two different curves.
    const v0 = this._adsRaw;
    if (adsGoal) {
      const T = Math.max(0.05, cfg.adsTime * 0.88);
      this._adsRaw = Math.min(1, v0 + (ADS_IN_P / T) * Math.pow(1 - v0, 1 - 1 / ADS_IN_P) * dt);
    } else {
      const T = Math.max(0.05, cfg.adsTime * 0.74);
      this._adsRaw = Math.max(0, v0 - (ADS_OUT_P / T) * Math.pow(v0, 1 - 1 / ADS_OUT_P) * dt);
    }
    this.adsT = this._adsRaw;
    this._adsWritten = this.adsT;
    const a = this.adsT;

    // --- FOV compression. Aiming has to magnify the world or it is just the gun
    // moving; this is the half of ADS the player actually reads. The camera
    // belongs to main.js, so it is written only when the value really changes and
    // is handed back untouched on dispose().
    if (this._fovBase > 0) {
      const f = this._fovBase + (this._fovAds - this._fovBase) * a;
      if (Math.abs(f - this.camera.fov) > 1e-4) {
        this.camera.fov = f;
        this.camera.updateProjectionMatrix();
        this._fovWritten = f;
      }
    }

    // --- fire control. The cooldown is floored: it may bank a couple of rounds
    // of catch-up credit after a frame stall, never a whole magazine's worth
    // after a 2.6s reload. Without the floor a held trigger repays the entire
    // reload as a 3-round-a-frame dump the instant the gun comes back.
    this._cool = Math.max(this._cool - dt, -0.18);
    this._sinceShot += dt;
    this._heat = Math.max(0, this._heat - dt * (this._sinceShot > 0.25 ? 1.6 : 0));
    this._bloom = Math.max(0, this._bloom - dt * (this._sinceShot > 0.12 ? 2.4 : 0));
    this._boltCycle = Math.min(1, this._boltCycle + dt / Math.max(0.03, this._boltTime()));
    this._muzzleFlash = Math.max(0, this._muzzleFlash - dt * 22);

    // Sprinting locks the trigger the instant the key goes down, not when the
    // pose finishes blending — a shot squeezed out of a lowering weapon reads as
    // a bug even when the maths says the muzzle was still up.
    this.ready = !this.reloading && this._raise <= 0 && !sprinting && this._sprintT < 0.35;
    if (this.reloading) this._stepReload(dt);
    else this._tryFire(dt, crouch, grounded, speed);

    // --- dry-magazine auto-reload. A held trigger must never produce a dead
    // frame. This routes through reload() itself rather than duplicating it, so
    // the RELOAD phases, the RELOAD_EMPTY choreography and the WEAPON_STATE
    // snapshots are indistinguishable from a manual reload. The short delay is
    // deliberate: the last muzzle flash and the bolt locking back both have to be
    // seen before the hands move, or the mag change looks like a dropped frame.
    if (this._autoReload >= 0) {
      if (this.reloading || this.ammo > 0 || this.reserve <= 0) this._autoReload = -1;
      else {
        this._autoReload = Math.max(0, this._autoReload - dt);
        // Not ready (sprinting, or still bringing the weapon back up) just holds
        // the request; it fires the moment the gun is available again.
        if (this._autoReload === 0 && this.ready) { this._autoReload = -1; this.reload(); }
      }
    }
    // The charge cycle runs on its own clock so it can finish even if the
    // reload is cancelled out from under it.
    if (this._chargeT > 0) this._chargeT += dt;

    // --- idle escalation: low ready, then an inspect. Any input kills both.
    const busy = this.firing || this.reloading || this.ads || speed > 0.6 || !grounded;
    if (busy) { this._idle = 0; if (this._inspectT >= 0 && (this.firing || this.reloading)) this._inspectT = -1; }
    else this._idle += dt;
    if (this._inspectT >= 0) {
      this._inspectT += dt / 2.6;
      if (this._inspectT >= 1) this._inspectT = -1;
    } else if (this._idle > 14) {
      this._inspectT = 0; this._idle = 0;
    }
    const lrGoal = !busy && this._idle > 5.5 && this._inspectT < 0 ? 1 : 0;
    this._lowReady = clamp(this._lowReady + (lrGoal ? dt / 0.9 : -dt / 0.35), 0, 1);

    // --- stride-locked bob
    this._stepBob(dt, speed, grounded, sprinting);

    // --- spring targets, then a fixed-step integration
    this._driveSprings(a);
    let acc = (this._acc || 0) + dt;
    let n = 0;
    while (acc >= SUB && n < MAX_SUB) { this._s.step(SUB); acc -= SUB; n++; }
    if (n === MAX_SUB) acc = 0;
    this._acc = acc;

    // --- camera recoil is applied as a delta, so the player is free to fight it
    this._applyCameraRecoil(dt);

    this._pose(a, crouch);
    this._animateParts(dt);

    this._prevYaw = pl.yaw || 0;
    this._prevPitch = pl.pitch || 0;
    bus.emit(EV.WEAPON_STATE, this.snapshot());
  }

  snapshot() {
    const s = this._snap;
    s.ammo = this.ammo; s.reserve = this.reserve; s.reloading = this.reloading;
    s.ads = this.ads; s.adsT = this.adsT; s.firing = this.firing;
    s.fireMode = this.fireMode; s.ready = this.ready; s.spread = this.spread;
    s.bloom = this._bloom; s.sprint = this._sprintT; s.flash = this._muzzleFlash;
    s.reloadPhase = this.reloading ? this._reloadSeq[Math.max(0, this._reloadPhase - 1)][0] : '';
    return s;
  }

  // ------------------------------------------------------------------ firing
  _boltTime() { return Math.min(0.075, (60 / this.cfg.rpm) * 0.85); }

  _tryFire(dt, crouch, grounded, speed) {
    const interval = 60 / this.cfg.rpm;
    const mode = this.fireMode;

    if (!this.ready) { this._triggerEdge = false; return; }

    // Semi and burst are edge-triggered; auto is level-triggered.
    let wants = false;
    if (mode === 'auto') wants = this.firing;
    else if (mode === 'semi') wants = this._triggerEdge;
    else if (mode === 'burst') wants = this._triggerEdge || this._burst > 0;
    this._triggerEdge = false;

    if (!wants) return;
    if (this.ammo <= 0) {
      // Pulling on an empty magazine with rounds left in reserve is a request to
      // reload, not a request to hear a click. If the round that emptied the mag
      // has already armed the auto path, let its short beat play out — the bolt
      // locking back is the feedback that says *why* the gun stopped.
      if (this.reserve > 0) {
        if (this._autoReload < 0) this.reload();
        return;
      }
      // Genuinely out. Dry fire: the trigger still moves and the click is worth
      // hearing.
      if (mode !== 'auto' || this._cool <= -0.12) {
        this._trigPull = 1;
        this._cool = 0.12;
        bus.emit(EV.WEAPON_STATE, this.snapshot());
      }
      return;
    }
    // Catch-up is capped: at 30fps a 900rpm gun still owes two rounds a frame,
    // but an unbounded loop after a stall would empty the magazine instantly.
    let fired = 0;
    while (this._cool <= 0 && this.ammo > 0 && fired < 3) {
      this._fire(crouch, grounded, speed);
      this._cool += interval;
      fired++;
      if (mode === 'burst') {
        this._burst = this._burst > 0 ? this._burst - 1 : 2;
        if (this._burst === 0) { this._cool += interval * 1.6; break; }
      }
      if (mode === 'semi') break;
    }
  }

  _fire(crouch, grounded, speed) {
    const cfg = this.cfg;
    this.ammo--;
    this._sinceShot = 0;
    this._boltCycle = 0;
    this._trigPull = 1;
    this._muzzleFlash = 1;
    this._idle = 0;

    // --- spread: base by stance, grown by bloom, capped so it stays learnable
    const moveMul = 1 + Math.min(1, speed / 5) * 0.55 + (grounded ? 0 : 0.9);
    const stanceMul = 1 - crouch * 0.28;
    const base = this.adsT > 0.6 ? cfg.spreadAds : cfg.spreadHip;
    const bloomMul = this.adsT > 0.6 ? 1 + this._bloom * 0.35 : 1 + this._bloom * 0.9;
    const spread = base * stanceMul * bloomMul * (this.adsT > 0.6 ? 1 : moveMul);
    this.spread = spread;
    this._bloom = Math.min(1.6, this._bloom + 0.22);
    this._heat = Math.min(1, this._heat + 0.14);

    // --- pattern lookup
    const i = this._shotIndex % this._patN;
    this._shotIndex++;
    let kickH = this._pattern[i * 2];
    let kickV = this._pattern[i * 2 + 1];
    // Random component is deliberately tiny cold and meaningful hot: the first
    // three rounds of any burst are effectively deterministic.
    const jitter = 0.04 + this._heat * 0.30;
    kickV *= 1 + this._rng.gauss() * jitter * 0.4;
    kickH += cfg.recoil.horizontal * this._rng.gauss() * jitter * 0.55;

    // --- stance multipliers on muzzle climb
    const climbMul = (1 - crouch * 0.24) * (this.adsT > 0.5 ? 0.86 : 1) *
      (grounded ? 1 : 1.35) * (1 + Math.min(1, speed / 6) * 0.12);
    kickV *= climbMul;
    kickH *= climbMul;

    // --- viewmodel kick: big, fast, fully recovers. Impulses are in spring
    // units of metres and radians per second; peak displacement for an impulse
    // J on these springs is roughly J/75, so 0.65 back is about 9mm of travel.
    const s = this._s;
    const vmScale = 1 - this.adsT * 0.45;
    s.v[SP.KICK_Z] += 0.82 * kickV * vmScale;              // straight back into the shoulder
    s.v[SP.KICK_Y] += 0.26 * kickV * vmScale;
    s.v[SP.KICK_RX] += 3.1 * kickV * vmScale;              // muzzle rises ~3 degrees
    s.v[SP.KICK_RY] += 3.5 * kickH * vmScale;
    s.v[SP.KICK_RZ] += 3.0 * kickH * vmScale;              // and rolls with it

    // --- camera: smaller, slower, only partly given back
    const camP = kickV * 0.0155;
    const camY = kickH * 0.0105;
    this._camRecP += camP;
    this._camRecY += camY;
    const keep = 0.45;                                     // 55% auto-recovers
    this._camPermP += camP * keep;
    this._camPermY += camY * keep;
    this._shakeAmp = Math.min(0.0042, this._shakeAmp + 0.0018 + this._heat * 0.0010);

    // --- shot event, pooled payload
    const cam = this.camera;
    if (cam.parent && cam.parent.isScene) this._shotOrigin.copy(cam.position);
    else this._shotOrigin.setFromMatrixPosition(cam.matrixWorld);
    this._shotDir.set(0, 0, -1).applyQuaternion(cam.quaternion);
    this._shotMsg.spread = spread;
    this._shotMsg.index = i;
    bus.emit(EV.SHOT, this._shotMsg);

    // That was the last round: arm the auto-reload rather than letting the next
    // trigger frame find a dead gun.
    if (this.ammo === 0 && this.reserve > 0 && this._autoReload < 0) {
      this._autoReload = AUTO_RELOAD_DELAY;
    }
  }

  /**
   * Fire one round immediately, bypassing the fire-rate cooldown, the trigger
   * state and the ready gate. Returns true when a round left the barrel.
   *
   * This exists for the screenshot harness. A muzzle flash lives for one or two
   * frames, so a settled capture lands on the gap between rounds almost every
   * time and the frame that is supposed to sell a firefight shows a quiet gun.
   * It is therefore contractually required to produce a shot: an empty magazine
   * is reloaded from reserve first, an in-flight reload is resolved rather than
   * waited on, and in the degenerate case of no reserve at all a round is staged
   * so the harness still gets its flash.
   */
  forceShot() {
    if (this.reloading) {
      this._loadMag();
      this.reloading = false;
      this._chargeT = 0;
      this._reloadPhase = this._reloadSeq.length;
      bus.emit(EV.RELOAD, { phase: 'end', weapon: this.cfg });
    }
    if (this.ammo <= 0) {
      this._loadMag();
      // Two, not one, so the bolt cycles home after the shot instead of locking
      // back — a captured frame should not show a gun that has just run dry.
      if (this.ammo <= 0) this.ammo = 2;
    }
    this._autoReload = -1;
    this._raise = 0;
    const pl = this.player;
    const crouch = typeof pl.crouch === 'number' ? sat(pl.crouch) : (pl.crouch ? 1 : 0);
    this._fire(crouch, pl.grounded !== false, pl.speed || 0);
    // Rejoin the normal cadence: the next scheduled round still waits its turn.
    this._cool = 60 / this.cfg.rpm;
    bus.emit(EV.WEAPON_STATE, this.snapshot());
    return true;
  }

  // ------------------------------------------------------------------ reload
  _cancelReload() {
    if (!this.reloading) return;
    this.reloading = false;
    this._cool = Math.max(this._cool, 0);
    this._chargeT = 0;
    bus.emit(EV.RELOAD, { phase: 'end', cancelled: true, weapon: this.cfg });
  }

  _stepReload(dt) {
    this._reloadT += dt;
    const u = this._reloadT / this._reloadLen;
    const seq = this._reloadSeq;
    while (this._reloadPhase < seq.length && u >= seq[this._reloadPhase][1]) {
      const phase = seq[this._reloadPhase][0];
      this._reloadPhase++;
      if (phase === 'magin' && !this._reloadEmpty) this._loadMag();
      if (phase === 'charge') { this._loadMag(); this._chargeT = 0.0001; }
      if (phase === 'magin') this._s.v[SP.JOLT] += 0.9;       // the seating bump
      if (phase === 'end') {
        this._loadMag();                                     // safety net
        this.reloading = false;
        // A fresh magazine starts the fire clock, so the first round out of it
        // is one round and not the catch-up the reload would otherwise owe.
        this._cool = Math.max(this._cool, 0);
        bus.emit(EV.RELOAD, { phase: 'end', weapon: this.cfg });
        return;
      }
      bus.emit(EV.RELOAD, { phase, empty: this._reloadEmpty, weapon: this.cfg });
    }
  }

  _loadMag() {
    const need = this.cfg.magSize - this.ammo;
    if (need <= 0 || this.reserve <= 0) return;
    const take = Math.min(need, this.reserve);
    this.ammo += take;
    this.reserve -= take;
  }

  // ------------------------------------------------------------------ motion
  _syncStride() {
    // A footstep means a foot just planted, which is phase 0 or PI. Nudge rather
    // than snap so a mistimed step does not visibly jump the weapon.
    const p = this._bobPhase % Math.PI;
    const err = p < Math.PI / 2 ? -p : Math.PI - p;
    this._bobPhase += err * 0.35;
  }

  _land(speed) {
    const s = Math.min(1, (speed || 0) / 9);
    if (s <= 0.05) return;
    this._s.v[SP.JOLT] -= 2.6 * s;
    this._s.v[SP.KICK_RX] -= 1.1 * s;
    this._idle = 0;
  }

  _stepBob(dt, speed, grounded, sprinting) {
    const stride = sprinting ? 4.10 : 3.10;    // metres per full gait cycle
    if (grounded && speed > 0.5) {
      this._bobPhase += (TAU * speed * dt) / stride;
      if (this._bobPhase > TAU * 64) this._bobPhase -= TAU * 64;
    }
    const goal = grounded ? Math.min(1, speed / (sprinting ? 7.2 : 4.6)) : 0;
    this._bobAmp += (goal - this._bobAmp) * Math.min(1, dt * 6);
  }

  _driveSprings(a) {
    const t = this._s.t;
    // Lag targets are proportional to turn rate. Sign: the rig is a child of the
    // camera, so a camera yaw of +d leaves a world-fixed gun at -d in rig space.
    // Rotational lag survives ADS almost intact — it pivots on the optic, so it
    // costs no sight picture. Positional lag does move the dot, so it is nearly
    // switched off when aiming.
    const rotScale = 1 - a * 0.55;
    const posScale = 1 - a * 0.88;
    const yv = this._yawVel, pv = this._pitchVel;
    t[SP.LAG_RY] = clamp(-yv * 0.115, -0.20, 0.20) * rotScale;
    t[SP.LAG_RX] = clamp(-pv * 0.100, -0.16, 0.16) * rotScale;
    t[SP.LAG_RZ] = clamp(yv * 0.070, -0.13, 0.13) * rotScale;
    // Clamps are pre-NEAR_COMP metres; on screen they are the same swing they
    // always were, because _pose scales the whole offset sum.
    t[SP.LAG_X] = clamp(yv * 0.0125, -0.030, 0.030) * posScale;
    t[SP.LAG_Y] = clamp(-pv * 0.0090, -0.022, 0.022) * posScale;
    t[SP.ADS] = a;
    t[SP.JOLT] = 0;
    t[SP.HEAVE] = this._sprintT * 0.4;
    t[SP.KICK_Z] = 0; t[SP.KICK_Y] = 0;
    t[SP.KICK_RX] = 0; t[SP.KICK_RY] = 0; t[SP.KICK_RZ] = 0;
  }

  _applyCameraRecoil(dt) {
    // Exponential return toward the permanent residue, then a rebase once it has
    // arrived: the residue stays in the camera and is never handed back, so the
    // player has to pull down for it.
    const k = 1 - Math.exp(-dt * this.cfg.recoil.recovery * 0.55);
    this._camRecP += (this._camPermP - this._camRecP) * k;
    this._camRecY += (this._camPermY - this._camRecY) * k;

    // Muzzle-blast shake: a few pixels, high frequency, always fully recovers.
    // Two sines rather than noise, so it is deterministic between rounds.
    this._shakeAmp = Math.max(0, this._shakeAmp - dt * 0.035);
    this._shakePh += dt * 108;
    const sh = this._shakeAmp;
    this._shakeP = Math.sin(this._shakePh) * sh * 0.6;
    this._shakeY = Math.sin(this._shakePh * 0.71 + 1.1) * sh * 0.45;

    const wantP = this._camRecP + this._shakeP;
    const wantY = this._camRecY + this._shakeY;
    const dP = wantP - this._camAppP;
    const dY = wantY - this._camAppY;
    if (dP !== 0 || dY !== 0) this.player.applyRecoil(dP, dY);
    this._camAppP = wantP;
    this._camAppY = wantY;

    if (this._sinceShot > 0.30 && Math.abs(this._camRecP - this._camPermP) < 0.0004) {
      // Rebase without applying anything: the offsets vanish from bookkeeping
      // while staying in the camera.
      this._camAppP -= this._camPermP; this._camAppY -= this._camPermY;
      this._camRecP -= this._camPermP; this._camRecY -= this._camPermY;
      this._camPermP = 0; this._camPermY = 0;
    }
  }

  /** Compose every contribution into one rig transform. Allocation-free. */
  _pose(a, crouch) {
    const s = this._s.x;
    const T = this._t;
    const sp = smooth(this._sprintT);
    const lr = this._lowReady;

    // --- idle: a breathing figure eight (y at twice the x rate) plus a much
    // slower drift on an incommensurate period, so the loop never closes. Every
    // term is a bare sine of the clock, so at t=0 the pose is exactly rest and
    // captured frames are reproducible.
    const idleAmp = (1 - a * 0.88) * (1 - sp) * (1 - this._bobAmp * 0.7);
    const b1 = T * 0.62, b2 = T * 0.1731;
    const swayX = (Math.sin(b1) * 0.0042 + Math.sin(b2) * 0.0026) * idleAmp;
    const swayY = (Math.sin(b1 * 2) * 0.0026 + Math.sin(b2 * 1.37) * 0.0019) * idleAmp;
    const swayRZ = (Math.sin(b1 * 0.5) * 0.010 + Math.sin(b2 * 0.83) * 0.006) * idleAmp;
    const swayRX = Math.sin(b1 * 2) * 0.007 * idleAmp;
    const breathe = Math.sin(T * 1.15) * 0.0022 * idleAmp;

    // --- walk bob: the same figure eight, but locked to the stride
    const amp = this._bobAmp * (1 - a * 0.72) * (1 + sp * 0.55);
    const ph = this._bobPhase;
    const bobX = Math.sin(ph) * 0.0135 * amp;
    const bobY = -Math.abs(Math.cos(ph)) * 0.0105 * amp + 0.004 * amp;
    const bobZ = Math.sin(ph * 2 + 0.6) * 0.0055 * amp;
    const bobRZ = Math.sin(ph) * 0.020 * amp;
    const bobRX = Math.cos(ph * 2) * 0.011 * amp;

    // --- sprint pose: canted inward and down, weapon out of the sight line
    const spX = -0.052 * sp, spY = -0.058 * sp, spZ = 0.055 * sp;
    const spRX = -0.30 * sp, spRY = 0.46 * sp, spRZ = 0.62 * sp;

    // --- low ready and inspect
    const lrRX = -0.30 * lr, lrY = -0.055 * lr, lrZ = 0.02 * lr;
    let insRX = 0, insRY = 0, insRZ = 0, insX = 0, insY = 0, insZ = 0;
    if (this._inspectT >= 0) {
      const u = this._inspectT;
      const e = bell(u, 0.28, 0.30);
      const turn = smooth(seg(u, 0.10, 0.45)) * (1 - smooth(seg(u, 0.62, 0.95)));
      insRY = (0.30 * e + 0.55 * turn);
      insRZ = (0.22 * e + 0.40 * turn);
      insRX = -0.10 * e + 0.16 * turn;
      insX = -0.045 * e; insY = 0.030 * e; insZ = 0.048 * e;
    }

    // --- reload pose
    let rlX = 0, rlY = 0, rlZ = 0, rlRX = 0, rlRY = 0, rlRZ = 0;
    if (this.reloading) {
      const u = sat(this._reloadT / this._reloadLen);
      const e = bell(u, 0.16, 0.18);
      // The rifle rotates in toward the centre line and tips the magwell up so
      // the off hand can actually reach it.
      rlRY = 0.30 * e; rlRZ = 0.42 * e; rlRX = -0.20 * e;
      rlX = -0.036 * e; rlY = -0.030 * e; rlZ = 0.052 * e;
      // seating bump and, on an empty reload, the yank on the charging handle
      const seatU = seg(u, this._seatStart(), this._seatEnd());
      const bump = Math.sin(seatU * Math.PI) * (seatU > 0 && seatU < 1 ? 1 : 0);
      rlY -= bump * 0.010; rlRX -= bump * 0.05;
      if (this._chargeT > 0) {
        const c = sat(this._chargeT / 0.26);
        const pull = Math.sin(c * Math.PI);
        rlRZ += pull * 0.14; rlX += pull * 0.012;
      }
    }

    // Springs already carry metres and radians; no second scale factor here.
    const kickZ = s[SP.KICK_Z];
    const kickY = s[SP.KICK_Y];
    const jolt = s[SP.JOLT] * 0.5;
    const heave = s[SP.HEAVE];
    const cr = crouch * (1 - sp);

    // --- ADS blend parameter. Allowed slightly out of range so the settle
    // spring's overshoot survives: the gun rolls a touch past level and comes
    // back, which is most of what sells the raise as weighted.
    const ap = clamp(s[SP.ADS], -0.12, 1.12);
    const hip = 1 - ap;

    // --- the raise is an arc, not a slide. A bell over the transition, zero at
    // both settled ends so neither rest pose is disturbed: the muzzle dips and
    // the cant rolls out early while the mass is still catching up, then the
    // rifle comes up into the eye. Without this the sight simply translates.
    const arc = sat(4 * ap * (1 - ap));
    const arcRX = -0.055 * arc, arcRZ = -0.045 * arc;
    const arcY = -0.010 * arc, arcZ = 0.014 * arc;

    // --- rotation. The hip rest angles blend out completely by full ADS: any
    // yaw or pitch left in at the sight picture walks the dot off centre.
    this._euler.set(
      HIP_RX * hip + arcRX
        + s[SP.LAG_RX] + swayRX + bobRX + spRX + lrRX + insRX + rlRX + s[SP.KICK_RX] + cr * 0.035,
      HIP_RY * hip
        + s[SP.LAG_RY] + spRY + insRY + rlRY + s[SP.KICK_RY],
      HIP_RZ * hip + arcRZ
        + s[SP.LAG_RZ] + swayRZ + bobRZ + spRZ + insRZ + rlRZ + s[SP.KICK_RZ],
      'YXZ',
    );
    this._quat.setFromEuler(this._euler);

    // --- pivot blend: shoulder at the hip, dead on the optic at full ADS.
    this._pivot.lerpVectors(this._hipPivot, this._opticLocal, ap);
    this._pos.lerpVectors(this._hipBase, this._adsBase, ap);
    // pos = base + pivot - R*pivot  →  the rotation happens about `pivot`
    this._pos.add(this._pivot);
    _tmp.copy(this._pivot).applyQuaternion(this._quat);
    this._pos.sub(_tmp);

    // Everything from here is a small positional offset in metres, and the rest
    // pose sits near enough to the eye that a millimetre is worth ~1.4x the
    // screen angle these numbers were tuned against. NEAR_COMP takes the angular
    // amplitude back to the tuned value in one place, so the relative weighting
    // of sway against bob against kick is untouched.
    this._pos.x += (s[SP.LAG_X] + swayX + bobX + spX + insX + rlX) * NEAR_COMP;
    this._pos.y += (s[SP.LAG_Y] + swayY + breathe + bobY + kickY + jolt + spY + lrY + insY
      + rlY + arcY - cr * 0.010) * NEAR_COMP;
    this._pos.z += (bobZ + kickZ + spZ + lrZ + insZ + rlZ + arcZ
      - heave * 0.012 - cr * 0.006) * NEAR_COMP;

    this.rig.position.copy(this._pos);
    this.rig.quaternion.copy(this._quat);
  }

  _seatStart() { return this._reloadEmpty ? 0.52 : 0.60; }
  _seatEnd() { return this._seatStart() + 0.12; }

  // ------------------------------------------------------------- moving parts
  _animateParts(dt) {
    const P = this.parts;
    if (!P) return;
    const R = this._rest;

    // --- bolt: back fast, forward slower, then locked open on an empty gun
    let bolt = 0;
    if (this._boltCycle < 1) {
      const c = this._boltCycle;
      bolt = c < 0.38 ? smooth(c / 0.38) : 1 - smoother((c - 0.38) / 0.62);
    }
    const holdOpen = this.ammo <= 0 && !(this.reloading && this._chargeT > 0) ? 1 : 0;
    if (holdOpen) bolt = Math.max(bolt, 1);
    let chg = 0;
    if (this._chargeT > 0) {
      const c = sat(this._chargeT / 0.26);
      const pull = c < 0.55 ? smooth(c / 0.55) : 1 - smoother((c - 0.55) / 0.45);
      chg = pull;
      bolt = Math.max(bolt, pull);
      if (this._chargeT > 0.30) this._chargeT = 0;
    }
    if (P.bolt) P.bolt.position.z = R.boltZ + bolt * 0.034;
    if (P.charging) P.charging.position.z = R.chgZ + chg * 0.052;

    // --- trigger finger
    const pullGoal = this.firing && this.ready && this.ammo > 0 ? 1 : 0;
    this._trigPull += (pullGoal - this._trigPull) * Math.min(1, dt * 26);
    if (P.trigger) P.trigger.rotation.x = -0.30 * this._trigPull;

    // --- magazine: drops away, gone, then a new one comes up and seats
    if (P.mag) {
      let my = 0, mz = 0, mx = 0, mrx = 0, vis = true;
      if (this.reloading) {
        const u = sat(this._reloadT / this._reloadLen);
        const out = this._reloadSeq[1][1];              // 'magout'
        const inn = this._reloadSeq[2][1];              // 'magin'
        if (u >= out && u < inn) {
          const d = seg(u, out, out + 0.15);
          const rise = seg(u, inn - 0.18, inn);
          if (rise <= 0) {
            my = -0.30 * smooth(d); mz = -0.02 * d; mrx = -0.75 * smooth(d);
            vis = d < 0.98;
          } else {
            my = -0.26 * (1 - smoother(rise));
            mx = 0.030 * (1 - smooth(rise));
            mrx = -0.30 * (1 - smooth(rise));
            vis = true;
          }
        } else if (u >= inn) {
          const bump = Math.sin(seg(u, inn, inn + 0.10) * Math.PI);
          my = -0.006 * bump;
        }
      }
      P.mag.visible = vis;
      P.mag.position.set(R.magX + mx, R.magY + my, R.magZ + mz);
      P.mag.rotation.x = mrx;
    }
  }
}
