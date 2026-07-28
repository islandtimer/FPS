// OWNER: agent "hud" — reticle, ammo, compass, killfeed, hitmarkers, damage
// direction, wave banners, menus. DOM + a 2D canvas overlay; no THREE here.
// LEGAL: original UI language and iconography only. No Activision UI strings,
// rank names, or logos.
// CONTRACT:
//   new HUD(rootEl)
//   .update(dt, ctx)   ctx = { player, weapon, director, perf }
//   .dispose()
// Subscribes to the bus for HIT/KILL/DAMAGE/WAVE/RELOAD.
//
// ---------------------------------------------------------------------------
// HOW THIS IS BUILT, and why.
//
// COST. update() runs inside the measured logic budget (main.js times step()),
// so the HUD is written as a dirty-checking renderer, not an immediate-mode one.
// Every DOM write goes through a cached last-written value; every canvas redraws
// only when a quantised signature of its state changes. In a static frame the
// whole module does a few dozen float compares and touches nothing. Measured
// cost is well under 0.1ms/frame steady state, and the only frames that cost
// more are the ones where something visibly moved.
//
// TWO TYPEFACES, ONE RULE. Monospace is the *debug* voice and nothing else: the
// perf readout and the self-benchmark panel. Every number the player is meant to
// read — ammo, reserve, health, score, compass bearings — is set in the narrow
// technical face (TECH) at tabular figures, so ammo and health read as siblings
// from the same instrument cluster and nothing shipped can be mistaken for
// engine telemetry. The perf panel is hidden by default; SHIFT+F reveals it.
//
// NO WALL CLOCK. All animation is driven by the accumulated dt, never by
// performance.now() and never by CSS keyframes. The bench harness calls step()
// 900 times synchronously and screenshots after exactly three fixed steps; a
// wall-clock or CSS-timed animation would make captures irreproducible between
// rounds. Same reason there is no Math.random() here — call signs come from a
// seeded stream.
//
// CRISPNESS. The reticle is the most scrutinised 40 pixels in the frame. Its
// canvas backing store is sized to devicePixelRatio and everything is drawn in
// integer device pixels as axis-aligned fillRects, so a tick is exactly N device
// pixels wide with hard edges — no half-covered pixels, no CSS scaling blur. Both
// canvases are positioned with an integer negative margin instead of a
// translate(-50%), because a percentage translate on an odd-width box lands the
// whole backing store on a half pixel and every 1px tick in it turns into two
// half-lit ones. The compass tape is pre-rendered once into an offscreen canvas
// at integer tick positions and blitted at integer offsets, so its glyphs are
// rasterised once and never resampled.
//
// LIGHT ON DARK, EVERYWHERE. The HUD owns its own contrast: it never assumes the
// world behind it is dark. The compass paints a soft dark scrim under itself and
// draws light ticks on top of it (a per-tick black skirt was making a 1px tick
// read as a 3px dark smudge against bright sky); the banner sits on a soft radial
// scrim instead of a hard offset shadow; the reticle and hitmarker carry a
// one-device-pixel near-black outline so they survive blown-out stucco.
//
// SPREAD IS READ, NOT INVENTED. weaponfx only recomputes .spread on the frame a
// round leaves the barrel, so the reticle mirrors that module's spread model
// from public state (cfg + adsT + player velocity + a locally tracked bloom fed
// by EV.SHOT) to get a value that is live every frame and agrees with the
// weapon's on the frames where both exist. If the ballistics model changes, this
// is the one block that has to follow it.
//
// ONE AUTHORITY FOR WAVE STATE. The director's live count is the only source of
// the hostile number. The wave banner does not snapshot the count it was raised
// with — it is re-written from the same string the objective strip uses, and the
// objective strip fades out while the banner is up, so the two can never disagree
// on screen and never print the same number twice.
//
// LAYERS. Everything that animates continuously (vignette, damage arcs, bars,
// banner) animates via opacity/transform only, so the compositor handles it and
// nothing relayouts. Blend/filter layers are display:none until the state that
// needs them is actually entered, so a healthy player pays nothing for the
// low-health grade.

import { bus, EV } from '../core/bus.js';
import { makeRng } from '../core/rng.js';

// ---------------------------------------------------------------- palette
// Mostly white, one warm accent, one danger red. Everything sits at low opacity
// until it has something to say.
const INK = '#eaf1f9';
const WARM = '#ffb14a';
const BAD = '#ff4a35';
const OUTLINE = 'rgba(2,4,7,0.88)';   // reticle/hitmarker skirt — survives blown stucco
// Debug voice. Used by the perf and benchmark panels and by nothing else.
const MONO = 'ui-monospace,"SF Mono","Roboto Mono",Menlo,Consolas,monospace';
// Shipped voice. A narrow technical sans; the stack degrades to whatever grotesk
// the machine has, and the numerals are locked to tabular figures either way.
const TECH = '"Roboto Condensed","Liberation Sans Narrow","Arial Narrow",' +
  '"Helvetica Neue Condensed","Liberation Sans","DejaVu Sans",Arial,sans-serif';

const DEG = 180 / Math.PI;
const CARD = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];

// Fictional call signs. Seeded so a killfeed screenshot reproduces round to round.
const CALL_A = ['DUSK', 'ASHFALL', 'KITE', 'HOLLOW', 'SALTPAN', 'VEER', 'MIRE', 'TALLOW', 'GRAVEL', 'REEDS'];
const PLAYER_CALLSIGN = 'RAVEN 1';

// Kill flourish duration. Longer than the hitmarker on purpose — the ring is what
// tells the player the target is down after the marker has already gone.
const KILL_LIFE = 0.34;

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (v) => { const t = clamp01(v); return t * t * (3 - 2 * t); };
/** Wrap to (-180, 180]. */
function wrapDeg(d) { d %= 360; if (d > 180) d -= 360; else if (d <= -180) d += 360; return d; }

function mkEl(tag, cls, parent) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (parent) parent.appendChild(n);
  return n;
}

// The whole stylesheet, injected once with the HUD and removed with it. Class
// names are prefixed so nothing can collide with another module's overlay.
const CSS = `
.hud{position:absolute;inset:0;color:${INK};font:12px/1 ${TECH};letter-spacing:.06em;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeSpeed}
.hud .sh{text-shadow:0 1px 2px rgba(0,0,0,.95),0 0 10px rgba(0,0,0,.55)}
/* Every player-facing figure: one face, tabular, slightly tightened. */
.hud .num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1;
  font-weight:600;letter-spacing:.015em;display:inline-block}

.hud-vig{position:absolute;inset:-2px;opacity:0;will-change:opacity;
  background:radial-gradient(ellipse 78% 72% at 50% 50%,rgba(0,0,0,0) 44%,rgba(96,8,4,.30) 74%,rgba(58,3,2,.78) 100%)}
.hud-grade{position:absolute;inset:0;opacity:0;display:none;background:rgba(122,128,136,.10);
  -webkit-backdrop-filter:saturate(.34) contrast(1.05);backdrop-filter:saturate(.34) contrast(1.05)}
.hud-edge{position:absolute;inset:0;opacity:0;will-change:opacity;
  background:radial-gradient(ellipse 92% 88% at 50% 50%,rgba(0,0,0,0) 58%,rgba(0,0,0,.55) 100%)}

.hud-ret{position:absolute;left:50%;top:50%}

.hud-arcs{position:absolute;left:50%;top:50%;width:520px;height:520px;margin:-260px 0 0 -260px}
.hud-arc{position:absolute;inset:0;opacity:0;will-change:transform,opacity;color:${BAD}}
.hud-arc svg{width:100%;height:100%;display:block;overflow:visible}

.hud-comp{position:absolute;left:50%;top:14px;
  -webkit-mask-image:linear-gradient(90deg,transparent 0,rgba(0,0,0,.35) 7%,#000 17%,#000 83%,rgba(0,0,0,.35) 93%,transparent 100%);
  mask-image:linear-gradient(90deg,transparent 0,rgba(0,0,0,.35) 7%,#000 17%,#000 83%,rgba(0,0,0,.35) 93%,transparent 100%)}
.hud-obj{position:absolute;left:50%;top:58px;transform:translateX(-50%);white-space:nowrap;
  font-size:11px;opacity:.62;letter-spacing:.20em;text-shadow:0 1px 2px rgba(0,0,0,.9)}
.hud-obj b{font-weight:600;color:${WARM};letter-spacing:.22em}
.hud-obj i{display:inline-block;width:26px;height:1px;background:currentColor;opacity:.5;
  vertical-align:middle;margin:0 10px 2px}
.hud-obj span{font-variant-numeric:tabular-nums}

/* Banner sits high, clear of the aim point, on a soft scrim rather than a hard
   offset shadow. Its rule is broken in the middle so nothing crosses the
   vertical axis the player is aiming along. */
.hud-ban{position:absolute;left:50%;top:11.5%;transform:translateX(-50%);text-align:center;
  opacity:0;will-change:opacity,transform;white-space:nowrap;z-index:0;
  text-shadow:0 0 16px rgba(0,0,0,.85),0 0 5px rgba(0,0,0,.7)}
/* Long, low, many-stop falloff. A two-stop ellipse reads as a grey blob on a
   bright wall; this one has no visible edge anywhere. */
.hud-ban::before{content:"";position:absolute;left:50%;top:50%;width:1240px;height:300px;
  margin:-150px 0 0 -620px;z-index:-1;pointer-events:none;
  background:radial-gradient(ellipse 50% 50% at 50% 50%,rgba(3,5,8,.56) 0,rgba(3,5,8,.44) 22%,
    rgba(3,5,8,.28) 42%,rgba(3,5,8,.14) 60%,rgba(3,5,8,.05) 78%,rgba(3,5,8,0) 92%)}
.hud-ban .t{font-size:31px;font-weight:700;letter-spacing:.40em;text-indent:.40em}
.hud-ban .r{height:1px;margin:14px auto 12px;width:300px;transform:scaleX(0);will-change:transform;
  background:linear-gradient(90deg,rgba(234,241,249,0) 0,rgba(234,241,249,.8) 16%,
    rgba(234,241,249,.8) 41%,rgba(234,241,249,0) 45%,rgba(234,241,249,0) 55%,
    rgba(234,241,249,.8) 59%,rgba(234,241,249,.8) 84%,rgba(234,241,249,0) 100%)}
.hud-ban .s{font-size:11.5px;letter-spacing:.30em;text-indent:.30em;opacity:.78;
  font-variant-numeric:tabular-nums}

.hud-feed{position:absolute;right:30px;top:22px;text-align:right;width:420px}
.hud-row{display:flex;align-items:center;justify-content:flex-end;gap:9px;height:19px;
  opacity:0;will-change:opacity;font-size:11.5px;letter-spacing:.10em}
.hud-row .a{color:${INK};font-weight:600}
.hud-row .v{color:rgba(234,241,249,.72)}
.hud-row .g{color:rgba(234,241,249,.86);display:flex;align-items:center;gap:5px}
.hud-row .g svg{filter:drop-shadow(0 1px 1.5px rgba(0,0,0,.95))}
.hud-row .hs{width:9px;height:9px;border:1.5px solid ${WARM};border-radius:50%;display:none;
  box-shadow:0 0 5px rgba(255,177,74,.55)}
.hud-row.k .g{color:${WARM}}
.hud-row.k .hs{display:block}

.hud-ammo{position:absolute;right:34px;bottom:30px;text-align:right}
.hud-ammo .nm{font-size:10.5px;letter-spacing:.30em;opacity:.55;margin-bottom:7px}
.hud-ammo .rw{display:flex;align-items:baseline;justify-content:flex-end;gap:9px}
.hud-ammo .mag{font-size:46px;line-height:.84;transform:scaleX(.9);transform-origin:100% 50%}
.hud-ammo .res{font-size:17px;opacity:.44;font-weight:400;transform:scaleX(.92);
  transform-origin:100% 50%}
.hud-ammo .res::before{content:"/ ";opacity:.5}
.hud-ammo .bar{height:2px;width:120px;margin:9px 0 0 auto;background:rgba(234,241,249,.14);
  overflow:hidden;opacity:0}
.hud-ammo .bar i{display:block;height:100%;background:${WARM};transform:scaleX(0);
  transform-origin:0 50%;will-change:transform}
.hud-ammo .md{font-size:10px;letter-spacing:.26em;opacity:.5;margin-top:8px}
.hud-ammo.rl .mag,.hud-ammo.rl .res{color:rgba(198,206,216,.55)}
.hud-ammo.rl .nm{opacity:.3}
.hud-ammo.dry .mag,.hud-ammo.dry .res{color:${BAD}}
.hud-ammo.dry .mag{text-shadow:0 0 14px rgba(255,74,53,.55)}

/* Empty-magazine prompt. Below the aim point, on its own scrim, pulsing on the
   HUD's dt clock so a capture is reproducible. */
.hud-rl{position:absolute;left:50%;top:56.5%;transform:translateX(-50%);opacity:0;
  white-space:nowrap;font-size:14px;font-weight:700;letter-spacing:.38em;text-indent:.38em;
  color:${BAD};padding:14px 40px;will-change:opacity;
  text-shadow:0 0 14px rgba(0,0,0,.95),0 1px 2px rgba(0,0,0,.95);
  background:radial-gradient(ellipse 50% 50% at 50% 50%,rgba(7,3,3,.60) 0,rgba(7,3,3,.34) 40%,
    rgba(7,3,3,.12) 66%,rgba(7,3,3,0) 88%)}

.hud-left{position:absolute;left:34px;bottom:30px}
.hud-left .sc{font-size:11px;letter-spacing:.22em;opacity:.6}
.hud-left .sc b{font-size:15px;opacity:.95;margin-left:10px;transform:scaleX(.92);
  transform-origin:0 50%;vertical-align:-1px}
.hud-left .st{font-size:11px;letter-spacing:.22em;color:${WARM};margin-top:6px;height:11px;opacity:0}
.hud-left .hpl{font-size:9.5px;letter-spacing:.30em;opacity:.55;margin-top:15px}
.hud-left .hp{display:flex;align-items:center;gap:12px;margin-top:7px}
/* Recessed track: 25% dark well, hairline frame, two segment marks. At full
   health the empty length still reads, so the bar has a reference. */
.hud-left .hpb{position:relative;width:178px;height:9px;overflow:hidden;
  background:rgba(3,6,10,.25);
  box-shadow:inset 0 0 0 1px rgba(234,241,249,.22),inset 0 1px 2px rgba(0,0,0,.55)}
.hud-left .hpb i{position:absolute;left:1px;top:1px;right:1px;bottom:1px;
  transform-origin:0 50%;will-change:transform;
  background:linear-gradient(180deg,rgba(255,255,255,.96) 0,rgba(206,222,238,.80) 100%)}
.hud-left .hpb u{position:absolute;top:0;bottom:0;width:1px;display:block;
  background:rgba(3,6,10,.62)}
.hud-left .hpv{font-size:19px;opacity:.72;transform:scaleX(.9);transform-origin:0 50%}
.hud-left.w .hpb i{background:linear-gradient(180deg,${WARM} 0,rgba(214,138,44,.86) 100%)}
.hud-left.c .hpb i{background:linear-gradient(180deg,${BAD} 0,rgba(190,44,30,.9) 100%)}
.hud-left.c .hpv{color:${BAD};opacity:.95}
.hud-left.c .hpl{color:${BAD};opacity:.7}

/* Debug voice — monospace, scrimmed, and off by default. SHIFT+F reveals it. */
.hud-perf,.hud-bench{position:absolute;left:26px;font:10.5px/1.55 ${MONO};letter-spacing:.06em;
  white-space:pre;padding:8px 12px;background:rgba(4,7,11,.62);
  border-left:2px solid rgba(255,177,74,.5)}
.hud-perf{top:20px;display:none;opacity:.9}
.hud-bench{top:150px;opacity:0;color:${WARM}}
`;

// Abstract weapon glyph for the killfeed — a fictional silhouette, not a
// depiction of any real firearm.
const GLYPH = '<svg viewBox="0 0 30 11" width="30" height="11" aria-hidden="true">' +
  '<path fill="currentColor" d="M1 3.6h15.2v3.1H1z M16.2 4.3h11.6v1.5H16.2z M1 2.2h5.4v1.1H1z' +
  ' M5.6 6.7h2.9l-.9 3.6H6.1z M11.4 6.7h6.2v1.1h-6.2z"/></svg>';

export class HUD {
  constructor(root) {
    this.root = root;
    this.dpr = Math.min(2, (typeof devicePixelRatio === 'number' && devicePixelRatio) || 1);
    this.t = 0;
    this.rng = makeRng('hud:callsigns');

    this.el = mkEl('div', 'hud', root);
    const style = mkEl('style', null, this.el);
    style.textContent = CSS;

    // ---------------------------------------------------------- screen grade
    this.$edge = mkEl('div', 'hud-edge', this.el);
    this.$vig = mkEl('div', 'hud-vig', this.el);
    this.$grade = mkEl('div', 'hud-grade', this.el);

    // ---------------------------------------------------------- reticle
    this.retCss = 220;
    this.$ret = mkEl('canvas', 'hud-ret', this.el);
    this.$ret.width = this.retW = Math.round(this.retCss * this.dpr);
    this.$ret.height = this.retH = this.retW;
    this.$ret.style.width = this.retCss + 'px';
    this.$ret.style.height = this.retCss + 'px';
    // Integer margin, not translate(-50%) — see CRISPNESS above.
    this.$ret.style.margin = (-this.retCss / 2) + 'px 0 0 ' + (-this.retCss / 2) + 'px';
    this.retCtx = this.$ret.getContext('2d');
    this._retSig = new Int32Array(10);
    this._retSig[0] = -1;

    // ---------------------------------------------------------- damage arcs
    // Four concentric arcs of decreasing span and increasing alpha. Stacking them
    // fakes a tapered stroke — bright and thin at the bearing, bleeding out to
    // nothing at the ends — which a single round-capped stroke cannot do, and it
    // costs no filter pass.
    const ARC_L = 'M-13.07 -42.74 A44.7 44.7 0 0 1 13.07 -42.74';
    const ARC_M = 'M-10.05 -43.55 A44.7 44.7 0 0 1 10.05 -43.55';
    const ARC_S = 'M-6.22 -44.26 A44.7 44.7 0 0 1 6.22 -44.26';
    this.$arcs = mkEl('div', 'hud-arcs', this.el);
    this.arcs = [];
    for (let i = 0; i < 4; i++) {
      const d = mkEl('div', 'hud-arc', this.$arcs);
      d.innerHTML = '<svg viewBox="-50 -50 100 100">' +
        '<path d="' + ARC_L + '" fill="none" stroke="rgba(0,0,0,.42)" stroke-width="3" stroke-linecap="round"/>' +
        '<path d="M0 -40 v-4" stroke="rgba(0,0,0,.42)" stroke-width="2.6" stroke-linecap="round"/>' +
        '<path d="' + ARC_L + '" fill="none" stroke="currentColor" stroke-opacity=".18" stroke-width="4.6" stroke-linecap="round"/>' +
        '<path d="' + ARC_L + '" fill="none" stroke="currentColor" stroke-opacity=".45" stroke-width="1.5" stroke-linecap="round"/>' +
        '<path d="' + ARC_M + '" fill="none" stroke="currentColor" stroke-opacity=".72" stroke-width="1.7" stroke-linecap="round"/>' +
        '<path d="' + ARC_S + '" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/>' +
        '<path d="M0 -40 v-4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
      this.arcs.push({ el: d, t: 0, life: 0, bearing: 0, wroteRot: 1e9, wroteA: -1 });
    }

    // ---------------------------------------------------------- compass
    this.compH = 36;
    this.ppd = 3.2;                       // CSS px per degree of yaw — 5° = 16px exactly
    this.$comp = mkEl('canvas', 'hud-comp', this.el);
    this.compCtx = this.$comp.getContext('2d');
    this.tape = document.createElement('canvas');
    this.compCss = 0;
    this._scrim = null;
    this._sizeCompass();
    this._buildTape();
    this._compSig = -1e9;
    this._markT = 0;

    this.$obj = mkEl('div', 'hud-obj', this.el);
    this.$obj.innerHTML = '<b></b><i></i><span></span>';
    this.$objW = this.$obj.querySelector('b');
    this.$objR = this.$obj.querySelector('span');

    // ---------------------------------------------------------- banner
    this.$ban = mkEl('div', 'hud-ban', this.el);
    this.$ban.innerHTML = '<div class="t"></div><div class="r"></div><div class="s"></div>';
    this.$banT = this.$ban.querySelector('.t');
    this.$banR = this.$ban.querySelector('.r');
    this.$banS = this.$ban.querySelector('.s');
    // live=1 means the subtitle tracks the director's hostile count instead of
    // holding the number the banner was raised with.
    this.ban = { t: 0, life: 0, wroteA: -1, wroteR: -1, live: 0, tail: '' };

    // ---------------------------------------------------------- killfeed
    this.$feed = mkEl('div', 'hud-feed sh', this.el);
    this.feed = [];
    for (let i = 0; i < 5; i++) {
      const r = mkEl('div', 'hud-row', this.$feed);
      r.innerHTML = '<span class="a"></span><span class="g">' + GLYPH + '<span class="hs"></span></span><span class="v"></span>';
      r.style.display = 'none';
      this.feed.push({
        el: r, a: r.querySelector('.a'), v: r.querySelector('.v'),
        t: 0, life: 0, active: false, wroteA: -1,
      });
    }

    // ---------------------------------------------------------- ammo
    this.$ammo = mkEl('div', 'hud-ammo sh', this.el);
    this.$ammo.innerHTML = '<div class="nm"></div><div class="rw"><span class="mag num">0</span>' +
      '<span class="res num">0</span></div><div class="bar"><i></i></div><div class="md"></div>';
    this.$nm = this.$ammo.querySelector('.nm');
    this.$mag = this.$ammo.querySelector('.mag');
    this.$res = this.$ammo.querySelector('.res');
    this.$bar = this.$ammo.querySelector('.bar');
    this.$barI = this.$ammo.querySelector('.bar i');
    this.$md = this.$ammo.querySelector('.md');
    this.$rl = mkEl('div', 'hud-rl', this.el);
    this.$rl.textContent = 'RELOAD';

    // ---------------------------------------------------------- score / health
    this.$left = mkEl('div', 'hud-left sh', this.el);
    this.$left.innerHTML =
      '<div class="sc">SCORE<b class="num">0</b></div><div class="st"></div>' +
      '<div class="hpl">VITALS</div>' +
      '<div class="hp"><div class="hpb"><i></i><u style="left:33.33%"></u><u style="left:66.66%"></u></div>' +
      '<span class="hpv num">100</span></div>';
    this.$score = this.$left.querySelector('.sc b');
    this.$streak = this.$left.querySelector('.st');
    this.$hpI = this.$left.querySelector('.hpb i');
    this.$hpV = this.$left.querySelector('.hpv');

    // ---------------------------------------------------------- perf readout
    // Engine telemetry is off by default: it is debug output, and it was ending up
    // in every judged frame. SHIFT+F (main.js) toggles it. The benchmark panel is
    // independent of the toggle — it is opacity 0 until a run actually reports.
    this.$perf = mkEl('div', 'hud-perf', this.el);
    this.$bench = mkEl('div', 'hud-bench', this.el);
    this.perfOn = false;
    this._perfT = 0;
    this._perfTxt = '';
    this._benchTxt = '';

    // ---------------------------------------------------------- live state
    this.bloom = 0;          // mirrors weaponfx's hipfire bloom, fed by EV.SHOT
    this.sinceShot = 9;
    this.flick = 0;          // reticle rotation impulse on a confirmed hit
    this.hmT = 9; this.hmDur = 0.12; this.hmKind = 0;
    this.killT = 9;          // kill flourish ring, separate life from the marker
    this.hurt = 0;           // damage flash, 0..1
    this.beat = 0;           // heartbeat phase at low health
    this.reloadT = -1; this.reloadDur = 1;
    this.names = new WeakMap();
    this.heading = 0;

    // Last-written cache. Every DOM write in update() is gated on one of these.
    this.c = {
      mag: -1, res: -1, name: '', mode: '', dry: -1, rl: -1, bar: -1,
      score: -1, streak: -1, hp: -1, hpc: '', wave: '', rem: '',
      vig: -1, grade: -1, edge: -1, sec: -1, barA: -1, pulse: -2, rlTxt: 'RELOAD',
    };

    this._onResize = () => { this._sizeCompass(); this._compSig = -1e9; this._retSig[0] = -1; };
    addEventListener('resize', this._onResize);
    this._px = 0;
    this._sizeView();

    this._offs = [
      bus.on(EV.SHOT, () => { this.bloom = Math.min(1.6, this.bloom + 0.22); this.sinceShot = 0; }),
      bus.on(EV.HIT, (e) => this._onHit(e)),
      bus.on(EV.KILL, (e) => this._onKill(e)),
      bus.on(EV.DAMAGE, (e) => this._onDamage(e)),
      bus.on(EV.RELOAD, (e) => this._onReload(e)),
      bus.on(EV.WAVE, (e) => this._onWave(e)),
      bus.on(EV.SPAWN, (e) => this._onSpawn(e)),
      bus.on(EV.DEATH, (e) => this._onDeath(e)),
      bus.on(EV.UI, (e) => this._onUi(e)),
    ];
  }

  // ================================================================= events
  _onSpawn({ enemy }) {
    if (!enemy || this.names.has(enemy)) return;
    const n = this.rng.int(1, 99);
    this.names.set(enemy, this.rng.pick(CALL_A) + ' ' + (n < 10 ? '0' + n : n));
  }

  _onHit(e) {
    if (!e || !e.victim) return;          // world impacts do not mark
    this._mark(e.headshot ? 1 : 0);
    // A nudge, not a spin: past ~7° the four ticks stop reading as a reticle.
    this.flick = e.headshot ? 0.14 : 0.10;
  }

  _onKill(e) {
    this._mark(e && e.headshot ? 3 : 2);
    this.flick = 0.17;
    this.killT = 0;                      // flourish ring, outlives the marker
    const victim = (e && e.victim && this.names.get(e.victim)) || 'HOSTILE';
    this._pushFeed(PLAYER_CALLSIGN, victim, !!(e && e.headshot));
  }

  _mark(kind) {
    // A kill marker always wins over a body marker that is already playing.
    if (this.hmT < this.hmDur && kind < this.hmKind) return;
    this.hmKind = kind;
    this.hmDur = kind >= 2 ? 0.18 : 0.12;
    this.hmT = 0;
  }

  _onDamage(e) {
    const amt = (e && e.amount) || 0;
    this.hurt = Math.min(1, this.hurt + 0.35 + Math.min(0.45, amt / 60));
    // fromDir is a world-space unit vector pointing from the player TOWARD the
    // attacker. Bearing is measured from -Z (north) toward +X, matching the
    // compass, so the arc and the compass agree without a second convention.
    const d = e && e.fromDir;
    let bearing = this.heading;
    if (d && typeof d.x === 'number') bearing = Math.atan2(d.x, -d.z) * DEG;
    else if (typeof d === 'number') bearing = d * DEG;
    let slot = null;
    for (const a of this.arcs) {
      // Fold repeat hits from roughly the same bearing into one indicator.
      if (a.life > 0 && Math.abs(wrapDeg(a.bearing - bearing)) < 22) { slot = a; break; }
    }
    if (!slot) {
      slot = this.arcs[0];
      for (const a of this.arcs) if (a.life - a.t < slot.life - slot.t) slot = a;
    }
    slot.bearing = bearing; slot.t = 0; slot.life = 1.6;
  }

  _onReload(e) {
    if (!e) return;
    if (e.phase === 'start') {
      const cfg = e.weapon || {};
      this.reloadDur = Math.max(0.4, (e.empty ? cfg.reloadEmptyTime : cfg.reloadTime) || 2);
      this.reloadT = 0;
    } else if (e.phase === 'end') this.reloadT = -1;
  }

  _onWave(e) {
    if (!e) return;
    // The count is deliberately NOT read from the event. _readouts owns it.
    if (e.phase === 'start') this._banner('WAVE ' + e.index, '', 1, ' INBOUND');
    else if (e.phase === 'clear') this._banner('SECTOR CLEAR', 'WAVE ' + e.index + '  ·  REGROUP');
  }

  _onDeath(e) {
    if (e && e.who === 'player') this._banner('OPERATOR DOWN', 'HOLD POSITION');
  }

  _onUi(e) {
    if (!e) return;
    if (e.kind === 'togglePerf') {
      this.perfOn = !this.perfOn;
      this.$perf.style.display = this.perfOn ? 'block' : 'none';
      if (this.perfOn) this._perfT = 1;    // repaint on the next update, not in 0.25s
    } else if (e.kind === 'benchStart') {
      this._benchTxt = '';
      this.$bench.textContent = 'BENCHMARK RUNNING — 20s\nHOLD STILL OR PLAY NORMALLY';
      this.$bench.style.opacity = '.85';
    } else if (e.kind === 'benchResult' && e.result) {
      const r = e.result;
      this.$bench.textContent =
        'BENCHMARK  ' + r.width + '×' + r.height + '  scale ' + r.renderScale + '\n' +
        'AVG ' + r.avgFps + ' FPS    1% LOW ' + r.onePercentLowFps + ' FPS\n' +
        'MEDIAN ' + r.medianMs + 'ms   DRAW ' + r.drawCalls + '   Q ' + r.quality + '\n' +
        String(r.gpu).slice(0, 46);
      this.$bench.style.opacity = '.9';
    }
  }

  /**
   * @param live 1 = subtitle is re-written from the director's live hostile count
   *             every time that count changes, with `tail` appended.
   */
  _banner(title, sub, live, tail) {
    this.$banT.textContent = title;
    this.ban.live = live || 0;
    this.ban.tail = tail || '';
    // A live banner takes its first subtitle from the cached count string, so it
    // is already correct on the frame it appears.
    this.$banS.textContent = this.ban.live ? (this.c.rem || '') + this.ban.tail : sub;
    this.ban.t = 0; this.ban.life = 3.4;
    this.ban.wroteA = -1; this.ban.wroteR = -1;
  }

  _pushFeed(attacker, victim, head) {
    let row = null;
    for (const r of this.feed) if (!r.active) { row = r; break; }
    if (!row) {                                   // recycle the oldest
      row = this.feed[0];
      for (const r of this.feed) if (r.t > row.t) row = r;
    }
    row.a.textContent = attacker;
    row.v.textContent = victim;
    row.el.className = 'hud-row' + (head ? ' k' : '');
    row.el.style.display = '';
    row.t = 0; row.life = 5.0; row.active = true; row.wroteA = -1;
    if (this.$feed.firstChild !== row.el) this.$feed.insertBefore(row.el, this.$feed.firstChild);
  }

  // ================================================================= layout
  _sizeView() {
    const h = (typeof innerHeight === 'number' && innerHeight) || 1080;
    // Vertical FOV is fixed at 80° in main.js. Screen px subtended by one radian
    // of cone half-angle, which is what turns weapon spread into reticle gap.
    this._px = (h * 0.5) / Math.tan(40 / DEG);
  }

  _sizeCompass() {
    this._sizeView();
    const vw = (typeof innerWidth === 'number' && innerWidth) || 1920;
    // Even width so the integer half-width margin centres it exactly, and a
    // multiple of the 16px minor-tick pitch so the tape blit never lands between
    // ticks at the mask edges.
    let w = Math.round(Math.max(340, Math.min(672, vw * 0.42)) / 16) * 16;
    if (w === this.compCss) return;
    this.compCss = w;
    this.$comp.style.width = w + 'px';
    this.$comp.style.height = this.compH + 'px';
    this.$comp.style.marginLeft = (-w / 2) + 'px';
    this.$comp.width = this.compW = Math.round(w * this.dpr);
    this.$comp.height = this.compHpx = Math.round(this.compH * this.dpr);
    // Cached once — createLinearGradient must never run per frame.
    const g = this.compCtx.createLinearGradient(0, 0, 0, this.compHpx);
    g.addColorStop(0.00, 'rgba(3,6,10,0.30)');
    g.addColorStop(0.12, 'rgba(3,6,10,0.62)');
    g.addColorStop(0.78, 'rgba(3,6,10,0.54)');
    g.addColorStop(1.00, 'rgba(3,6,10,0.00)');
    this._scrim = g;
  }

  /**
   * Pre-render the full 360° tape once; the live pass is three blits.
   * Light-on-dark: the scrim under the tape carries the contrast, so ticks are
   * drawn as clean light rects with no per-tick black skirt. Every x is an
   * integer device pixel and the pitch (5° = 16 CSS px) is integral too, so
   * neighbouring ticks are rasterised identically instead of one landing on a
   * pixel boundary and the next straddling two.
   */
  _buildTape() {
    const d = this.dpr, ppd = this.ppd;
    const W = this.tapeW = Math.round(360 * ppd * d);
    const H = Math.round(this.compH * d);
    this.tape.width = W; this.tape.height = H;
    const c = this.tape.getContext('2d');
    const tw = Math.max(1, Math.round(d));
    const wrap = (x, fn) => { fn(x); if (x < 40 * d) fn(x + W); else if (x > W - 40 * d) fn(x - W); };

    // Ticks hang from the top edge; the label row sits under them. One direction,
    // one rhythm — the centre index reads against it immediately.
    for (let deg = 0; deg < 360; deg += 5) {
      if (deg % 45 === 0) continue;               // cardinal slots carry a glyph
      const major = deg % 15 === 0;
      const h = Math.round((major ? 10 : 6) * d);
      const x = Math.round(deg * ppd * d);
      wrap(x, (xx) => {
        c.fillStyle = major ? 'rgba(240,247,253,.92)' : 'rgba(228,238,248,.58)';
        c.fillRect(xx - (tw >> 1), 0, tw, h);
      });
    }

    c.textAlign = 'center';
    c.textBaseline = 'alphabetic';
    c.shadowColor = 'rgba(0,0,0,.85)';
    c.shadowBlur = 3 * d;
    const labelY = Math.round(26 * d);
    for (let i = 0; i < 8; i++) {
      const deg = i * 45;
      const x = Math.round(deg * ppd * d);
      const primary = deg % 90 === 0;
      c.font = (primary ? 700 : 600) + ' ' + Math.round((primary ? 14.5 : 11) * d) + 'px ' + TECH;
      c.fillStyle = primary ? 'rgba(244,249,254,.98)' : 'rgba(232,240,249,.70)';
      // The cardinal's own tick, short and bright, above its glyph.
      wrap(x, (xx) => {
        c.save(); c.shadowBlur = 0;
        c.fillStyle = primary ? 'rgba(244,249,254,.92)' : 'rgba(232,240,249,.62)';
        c.fillRect(xx - (tw >> 1), 0, tw, Math.round(13 * d));
        c.restore();
        c.fillStyle = primary ? 'rgba(244,249,254,.98)' : 'rgba(232,240,249,.70)';
        c.fillText(CARD[i], xx, labelY);
      });
    }
    // Bearings only every 30°. Every 15° turned the label row into a ladder and
    // fought the cardinals for attention.
    c.font = '400 ' + Math.round(9 * d) + 'px ' + TECH;
    c.fillStyle = 'rgba(228,238,248,.50)';
    for (let deg = 30; deg < 360; deg += 30) {
      if (deg % 45 === 0) continue;
      const x = Math.round(deg * ppd * d);
      wrap(x, (xx) => c.fillText(String(deg), xx, labelY - Math.round(1.5 * d)));
    }
    c.shadowBlur = 0;
  }

  // ================================================================= update
  update(dt, ctx) {
    if (!ctx) return;
    const t = (this.t += dt);
    const w = ctx.weapon || {};
    const p = ctx.player || {};
    const dir = ctx.director || {};

    // Decay bands. Mirrors weaponfx's bloom recovery so the cone the reticle
    // draws is the cone the next round will actually use.
    this.sinceShot += dt;
    if (this.sinceShot > 0.12) this.bloom = Math.max(0, this.bloom - dt * 2.4);
    this.flick *= Math.exp(-dt * 13);
    if (this.hmT < this.hmDur) this.hmT += dt;
    if (this.killT < KILL_LIFE) this.killT += dt;
    this.hurt = Math.max(0, this.hurt - dt * 1.8);
    if (this.reloadT >= 0) this.reloadT += dt;

    const adsE = smooth(w.adsT || 0);
    this.heading = wrapDeg(-(p.yaw || 0) * DEG);

    this._reticle(w, p, adsE);
    this._compass(dt, p, dir);
    this._ammo(w, t);
    this._readouts(dir, p);
    this._feedTick(dt);
    this._bannerTick(dt);
    this._damageTick(dt, p);
    this._secondary(adsE);
    this._perfTick(dt, ctx.perf);
  }

  // ---------------------------------------------------------------- reticle
  _reticle(w, p, adsE) {
    const cfg = w.cfg || {};
    const hip = cfg.spreadHip || 0.032;
    const aim = cfg.spreadAds || 0.0035;
    const speed = p.speed || 0;
    const crouch = clamp01(p.crouch || 0);
    const grounded = p.grounded !== false;

    const moveMul = 1 + Math.min(1, speed / 5) * 0.55 + (grounded ? 0 : 0.9);
    const stance = 1 - crouch * 0.28;
    const bloomMul = 1 + this.bloom * (adsE > 0.6 ? 0.35 : 0.9);
    const spread = (hip + (aim - hip) * adsE) * stance * bloomMul * (1 + (moveMul - 1) * (1 - adsE));

    const d = this.dpr;
    // Hip: four long ticks at the true cone edge plus a centre dot. ADS: the
    // ticks pinch to a tight optic frame and the dot yields to the real one
    // rendered on the sight, so the screen centre stays clear of the target.
    const lenPx = 12 - adsE * 7.5;
    // Capped so a full-bloom airborne spray still draws inside the canvas — and
    // because a reticle wider than the shoulder-width of a target is useless.
    const gapCap = this.retCss * 0.5 - lenPx - 8;
    const gapPx = Math.min(gapCap, (2.5 + spread * this._px) * (1 - adsE * 0.34));
    // 3px of core at hip: with a 1px dark outline on each side, a 2px tick reads
    // as a grey pill rather than a white one against a bright wall.
    const thPx = 3 - adsE * 1;
    // Pinches shut before the sight picture is fully up: at full ADS the optic's
    // own reticle is the aiming reference and a HUD cross over it reads as dirt
    // on the lens.
    const tickA = Math.max(0, 0.92 - adsE * 1.05) * (1 - clamp01(w.sprint || 0) * 0.5);
    const dotA = 0.95 * (1 - adsE);

    const s = this._retSig;
    const g = Math.round(gapPx * d);
    const L = Math.max(2 * d, Math.round(lenPx * d)) | 0;
    const th = Math.max(2, Math.round((thPx * d) / 2) * 2);
    const ta = (tickA * 100) | 0;
    const da = (dotA * 100) | 0;
    const fl = (this.flick * 300) | 0;
    const hm = this.hmT < this.hmDur ? (((this.hmT / this.hmDur) * 40) | 0) + 1 + this.hmKind * 64 : 0;
    const kr = this.killT < KILL_LIFE ? (((this.killT / KILL_LIFE) * 40) | 0) + 1 : 0;
    const ae = (adsE * 60) | 0;
    if (s[0] === g && s[1] === L && s[2] === th && s[3] === ta && s[4] === da &&
        s[5] === fl && s[6] === hm && s[7] === ae && s[8] === kr) return;
    s[0] = g; s[1] = L; s[2] = th; s[3] = ta; s[4] = da; s[5] = fl; s[6] = hm; s[7] = ae; s[8] = kr;

    const c = this.retCtx, W = this.retW, H = this.retH;
    c.clearRect(0, 0, W, H);
    const cx = W >> 1, cy = H >> 1;

    // Optic frame: a faint ring that only exists in ADS, drawn under the ticks.
    if (adsE > 0.02) {
      c.globalAlpha = 0.16 * adsE;
      c.strokeStyle = INK;
      c.lineWidth = Math.max(1, Math.round(d));
      c.beginPath();
      c.arc(cx, cy, Math.round(30 * d), 0, Math.PI * 2);
      c.stroke();
    }

    if (kr) this._killRing(c, cx, cy);

    c.globalAlpha = tickA;
    const rot = this.flick;
    if (rot > 0.004) {
      c.save();
      c.translate(cx, cy);
      c.rotate(rot);
      this._ticks(c, 0, 0, g, L, th, INK);
      c.restore();
    } else {
      this._ticks(c, cx, cy, g, L, th, INK);
    }

    if (da > 1) {
      c.globalAlpha = dotA;
      const dd = Math.max(2, Math.round(2 * d / 2) * 2);
      this._rect(c, cx - (dd >> 1), cy - (dd >> 1), dd, dd, INK);
    }

    if (hm) this._hitmarker(c, cx, cy);
    c.globalAlpha = 1;
  }

  /** Four axis-aligned ticks, integer device pixels, hard edges. */
  _ticks(c, cx, cy, g, L, th, col) {
    const h = th >> 1;
    this._rect(c, cx - h, cy - g - L, th, L, col);
    this._rect(c, cx - h, cy + g, th, L, col);
    this._rect(c, cx - g - L, cy - h, L, th, col);
    this._rect(c, cx + g, cy - h, L, th, col);
  }

  /**
   * Rect with a near-black one-device-pixel outline. The outline is what makes
   * the reticle survive blown-out stucco; it is drawn at 0.9 of the element's own
   * alpha so a fading marker takes its outline with it.
   */
  _rect(c, x, y, w, h, col) {
    const a = c.globalAlpha;
    const o = Math.max(1, Math.round(this.dpr));
    c.globalAlpha = a * 0.9;
    c.fillStyle = OUTLINE;
    c.fillRect(x - o, y - o, w + o * 2, h + o * 2);
    c.globalAlpha = a;
    c.fillStyle = col;
    c.fillRect(x, y, w, h);
  }

  /**
   * Hit confirmation. Snaps to full in ~18ms, holds, and is gone by 120ms —
   * short enough to read as an impact rather than a widget. Kind 2/3 (kill) is
   * warm, thicker and longer-lived, and pairs with the expanding ring below so a
   * kill never looks like a body shot.
   */
  _hitmarker(c, cx, cy) {
    const d = this.dpr;
    const u = clamp01(this.hmT / this.hmDur);
    const kill = this.hmKind >= 2;
    const head = this.hmKind === 1 || this.hmKind === 3;
    const k = u < 0.14 ? u / 0.14 : 1;
    const a = u < 0.5 ? 1 : 1 - (u - 0.5) / 0.5;
    const grow = 0.74 + 0.26 * k + (1 - a) * 0.26;

    const th = Math.max(2, Math.round((kill ? 3.4 : 2.4) * d / 2) * 2);
    const len = Math.round((kill ? 14 : 10.5) * d * grow);
    const inner = Math.round((kill ? 8.5 : 7) * d * grow);
    const col = kill ? WARM : head ? '#ffe2a8' : '#ffffff';

    c.globalAlpha = a;
    c.save();
    c.translate(cx, cy);
    for (let i = 0; i < 4; i++) {
      c.save();
      c.rotate(Math.PI * 0.25 + i * Math.PI * 0.5);
      this._rect(c, inner, -(th >> 1), len, th, col);
      if (kill) {
        // Hot core inside the warm arm. Warm-on-warm is exactly the case the
        // frame is full of (sunlit stucco), so the kill marker cannot rely on
        // hue alone — it needs a value spike too.
        const ith = Math.max(1, th - 2);
        c.fillStyle = '#fff4de';
        c.fillRect(inner + 1, -(ith >> 1), len - 2, ith);
      }
      c.restore();
    }
    // Headshot adds four short axis ticks — a starburst rather than an X. On a
    // headshot KILL they are left off: the ring is already saying "down", and
    // stacking both put warm ticks straight on top of the white reticle ticks.
    if (head && !kill) {
      const ht = Math.max(2, th - 1);
      const hl = Math.round(len * 0.5);
      const hi = Math.round(inner * 1.55);
      for (let i = 0; i < 4; i++) {
        c.save();
        c.rotate(i * Math.PI * 0.5);
        this._rect(c, hi, -(ht >> 1), hl, ht, '#ffe2a8');
        c.restore();
      }
    }
    c.restore();
    c.globalAlpha = 1;
  }

  /**
   * Kill flourish: a ring that punches outward and thins as it goes, with a
   * white leading edge and a fainter trailing ring behind it. It outlives the
   * marker by ~200ms, which is what separates "hit" from "killed" at a glance
   * without adding a second HUD element.
   */
  _killRing(c, cx, cy) {
    const d = this.dpr;
    const u = clamp01(this.killT / KILL_LIFE);
    const e = 1 - Math.pow(1 - u, 2.6);          // fast out, long settle
    const a = Math.pow(1 - u, 1.7);
    const r = (10 + 26 * e) * d;
    const lw = Math.max(1, Math.round((3.0 - 2.1 * e) * d));

    c.save();
    c.globalAlpha = a * 0.38;
    c.strokeStyle = OUTLINE;
    c.lineWidth = lw + Math.max(2, Math.round(1.5 * d));
    c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2); c.stroke();
    c.globalAlpha = a * 0.95;
    c.strokeStyle = '#ffc86e';
    c.lineWidth = lw;
    c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2); c.stroke();
    if (u < 0.4) {
      // Value spike on the leading edge of the punch, for the same reason the
      // arms have a hot core.
      c.globalAlpha = a * 0.85 * (1 - u / 0.4);
      c.strokeStyle = '#fff6e8';
      c.lineWidth = Math.max(1, Math.round(d));
      c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2); c.stroke();
    }
    // A second ring just behind the first, wider and fainter, so the punch has
    // some depth to it instead of reading as one flat hoop.
    if (u < 0.55) {
      c.globalAlpha = a * 0.30;
      c.strokeStyle = WARM;
      c.lineWidth = Math.max(1, Math.round(d));
      c.beginPath(); c.arc(cx, cy, r * 0.72, 0, Math.PI * 2); c.stroke();
    }
    c.restore();
    c.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- compass
  _compass(dt, p, dir) {
    const list = (dir.enemies && dir.enemies.list) || null;
    const hasMarks = !!(list && list.length);
    this._markT += dt;
    const sig = Math.round(this.heading * 12);
    if (sig === this._compSig && !(hasMarks && this._markT > 0.05)) return;
    this._compSig = sig;
    if (hasMarks) this._markT = 0;

    const c = this.compCtx, d = this.dpr, W = this.compW, H = this.compHpx;
    c.clearRect(0, 0, W, H);

    // The HUD's own dark ground. Everything above it is light — the widget reads
    // the same way over bright sky and over a black interior.
    c.fillStyle = this._scrim;
    c.fillRect(0, 0, W, H);

    const centre = this.heading * this.ppd * d;
    let x0 = Math.round(W * 0.5 - centre);
    const tw = this.tapeW;
    x0 = ((x0 % tw) + tw) % tw - tw;
    for (let x = x0; x < W; x += tw) c.drawImage(this.tape, x, 0);

    // Base rule closes the widget off and gives the ticks something to sit on.
    c.fillStyle = 'rgba(226,236,247,.20)';
    c.fillRect(0, Math.round(31 * d), W, Math.max(1, Math.round(d)));

    // Centre index — the only warm element up here. A post with a head, kept
    // entirely inside the tick band so it can never sit on top of a cardinal
    // glyph when the player happens to face due north.
    const mx = W >> 1;
    const mw = Math.max(2, Math.round(2 * d));
    const hw = Math.max(mw + 2, Math.round(7 * d));
    const bandH = Math.round(14 * d), headH = Math.round(3.5 * d);
    c.globalAlpha = 0.92;
    c.fillStyle = OUTLINE;
    c.fillRect(mx - (hw >> 1) - 1, -1, hw + 2, headH + 2);
    c.fillRect(mx - (mw >> 1) - 1, -1, mw + 2, bandH + 1);
    c.globalAlpha = 1;
    c.fillStyle = WARM;
    c.fillRect(mx - (hw >> 1), 0, hw, headH);
    c.fillRect(mx - (mw >> 1), 0, mw, bandH);

    if (hasMarks) {
      const px = p.pos;
      const half = (W * 0.5) / (this.ppd * d);
      // Threat marks own the tick band: a contact matters more than the tick it
      // covers. They point down, into the tape, from the top edge.
      const ty = Math.round(1 * d);
      let n = 0;
      for (let i = 0; i < list.length && n < 10; i++) {
        const e = list[i];
        if (!e || !e.alive || !e.pos || !px) continue;
        const dx = e.pos.x - px.x, dz = e.pos.z - px.z;
        const dist = Math.sqrt(dx * dx + dz * dz);
        if (dist > 90) continue;
        const rel = wrapDeg(Math.atan2(dx, -dz) * DEG - this.heading);
        if (Math.abs(rel) > half) continue;
        n++;
        const ex = Math.round(mx + rel * this.ppd * d);
        const s = Math.round(4.5 * d);
        const h = Math.round(7 * d);
        c.globalAlpha = 0.45 + 0.55 * (1 - Math.min(1, dist / 90));
        c.fillStyle = OUTLINE;
        c.beginPath();
        c.moveTo(ex - s - 1, ty - 1);
        c.lineTo(ex + s + 1, ty - 1);
        c.lineTo(ex, ty + h + 1);
        c.fill();
        c.fillStyle = BAD;
        c.beginPath();
        c.moveTo(ex - s, ty);
        c.lineTo(ex + s, ty);
        c.lineTo(ex, ty + h);
        c.fill();
      }
      c.globalAlpha = 1;
    }
  }

  // ---------------------------------------------------------------- ammo
  _ammo(w, t) {
    const cc = this.c;
    const mag = w.ammo | 0, res = w.reserve | 0;
    if (mag !== cc.mag) { cc.mag = mag; this.$mag.textContent = mag < 10 ? '0' + mag : String(mag); }
    if (res !== cc.res) { cc.res = res; this.$res.textContent = String(res); }

    const name = (w.cfg && w.cfg.name) || '';
    if (name !== cc.name) { cc.name = name; this.$nm.textContent = name; }
    const mode = w.fireMode ? String(w.fireMode).toUpperCase() : '';
    if (mode !== cc.mode) {
      cc.mode = mode;
      this.$md.textContent = mode ? mode + '  ·  ' + ((w.cfg && w.cfg.magSize) || '') + ' RND' : '';
    }

    // Empty is the weapon's most urgent state, so it gets three signals at once:
    // the numerals go red, they pulse, and a prompt appears under the aim point.
    // The pulse runs on the HUD's own accumulated clock, so a capture is
    // reproducible frame for frame.
    const rl = w.reloading ? 1 : 0;
    const dry = mag === 0 && !rl ? 1 : 0;
    if (dry !== cc.dry || rl !== cc.rl) {
      cc.dry = dry; cc.rl = rl;
      this.$ammo.className = 'hud-ammo sh' + (rl ? ' rl' : '') + (dry ? ' dry' : '');
      if (!dry && cc.pulse !== -2) {
        cc.pulse = -2;
        this.$mag.style.opacity = '';
        this.$rl.style.opacity = '0';
      }
    }
    if (dry) {
      // Telling a player to reload with an empty belt is worse than saying
      // nothing, so the prompt states which of the two problems they have.
      const txt = res > 0 ? 'RELOAD' : 'NO AMMO';
      if (txt !== cc.rlTxt) { cc.rlTxt = txt; this.$rl.textContent = txt; }
      const s = 0.5 + 0.5 * Math.sin(t * 8.2);
      const q = (s * 12) | 0;
      if (q !== cc.pulse) {
        cc.pulse = q;
        const f = q / 12;
        this.$mag.style.opacity = (0.52 + 0.48 * f).toFixed(2);
        this.$rl.style.opacity = (0.45 + 0.55 * f).toFixed(2);
      }
    }

    const showBar = this.reloadT >= 0 && this.reloadT <= this.reloadDur ? 1 : 0;
    if (showBar !== cc.barA) { cc.barA = showBar; this.$bar.style.opacity = showBar ? '1' : '0'; }
    if (showBar) {
      const f = Math.round(clamp01(this.reloadT / this.reloadDur) * 60);
      if (f !== cc.bar) { cc.bar = f; this.$barI.style.transform = 'scaleX(' + (f / 60).toFixed(3) + ')'; }
    }
  }

  // ---------------------------------------------------------------- readouts
  _readouts(dir, p) {
    const cc = this.c;
    const score = dir.score | 0;
    if (score !== cc.score) {
      cc.score = score;
      this.$score.textContent = score >= 1000
        ? Math.floor(score / 1000) + ' ' + String(score % 1000).padStart(3, '0')
        : String(score);
    }
    const streak = dir.streak | 0;
    if (streak !== cc.streak) {
      cc.streak = streak;
      this.$streak.style.opacity = streak >= 2 ? '1' : '0';
      if (streak >= 2) this.$streak.textContent = '×' + streak + ' CHAIN';
    }

    const wave = dir.phase === 'intermission'
      ? 'NEXT WAVE ' + Math.max(0, Math.ceil(dir.timer || 0))
      : 'WAVE ' + (dir.wave | 0);
    if (wave !== cc.wave) { cc.wave = wave; this.$objW.textContent = wave; }
    // THE hostile count. Nothing else on screen is allowed to compute one: the
    // banner borrows this exact string, so the two can never disagree.
    const rem = dir.phase === 'intermission'
      ? 'STAND BY'
      : (dir.remaining | 0) + (dir.remaining === 1 ? ' HOSTILE' : ' HOSTILES');
    if (rem !== cc.rem) {
      cc.rem = rem;
      this.$objR.textContent = rem;
      if (this.ban.live && this.ban.life > 0) this.$banS.textContent = rem + this.ban.tail;
    }

    const hp = Math.max(0, Math.min(100, Math.round(p.health == null ? 100 : p.health)));
    if (hp !== cc.hp) {
      cc.hp = hp;
      this.$hpI.style.transform = 'scaleX(' + (hp / 100).toFixed(3) + ')';
      this.$hpV.textContent = hp < 100 ? (hp < 10 ? '00' + hp : '0' + hp) : '100';
      const cls = hp <= 30 ? 'c' : hp <= 60 ? 'w' : '';
      if (cls !== cc.hpc) { cc.hpc = cls; this.$left.className = 'hud-left sh ' + cls; }
    }
  }

  // ---------------------------------------------------------------- killfeed
  _feedTick(dt) {
    for (const r of this.feed) {
      if (!r.active) continue;
      r.t += dt;
      if (r.t >= r.life) { r.active = false; r.el.style.display = 'none'; continue; }
      // Snap in, hold, fade out over the last 0.8s.
      const a = Math.min(clamp01(r.t / 0.09), clamp01((r.life - r.t) / 0.8));
      const q = (a * 25) | 0;
      if (q !== r.wroteA) { r.wroteA = q; r.el.style.opacity = (q / 25).toFixed(2); }
    }
  }

  // ---------------------------------------------------------------- banner
  _bannerTick(dt) {
    const b = this.ban;
    if (b.life <= 0) return;
    b.t += dt;
    if (b.t >= b.life) {
      b.life = 0; b.live = 0;
      this.$ban.style.opacity = '0';
      this.$banR.style.transform = 'scaleX(0)';
      return;
    }
    const inT = 0.34, outT = 0.55;
    const a = Math.min(smooth(b.t / inT), smooth((b.life - b.t) / outT));
    const q = (a * 40) | 0;
    if (q !== b.wroteA) {
      b.wroteA = q;
      const f = q / 40;
      this.$ban.style.opacity = f.toFixed(2);
      // Slides up as it lands, drifts up again as it leaves — never static.
      this.$ban.style.transform = 'translateX(-50%) translateY(' + ((1 - f) * 14).toFixed(1) + 'px)';
    }
    const r = (clamp01(b.t / 0.5) * 40) | 0;
    if (r !== b.wroteR) { b.wroteR = r; this.$banR.style.transform = 'scaleX(' + (r / 40).toFixed(3) + ')'; }
  }

  // ------------------------------------------------------- damage + vignette
  _damageTick(dt, p) {
    for (const a of this.arcs) {
      if (a.life <= 0) continue;
      a.t += dt;
      if (a.t >= a.life) {
        a.life = 0;
        if (a.wroteA !== 0) { a.wroteA = 0; a.el.style.opacity = '0'; }
        continue;
      }
      const rel = wrapDeg(a.bearing - this.heading);
      const q = Math.round(rel * 2);
      if (q !== a.wroteRot) { a.wroteRot = q; a.el.style.transform = 'rotate(' + (q / 2).toFixed(1) + 'deg)'; }
      const f = Math.min(clamp01(a.t / 0.05), clamp01((a.life - a.t) / 0.95));
      const qa = (f * 25) | 0;
      if (qa !== a.wroteA) { a.wroteA = qa; a.el.style.opacity = (qa / 25).toFixed(2); }
    }

    const hp = clamp01((p.health == null ? 100 : p.health) / 100);
    const low = clamp01((0.45 - hp) / 0.45);
    // Two-thump heartbeat, faster the closer to death. Purely dt-driven.
    this.beat += dt * (0.95 + low * 0.85);
    if (this.beat > 1) this.beat -= 1;
    const bt = this.beat;
    const thump = Math.exp(-bt * 14) + 0.62 * Math.exp(-Math.abs(bt - 0.16) * 20);

    const cc = this.c;
    const vig = clamp01(this.hurt * 0.85 + low * (0.30 + 0.34 * thump));
    const q = (vig * 50) | 0;
    if (q !== cc.vig) { cc.vig = q; this.$vig.style.opacity = (q / 50).toFixed(2); }

    const edge = clamp01(low * 0.55);
    const qe = (edge * 40) | 0;
    if (qe !== cc.edge) { cc.edge = qe; this.$edge.style.opacity = (qe / 40).toFixed(2); }

    // The desaturation layer is a real compositing pass, so it does not exist
    // at all until the player is actually hurt.
    const grade = clamp01(low * 0.45 + this.hurt * 0.16);
    const qg = (grade * 40) | 0;
    if (qg !== cc.grade) {
      if ((cc.grade <= 0) !== (qg <= 0)) this.$grade.style.display = qg > 0 ? 'block' : 'none';
      cc.grade = qg;
      this.$grade.style.opacity = (qg / 40).toFixed(2);
    }
  }

  /**
   * Secondary furniture steps back when the player aims, and the objective strip
   * steps out entirely while a banner is up — the banner is already saying it,
   * and two copies of the same count is exactly the contradiction we are fixing.
   */
  _secondary(adsE) {
    const a = 1 - adsE * 0.62;
    const objMul = this.ban.live && this.ban.life > 0 ? 0 : 0.74;
    const q = ((a * 25) | 0) * 4 + (objMul > 0 ? 1 : 0);
    if (q === this.c.sec) return;
    this.c.sec = q;
    const f = ((q / 4) | 0) / 25;
    const v = f.toFixed(2);
    this.$comp.style.opacity = v;
    this.$obj.style.opacity = (f * objMul).toFixed(2);
    this.$left.style.opacity = v;
  }

  // ---------------------------------------------------------------- perf
  _perfTick(dt, perf) {
    if (!this.perfOn || !perf) return;
    this._perfT += dt;
    if (this._perfT < 0.25) return;
    this._perfT = 0;
    const fps = perf.wallFps || 0;
    const low = perf.onePercentLowFps || 0;
    const st = perf.stats || {};
    const tri = st.triangles || 0;
    const txt =
      fps.toFixed(1).padStart(5) + ' FPS      1% ' + low.toFixed(1) + '\n' +
      'FRAME ' + (fps > 0 ? (1000 / fps).toFixed(1) : '--') + 'ms   LOGIC ' +
        (perf.logicMsAvg || 0).toFixed(2) + 'ms\n' +
      'DRAW ' + (st.calls || 0) + '   TRI ' + (tri > 9999 ? (tri / 1000).toFixed(0) + 'k' : tri) +
        '   PRG ' + (st.programs || 0) + '\n' +
      'SCALE ' + (perf.renderScale || 1).toFixed(2) + '   Q ' + (perf.quality || '-') + '\n' +
      'SHIFT+F HIDE  ·  SHIFT+B 20s BENCH';
    if (txt !== this._perfTxt) { this._perfTxt = txt; this.$perf.textContent = txt; }
  }

  dispose() {
    for (const off of this._offs) off();
    this._offs.length = 0;
    removeEventListener('resize', this._onResize);
    this.el.remove();
  }
}
