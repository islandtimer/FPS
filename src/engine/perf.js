// Performance instrumentation + adaptive quality.
//
// Bar #2 of this project is "1080p @ 60fps on an M1 / GTX 1660". The CI container
// this game is built in has NO GPU (Chromium falls back to SwiftShader, a CPU
// rasterizer), so raw FPS measured there is meaningless in absolute terms.
// We therefore track three separate things and never conflate them:
//
//   1. wallFps       - real measured fps in whatever browser is running. On the
//                      user's laptop this IS the answer. In CI it is a *relative*
//                      regression signal only.
//   2. cpuMs         - JS main-thread time per frame excluding the GL submit.
//                      Hardware-independent-ish; a hard 60fps blocker if > ~8ms.
//   3. gpuBudget     - a static cost model (draw calls, triangles, shaded
//                      fragments at 1080p, shadow texels, VRAM) converted to an
//                      ESTIMATED ms/frame on the two reference GPUs. Labelled as
//                      an estimate everywhere it is surfaced.
//
// Anything that claims "60fps" without saying which of the three it came from is
// a lie, and this file exists so we never have to tell one.

const REF = {
  // Rough, published-spec-derived throughput used by the cost model.
  // fragGpix: effective shaded gigapixels/sec for a ~120-ALU PBR fragment.
  // triG:     effective giga-triangles/sec.
  // drawUs:   CPU microseconds per draw call through three.js + WebGL2.
  m1:   { name: 'Apple M1 (8c GPU)', fragGpix: 3.1, triG: 1.6, drawUs: 7.0, vramMB: 4096 },
  gtx1660: { name: 'GTX 1660', fragGpix: 5.6, triG: 3.2, drawUs: 9.0, vramMB: 6144 },
};

const QUALITY = ['low', 'medium', 'high', 'ultra'];

export class Perf {
  constructor(renderer, { targetFps = 60, adaptive = true } = {}) {
    this.renderer = renderer;
    this.targetFps = targetFps;
    this.adaptive = adaptive;

    this.frames = 0;
    this.wallFps = 0;
    this.logicMs = 0;
    this.logicMsAvg = 0;
    this._logicEma = null;
    this.cpuMs = 0;
    this.gpuSubmitMs = 0;
    this.frameMs = 0;

    this._hist = new Float32Array(240);
    this._histI = 0;
    this._histN = 0;
    this._acc = 0;
    this._accFrames = 0;
    this._last = performance.now();

    // Adaptive render scale (DRS). 1.0 = native.
    this.renderScale = 1.0;
    this.minScale = 0.66;
    this.maxScale = 1.0;
    this._scaleCooldown = 0;

    this.quality = 'high';
    this.stats = { calls: 0, triangles: 0, programs: 0, textures: 0, geometries: 0 };
    this.pipelineCost = { fullscreenPasses: 1, shadowTexels: 0, overdraw: 1.6 };

    if (typeof window !== 'undefined') window.__perf = this;
  }

  /**
   * Called once per frame, before any game work.
   * `now` is the rAF vsync timestamp and is used ONLY for the frame delta.
   * CPU accounting must use performance.now() at callback entry — the gap between
   * the vsync stamp and the callback actually running is the *previous* frame's
   * GPU/present time, and charging it to CPU makes a fast build look slow.
   */
  beginFrame(now) {
    this._frameStart = performance.now();
    this.frameMs = now - this._last;
    this._last = now;
    return this.frameMs;
  }

  /** Called after game logic, before renderer.render(). */
  markCpuDone(now) { this._cpuEnd = now; }

  /** Game-logic-only cost, set by main.js. The hardware-transferable CPU number. */
  setLogicMs(ms) {
    this.logicMs = ms;
    this._logicEma = this._logicEma == null ? ms : this._logicEma * 0.92 + ms * 0.08;
    this.logicMsAvg = this._logicEma;
  }

  /** Called after renderer.render() returns. */
  endFrame(now) {
    const total = now - this._frameStart;
    this.cpuMs = (this._cpuEnd ?? now) - this._frameStart;
    this.gpuSubmitMs = now - (this._cpuEnd ?? now);

    this._hist[this._histI] = this.frameMs;
    this._histI = (this._histI + 1) % this._hist.length;
    this._histN = Math.min(this._histN + 1, this._hist.length);

    this.frames++;
    this._acc += this.frameMs;
    this._accFrames++;
    if (this._acc >= 500) {
      this.wallFps = (this._accFrames * 1000) / this._acc;
      this._acc = 0;
      this._accFrames = 0;
      this._sampleRenderer();
    }

    if (this.adaptive) this._adapt(total);
  }

  _sampleRenderer() {
    const info = this.renderer?.info;
    if (!info) return;
    this.stats.calls = info.render.calls;
    this.stats.triangles = info.render.triangles;
    this.stats.programs = info.programs?.length ?? 0;
    this.stats.textures = info.memory.textures;
    this.stats.geometries = info.memory.geometries;
  }

  /** Frame-time percentile in ms. p=0.99 -> "1% low". */
  percentile(p) {
    if (!this._histN) return 0;
    const a = Array.from(this._hist.subarray(0, this._histN)).sort((x, y) => x - y);
    return a[Math.min(a.length - 1, Math.floor(p * a.length))];
  }

  get onePercentLowFps() {
    const ms = this.percentile(0.99);
    return ms > 0 ? 1000 / ms : 0;
  }

  _adapt(totalMs) {
    const budget = 1000 / this.targetFps;
    this._scaleCooldown -= 1;
    if (this._scaleCooldown > 0) return;
    if (totalMs > budget * 1.25 && this.renderScale > this.minScale) {
      this.renderScale = Math.max(this.minScale, this.renderScale - 0.05);
      this._scaleCooldown = 45;
    } else if (totalMs < budget * 0.72 && this.renderScale < this.maxScale) {
      this.renderScale = Math.min(this.maxScale, this.renderScale + 0.025);
      this._scaleCooldown = 90;
    }
  }

  /**
   * Static cost model -> estimated ms/frame on reference hardware.
   * Deliberately conservative; treat as a budget check, not a measurement.
   */
  estimate(width = 1920, height = 1080) {
    const px = width * height;
    const c = this.pipelineCost;
    const shadedFrag = px * (c.overdraw + c.fullscreenPasses);
    const out = {};
    for (const [key, hw] of Object.entries(REF)) {
      const fragMs = (shadedFrag / (hw.fragGpix * 1e9)) * 1000;
      const shadowMs = (c.shadowTexels / (hw.fragGpix * 4e9)) * 1000; // depth-only is cheap
      const triMs = (this.stats.triangles / (hw.triG * 1e9)) * 1000;
      const cpuMs = (this.stats.calls * hw.drawUs) / 1000;
      const gpuMs = fragMs + shadowMs + triMs;
      // CPU and GPU overlap on a real pipeline; the frame is gated by the slower side.
      const est = Math.max(gpuMs, cpuMs + this.logicMsAvg);
      out[key] = {
        name: hw.name,
        fragMs: +fragMs.toFixed(2),
        shadowMs: +shadowMs.toFixed(2),
        triMs: +triMs.toFixed(2),
        drawCpuMs: +cpuMs.toFixed(2),
        estMs: +est.toFixed(2),
        estFps: Math.round(1000 / Math.max(0.1, est)),
      };
    }
    return out;
  }

  /** Hardware-neutral budget check. These are the numbers reviews should cite. */
  budgets(width = 1920, height = 1080) {
    const c = this.pipelineCost;
    const B = [
      ['drawCalls', this.stats.calls, 260],
      ['triangles', this.stats.triangles, 950_000],
      ['fullscreenPasses', c.fullscreenPasses, 8],
      ['shadowTexelsM', +(c.shadowTexels / 1e6).toFixed(1), 14],
      ['logicMs', +this.logicMsAvg.toFixed(2), 4.0],
      ['programs', this.stats.programs, 90],
    ];
    const rows = B.map(([k, v, limit]) => ({ k, v, limit, ok: v <= limit }));
    return { rows, pass: rows.every((r) => r.ok) };
  }

  snapshot(width = 1920, height = 1080) {
    return {
      wallFps: +this.wallFps.toFixed(1),
      onePercentLowFps: +this.onePercentLowFps.toFixed(1),
      logicMs: +this.logicMsAvg.toFixed(3),
      cpuMs: +this.cpuMs.toFixed(2),
      gpuSubmitMs: +this.gpuSubmitMs.toFixed(2),
      renderScale: +this.renderScale.toFixed(3),
      quality: this.quality,
      stats: { ...this.stats },
      pipelineCost: { ...this.pipelineCost },
      budgets: this.budgets(width, height),
      estimate: this.estimate(width, height),
    };
  }
}

export { QUALITY, REF };
