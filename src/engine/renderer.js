// OWNER: agent "render" — post-processing & framebuffer pipeline.
// CONTRACT (do not change signatures; main.js depends on them):
//   new RenderPipeline(canvas)         -> pipeline
//   .setSize(w, h, renderScale)        -> void
//   .setQuality('low'|'medium'|'high'|'ultra')
//   .render(scene, camera, dt)         -> void
//   .reportCost(perf)                  -> fills perf.pipelineCost {fullscreenPasses, shadowTexels, overdraw}
//   .renderer                          -> THREE.WebGLRenderer (read-only for others)
//   .dispose()

import * as THREE from 'three';

export class RenderPipeline {
  constructor(canvas) {
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      powerPreference: 'high-performance',
      stencil: false,
      depth: true,
    });
    renderer.setPixelRatio(1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer = renderer;
    this.quality = 'high';
    this.width = 1920;
    this.height = 1080;
    this.renderScale = 1;
  }

  setSize(w, h, renderScale = 1) {
    this.width = w; this.height = h; this.renderScale = renderScale;
    this.renderer.setSize(Math.round(w * renderScale), Math.round(h * renderScale), false);
    const c = this.renderer.domElement;
    c.style.width = w + 'px';
    c.style.height = h + 'px';
  }

  setQuality(q) { this.quality = q; }

  render(scene, camera, dt) { this.renderer.render(scene, camera); }

  reportCost(perf) {
    perf.pipelineCost.fullscreenPasses = 1;
    perf.pipelineCost.overdraw = 1.6;
    const sm = this.renderer.shadowMap.enabled ? 2048 * 2048 * 3 : 0;
    perf.pipelineCost.shadowTexels = sm;
  }

  dispose() { this.renderer.dispose(); }
}
