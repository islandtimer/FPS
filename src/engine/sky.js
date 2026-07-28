// OWNER: agent "sky" — atmosphere, sun, IBL, fog, shadow cascades.
// CONTRACT:
//   new Atmosphere(renderer, scene)
//   .update(dt)
//   .sunDirection  -> THREE.Vector3 (normalized, pointing FROM sun TO scene is `-sunDirection`)
//   .sunLight      -> THREE.DirectionalLight (shadow caster)
//   .shadowTexels  -> number, reported into the perf budget
//   .setQuality(q)

import * as THREE from 'three';

export class Atmosphere {
  constructor(renderer, scene) {
    this.scene = scene;
    this.renderer = renderer;
    this.sunDirection = new THREE.Vector3(0.4, 0.65, 0.3).normalize();

    scene.background = new THREE.Color(0x8fa6bd);
    scene.fog = new THREE.FogExp2(0x9fb2c6, 0.012);

    const hemi = new THREE.HemisphereLight(0xa8c4e0, 0x4a4238, 1.1);
    scene.add(hemi);
    this.hemi = hemi;

    const sun = new THREE.DirectionalLight(0xffe9c8, 3.0);
    sun.position.copy(this.sunDirection).multiplyScalar(60);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 140;
    const s = 40;
    Object.assign(sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s });
    sun.shadow.bias = -0.0006;
    sun.shadow.normalBias = 0.04;
    scene.add(sun);
    scene.add(sun.target);
    this.sunLight = sun;
    this.shadowTexels = 2048 * 2048;
  }

  setQuality() {}
  update() {}
  dispose() {}
}
