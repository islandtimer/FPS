// OWNER: agent "level" — the map: geometry, blockout, modular kit, layout.
// CONTRACT:
//   buildLevel(materials, rng) -> {
//     group:        THREE.Group        (added to scene)
//     colliders:    Array<{min:Vec3,max:Vec3}>  AABBs for player/AI collision
//     raycastables: THREE.Object3D[]   meshes hitscan may test against
//     playerStart:  {pos:Vec3, yaw:number}
//     spawnPoints:  Array<{pos:Vec3, tag:string}>
//     bounds:       {min:Vec3, max:Vec3}
//     surfaceOf(mesh) -> 'concrete'|'sand'|'metal'|'wood'|'glass'|'fabric'  (for impact fx/audio)
//   }

import * as THREE from 'three';

export function buildLevel(materials, rng) {
  const group = new THREE.Group();
  group.name = 'level';
  const colliders = [];
  const raycastables = [];
  const surfaces = new WeakMap();

  const add = (mesh, surface) => {
    mesh.castShadow = true; mesh.receiveShadow = true;
    group.add(mesh); raycastables.push(mesh); surfaces.set(mesh, surface);
    return mesh;
  };

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(120, 120),
    materials.get('sand', { repeat: 24 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  add(ground, 'sand');

  // Placeholder blockout — replaced by the level agent.
  const box = new THREE.BoxGeometry(1, 1, 1);
  for (let i = 0; i < 24; i++) {
    const w = 2 + (i % 4) * 1.5, h = 3 + (i % 3) * 1.2, d = 2 + (i % 5);
    const m = new THREE.Mesh(box, materials.get('concrete', { repeat: 2 }));
    m.scale.set(w, h, d);
    const a = (i / 24) * Math.PI * 2;
    m.position.set(Math.cos(a) * (14 + (i % 3) * 6), h / 2, Math.sin(a) * (14 + (i % 4) * 5));
    add(m, 'concrete');
    colliders.push(aabbOf(m));
  }

  return {
    group, colliders, raycastables,
    playerStart: { pos: new THREE.Vector3(0, 1.7, 0), yaw: 0 },
    spawnPoints: Array.from({ length: 12 }, (_, i) => {
      const a = (i / 12) * Math.PI * 2;
      return { pos: new THREE.Vector3(Math.cos(a) * 26, 0, Math.sin(a) * 26), tag: 'enemy' };
    }),
    bounds: { min: new THREE.Vector3(-58, 0, -58), max: new THREE.Vector3(58, 24, 58) },
    surfaceOf: (mesh) => surfaces.get(mesh) || 'concrete',
  };
}

export function aabbOf(mesh) {
  mesh.updateMatrixWorld(true);
  const b = new THREE.Box3().setFromObject(mesh);
  return { min: b.min.clone(), max: b.max.clone() };
}
