// OWNER: agent "char" — enemy/ally characters: skinned mesh + skeleton + clips,
// all generated in code. Fictional faction, no real insignia or trademarks.
// CONTRACT:
//   buildCharacter(materials, variant, rng) -> {
//     group:  THREE.Group
//     skeleton: THREE.Skeleton | null
//     hitboxes: Array<{name:'head'|'chest'|'stomach'|'limb', mult:number, obj:THREE.Object3D, radius:number, halfHeight:number}>
//     setPose(state) -> void   state: {move:0..1, aim:0..1, crouch:0..1, dead:number, hitDir?:Vec3}
//     update(dt)
//     dispose()
//   }
//   CHARACTER_VARIANTS -> string[]

import * as THREE from 'three';

export const CHARACTER_VARIANTS = ['rifleman', 'shotgunner', 'sniper'];

export function buildCharacter(materials, variant = 'rifleman', rng) {
  const group = new THREE.Group();
  const body = materials.get('fabric', { color: 0x6b6250, roughness: 0.95 });

  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.24, 0.5, 4, 10), body);
  torso.position.y = 1.15;
  group.add(torso);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 10), body);
  head.position.y = 1.62;
  group.add(head);

  group.traverse((o) => { o.castShadow = true; o.receiveShadow = true; });

  return {
    group, skeleton: null,
    hitboxes: [
      { name: 'head', mult: 1.9, obj: head, radius: 0.13, halfHeight: 0.0 },
      { name: 'chest', mult: 1.0, obj: torso, radius: 0.26, halfHeight: 0.3 },
    ],
    setPose() {},
    update() {},
    dispose() {},
  };
}
