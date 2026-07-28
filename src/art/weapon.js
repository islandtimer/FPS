// OWNER: agent "weapon" — first-person weapon meshes, built in code.
// LEGAL: fictional designs and fictional names only. No real-world manufacturer
// names, no Activision/Infinity Ward/Treyarch weapon names, no trademarks.
// CONTRACT:
//   buildWeapon(id, materials) -> {
//     group:  THREE.Group      (origin at the eye-space rest position)
//     parts:  { receiver, barrel, bolt, mag, stock, foregrip, optic, muzzle, trigger, charging }
//     muzzleTip: THREE.Object3D   (world position used for tracers/flash)
//     ejectPort: THREE.Object3D
//     config: {
//       name, class, rpm, damage, headMult, magSize, reserve,
//       spreadHip, spreadAds, recoil:{vertical,horizontal,recovery},
//       adsTime, reloadTime, reloadEmptyTime, muzzleVelocity, falloff:[[m,mult],...]
//     }
//   }
//   WEAPON_IDS -> string[]

import * as THREE from 'three';

export const WEAPON_IDS = ['ar_vector', 'smg_wasp', 'dmr_ridge'];

export function buildWeapon(id, materials) {
  const group = new THREE.Group();
  const gun = materials.get('gunmetal', { roughness: 0.42, metalness: 0.9 });

  const receiver = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.09, 0.34), gun);
  receiver.position.set(0, -0.02, -0.1);
  group.add(receiver);

  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.4, 12), gun);
  barrel.rotation.x = Math.PI / 2;
  barrel.position.set(0, 0.005, -0.42);
  group.add(barrel);

  const mag = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.16, 0.07), gun);
  mag.position.set(0, -0.13, -0.08);
  group.add(mag);

  const muzzleTip = new THREE.Object3D();
  muzzleTip.position.set(0, 0.005, -0.62);
  group.add(muzzleTip);

  const ejectPort = new THREE.Object3D();
  ejectPort.position.set(0.045, 0.01, -0.14);
  group.add(ejectPort);

  group.traverse((o) => { o.castShadow = false; o.receiveShadow = false; });

  return {
    group,
    parts: { receiver, barrel, mag, bolt: null, stock: null, foregrip: null, optic: null, muzzle: null },
    muzzleTip, ejectPort,
    config: {
      name: 'VK-7 CARBINE', class: 'assault', rpm: 720,
      damage: 28, headMult: 1.9, magSize: 30, reserve: 210,
      spreadHip: 0.032, spreadAds: 0.0035,
      recoil: { vertical: 1.0, horizontal: 0.45, recovery: 7.5 },
      adsTime: 0.22, reloadTime: 1.9, reloadEmptyTime: 2.6,
      muzzleVelocity: 780, falloff: [[0, 1], [28, 1], [55, 0.72], [90, 0.55]],
    },
  };
}
