// Canonical camera poses. Every round's screenshots come from exactly these, so
// A/B comparisons between rounds and against reference material are apples-to-apples.
// Agents may ADD shots. Never silently change an existing pose — that breaks the
// historical comparison on the progress page.

export const CAMERA_SHOTS = {
  // Camera forward for a YXZ euler is (-sin(yaw), sin(pitch), -cos(yaw)) — yaw 0
  // looks down -Z. Poses are re-anchored once against the finished level and then
  // frozen, so every round after that is a like-for-like comparison.
  establish: {
    pos: [0, 0, 30], yaw: 0, pitch: -0.04,
    desc: 'Wide exterior. Judges: sky, sun, aerial perspective, silhouette, composition.',
  },
  street: {
    pos: [-6, 0, 12], yaw: -0.35, pitch: -0.02,
    desc: 'Mid-distance approach down cover. Judges: material variety, shadow contact, depth.',
  },
  interior: {
    pos: [8.5, 0, -3], yaw: 1.9, pitch: 0.02,
    desc: 'Enclosed space. Judges: indirect light, ambient occlusion, interior/exterior exposure balance.',
  },
  hipfire: {
    pos: [2, 0, 6], yaw: -0.2, pitch: -0.01,
    desc: 'Weapon at rest, HUD visible. Judges: viewmodel silhouette, materials, HUD hierarchy.',
  },
  ads: {
    pos: [2, 0, 6], yaw: -0.2, pitch: -0.01, ads: true,
    desc: 'Aiming down sights. Judges: optic, sight picture, DOF, weapon detail at close range.',
  },
  firefight: {
    pos: [0, 0, 8], yaw: -0.1, pitch: 0,
    desc: 'Combat moment. Judges: enemies, muzzle flash, tracers, impact FX, readability under load.',
    setup({ enemies, level, viewmodel }) {
      enemies.clear();
      for (let i = 0; i < 5; i++) enemies.spawn(level.spawnPoints[i % level.spawnPoints.length]);
      viewmodel.triggerDown(true);
    },
  },
  material_detail: {
    pos: [12.5, 0, 1.4], yaw: -1.4, pitch: -0.15,
    desc: 'Close surface read. Judges: texel density, normal detail, roughness variation, tiling artefacts.',
  },
  silhouette: {
    pos: [0, 0, 18], yaw: 0.2, pitch: 0.06,
    desc: 'Enemy against bright background. Judges: character readability, rim light, animation pose.',
    setup({ enemies, level }) {
      enemies.clear();
      enemies.spawn({ pos: level.spawnPoints[0].pos.clone().setLength(10) });
    },
  },
};

export const SHOT_NAMES = Object.keys(CAMERA_SHOTS);
