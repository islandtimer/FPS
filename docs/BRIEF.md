# OPERATION BLACKSAND — build brief

Read this before touching anything. It is the shared contract for every agent
working on this project.

## What we are making

A first-person shooter in Three.js whose visual and tactile bar is a modern
AAA military shooter — the Call of Duty tier. Not "good for a web game".
Good, full stop, against the actual thing.

Everything is generated in code. Every texture, mesh, animation and sound is
synthesised at runtime or build time. There are no downloaded images, no audio
files, no imported models. This is deliberate: a shader can be iterated across
rounds in a way a downloaded PNG cannot.

## Two hard bars

**Bar 1 — it looks and feels AAA.** Judged by critics who see only rendered
output, never the code or the reasoning behind it.

**Bar 2 — 1080p at 60fps on an M1 / GTX 1660.** Non-negotiable, and the visual
critics cannot see it, so it is enforced mechanically by `tools/bench.mjs`.
A build that looks flawless and runs at 25fps has failed.

### Measuring bar 2 honestly

The build container has **no GPU**. Chromium falls back to SwiftShader, a CPU
rasteriser. Full-resolution frame rate measured here is therefore *not* a
prediction of an M1 or a GTX 1660, and must never be quoted as one. What we
measure instead:

| signal | how | meaning |
|---|---|---|
| `logicMs` | 900 synchronous `step()` calls, timed directly | pure JS cost/frame. Transfers to real hardware. Budget **≤ 4.0ms p95**. |
| `drawCalls` | `renderer.info.render.calls` | budget **≤ 260** |
| `triangles` | `renderer.info.render.triangles` | budget **≤ 950k** |
| `fullscreenPasses` | declared by the render pipeline | budget **≤ 8** at 1080p |
| `shadowTexels` | declared by the atmosphere module | budget **≤ 14M/frame** |
| `programs` | compiled shader count | budget **≤ 90** |
| `swFps` | SwiftShader at 1080p | round-over-round regression signal ONLY |
| `estimate` | cost model → ms/frame on M1 / 1660 | always labelled an estimate |

If your change pushes a budget over its limit, it is not done, no matter how
good it looks. Budgets are in `src/engine/perf.js`.

## The reference standard

This is what the critics score against. It is a description of how the target
class of game actually renders, written down so judgement is consistent.

**Light and atmosphere.** A single strong key with a warm cast and a cool
ambient fill; the sky is a gradient with real aerial perspective, not a flat
colour. Distance desaturates and lifts toward the horizon. Sun position creates
long readable shadows with sharp contact and soft falloff. Interiors are
markedly darker than exteriors and the eye adapts between them. Fog is height-
and distance-aware, thicker at ground level, with visible light shafts where the
sun rakes through a gap.

**Materials.** Nothing is uniform. Concrete has patchy roughness, water
staining, chipped edges with lighter aggregate underneath, and grime that
accumulates in crevices. Metal has directional anisotropy, edge wear that
exposes bright base metal, and rust that blooms from fixings outward. Every
surface reads at three distances: silhouette at 40m, pattern at 10m, texel
detail at 0.5m. Tiling must not be visible — break it with large-scale
variation, detail maps, and per-instance colour.

**Tone.** Filmic curve, never a clipped highlight or a crushed black without
intent. Slight desaturation overall, warm highlights, cool shadows. Bloom is a
tight veil around genuinely bright pixels, not a haze over everything.
Sub-pixel edge quality matters — aliasing on a rifle rail reads as cheap
instantly.

**The weapon.** Occupies the lower-right quadrant, held at a believable angle,
close enough that its material detail carries the frame. Distinct machined,
polymer, and coated-steel surfaces. Rails, fasteners, seams and sling points
read as manufactured, not extruded. It moves constantly — never rigid.

**Feel.** Sub-100ms response on everything. ADS is a fast eased transition with
FOV compression, not a lerp. Recoil is a repeatable pattern with a visual kick
that recovers faster than the pattern, so skilled players can control it. Firing
shakes the camera a few pixels, punches the viewmodel back and up, flashes the
muzzle for one to two frames, ejects a shell, and leaves a hot barrel.

**Readability under load.** In a firefight the player must instantly parse:
where the enemies are, where damage is coming from, how much ammo is left,
whether the last shot hit. Everything else gets out of the way.

## Legal — absolute

- No Activision / Infinity Ward / Treyarch / Sledgehammer assets, audio,
  textures, meshes, model names, map names, faction names, UI strings, rank
  names, killstreak names, sound design lifted from recordings, or trademarks.
- No real-world firearm manufacturer or model designations. Weapons are
  fictional designs with fictional names.
- No real military insignia, flags, or unit markings.
- Everything procedural, everything original.

## Module ownership

Each agent owns exactly the files listed for it and **edits nothing else**.
Contracts at the top of each file are load-bearing — other modules import
against them. If you need a contract changed, say so in your report; do not
change it unilaterally.

| agent | owns |
|---|---|
| render | `src/engine/renderer.js` |
| sky | `src/engine/sky.js` |
| tex | `src/art/textures.js`, `src/art/materials.js` |
| level | `src/art/level.js` |
| weapon | `src/art/weapon.js` |
| char | `src/art/character.js` |
| player | `src/game/player.js` |
| feel | `src/game/weaponfx.js` |
| combat | `src/game/combat.js` |
| ai | `src/game/ai.js` |
| loop | `src/game/director.js` |
| hud | `src/ui/hud.js` |
| audio | `src/audio/audio.js` |

`src/main.js`, `src/core/*`, `src/engine/perf.js`, `src/bench/shots.js`,
`tools/*` and `progress/*` are owned by the orchestrator.

## Verifying your work

You may not run `npx vite build` into the shared `dist/`. Build into your own
directory so parallel agents do not collide:

```
npx vite build --outDir .build/<your-agent-name> --emptyOutDir --logLevel error
```

A clean exit means your module at least parses, imports and bundles. Report the
result. If you cannot make it build, say so plainly rather than reporting success.

## Style

Match the surrounding code: ES modules, no framework, no TypeScript, no build
plugins, no new dependencies beyond `three`. Comments explain *why*, not *what*.
Allocate nothing per-frame in hot paths — pool and reuse. Seeded randomness only,
via `src/core/rng.js`, so screenshots are reproducible between rounds.
