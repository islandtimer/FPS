export const meta = {
  name: 'blacksand-build',
  description: 'Execute a ranked work order: one builder per owned module, disjoint files, budget-aware',
  phases: [{ title: 'Build', detail: 'one builder per module' }],
}

// args: { items: [{module, title, task, why, budgetCost}], round: number }
// Tolerate args arriving as a JSON string — a silently empty ITEMS list spawns zero
// agents and reports success, which is the worst possible failure mode here.
const A = (typeof args === 'string' ? JSON.parse(args) : args) || {}
const ITEMS = A.items || []
if (!ITEMS.length) throw new Error('build workflow received no work items — refusing to no-op')

const OWNS = {
  render: ['src/engine/renderer.js'],
  sky: ['src/engine/sky.js'],
  tex: ['src/art/textures.js', 'src/art/materials.js'],
  level: ['src/art/level.js'],
  weapon: ['src/art/weapon.js'],
  char: ['src/art/character.js'],
  player: ['src/game/player.js'],
  feel: ['src/game/weaponfx.js'],
  combat: ['src/game/combat.js'],
  ai: ['src/game/ai.js'],
  loop: ['src/game/director.js'],
  hud: ['src/ui/hud.js'],
  audio: ['src/audio/audio.js'],
}

const REPORT = {
  type: 'object',
  required: ['agent', 'built', 'summary', 'changes', 'budgetImpact'],
  properties: {
    agent: { type: 'string' },
    built: { type: 'boolean' },
    buildError: { type: 'string' },
    summary: { type: 'string' },
    changes: { type: 'array', items: { type: 'string' } },
    budgetImpact: {
      type: 'object',
      properties: {
        drawCalls: { type: 'string' }, triangles: { type: 'string' },
        fullscreenPasses: { type: 'string' }, shadowTexels: { type: 'string' },
        logicMs: { type: 'string' }, notes: { type: 'string' },
      },
    },
    contractChangesNeeded: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

// Several work-order items can land on the same module. Group them so one agent
// owns one file — two agents editing the same file would lose each other's work.
const byModule = new Map()
for (const it of ITEMS) {
  if (!byModule.has(it.module)) byModule.set(it.module, [])
  byModule.get(it.module).push(it)
}

const COMMON = `
You are one of several agents working in parallel on /home/user/FPS, an all-procedural
Three.js first-person shooter whose bar is a modern AAA military shooter.

FIRST read /home/user/FPS/docs/BRIEF.md in full — reference standard, hard performance
budgets, legal constraints, ownership. Then read the file(s) you own, in full, before editing.

RULES:
- Edit ONLY your own file(s). Other agents are editing other files right now.
- Preserve the exported contract at the head of your file exactly. Report any needed
  contract change instead of making it.
- Everything procedural, no new dependencies, no downloaded assets.
- No Activision/Infinity Ward/Treyarch trademarks, weapon names, map names, UI strings or
  sound design. No real firearm manufacturer or model designations. Fictional only.
- Seeded randomness only (src/core/rng.js) so screenshots reproduce between rounds.
- No per-frame allocation in hot paths.
- Respect the frame budget. Report your true cost, measured where you can measure it.

This work order came from critics who saw ONLY the rendered frames, never the code. If a
critic describes something as missing that you believe is present, the honest reading is
that it is not reading as present on screen — fix the perception, not the critic.

VERIFY before reporting, from /home/user/FPS:
  npx vite build --outDir .build/AGENT --emptyOutDir --logLevel error
If it will not build, fix it; if you truly cannot, report built:false with the error.

Your final message is a machine-consumed structured report.
`

phase('Build')

const reports = await parallel(
  [...byModule.entries()].map(([mod, items]) => () =>
    agent(
      `${COMMON}
YOUR AGENT NAME: ${mod}
YOU OWN: ${(OWNS[mod] || []).join(', ')}

The art director assigned you ${items.length} item(s) this round. Do all of them.

${items.map((it, i) => `--- ITEM ${i + 1}: ${it.title} ---
${it.task}

Why this matters: ${it.why}
${it.budgetCost ? `Budget cost expected: ${it.budgetCost}` : ''}`).join('\n\n')}

Do the deepest version of this work that stays inside the frame budget. Shallow changes
will be caught by the critics next round and sent straight back.`,
      { label: `build:${mod}`, phase: 'Build', schema: REPORT, effort: 'high' }
    )
  )
)

const ok = reports.filter(Boolean)
log(`build: ${ok.filter((r) => r.built).length}/${byModule.size} modules built clean`)
return { reports: ok }
