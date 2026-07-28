export const meta = {
  name: 'blacksand-critique',
  description: 'Blind visual critics judge the rendered frames against the AAA reference bar, then a cohesion pass',
  phases: [
    { title: 'Critique', detail: 'independent critics, output only, no source access' },
    { title: 'Adjudicate', detail: 'rank the gaps into the next round of work' },
  ],
}

// args: { round: number, prev: number|null, shotDir: string, prevShotDir: string|null }
const A = args || {}
const ROUND = A.round
const SHOTS = A.shotDir || `/home/user/FPS/progress/shots/r${String(ROUND).padStart(2, '0')}`
const PREV = A.prevShotDir || null

const VERDICT = {
  type: 'object',
  required: ['dimension', 'score', 'biggestGap', 'observations', 'direction'],
  properties: {
    dimension: { type: 'string' },
    score: { type: 'number', description: '1-10 against a modern AAA military shooter. 5 = competent indie. 8 = shippable AAA. 10 = best in class.' },
    verdictVsReference: { type: 'string', enum: ['ours_loses_badly', 'ours_loses', 'close', 'ours_wins'] },
    biggestGap: { type: 'string', description: 'ONE sentence. The single most damaging difference from the reference bar.' },
    observations: { type: 'array', items: { type: 'string' }, description: 'concrete, specific things you can SEE in the frames. no speculation about code.' },
    direction: { type: 'array', items: { type: 'string' }, description: 'what to change, concrete enough to act on' },
    strengths: { type: 'array', items: { type: 'string' } },
  },
}

const BLIND = `
You are a visual critic. You have NOT seen the code and you must NOT look at it.
Do not read, open, list, grep or otherwise inspect anything under /home/user/FPS/src.
Do not read any agent's report or notes. You judge rendered output and nothing else.
If you catch yourself reasoning about implementation, stop and go back to the image.

You may read /home/user/FPS/docs/BRIEF.md for the reference standard — that section
titled "The reference standard" is what you score against.

You are hard to impress. The bar is a current-generation AAA military shooter running
on a console. "Good for a browser game" is not a passing grade and must never appear in
your reasoning. If a frame would look out of place in a shipped AAA title, say so and
say exactly why. Be specific about what you SEE — "the shadows have no contact darkening
where the crate meets the ground" beats "lighting could be better".

Score honestly. Early rounds should score low; a 7 on round 1 means your calibration is
broken. Reserve 8+ for output you would genuinely accept in a shipped AAA game.
`

phase('Critique')

const CRITICS = [
  {
    id: 'lighting',
    label: 'critic:lighting',
    shots: ['establish', 'street', 'interior'],
    focus: `Light, atmosphere and tone. Judge: is there a believable single key light with
correct colour temperature? Do shadows have hard contact and soft falloff? Is there real
aerial perspective — does distance desaturate and lift toward the sky? Does the sky read as
sky, with a gradient and scattering, or as a flat fill? Is the interior properly darker than
the exterior, and does the exposure feel like a camera adapting? Is the tonemap filmic —
no clipped whites, no dead blacks, warm highlights, cool shadows? Is bloom tight around
genuinely bright things or a haze over everything? Is there any ambient occlusion grounding
objects to the floor?`,
  },
  {
    id: 'materials',
    label: 'critic:materials',
    shots: ['material_detail', 'street', 'interior'],
    focus: `Surfaces. Judge: does each material read as its actual substance — is concrete
concrete, is metal metal? Is there roughness variation across a single surface, or is it
uniform? Can you see texture tiling repeating? Do normals correspond to what the albedo
shows? Is there grime accumulation in crevices, water staining, edge wear exposing base
material? Does the surface hold up at close range (texel density) and at distance
(large-scale variation)? Flat, uniform, plastic-looking surfaces are the number one tell
of a non-AAA renderer — call them out precisely.`,
  },
  {
    id: 'level',
    label: 'critic:level-art',
    shots: ['establish', 'street', 'silhouette'],
    focus: `Environment art and composition. Judge: does this read as a designed place or as
a blockout? Are hard edges chamfered so they catch a highlight, or razor-sharp? Is there
foreground / midground / background layering, or is everything at one depth? Is the skyline
against the sky interesting? Is there clutter, wear, asymmetry, evidence of use — or is it
clean primitives? Does the space suggest how you would fight in it — cover, lanes, sightlines,
verticality? Is scale believable (doors, steps, walls at human proportion)?`,
  },
  {
    id: 'weapon',
    label: 'critic:weapon',
    shots: ['hipfire', 'ads'],
    focus: `The first-person weapon. This is the object on screen 100% of the time so it is
held to the highest standard. Judge: does it read as a manufactured object — an assembly of
parts with seams, fasteners, chamfers, rails — or as extruded boxes? Are there at least three
visibly distinct materials (machined steel, polymer, rubber, glass)? Does it sit naturally in
frame at a believable angle and scale, and does its silhouette read? At ADS, is the sight
picture correct — is the optic actually centred, is the sight readable? Is there aliasing on
the edges? Is the near-camera detail dense enough to carry the frame?`,
  },
  {
    id: 'combat',
    label: 'critic:combat-fx',
    shots: ['firefight', 'silhouette'],
    focus: `Combat legibility and effects. Judge: can you instantly tell where the enemies
are and what is happening? Do the characters read as people — silhouette, proportion, pose —
or as capsules? Is the muzzle flash convincing in shape and brightness, and does it light the
scene? Are there tracers, impact effects, dust, debris? Is there any sense of weight and
violence to the moment, or does it look inert? Does anything on screen fight for attention
that should not?`,
  },
  {
    id: 'hud',
    label: 'critic:hud',
    shots: ['hipfire', 'firefight'],
    focus: `The interface. Judge: is the visual language coherent and confident, or default
browser text? Is the hierarchy right — does the most urgent information dominate? Is the
reticle crisp at the pixel level and does it communicate weapon state? Does the HUD read over
both bright sky and dark interior? Is anything blurry, mis-aligned, or fighting the centre of
frame? Does it look like a shipped game's HUD or like debug output?`,
  },
  {
    id: 'gestalt',
    label: 'critic:gestalt',
    shots: ['establish', 'street', 'hipfire', 'firefight', 'interior'],
    focus: `The whole frame, as a player would see it in a store page screenshot. One
question above all: would a person scrolling past this assume it is a AAA game, an indie
game, or a tech demo? Say which, and say what specifically gives it away. Then name the
single change that would move it up one category. Judge cohesion too — do the weapon, the
world, the characters and the interface look like they come from the same game, or like
separately-made parts bolted together?`,
  },
]

const verdicts = await parallel(
  CRITICS.map((c) => () =>
    agent(
      `${BLIND}

YOUR DIMENSION: ${c.id}

${c.focus}

Look at these rendered frames (use the Read tool on each path — they are images and you
will see them):
${c.shots.map((s) => `  ${SHOTS}/${s}.png`).join('\n')}

Each frame is a fixed camera pose that will be re-shot every round, so be specific about
what is in front of you.

Score against the reference standard in docs/BRIEF.md. Then name the SINGLE biggest gap —
not a list, the one thing that costs the most. Return the structured verdict.`,
      { label: c.label, phase: 'Critique', schema: VERDICT, effort: 'high' }
    )
  )
)

const got = verdicts.filter(Boolean)
log(`critics returned ${got.length}/${CRITICS.length}; scores: ${got.map(v => `${v.dimension}=${v.score}`).join(' ')}`)

// Blind A/B against the previous round, if there is one.
let ab = null
if (PREV) {
  ab = await agent(
    `${BLIND}

Two builds of the same game, same camera poses. They are labelled A and B and you are not
told which is newer. Judge which is better and by how much, per pose and overall.

BUILD A:
${['establish', 'street', 'hipfire', 'firefight'].map((s) => `  ${PREV}/${s}.png`).join('\n')}

BUILD B:
${['establish', 'street', 'hipfire', 'firefight'].map((s) => `  ${SHOTS}/${s}.png`).join('\n')}

For each pose say which is better and why, in terms of what you can see. Then give an
overall winner and a margin (decisive / clear / slight / indistinguishable). If B is worse
than A in any specific respect, say so explicitly — a regression that nobody names gets shipped.`,
    {
      label: 'critic:ab', phase: 'Critique', effort: 'high',
      schema: {
        type: 'object',
        required: ['winner', 'margin', 'perPose', 'regressions'],
        properties: {
          winner: { type: 'string', enum: ['A', 'B', 'tie'] },
          margin: { type: 'string', enum: ['decisive', 'clear', 'slight', 'indistinguishable'] },
          perPose: { type: 'array', items: { type: 'string' } },
          regressions: { type: 'array', items: { type: 'string' } },
          reasoning: { type: 'string' },
        },
      },
    }
  )
}

phase('Adjudicate')

const plan = await agent(
  `You are the art director on this project. Seven independent critics have each judged one
dimension of the same rendered frames, blind to the code and to each other. Their verdicts:

${JSON.stringify(got, null, 2)}

${ab ? `A blind A/B against the previous build returned:\n${JSON.stringify(ab, null, 2)}\n` : ''}

You may read /home/user/FPS/docs/BRIEF.md and you MAY read the source under
/home/user/FPS/src to judge feasibility — you are the one person here who sees both sides.

Produce the work order for the next round. Rules:
- Rank by how much each fix moves the overall "is this AAA" verdict per unit of effort.
  A critic scoring 3/10 on something nobody looks at outranks nothing.
- Assign each item to exactly one owning module from the ownership table in the brief.
  Two items must never require the same file to be edited by two agents.
- Every item must be concrete enough that a builder can act on it without asking questions.
- Respect the frame budget. If an item costs budget, say where it comes from.
- Cap it at 8 items. Fewer, deeper items beat many shallow ones.`,
  {
    label: 'director:work-order', phase: 'Adjudicate', effort: 'high',
    schema: {
      type: 'object',
      required: ['overallScore', 'headline', 'workOrder'],
      properties: {
        overallScore: { type: 'number' },
        headline: { type: 'string', description: 'one sentence: where this build actually stands' },
        biggestSingleGap: { type: 'string' },
        workOrder: {
          type: 'array',
          items: {
            type: 'object',
            required: ['module', 'title', 'task', 'why'],
            properties: {
              module: { type: 'string' },
              title: { type: 'string' },
              task: { type: 'string', description: 'the full instruction to hand a builder' },
              why: { type: 'string' },
              budgetCost: { type: 'string' },
              priority: { type: 'number' },
            },
          },
        },
      },
    },
  }
)

return { round: ROUND, verdicts: got, ab, plan }
