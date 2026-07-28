// Minimal synchronous event bus. Modules never import each other directly for
// gameplay events — they publish here. Keeps module ownership clean.

const handlers = new Map();

export const bus = {
  on(evt, fn) {
    let s = handlers.get(evt);
    if (!s) handlers.set(evt, (s = new Set()));
    s.add(fn);
    return () => s.delete(fn);
  },
  off(evt, fn) { handlers.get(evt)?.delete(fn); },
  emit(evt, payload) {
    const s = handlers.get(evt);
    if (!s) return;
    for (const fn of s) {
      try { fn(payload); } catch (e) { console.error('[bus]', evt, e); }
    }
  },
  clear() { handlers.clear(); },
};

/**
 * Canonical event names. Add here, not ad hoc, so modules stay decoupled.
 *   shot        {origin,dir,weapon,spread}
 *   hit         {point,normal,material,surface,damage,headshot,victim}
 *   kill        {victim,headshot,distance}
 *   damage      {amount,fromDir,source}
 *   reload      {phase:'start'|'magout'|'magin'|'end', weapon}
 *   footstep    {surface,speed,foot}
 *   wave        {index,phase:'start'|'clear'|'intermission', remaining}
 *   ui          {kind,...}
 */
export const EV = {
  SHOT: 'shot', HIT: 'hit', KILL: 'kill', DAMAGE: 'damage',
  RELOAD: 'reload', FOOTSTEP: 'footstep', WAVE: 'wave',
  WEAPON_STATE: 'weaponState', EXPLOSION: 'explosion', UI: 'ui',
  SPAWN: 'spawn', DEATH: 'death', ADS: 'ads', LAND: 'land',
};
