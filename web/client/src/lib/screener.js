// Demo-mode mirror of src/maplemind/skills/screening.py (used ONLY when no server is running, so the hosted
// preview's check-in still works). The Python implementation is the source of truth and is what live mode uses.
export function newState(order, max) {
  return { order, max, current: 0, item: 0, answers: {}, asked: 0, done: false, partial: false, safety: false };
}
export function answer(s, inst, score) {
  s = structuredClone(s);
  const name = s.order[s.current];
  const I = inst[name];
  (s.answers[name] ||= []).push(score);
  s.asked += 1;
  if ((I.safety_items || []).includes(s.item) && score > 0) { s.safety = true; s.done = true; return s; }
  s.item += 1;
  const gate = I.gate;
  const atGateEnd = s.item === Math.max(...gate.items) + 1;
  const next = () => {
    s.current += 1; s.item = 0;
    if (s.current >= s.order.length) { s.done = true; return; }
    const nx = inst[s.order[s.current]];
    if (s.asked + nx.gate.items.length > s.max) { s.done = true; s.partial = true; }
  };
  if (atGateEnd && gate.items.reduce((a, i) => a + s.answers[name][i], 0) < gate.threshold) next();
  else if (s.item >= I.items.length) next();
  if (!s.done && s.asked >= s.max) { s.done = true; s.partial = true; }
  return s;
}
export function summary(s, inst) {
  const results = {};
  for (const [name, scores] of Object.entries(s.answers)) {
    const I = inst[name], total = scores.reduce((a, b) => a + b, 0);
    const brief = I.gate.items.filter((i) => i < scores.length).reduce((a, i) => a + scores[i], 0);
    const complete = scores.length === I.items.length;
    const r = { instrument: I.name, items_answered: scores.length, brief_score: brief, brief_positive: brief >= I.gate.threshold, complete };
    if (complete) Object.assign(r, { total, band: I.bands.find((b) => b[0] <= total && total <= b[1])[2], positive: total >= I.positive_cutoff });
    results[name] = r;
  }
  return { partial: s.partial, questions_asked: s.asked, safety_trigger: s.safety, results };
}
