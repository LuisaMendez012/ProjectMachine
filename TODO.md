# UI/UX Redesign TODO (Smart City Platform) — Do not break backend

## Phase 1 — Emoji removal (no functionality changes)
- [ ] Search all templates for emoji characters and replace them with professional icon HTML/SVG (no emojis anywhere).
- [ ] Update `templates/base.html` brand icon (remove 🚲) with inline SVG.
- [ ] Update Home (`templates/fase1.html`) remove emoji icons (🗺️⏱️🌦️🤖📊🔬📈📍) and ✓/⚠.
- [ ] Update Methodology (`templates/fase2.html`) remove 1️⃣..6️⃣ and ✓.
- [ ] Update Prediction (`templates/prediction.html`) remove ✓/⚠/📍 and implement required new UI pieces later.
- [ ] Update About (`templates/about.html`) remove 🚀/📍/✓.
- [ ] Re-run emoji scan to verify zero matches.

## Phase 2 — Navbar redesign & UX order
- [ ] Confirm navbar order and active highlight matches approved IA.
- [ ] Ensure Prediction System is visually highlighted.

## Phase 3 — Home page redesign
- [ ] Redesign `templates/fase1.html` into a clean landing page: hero, benefits, short overview, quick preview cards, impact stats.
- [ ] Replace inline styles with CSS classes where feasible.

## Phase 4 — Prediction System redesign (highest priority)
- [ ] Redesign `templates/prediction.html` layout: two-column form/result.
- [ ] Preserve backend POST form fields `lugar` and `zona` exactly.
- [ ] Add UI fields for Time of Day + Weather Condition without breaking backend by mapping client-side into `lugar` (or keep hidden if backend can’t consume).
- [ ] Create occupancy indicator + recommendation text with professional styling.
- [ ] Add confidence/reliability section.

## Phase 5 — Methodology (CRISP-ML) presentation only
- [ ] Redesign `templates/fase2.html` into elegant step/timeline/accordion style.
- [ ] Preserve all academic content; do not remove methodology text.

## Phase 6 — Model Engineering + Evaluation + About
- [ ] Ensure required academic info is preserved but presented cleanly.
- [ ] Improve `templates/fase3.html` dashboard layout; ensure Random Forest highlighted.
- [ ] Polish `templates/about.html` into clean card-based layout.

## Phase 7 — Final polish & QA
- [ ] Responsive checks: desktop/tablet/mobile.
- [ ] Visual consistency and accessibility review.
- [ ] Run Flask and validate routes + prediction workflow.
- [ ] Ensure Render deployment compatibility.

