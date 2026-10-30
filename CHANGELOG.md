# Changelog

## v0.13.0 — 2026-10-30

- Tride 0.13.0 writes format 0.72 (0.12.x wrote 0.71). A scene that uses one of the new tone mapping operators (ACES Neutral, PBR Neutral, AgX Punchy) needs Tride 0.13.0 or later on every player; a 0.12.x player refuses it with a "newer format" error that asks to update Tride. A scene in Standard, ACES, ACES Legacy or Filmic still opens in 0.12.x; scenes saved by 0.12.x open in 0.13.0 unchanged.
- An imported model with several materials keeps each of them when a Bend deforms it or a cloner repeats it; before, the whole bent copy wore its first material.
- The player draws the Standard tone mapping exactly: a scene's background colour, or an unlit object glowing at strength 1, comes out as the same sRGB hex the designer picked (before, white came out #f9f7f8). It also draws the three new operators ACES Neutral, PBR Neutral and AgX Punchy.
- The repository ships no sample scenes any more: the TridePlayer example opens any `.trd` exported from Tride Editor ("Open .trd…") and lists its bindings and triggers.
- A very quick tap, pressed and lifted within one frame, now writes its point in the scene the way a click does.
- Scene work that is thinned out to save time (a live cloner over its budget, a growing L-system, a parametric shape being rebuilt) now updates on the scene's 60 Hz step rather than on every display refresh: a 120 Hz screen shows the same frames as a 60 Hz one and no longer does twice the work. A live cloner also reuses its buffers instead of allocating new ones on every update.
