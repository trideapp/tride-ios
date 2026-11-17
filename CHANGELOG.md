# Changelog

## v0.18.0 — 2026-11-17

- Tride 0.18.0 writes format 0.78 (0.17.x wrote 0.77), which adds an environment source that shows and lights nothing, None. A scene with an environment set to None needs Tride 0.18.0 or later on every player; Tride 0.17.x refuses it with the "newer format" error that asks to update Tride. Any other scene saved by 0.18.0 still opens on Tride 0.17.x, with a warning each time it loads that the file comes from a newer format. Scenes saved by earlier Tride open in 0.18.0 with the look they had.
- An environment can be set to None: it shows nothing and lights nothing. The light passes to the next environment down the list that lights the scene, and where nothing below shows, the background is transparent.
