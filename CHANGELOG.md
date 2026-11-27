# Changelog

## v0.20.0 — 2026-11-27

- TrideKit is quiet by default: `TrideKitLog.level` (`off`, `error`, `warning` by default, `info`, `debug`) sets how much it writes to the system log, `TrideKitLog.handler` takes the lines instead, and the launch argument `-TrideKitLogLevel debug` raises the level without changing code. A playing scene writes no line at any level (the generator line every frame is gone). The runtime's notices are written once, in the `studio.tride.kit` subsystem, and reach `subscribeDiagnostics` with code `core.notice`.
