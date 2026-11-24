# Changelog

## v0.19.1 — 2026-11-24

- TrideKit added by URL links again. In 0.19.0 the prebuilt Filament bridge was packaged without the shared engine, so an app that used TrideKit failed to link with undefined SharedEngine symbols.
- An object whose compiled material package is missing from the scene file is drawn in its fallback look again. In 0.19.0 it could be left out as fully transparent, judged by the package it did not have.
