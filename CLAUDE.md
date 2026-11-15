# musicvis

Generic toolkit for scripted music visualisers, one per track. Resources: `docs/RESOURCES.md`.

## Pipeline
1. Track in `audio/` (gitignored).
2. `uv run tools/analyze.py audio/<track>.wav` -> Demucs `htdemucs_6s` stems + `analysis/<track>.json` (Python 3.12 pinned: torch has no 3.13 wheel). For Suno tracks, export its stems and pass `--stems <dir>`.
3. Per track: write a brief, a `sections.json` (section type + start/end seconds), and a data builder that emits `<project>/audio-data.js` (per-frame channels; all decisions precomputed so the shader is a pure function of the frame).
4. `<project>/index.html` is a HyperFrames composition: one WebGL2 fragment shader driven by `audio-data.js` via a single seekable GSAP tween.
5. `npx hyperframes check`, `snapshot`, then `render`.

## Rules
- Use only the palette in the track's brief. No text, faces or logos. 16:9, 30 fps.
- Map each stem to its own layer.
- Track-specific projects (`video*/`, briefs, builders) live outside this repo.
