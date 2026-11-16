# musicvis

Scripted, repeatable music visualisers, one per track. Resources: `docs/RESOURCES.md`.

| Track | Brief | Sections | Data builder | HyperFrames project |
| --- | --- | --- | --- | --- |
| glitch-metal | `docs/BRIEF.md` | `docs/sections.json` | `tools/build_data.py` | `video/` |
| one-fuzz | `docs/one-fuzz/BRIEF.md` | `docs/one-fuzz/sections.json` | `tools/build_one_fuzz.py` | `video-one-fuzz/` |

## Pipeline
1. Track in `audio/` (gitignored); copy to `<project>/assets/` for the composition.
2. `uv run tools/analyze.py audio/<track>.wav` -> Demucs `htdemucs_6s` stems + `analysis/<track>.json` (Python 3.12 pinned: torch has no 3.13 wheel). If the track came from Suno, export its stems instead and pass `--stems audio/stems/suno/<track>` (cleaner than Demucs; one-fuzz uses this).
3. Edit the track's `sections.json` (section type + start/end seconds; PROVISIONAL until heard).
4. Run the track's data builder -> `<project>/audio-data.js` (per-frame channels; all decisions precomputed so the shader is a pure function of the frame). Stereo pan/width per stem comes from `tools/stereo.py`, cached in `analysis/<track>.stereo.npz`.
5. `<project>/index.html` is the HyperFrames composition: one WebGL2 fragment shader driven by `audio-data.js` through a single seekable GSAP tween. Palette hexes live as consts in the shader.
6. `cd <project> && npx hyperframes@0.9.30 check`, `snapshot --at ...`, then `render -w 4 -o ../renders/<name>.mp4` (`--start/--end -q draft` for quick looks). Don't edit `audio-data.js`/`index.html` while a render is running.

## Rules (all tracks)
- Use only the palette hexes in that track's brief.
- No text, faces or logos. 16:9, 30 fps. Keep section timings in the track's `SECTIONS.md`.
- Map each stem to its own layer.

### glitch-metal
- Magenta/cyan for channel splits, never raw red/blue. Ribbon is never broken by glitch effects.
- drums->grid/glitch, other/guitar->shards, synth->ribbon, bass->background swell.

### one-fuzz
- Suno tempo drifts (140.3-142.2 BPM): use the tempo-following grid in `tools/build_one_fuzz.py`, never a fixed BPM or librosa's `beat_track`.
- Flat spot inks multiplied over paper; no glow, gradients or motion blur. Any frame must read as a print.
- drums->halftone stamps + strip shreds, fuzz riff->ink flood, arp synth->stencil, fretless->pen line, acid->teal tint.
