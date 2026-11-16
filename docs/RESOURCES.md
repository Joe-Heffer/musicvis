# Resources (researched 2026-11)

## Rendering / composition
- **HyperFrames** (HeyGen, open source): HTML + CSS + GSAP + WebGL -> deterministic MP4; CLI for init/preview/lint/render; has an audio-reactive workflow (pre-extract band curves, sample per frame). Docs: https://hyperframes.heygen.com ; vs Remotion: https://hyperframes.heygen.com/guides/hyperframes-vs-remotion . Installed as Claude skills here (`/hyperframes`, `hyperframes-registry` has glitch/chromatic aberration/CRT blocks).
- **Remotion**: React -> video, headless Chromium; source-available (licence needed for companies). https://www.remotion.dev
- **Revideo / Motion Canvas**: TS canvas animation; Revideo adds headless render + audio.
- **ShaderFlow** (Python, ShaderToy-like, audio-reactive): fallback if you want pure GLSL + Python.
- **TouchDesigner / After Effects**: manual alternatives from the brief; not scriptable by Claude Code.

## Audio analysis (Python)
- **Demucs** (htdemucs_6s: drums, bass, vocals, other, guitar, piano): stem split. https://github.com/adefossez/demucs
- **librosa**: RMS, STFT bands, onset detection, beat tracking, pYIN/YIN pitch. https://librosa.org
- **madmom**: neural beat/onset on the drum stem; better but hard to install (optional extra).
- Use `tools/analyze.py` -> `analysis/<track>.json`.

## Visual effects
- **three.js** + custom GLSL / postprocessing for grid, shards, ribbon (Line2/TubeGeometry), bloom.
- **@vysmo/effects**: WebGL2 filters incl. datamosh, VHS, scanlines, lens distortion (no pixel sort).
- Pixel sorting: approximate in a fragment shader (brightness threshold + directional sort) or do a CPU pass in Python/numpy on rendered frames, then ffmpeg.
- Glitch order tip: slice displacement first, then RGB split on displaced UVs. Use #FF2E93 / #19E3FF as the two split channels.
- Browser references: Codrops audio visualizer tutorials (tympanus.net/codrops, 2023 & 2025).

## Post / delivery
- **ffmpeg**: dropped-frame effect (`-vf "select=...,setpts"`, `fps`, `tblend`), bit-crush (`scale` down + `neighbor` up), encode `-c:v libx264 -crf 16 -pix_fmt yuv420p`.
