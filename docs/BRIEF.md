# Glitch Metal Instrumental: Visualiser Brief

Abstract, audio-reactive visualiser for an instrumental glitch metal track with a pop-melody hook.
Metal = cold steel grid that fragments with the riff. Pop = one smooth ribbon of light following the synth, never broken by glitches. Chorus: grid dissolves into depth, ribbon blooms into full palette. No characters, logos or text.

## Palette
| Role | Hex |
| --- | --- |
| Background | `#0A0B0F` |
| Grid/shards dark | `#2A2E38` |
| Grid/shards light | `#8A93A3` |
| Flash (kick, final beat) | `#F4F6FA` |
| Synth ribbon (magenta) | `#FF2E93` |
| Glitch fringes (cyan) | `#19E3FF` |
| Chorus gradient (violet) | `#7A3CFF` |
Proportions 80% base/greys, 15% magenta+cyan, 5% flash+violet. Channel-split uses magenta/cyan, never raw R/B.

## Audio mapping
- Double-kick: fast grid pulse, small camera shake on heaviest hits
- Stuttering snare: each stutter = frame repeat / datamosh smear
- Palm-muted riff: angular shards appear/vanish on the guitar syncopation
- Synth lead: single luminous ribbon, height follows pitch
- Clean chorused guitar: faint cyan echo trailing the ribbon in hooks
- Sub bass: slow background brightness swell
- Tape-stop fills: time-remapped slowdown + chromatic aberration
- Sidechain pump: whole frame breathes on the beat in choruses
- Stereo image: guitar pan drifts the shard lattice sideways; mix width widens the magenta/cyan split

## Sections
| Section | Image | Colour |
| --- | --- | --- |
| Intro | Ribbon alone on black, flickering on stutter edits | Magenta only |
| Verse 1 | Grid fragments/reassembles every 4 bars with bit-crush drop-outs | Steel + thin cyan fringes |
| Pre-chorus | Frame rate visibly drops (half-time); shards thin to lines | Colour drains to grey, faint magenta line |
| Chorus | Frame opens up, widescreen; grid dissolves into depth, ribbon blooms | Full palette + violet gradient |
| Verse 2 | Grid returns; ribbon answers riff between phrases | As verse 1, ribbon dim |
| Pre-chorus | As first | As first |
| Chorus | As first | Full palette |
| Instrumental break | Pixel sorting, resolution drop; ribbon higher in frame | Mono + single cyan line |
| Final chorus | As first, more layers, brighter bloom | Full palette, brighter |
| End | Stuttering tail, black on last beat | Colour cuts with sound |

## Format
16:9, 30 fps, deliberately dropped frames during glitches. Keep a version without the pre-chorus frame-rate drop for heavy recompression. Also produce cover art (steel shard grid split by one magenta ribbon with cyan fringe) and a YouTube description.

Fill in section timestamps in `docs/SECTIONS.md` once the track is in `audio/`.
