# one-fuzz section timestamps (approved by ear 2026-11-11 against renders/one-fuzz_check.wav)
Source of truth: `docs/one-fuzz/sections.json`. Check by ear with `uv run tools/build_one_fuzz.py --check`, which writes
`renders/one-fuzz_check.wav`: the mix + clicks on the beat grid (high click = downbeat) + a bell at each section start.

## Beat grid
The track is not at a fixed tempo: it drifts between 140.3 and 142.2 BPM (fastest around 110-135 s and 180-205 s).
`beat_grid()` in `tools/build_one_fuzz.py` tracks a smoothed local-tempo curve (Ellis DP) on the Suno drums + percussion
onsets. It's cached in `analysis/one-fuzz.grid.json`; delete that file to recompute. Validation: in every 8-bar
window the best extra shift is -2 ms (librosa's fixed-tempo tracker in v1 was off by up to ±80 ms). Downbeat = beat
phase 2 (most kick energy). 1 bar ≈ 1.70 s.

## Stems (Suno export, `audio/stems/suno/one-fuzz/`)
| Stem | Carries | Layer |
| --- | --- | --- |
| drums | kit; kick <150 Hz, snare/rim 1-4 kHz, hats 6-11 kHz | black / teal / pink stamps, shreds, kick jolts |
| percussion | shakers, rims | pink stamps |
| bass | fuzz-octave riff, ~65 Hz fundamental | ink flood (drops) |
| guitar | fretless runs, 4-7 onsets/s at 200-400 Hz in the solos | pen line |
| synth | acid line, sustained ~65 Hz, filter-swept | teal tint |
| keyboard + other | glassy arp / top line | stencil band |
| lead_vocals, brass, woodwinds | 15-40 dB below the rest | ignored |

## Sections (bar numbers count from the first downbeat, 0.91 s = bar 1)
| Section | Type | Start (s) | End (s) | Bars | Evidence |
| --- | --- | --- | --- | --- | --- |
| Intro | intro | 0.00 | 14.51 | -9 | percussion + other up, bass on/off |
| Verse 1 | verse | 14.51 | 41.84 | 9-25 | bass + synth (acid) enter, percussion drops; keyboard joins at bar 17 |
| Fretless solo 1 | solo | 41.84 | 55.49 | 25-33 | guitar on, bass/synth out |
| Drop 1 | drop | 55.49 | 69.18 | 33-41 | bass + synth full, guitar silent |
| Verse 2 | verse | 69.18 | 82.81 | 41-49 | guitar returns under bass + synth |
| Fretless answer 2 | solo | 82.81 | 91.28 | 49-54 | synth falls 20 dB, guitar runs |
| Drop 2 | drop | 91.28 | 96.38 | 54-57 | guitar out, synth + keyboard up |
| Half-time breakdown | breakdown | 96.38 | 103.17 | 57-61 | drums dip 5-10 dB |
| Build | solo | 103.17 | 109.93 | 61-65 | drums back, guitar runs, no percussion |
| Drop 3 | drop | 109.93 | 116.74 | 65-69 | guitar out, bass/synth/keyboard full |
| Fretless answer 3 | solo | 116.74 | 123.50 | 69-73 | guitar on, synth drops ~25 dB |
| Drop 3b | drop | 123.50 | 136.99 | 73-81 | as drop 3 |
| Verse 3 | verse | 136.99 | 150.62 | 81-89 | synth down to -40, keyboard out; bass out 147-150 |
| Verse 4 | verse | 150.62 | 160.84 | 89-95 | guitar + keyboard trade bars, drums dip |
| Final drop | drop (final) | 160.84 | 215.05 | 95-127 | loudest; guitar runs 191-201 |
| End chord | end | 215.05 | 222.80 | | last drum hit + fuzz chord at 215.05; chord rings to the end of the file |
| Silence | silence | 222.80 | 223.04 | | Suno leaves almost no tail |
