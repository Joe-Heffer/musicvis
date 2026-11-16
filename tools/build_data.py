#!/usr/bin/env python3
"""Turn analysis/<track>.json + docs/sections.json into video/audio-data.js (per-frame channels, 0..255).

Usage: uv run tools/build_data.py [--track glitch-metal] [--no-prechorus-drop]
All decisions (envelopes, section ramps, frame holds, tape-stop proxy) are made here so the
browser composition stays a pure function of the frame index (deterministic render).
"""
import argparse, json
from pathlib import Path
import numpy as np
from stereo import stereo, unit_pan, unit_width

ROOT = Path(__file__).resolve().parent.parent

def hash01(i, salt=0):
    x = np.sin((np.asarray(i, float) + 1) * 12.9898 + salt * 78.233) * 43758.5453
    return x - np.floor(x)

def smooth(x, k):
    k = max(1, int(k)); return np.convolve(x, np.ones(k) / k, "same")

def edge_env(curve, thr, decay, gain=3.0):
    """Peak-triggered decaying envelope from a 0..1 band curve."""
    d = np.diff(curve, prepend=curve[0])
    env = np.zeros(len(curve)); e = 0.0
    for i, v in enumerate(d):
        e *= decay
        if v > thr: e = max(e, min(1.0, v * gain))
        env[i] = e
    return env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="glitch-metal")
    ap.add_argument("--no-prechorus-drop", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "video" / "audio-data.js"))
    a = ap.parse_args()
    d = json.loads((ROOT / "analysis" / f"{a.track}.json").read_text())
    secs = json.loads((ROOT / "docs" / "sections.json").read_text())["sections"]
    fps = d["fps"]; S = d["stems"]; n = int(round(d["duration"] * fps))
    t = np.arange(n) / fps
    def fit(x): x = np.asarray(x, float); return np.interp(np.arange(n), np.arange(len(x)), x)

    drums, bass, gtr, pno = S["drums"], S["bass"], S["guitar"], S["piano"]
    kick = edge_env(fit(drums["bands"]["low"]), 0.10, 0.70)
    snare_raw = edge_env(fit(drums["bands"]["high"]), 0.10, 0.65)
    snare = snare_raw
    # stutter density: snare hits per 0.4s window
    hits = (np.diff(snare_raw, prepend=0) > 0.2).astype(float)
    stut = np.clip(np.convolve(hits, np.ones(12), "same") / 3.0, 0, 1)
    sub = np.clip(smooth(fit(bass["bands"]["sub"]), 20) * 1.3, 0, 1)
    riff = np.clip((smooth(fit(gtr["rms"]), 2) - 0.12) * 1.8, 0, 1)
    # stereo: guitar pan drifts the shard lattice sideways; mix width scales the magenta/cyan split
    sdir = ROOT / "audio" / "stems" / "htdemucs_6s" / a.track
    img = stereo({"guitar": sdir / "guitar.wav", "mix": ROOT / "audio" / f"{a.track}.wav"}, fps, n,
                 ROOT / "analysis" / f"{a.track}.stereo.npz")
    gpan = unit_pan(img["guitar"][0])
    sdx = np.cumsum(gpan * 0.25 / fps)                       # scene units; 0.25/s at full pan
    wid = unit_width(img["mix"][1])
    echo = np.clip(smooth(fit(pno["rms"]), 15) * 1.2, 0, 1)
    # melody: per frame take the loudest of other/piano/vocals as the "synth lead" and use its raw f0
    import librosa
    cache = ROOT / "analysis" / "pitch.npz"
    stem_dir = ROOT / "audio" / "stems" / "htdemucs_6s" / a.track
    if cache.exists():
        z = np.load(cache); f0s = {k: z[k] for k in z.files}
    else:
        f0s = {}
        for nm in ("other", "piano", "vocals"):
            y, sr = librosa.load(stem_dir / f"{nm}.wav", sr=22050, mono=True)
            f0s[nm] = np.log2(librosa.yin(y, fmin=80, fmax=1500, sr=sr, frame_length=2048, hop_length=sr // fps))
        np.savez(cache, **f0s)
    names = ["other", "piano", "vocals"]
    r = np.vstack([smooth(fit(S[k]["rms"]), 6) for k in names])
    f0m = np.vstack([fit(f0s[k][: n]) if len(f0s[k]) >= n else fit(np.pad(f0s[k], (0, n - len(f0s[k])), mode="edge")) for k in names])
    win = r.argmax(0)
    p = f0m[win, np.arange(n)]
    synth = np.clip(r.max(0) * 1.4, 0, 1)
    gate = synth > 0.12
    for i in range(1, n):
        if not gate[i]: p[i] = p[i - 1]
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(p, 3, mode="edge"); p = np.median(sliding_window_view(pad, 7), axis=1)
    lo, hi = np.percentile(p, 5), np.percentile(p, 95)
    pitch = np.clip(smooth(smooth((p - lo) / (hi - lo + 1e-9), 8), 8), 0, 1)

    # ---- sections: masks with 0.35s ramps ----
    def mask(pred):
        m = np.array([1.0 if any(pred(s) and s["start"] <= x < s["end"] for s in secs) else 0.0 for x in t])
        return np.clip(smooth(m, 10), 0, 1)
    intro = mask(lambda s: s["type"] == "intro"); verse = mask(lambda s: s["type"] == "verse")
    pre = mask(lambda s: s["type"] == "pre"); chorus = mask(lambda s: s["type"] in ("chorus",))
    brk = mask(lambda s: s["type"] == "break"); fin = mask(lambda s: s.get("final", False))
    endm = mask(lambda s: s["type"] == "end")

    # ---- beats / bar position / pump ----
    beats = np.array(d["beats"]); ivl = np.median(np.diff(beats))
    bt = np.concatenate([[beats[0] - ivl], beats, [beats[-1] + ivl]])
    bpos = np.interp(t, bt, np.arange(len(bt)) - 1)           # fractional beat index
    barpos = np.clip(bpos, 0, None) / 4.0
    b16 = np.floor(np.clip(bpos, 0, None) * 4).astype(int)
    phase = bpos - np.floor(bpos)
    pump = np.exp(-phase * 4.0) * chorus * (0.6 + 0.4 * fin)

    # ---- glitch amounts per section ----
    glitch = np.clip(stut * (0.55 * intro + 1.0 * verse + 0.5 * pre + 0.35 * chorus + 1.0 * brk + 0.5 * endm) + snare * 0.35 * (verse + brk), 0, 1)
    # ---- tape-stop proxy: last 0.5s before pre/chorus/break starts ----
    ts = np.zeros(n)
    for s in secs:
        if s["type"] in ("pre", "chorus", "break") and s["start"] > 1:
            w = (t > s["start"] - 0.5) & (t < s["start"]); ts[w] = np.maximum(ts[w], (t[w] - (s["start"] - 0.5)) / 0.5)
    ca = np.clip(0.30 * verse + 0.25 * chorus + 0.5 * snare * (verse + chorus) + 0.15 * brk + ts + 0.6 * glitch * 0.5, 0, 1)
    tm = np.cumsum((1 - 0.85 * ts) / fps) * 1000

    # ---- crush: drop-outs at 4-bar fragment peaks in verses; constant in break ----
    ph4 = (barpos / 4.0) % 1.0
    drop = ((ph4 > 0.80) & (ph4 < 0.84) | (ph4 > 0.93) & (ph4 < 0.95)).astype(float) * verse
    crush = np.clip(drop + 0.85 * brk + 0.5 * stut * intro * (hash01(np.arange(n), 3) > 0.6), 0, 1)
    crush = np.clip(crush + 0.7 * ts * (1 - chorus), 0, 1)

    # ---- shake + flash on the heaviest kicks ----
    heavy = np.clip((kick - 0.8) * 5, 0, 1)
    shx = 128 + 127 * (hash01(np.arange(n), 5) * 2 - 1) * heavy
    shy = 128 + 127 * (hash01(np.arange(n), 7) * 2 - 1) * heavy
    flash = 0.16 * heavy * (1 - 0.5 * fin * 0)

    # ---- ribbon gain / flicker ----
    answer = np.clip(1 - riff * 2.0, 0, 1) * np.clip(synth * 2, 0, 1)
    rib = np.clip(1.0 * intro + (0.12 + 0.55 * answer) * verse + 0.5 * pre + 1.0 * chorus + 0.9 * brk + 0.8 * endm, 0, 1)
    flick = 1 - 0.75 * ((stut > 0.3) & (hash01(np.arange(n), 11) > 0.45)) * (intro + endm * 0.8)
    flick = np.clip(flick, 0, 1)

    # ---- end: stutter then black on the last beat; flash one frame ----
    last = beats[-1]
    bright = np.ones(n)
    tail0 = last - 1.5
    inT = (t >= tail0) & (t < last)
    bright[inT] = np.where(hash01(np.arange(n)[inT] // 2, 13) < (0.2 + 0.7 * (t[inT] - tail0) / 1.5), 0.0, 1.0)
    bright[t >= last] = 0.0
    lf = int(round(last * fps))
    flash[max(lf - 1, 0):lf] = 0.9

    # ---- displayed-frame source (dropped / repeated frames) ----
    src = np.arange(n)
    if not a.no_prechorus_drop:
        m = pre > 0.5; src = np.where(m, (np.arange(n) // 3) * 3, src)       # pre-chorus: 10 fps
    for i in range(1, n):
        if glitch[i] > 0.5 and hash01(i, 17) < 0.40: src[i] = src[i - 1]       # frame repeats on stutters
        elif brk[i] > 0.5 and i % 2: src[i] = src[i - 1]                      # break: 15 fps crunch

    q = lambda x: np.clip(np.round(np.asarray(x) * 255), 0, 255).astype(int).tolist()
    ch = dict(kick=q(kick), snare=q(snare), glitch=q(glitch), sub=q(sub), riff=q(riff), synth=q(synth), echo=q(echo),
              pitch=q(pitch), intro=q(intro), verse=q(verse), pre=q(pre), chorus=q(chorus), brk=q(brk), fin=q(fin),
              bright=q(bright), wid=q(wid), crush=q(crush), ca=q(ca), flash=q(flash), pump=q(pump), rib=q(rib), flick=q(flick),
              shx=np.round(shx).astype(int).tolist(), shy=np.round(shy).astype(int).tolist())
    out = dict(fps=fps, n=n, ch=ch, src=src.astype(int).tolist(), tm=np.round(tm).astype(int).tolist(), sdx=np.round(sdx * 1000).astype(int).tolist(),
               bar=np.round(barpos * 100).astype(int).tolist(), b16=b16.tolist(), lastBeat=float(last))
    Path(a.out).write_text("window.AUDIO=" + json.dumps(out, separators=(",", ":")) + ";\n")
    print("stereo: guitar pan p5/95", np.percentile(gpan, [5, 95]).round(2), "shard drift range",
          round(float(sdx.min()), 2), round(float(sdx.max()), 2), "mix width p50", round(float(np.median(wid)), 2))
    print("wrote", a.out, "frames", n, "last beat", round(last, 2), "tempo", round(d["tempo"], 1))

if __name__ == "__main__":
    main()
