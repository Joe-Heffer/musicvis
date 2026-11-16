#!/usr/bin/env python3
"""one-fuzz: Suno stems + docs/one-fuzz/sections.json -> video-one-fuzz/audio-data.js

Usage: uv run tools/build_one_fuzz.py [--check]
  --check  also write renders/one-fuzz_check.wav: the mix + clicks on the beat grid
           (accented downbeats) + a bell at each section start, to verify by ear.

Stem roles (Suno export in audio/stems/suno/one-fuzz/, see docs/one-fuzz/SECTIONS.md):
  drums      -> kick (black) + snare/rim (teal) stamps, hats (pink) stamps, strip shreds on drill edits
  percussion -> pink stamps
  bass       -> fuzz riff: ink flood on the drops
  guitar     -> fretless runs: the pen line
  synth      -> acid: teal tint
  keyboard + other -> glassy arp: stencil band
  lead_vocals / brass / woodwinds: 15-40 dB down, ignored.
Timing comes from a tempo-following beat grid (the track drifts 140.3-142.2 BPM), never a fixed BPM.
Everything is decided here, so the shader is a pure function of the frame.
"""
import json, sys
from pathlib import Path
import numpy as np, librosa
from scipy.ndimage import median_filter, gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from build_data import hash01, smooth  # noqa: E402

TRACK = "one-fuzz"
ASP = 16 / 9
STEMS = ROOT / "audio" / "stems" / "suno" / TRACK
SR, HOP = 22050, 128          # 5.8 ms analysis hop for anything timing-critical
FR = SR / HOP


def load(name):
    return librosa.load(STEMS / f"{name}.wav", sr=SR)[0]


def beat_grid(drums, perc, bpm0=140.0, tight=400.0):
    """Ellis DP beat tracker that follows a smoothed local-tempo curve.
    Returns beat times (s) and the beat index of the first downbeat (0..3)."""
    od = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=HOP)
    op = librosa.onset.onset_strength(y=perc, sr=SR, hop_length=HOP)
    o = od / od.std() + 0.5 * op / op.std(); n = len(o)
    # local tempo: 1-bar autocorrelation lag in 4-bar windows, every half bar
    W = int(16 * 60 / bpm0 * FR); step = W // 8; cs, bp = [], []
    lo_, hi_ = int(4 * 60 / (bpm0 + 4) * FR), int(4 * 60 / (bpm0 - 4) * FR)
    for c in range(W // 2, n - W // 2, step):
        seg = o[c - W // 2:c + W // 2] - o[c - W // 2:c + W // 2].mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        k = lo_ + int(np.argmax(ac[lo_:hi_])); a, b, d = ac[k - 1], ac[k], ac[k + 1]
        cs.append(c); bp.append(4 * 60 * FR / (k + 0.5 * (a - d) / (a - 2 * b + d)))
    bp = gaussian_filter1d(median_filter(np.array(bp), 5, mode="nearest"), 2, mode="nearest")
    P = 60 * FR / np.interp(np.arange(n), cs, bp)
    cum = o.copy(); back = np.full(n, -1)
    for t in range(n):
        p = P[t]; lo, hi = int(t - 1.25 * p), int(t - 0.8 * p)
        if hi < 0: continue
        lo = max(lo, 0); tau = t - np.arange(lo, hi + 1)
        sc = cum[lo:hi + 1] - tight * np.log(tau / p) ** 2
        j = int(np.argmax(sc)); cum[t] = o[t] + sc[j]; back[t] = lo + j
    t = n - 1 - int(np.argmax(cum[::-1][:int(P[-1])]))
    beats = [t]
    while back[t] >= 0: t = back[t]; beats.append(t)
    beats = np.array(beats[::-1]) / FR
    # downbeat: the beat phase (mod 4) with the most kick energy
    Sx = np.abs(librosa.stft(drums, n_fft=1024, hop_length=HOP)); fq = librosa.fft_frequencies(sr=SR, n_fft=1024)
    kick = librosa.onset.onset_strength(S=librosa.amplitude_to_db(Sx[fq < 150]), sr=SR, hop_length=HOP)
    kb = np.array([kick[max(0, int(b * FR) - 2):int(b * FR) + 3].max() for b in beats])
    db = int(np.argmax([kb[m::4].mean() for m in range(4)]))
    # validation: per 8 bars, the shift that would best align the grid with the onsets
    drift = []
    for a in range(0, len(beats) - 32, 32):
        g = beats[a:a + 32]; sh = np.arange(-0.06, 0.0605, 0.002)
        sc = [o[np.round((g + s) * FR).astype(int).clip(0, n - 1)].mean() for s in sh]
        drift.append(round(float(sh[int(np.argmax(sc))] * 1000)))
    return beats, db, drift


def grid(drums, perc):
    cache = ROOT / "analysis" / f"{TRACK}.grid.json"
    if cache.exists():
        g = json.loads(cache.read_text())
    else:
        beats, db, drift = beat_grid(drums, perc)
        g = dict(beats=[round(float(b), 4) for b in beats], downbeat=db, drift_ms=drift)
        cache.write_text(json.dumps(g))
    return np.array(g["beats"]), g["downbeat"], g["drift_ms"]


def band_onsets(S, fq, lo, hi, delta, wait_s=0.05, floor=0.0):
    """Attack times (backtracked) + 0..1 strengths for one frequency band.
    Peaks below `floor` (fraction of the band's 99.5th pct flux) are dropped; the rest rescale to 0..1."""
    env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[(fq >= lo) & (fq < hi)]), sr=SR, hop_length=HOP)
    env = env / (np.percentile(env, 99.5) + 1e-9)
    pk = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="frames", backtrack=False,
                                    delta=delta, wait=int(wait_s * FR), pre_avg=int(0.1 * FR), post_avg=1,
                                    pre_max=int(0.03 * FR), post_max=1, normalize=False)
    v = np.clip(env[pk], 0, 1); keep = v >= floor
    bt = librosa.onset.onset_backtrack(pk[keep], env)
    return bt * HOP / SR, (v[keep] - floor) / (1 - floor + 1e-9)


def check_track(beats, db, secs, dst):
    import soundfile as sf
    y, sr = sf.read(ROOT / "audio" / f"{TRACK}.wav")
    y = y if y.ndim > 1 else y[:, None]
    out = y * 0.8
    def tone(f, dur, amp, decay):
        tt = np.arange(int(dur * sr)) / sr
        return amp * np.sin(2 * np.pi * f * tt) * np.exp(-tt / decay)
    for i, b in enumerate(beats):
        down = (i - db) % 4 == 0
        c = tone(1800 if down else 1200, 0.03, 0.22 if down else 0.12, 0.008)
        s = int(b * sr); e = min(len(out), s + len(c)); out[s:e] += c[:e - s, None]
    for s_ in secs:
        bell = tone(880, 0.8, 0.3, 0.25) + tone(1320, 0.8, 0.15, 0.18)
        s = int(s_["start"] * sr); e = min(len(out), s + len(bell)); out[s:e] += bell[:e - s, None]
    sf.write(dst, np.clip(out, -1, 1), sr)
    print("wrote", dst)


def main():
    d = json.loads((ROOT / "analysis" / f"{TRACK}.json").read_text())
    secs = json.loads((ROOT / "docs" / TRACK / "sections.json").read_text())["sections"]
    fps = d["fps"]; S = d["stems"]; dur = d["duration"]; n = int(round(dur * fps))
    t = np.arange(n) / fps
    fi = np.arange(n)

    def fit(x):
        x = np.asarray(x, float); return np.interp(np.arange(n), np.arange(len(x)), x)

    def sec_at(x):
        for s in secs:
            if s["start"] <= x < s["end"]: return s
        return secs[-1]

    def mask(pred, k=8):
        m = np.array([1.0 if pred(sec_at(x)) else 0.0 for x in t])
        return np.clip(smooth(m, k), 0, 1)

    drop = mask(lambda s: s["type"] == "drop"); fin = mask(lambda s: s.get("final", False))
    verse = mask(lambda s: s["type"] in ("verse", "intro"))
    brk = mask(lambda s: s["type"] == "breakdown")
    endm = np.array([1.0 if sec_at(x)["type"] == "end" else 0.0 for x in t])
    sil = np.array([1.0 if sec_at(x)["type"] == "silence" else 0.0 for x in t])
    stype = [sec_at(x)["type"] for x in t]

    # ---- beat grid (tempo-following) ----
    yd, yp = load("drums"), load("percussion")
    beats, db, drift = grid(yd, yp)
    if "--check" in sys.argv:
        check_track(beats, db, secs, ROOT / "renders" / f"{TRACK}_check.wav")
    beat_s = float(np.median(np.diff(beats)))
    off = 4 * int(np.ceil(db / 4)) - db                    # bpos multiples of 4 = downbeats, bpos >= 0 at t=0
    pre = beats[0] - beat_s * np.arange(1, 4)[::-1]
    post = beats[-1] + beat_s * np.arange(1, 4)
    bt = np.concatenate([pre, beats, post]); bidx = np.arange(len(bt)) - 3 + off
    bpos_of = lambda x: np.interp(x, bt, bidx)
    bpos = np.maximum(bpos_of(t), 0)
    bar = np.floor(bpos / 4).astype(int)
    b16 = np.floor(bpos * 4 + 0.02).astype(int)
    hitf = lambda x: int(np.floor(x * fps + 0.25))         # frame showing a hit at time x (<= 8 ms early)

    # ---- drum hits: attack times from the Suno drums + percussion stems ----
    Sd = np.abs(librosa.stft(yd, n_fft=1024, hop_length=HOP)); fq = librosa.fft_frequencies(sr=SR, n_fft=1024)
    Sp = np.abs(librosa.stft(yp, n_fft=1024, hop_length=HOP))
    hits = []
    # floors tuned to ~2-3 kicks, ~4 snare/rim and ~8 hats per bar
    for kind, (Sx, lo, hi, floor) in {2: (Sd, 0, 150, 0.6), 1: (Sd, 1000, 4000, 0.7), 0: (Sd, 6000, 11000, 0.5)}.items():
        ts, vs = band_onsets(Sx, fq, lo, hi, 0.3, floor=floor)
        hits += [(float(a), kind, 0.4 + 0.6 * float(v)) for a, v in zip(ts, vs)]
    ts, vs = band_onsets(Sp, fq, 300, 11000, 0.3, floor=0.6)
    hits += [(float(a), 0, 0.35 + 0.5 * float(v)) for a, v in zip(ts, vs)]
    hits.sort()
    # drop a pink hit that duplicates another pink hit within 40 ms (hat in both stems)
    hits = [h for i, h in enumerate(hits)
            if not (h[1] == 0 and any(g[1] == 0 and 0 < h[0] - g[0] < 0.04 for g in hits[max(0, i - 6):i]))]

    # ---- stamps: x = 16th within the bar, y = kind band + bar row ----
    mx, my = 0.07, 0.08
    st = []; bar_f = 4 * beat_s * fps
    for (ht, kind, s) in hits:
        f0 = hitf(ht)
        if f0 >= n: continue
        ty = stype[f0]
        if ty in ("end", "silence"): continue
        q16 = int(round(max(bpos_of(ht), 0) * 4)); br = q16 // 16; c16 = q16 % 16
        strength = min(1.0, s)
        if ty == "breakdown":                       # half-time: double cells, 8 per bar
            cols, rows = 8, 4
            col = c16 // 2; row = {0: 0, 1: 1, 2: 2 + br % 2}[kind]
            life = int(2 * (1.0 if kind == 2 else 0.5) * bar_f)
        else:
            cols, rows = 16, 9
            col = c16; row = {0: 0, 1: 3, 2: 6}[kind] + br % 3
            life = int((1.0 if kind == 2 else 0.5) * bar_f)
        if ty == "solo" and kind == 0: strength *= 0.6
        cw = (ASP - 2 * mx) / cols; rh = (1 - 2 * my) / rows
        span = 2 if (kind == 2 and strength > 0.75 and ty != "breakdown") else 1
        jx = (hash01(f0, 21 + kind) - 0.5) * 0.008; jy = (hash01(f0, 31 + kind) - 0.5) * 0.008
        x0 = mx + col * cw + cw * 0.07 + jx; y0 = my + row * rh + rh * 0.07 + jy
        w = cw * (span - 0.14); h = rh * 0.86
        cell = {2: 0.0175, 1: 0.0115, 0: 0.0085}[kind] * (1.5 if ty == "breakdown" else 1.0)
        cov = 0.35 + 0.65 * strength
        q = lambda v: int(round(v * 10000))
        st += [f0, life, q(x0), q(y0), q(w), q(h), kind, int(round(cov * 100)), q(cell), int(hash01(f0, kind) * 97)]

    # ---- shreds: drill-edit beats = beats with unusually dense drum flux ----
    oenv = librosa.onset.onset_strength(y=yd, sr=SR, hop_length=HOP)
    opk = librosa.util.peak_pick(oenv / oenv.max(), pre_max=2, post_max=2, pre_avg=8, post_avg=8, delta=0.04, wait=4) / FR
    med = np.median(oenv)
    score, cnt = [], []
    for i in range(len(beats) - 1):
        sl = slice(int(beats[i] * FR), int(beats[i + 1] * FR))
        score.append(oenv[sl].mean() / med); cnt.append(((opk >= beats[i]) & (opk < beats[i + 1])).sum())
    score = np.array(score); cnt = np.array(cnt); thr = np.percentile(score, 87)
    shred = np.zeros(n); sseed = np.zeros(n, int); sn = np.full(n, 8)
    gain_for = {"drop": 1.0, "verse": 0.65, "intro": 0.55, "solo": 0.6, "breakdown": 0.0}
    ev = 0
    for i in np.where((score > thr) | (cnt >= 5))[0]:
        f0 = hitf(beats[i]); f1 = hitf(beats[i + 1])
        if f0 >= n: continue
        g = gain_for.get(stype[f0], 0.0)
        if g <= 0: continue
        ev += 1
        g *= float(np.clip(0.55 + (score[i] - thr) * 0.3, 0.55, 1.0))
        strips = int(np.clip(8 + (score[i] - thr) * 14 + 6 * (cnt[i] >= 5), 8, 48))
        L = max(f1 - f0, 1)
        for k in range(L + 1):
            if f0 + k >= n: break
            x = k / L
            e = 1.0 if x < 0.55 else (1 - (x - 0.55) / 0.45) ** 2     # hold, then home by the next beat
            if k == 0 or e * g >= shred[f0 + k]:
                shred[f0 + k] = e * g; sseed[f0 + k] = ev; sn[f0 + k] = strips

    # ---- fuzz flood from the bass stem ----
    bass = fit(S["bass"]["rms"])
    env = np.zeros(n); e = 0.0
    for i in range(n):
        e = bass[i] if bass[i] > e else e * 0.88 + bass[i] * 0.12
        env[i] = e
    dropw = drop * (1 + 0.15 * fin)
    Sb = np.abs(librosa.stft(load("bass"), n_fft=2048, hop_length=HOP)); fqb = librosa.fft_frequencies(sr=SR, n_fft=2048)
    bon, _ = band_onsets(Sb, fqb, 30, 4000, 0.2, wait_s=0.08, floor=0.25)
    acc = np.zeros(n); L = int(beat_s * fps * 1.2)
    for ot in bon:
        f0 = hitf(ot)
        for k in range(L):
            if f0 + k < n: acc[f0 + k] = max(acc[f0 + k], (1 - k / L) ** 1.6)
    fb = np.clip((env - 0.42) * 2.2, 0, 1.1) * (0.25 + 0.75 * acc) * dropw
    fp = np.clip((env - 0.24) * 1.8, 0, 1.1) * (0.35 + 0.65 * acc) * dropw
    fseed = (bar * 7 + 3) % 9973                            # flood redraws on each downbeat

    # ---- acid: synth stem -> slow teal tint ----
    acid_src = fit(S["synth"]["rms"])
    acid = np.clip(smooth(acid_src, 20) * 0.9, 0, 1) * np.clip(verse + 0.35 * drop + 0.5 * brk, 0, 1)

    # ---- final chord: the hit, then the ink lifts with the decay ----
    end = np.zeros(n); lift = np.zeros(n)
    es = secs[[s["type"] for s in secs].index("end")]["start"]
    hit = hitf(es)
    if hit < n:
        e_ = endm > 0.5
        end[e_] = 1.0
        quiet = secs[[s["type"] for s in secs].index("silence")]["start"]
        hold = es + 0.6                                     # solid black while the chord blooms
        lift[e_] = np.clip((t[e_] - hold) / (quiet - hold), 0, 1) ** 1.4
    endm_any = np.clip(endm + sil, 0, 1)
    for a in (fb, fp, acid): a *= 1 - endm_any

    # ---- arp: keyboard + other -> stencil band, one tile per 16th, resets each bar ----
    syn = np.clip(smooth(np.maximum(fit(S["keyboard"]["rms"]), fit(S["other"]["rms"])), 6) * 1.6, 0, 1)
    ston = ((syn > 0.18) & (endm_any < 0.5)).astype(int)
    stp = np.where(brk > 0.5, (b16 // 2) % 16, b16 % 16)
    ypos = {"intro": 0.5, "verse": 0.24, "solo": 0.8, "drop": 0.78, "breakdown": 0.5}
    sty = np.array([ypos.get(s, 0.5) for s in stype])
    stt = np.full(n, 0.095)

    # ---- fretless pen line: guitar stem ----
    cache = ROOT / "analysis" / f"{TRACK}.suno.pitch.npz"
    if cache.exists():
        gf0 = np.load(cache)["guitar"]
    else:
        yg = load("guitar")
        gf0 = np.log2(librosa.yin(yg, fmin=60, fmax=1000, sr=SR, frame_length=2048, hop_length=SR // fps))
        np.savez(cache, guitar=gf0)
    gp = fit(gf0[:n] if len(gf0) >= n else np.pad(gf0, (0, n - len(gf0)), mode="edge"))
    gr = smooth(fit(S["guitar"]["rms"]), 4)
    act = np.clip((gr - 0.10) * 4, 0, 1)
    px = np.zeros(n); py = np.zeros(n); pstart = np.full(n, -1)
    x, yy, hd = ASP * 0.3, 0.5, 0.3; start = -1; prev_p = gp[0]
    for i in range(n):
        if fb[i] > 0.75 or endm_any[i] > 0.5: start = -1     # flood covers the trail
        a = act[i]
        if a > 0.05:
            if start < 0: start = i
            dp = np.clip(gp[i] - prev_p, -0.5, 0.5) * 12      # semitone change
            hd += dp * 0.10 * a + (hash01(i // 4, 41) - 0.5) * 0.22 * a
            cx, cy = ASP / 2 - x, 0.5 - yy
            edge = max(abs(x - ASP / 2) / (ASP / 2 - 0.1), abs(yy - 0.5) / 0.4)
            if edge > 0.8:
                want = np.arctan2(cy, cx); dh = (want - hd + np.pi) % (2 * np.pi) - np.pi
                hd += dh * min(1.0, (edge - 0.8) * 2.5)
            sp = (0.0035 + 0.006 * a)
            x += np.cos(hd) * sp; yy += np.sin(hd) * sp
            x = float(np.clip(x, 0.05, ASP - 0.05)); yy = float(np.clip(yy, 0.05, 0.95))
        prev_p = gp[i] if gr[i] > 0.08 else prev_p
        px[i], py[i], pstart[i] = x, yy, start
    pena = np.where(pstart >= 0, 1.0, 0.0)

    # ---- pops: page-wide responses so each kick / snare reads at a glance ----
    # kick plate: a coarse black halftone blotch slammed on the hit, gone in ~5 frames
    # snare band: a torn teal band across the page, gone in ~6 frames
    kp = np.zeros(n); kseed = np.zeros(n, int); spl = np.zeros(n); sy = np.zeros(n); sseed2 = np.zeros(n, int)
    for j, (ht, kind, s) in enumerate(hits):
        f0 = hitf(ht)
        if f0 >= n or stype[f0] in ("end", "silence"): continue
        if kind == 2 and s > 0.55:
            for k in range(7):
                if f0 + k < n and s * np.exp(-k / 1.6) >= kp[f0 + k]:
                    kp[f0 + k] = s * np.exp(-k / 1.6); kseed[f0 + k] = j % 997
        elif kind == 1 and s > 0.6 and stype[f0] != "breakdown":
            yc = 0.18 + 0.64 * hash01(j, 77)
            for k in range(8):
                if f0 + k < n and s * np.exp(-k / 2.2) >= spl[f0 + k]:
                    spl[f0 + k] = s * np.exp(-k / 2.2); sy[f0 + k] = yc; sseed2[f0 + k] = j % 997
    quiet_page = np.clip(1 - 0.5 * brk, 0, 1) * (1 - endm_any)      # breakdown: half-strength plates
    kp *= quiet_page; spl *= (1 - endm_any)

    # ---- misregistration: a new pull each bar, jolt on heavy kicks ----
    kick = np.zeros(n)
    for (ht, kind, s) in hits:
        f0 = hitf(ht)
        if kind == 2 and s > 0.7 and f0 < n:
            for k in range(6):
                if f0 + k < n: kick[f0 + k] = max(kick[f0 + k], (1 - k / 6) * s)
    amp = (0.22 + 0.4 * drop + 0.15 * fin) * (1 - endm_any)
    reg = {}
    for ink in range(3):
        sc = 0.55 if ink == 2 else 1.0
        for ax, salt in (("x", 50), ("y", 60)):
            base = (hash01(bar, salt + ink) * 2 - 1) * amp
            jolt = (hash01(fi // 2, salt + 10 + ink) * 2 - 1) * kick * 0.9
            reg[f"r{ax}{ink}"] = np.clip(base * sc + jolt * sc, -1, 1)

    q = lambda v: np.clip(np.round(np.asarray(v) * 255), 0, 255).astype(int).tolist()
    sgn = lambda v: np.clip(np.round(128 + 127 * np.asarray(v)), 0, 255).astype(int).tolist()
    ch = dict(shred=q(shred), sn=sn.tolist(), fb=q(np.clip(fb, 0, 1)), fp=q(np.clip(fp, 0, 1)), acid=q(acid),
              stp=stp.tolist(), ston=ston.tolist(), sty=q(sty), stt=np.round(stt * 2550).astype(int).tolist(),
              end=q(end), lift=q(lift), pena=q(pena))
    for k, v in reg.items(): ch[k] = sgn(v)
    ch.update(kp=q(kp), spl=q(spl), sy=q(sy))
    out = dict(fps=fps, n=n, bf=round(beat_s * fps, 3), ch=ch, st=st, sseed=sseed.tolist(), fseed=fseed.tolist(),
               kseed=kseed.tolist(), sseed2=sseed2.tolist(),
               bar=np.round(bpos / 4 * 100).astype(int).tolist(),
               px=np.round(px / ASP * 4096).astype(int).tolist(), py=np.round(py * 4096).astype(int).tolist(),
               ps=pstart.astype(int).tolist())
    dst = ROOT / "video-one-fuzz" / "audio-data.js"
    dst.write_text("window.AUDIO=" + json.dumps(out, separators=(",", ":")) + ";\n")
    kinds = np.bincount([h[1] for h in hits], minlength=3)
    print("wrote", dst, "frames", n, "beats", len(beats), "downbeat", db, "grid drift ms/8 bars", drift)
    print("hits pink/teal/black", kinds.tolist(), "stamps", len(st) // 10, "shreds", ev, "hit frame", hit,
          "pen frames", int((act > 0.05).sum()))


if __name__ == "__main__":
    main()
