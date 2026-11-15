#!/usr/bin/env python3
"""Split a track into stems (Demucs) and export per-frame features as JSON.

Usage: uv run tools/analyze.py audio/track.wav [--fps 30] [--stems audio/stems/suno/<track>]
Output: analysis/<track>.json  -> {fps, duration, beats[], stems:{name:{rms[], onsets[], bands{}}}}
All curves are sampled at `fps` and normalised 0..1 so a composition can index by frame.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, librosa

ROOT = Path(__file__).resolve().parent.parent
BANDS = {"sub": (20, 80), "low": (80, 250), "mid": (250, 2000), "high": (2000, 12000)}

def norm(x):
    x = np.asarray(x, float); lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

def resample(x, n):
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)

def features(path, fps, hop=512):
    y, sr = librosa.load(path, sr=None, mono=True)
    n = int(len(y) / sr * fps)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    out = {"rms": resample(norm(librosa.feature.rms(S=S)[0]), n).round(3).tolist(), "bands": {}}
    for k, (a, b) in BANDS.items():
        m = (freqs >= a) & (freqs < b)
        out["bands"][k] = resample(norm(S[m].mean(0)), n).round(3).tolist()
    on = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="time", backtrack=False)
    out["onsets"] = [round(float(t), 3) for t in on]
    # dominant pitch (for the synth ribbon height), 0..1 over the stem's own range
    f0 = librosa.yin(y, fmin=65, fmax=2000, sr=sr, hop_length=hop)
    out["pitch"] = resample(norm(np.log2(f0)), n).round(3).tolist()
    return out, len(y) / sr

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("track"); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", default="htdemucs_6s", help="htdemucs (4 stems) or htdemucs_6s (adds guitar, piano)")
    ap.add_argument("--stems", help="use an existing stem folder (e.g. Suno exports) instead of running Demucs")
    a = ap.parse_args(); track = Path(a.track).resolve(); name = track.stem
    stem_dir = Path(a.stems).resolve() if a.stems else ROOT / "audio" / "stems" / a.model / name
    if not a.stems and not stem_dir.exists():
        subprocess.run([sys.executable, "-m", "demucs", "-n", a.model, "-o", str(ROOT / "audio/stems"), str(track)], check=True)
    res = {"fps": a.fps, "stems": {}}
    mix, dur = features(track, a.fps); res["duration"] = dur; res["stems"]["mix"] = mix
    for wav in sorted(stem_dir.glob("*.wav")):
        res["stems"][wav.stem], _ = features(wav, a.fps)
    y, sr = librosa.load(stem_dir / "drums.wav", sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    res["tempo"] = float(np.atleast_1d(tempo)[0]); res["beats"] = [round(float(b), 3) for b in beats]
    out = ROOT / "analysis" / f"{name}.json"; out.write_text(json.dumps(res)); print("wrote", out)

if __name__ == "__main__":
    main()
