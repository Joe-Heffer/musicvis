"""Per-frame stereo image of each stem: pan (L/R energy balance) and width (side/mid ratio).

    pan   = (eR - eL) / (eR + eL)    -1 hard left .. 0 centre .. +1 hard right
    width = eS / (eM + eS)           0 mono .. ~0.5 decorrelated .. 1 out of phase
with e* the per-frame RMS of L, R, M = (L+R)/2 and S = (L-R)/2. Rests hold the last value,
so silence never jitters the image, and both curves are smoothed. Values are raw, so each
builder does its own scaling. Results are cached in an .npz (analysis/*.npz is gitignored).
"""
from pathlib import Path
import numpy as np, librosa
from scipy.ndimage import gaussian_filter1d

SR = 22050


def _rms(x, hop):
    return librosa.feature.rms(y=x, frame_length=2 * hop, hop_length=hop)[0]


def _image(wav, fps, gate=0.02, smooth_s=0.25):
    y, _ = librosa.load(wav, sr=SR, mono=False)
    if y.ndim == 1: y = np.vstack([y, y])
    hop = SR // fps
    L, R = y[0], y[1]
    eL, eR, eM, eS = (_rms(v, hop) for v in (L, R, (L + R) / 2, (L - R) / 2))
    pan = (eR - eL) / (eR + eL + 1e-9)
    width = eS / (eM + eS + 1e-9)
    quiet = eM + eS < gate * np.percentile(eM + eS, 99)
    for i in range(1, len(pan)):
        if quiet[i]: pan[i], width[i] = pan[i - 1], width[i - 1]
    sig = smooth_s * fps
    return gaussian_filter1d(pan, sig, mode="nearest"), gaussian_filter1d(width, sig, mode="nearest")


def stereo(wavs, fps, n, cache):
    """{name: (pan, width)} as n-frame float arrays, for wavs = {name: path}. Cached per name."""
    cache = Path(cache)
    z = dict(np.load(cache)) if cache.exists() else {}
    missing = [k for k in wavs if f"{k}_pan" not in z]
    for k in missing:
        z[f"{k}_pan"], z[f"{k}_width"] = _image(wavs[k], fps)
    if missing: np.savez(cache, **z)
    fit = lambda x: np.interp(np.arange(n), np.arange(len(x)), x)
    return {k: (fit(z[f"{k}_pan"]), fit(z[f"{k}_width"])) for k in wavs}


def unit_pan(pan, pct=98):
    """Rescale pan so the stem's own widest swing (pct of |pan|) reaches +-1."""
    return np.clip(pan / (np.percentile(np.abs(pan), pct) + 1e-6), -1, 1)


def unit_width(width, lo=5, hi=95):
    """Rescale width to 0..1 over the stem's own 5-95 percentile range."""
    a, b = np.percentile(width, [lo, hi])
    return np.clip((width - a) / (b - a + 1e-9), 0, 1)
