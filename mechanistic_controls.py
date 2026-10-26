"""
mechanistic_controls.py
=======================

HARDENED Mechanistic Experiment: Do wavelet detail subbands SPECIFICALLY encode
the tumour boundary, or just "any edge" (which any high-pass filter would trivially do)?

Methodology refinements over the original script:
  - [CRITICAL] Correct loading of matched FLAIR and SEG volumes.
  - Evaluation WITHOUT interpolation: detail energy exists at half-resolution;
    the boundary shell is downsampled to this grid (block-any) instead of 'zooming'
    the coefficients (which blurs and falsifies the metric).
  - CONTROL 1 (Sobel): same AUC using a generic gradient. If it matches the
    wavelet, the signal is merely a generic "edge detector", not something specific.
  - CONTROL 2 (null-shell): shell displaced to another brain region -> chance level
    floor. Measures SPECIFICITY for the tumour, not edges in general.
  - Evaluation region = tumour neighbourhood (not the whole head): compares
    "tumour border vs. contiguous brain tissue", excluding skull/background.
  - Subband breakdown (the 7 detail bands evaluated separately).
  - Removed arbitrary labels like "CONFIRMED if AUC>=0.80".
"""

import os, glob, argparse
import numpy as np
import pywt
from scipy.ndimage import (binary_dilation, binary_erosion, sobel,
                           binary_fill_holes, label as cc_label)
from sklearn.metrics import roc_auc_score

DETAIL_KEYS = ['aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']  # 7 detail bands (excluding 'aaa')


# --------------------------------------------------------------------------
def crop_even(*arrays):
    """Crops to even dimensions so coefficients are exactly half-size (no padding)."""
    shp = arrays[0].shape
    sl = tuple(slice(0, s - (s % 2)) for s in shp)
    return [a[sl] for a in arrays]


def block_any_half(mask):
    """Downsamples a binary mask by 2x: voxel=1 if ANY 2x2x2 sub-voxel is 1."""
    d, h, w = mask.shape
    m = mask[:d//2*2, :h//2*2, :w//2*2].reshape(d//2, 2, h//2, 2, w//2, 2)
    return m.max(axis=(1, 3, 5))


def block_mean_half(x):
    """Downsamples a continuous map by 2x using 2x2x2 block mean."""
    d, h, w = x.shape
    m = x[:d//2*2, :h//2*2, :w//2*2].reshape(d//2, 2, h//2, 2, w//2, 2)
    return m.mean(axis=(1, 3, 5))


def make_shell(mask):
    st = np.ones((3, 3, 3), bool)
    return ((binary_dilation(mask, st).astype(np.int8) -
             binary_erosion(mask, st).astype(np.int8)) > 0)


def brain_mask_from_flair(flair):
    """Approximate brain mask: quantile thresholding + hole filling + largest connected component."""
    thr = np.quantile(flair[flair > 0], 0.25) if (flair > 0).any() else 0
    bm = flair > thr
    bm = binary_fill_holes(bm)
    lab, n = cc_label(bm)
    if n > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        bm = lab == sizes.argmax()
    return bm


def wavelet_detail_energy(flair):
    """Energy of the 7 detail subbands (at HALF resolution, no zoom/interpolation)."""
    coeffs = pywt.dwtn(flair, wavelet='db1')   # db1 == Haar
    E = np.zeros_like(coeffs['aaa'], dtype=np.float32)
    per_band = {}
    for k in DETAIL_KEYS:
        per_band[k] = coeffs[k].astype(np.float32) ** 2
        E += per_band[k]
    return E, per_band  # shape ~ flair/2


def sobel_energy(flair):
    """3D Sobel gradient magnitude (control: generic edge detector)."""
    g = [sobel(flair, axis=a, mode='reflect') for a in range(3)]
    return np.sqrt(g[0]**2 + g[1]**2 + g[2]**2).astype(np.float32)


def auc_in_region(score_half, shell_half, region_half):
    y = shell_half[region_half].astype(np.int8)
    s = score_half[region_half]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, s)


def null_shell(mask, brain, rng, max_try=20):
    """Shell of the SAME size displaced to another brain region (chance-level floor)."""
    st = np.ones((3, 3, 3), bool)
    base = mask
    for _ in range(max_try):
        off = rng.integers(-40, 41, size=3)
        shifted = np.zeros_like(base)
        src = tuple(slice(max(0, -o), base.shape[i] - max(0, o)) for i, o in enumerate(off))
        dst = tuple(slice(max(0, o), base.shape[i] - max(0, -o)) for i, o in enumerate(off))
        shifted[dst] = base[src]
        if shifted.sum() == 0:
            continue
        # Ensure it falls within the brain and does not overlap the original tumour
        if (shifted & brain).sum() < 0.7 * shifted.sum():
            continue
        if (shifted & binary_dilation(base, st, iterations=3)).sum() > 0.1 * shifted.sum():
            continue
        return make_shell(shifted)
    return None


# --------------------------------------------------------------------------
def analyze_case(flair, seg, rng, neigh_iter=8):
    flair, seg = crop_even(flair.astype(np.float32), seg.astype(np.float32))
    mask = seg > 0
    if mask.sum() < 20:
        return None

    shell = make_shell(mask)
    brain = brain_mask_from_flair(flair)
    # Region = tumour neighbourhood within the brain (border vs contiguous tissue)
    neigh = binary_dilation(mask, np.ones((3, 3, 3), bool), iterations=neigh_iter) & brain

    Ew, per_band = wavelet_detail_energy(flair)        # half res
    Es = sobel_energy(flair)                            # full res

    shell_h = block_any_half(shell)
    neigh_h = block_any_half(neigh)
    brain_h = block_any_half(brain)
    Es_h = block_mean_half(Es)
    
    # Align shapes (dwtn may differ by 1 voxel)
    shp = tuple(min(a.shape[i] for a in (Ew, shell_h, neigh_h, brain_h, Es_h)) for i in range(3))
    cut = tuple(slice(0, s) for s in shp)
    Ew, Es_h, shell_h, neigh_h, brain_h = [a[cut] for a in (Ew, Es_h, shell_h, neigh_h, brain_h)]

    out = {}
    # AUC in tumour NEIGHBOURHOOD (primary metric, no skull)
    out['AUC_wavelet_neigh'] = auc_in_region(Ew, shell_h, neigh_h)
    out['AUC_sobel_neigh']   = auc_in_region(Es_h, shell_h, neigh_h)
    
    # AUC in WHOLE brain (secondary, includes confounding edges)
    out['AUC_wavelet_brain'] = auc_in_region(Ew, shell_h, brain_h)
    out['AUC_sobel_brain']   = auc_in_region(Es_h, shell_h, brain_h)

    # NULL Shell (specificity): Wavelet AUC against a displaced border
    ns = null_shell(mask, brain, rng)
    if ns is not None:
        ns_h = block_any_half(ns)[cut]
        # Evaluate in the neighbourhood of THAT null shell
        ns_neigh = block_any_half(binary_dilation(ns, np.ones((3,3,3),bool),
                                                  iterations=neigh_iter) & brain)[cut]
        out['AUC_wavelet_null'] = auc_in_region(Ew, ns_h, ns_neigh)
    else:
        out['AUC_wavelet_null'] = np.nan

    # Ratio R (mean energy on shell / mean energy in neighbourhood outside shell)
    inside = Ew[shell_h & neigh_h]
    outside = Ew[(~shell_h) & neigh_h]
    out['R_wavelet'] = float(inside.mean() / outside.mean()) if outside.mean() > 0 else np.nan

    # Breakdown by subband
    for k in DETAIL_KEYS:
        pb = per_band[k][cut]
        out[f'AUC_{k}'] = auc_in_region(pb, shell_h, neigh_h)

    return out


# --------------------------------------------------------------------------
def synth_case(size=96, seed=0, with_skull=True):
    """Phantom: textured 'brain' + tumour with sharp border + bright 'skull'."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
    c = np.array([size/2]*3)
    r = np.sqrt(((zz-c[0])**2 + (yy-c[1])**2 + (xx-c[2])**2))
    brain = r < size*0.42
    flair = np.where(brain, 0.5 + 0.15*rng.standard_normal((size,)*3), 0).astype(np.float32)
    
    if with_skull:
        skull = (r > size*0.42) & (r < size*0.46)
        flair[skull] = 1.5   # Very bright skull (must NOT contaminate if masked properly)
        
    # Tumour with sharp intensity border, off-center
    tc = c + rng.uniform(-10, 10, 3)
    rt = np.sqrt(((zz-tc[0])**2 + (yy-tc[1])**2 + (xx-tc[2])**2))
    tumor = rt < size*0.14
    flair[tumor] += 0.6
    seg = tumor.astype(np.int32)
    
    return flair, seg


def aggregate(rows):
    import statistics as st
    keys = [k for k in rows[0] if rows[0][k] is not None]
    print(f"\n{'Metric':<22}{'mean ± std':<20}")
    print("-" * 42)
    table = {}
    for k in keys:
        vals = [r[k] for r in rows if r is not None and not np.isnan(r[k])]
        if not vals:
            continue
        m = float(np.mean(vals)); s = float(np.std(vals))
        table[k] = (m, s)
        print(f"{k:<22}{m:.3f} ± {s:.3f}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    if args.data_dir:
        import nibabel as nib
        cases = sorted(d for d in glob.glob(os.path.join(args.data_dir, "*"))
                       if os.path.isdir(d))[:args.n]
        for c in cases:
            f = glob.glob(os.path.join(c, "*t2f.nii.gz"))   # FLAIR (correct!)
            s = glob.glob(os.path.join(c, "*seg.nii.gz"))   # SEG (correct!)
            if not f or not s:
                continue
            flair = nib.load(f[0]).get_fdata().astype(np.float32)
            seg = nib.load(s[0]).get_fdata().astype(np.int32)
            res = analyze_case(flair, seg, rng)
            if res:
                rows.append(res)
        print(f"Analysed {len(rows)} real cases.")
    else:
        print(f"No --data_dir provided: running {args.n} synthetic phantoms (with bright skull).")
        for i in range(args.n):
            flair, seg = synth_case(seed=i)
            res = analyze_case(flair, seg, rng)
            if res:
                rows.append(res)

    table = aggregate(rows)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv("mechanistic_controls_percase.csv", index=False)
        pd.DataFrame([{"metric": k, "mean": v[0], "std": v[1]}
                      for k, v in table.items()]).to_csv(
                          "mechanistic_controls_summary.csv", index=False)
        print("\nCSV -> mechanistic_controls_percase.csv | _summary.csv")
    except ImportError:
        pass