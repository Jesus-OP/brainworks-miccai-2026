"""
margin_stratification.py
========================

The experiment that validates Hypothesis 2 (H2): demonstrating that boundary 
detectability collapses as margin diffuseness increases.

Per patient:
  - sigma : margin sharpness = mean FLAIR gradient magnitude on the tumour 
            shell (sharp -> high sigma; diffuse -> low sigma).
  - AUC_wavelet, AUC_sobel : boundary detectability in the neighbourhood.

Cohort level:
  - Spearman correlation between sigma and AUC. H2 predicts a POSITIVE correlation:
    sharper margins -> more detectable boundary; infiltrative -> less detectable.

This reframes patient-to-patient variance (which otherwise appears as noise) into 
the central finding. 
"""
import numpy as np
import pywt
from scipy.ndimage import (binary_dilation, binary_erosion, sobel,
                           binary_fill_holes, label as cc_label,
                           uniform_filter)
from sklearn.metrics import roc_auc_score


def make_shell(mask):
    st = np.ones((3,3,3), bool)
    return (binary_dilation(mask,st).astype(np.int8) -
            binary_erosion(mask,st).astype(np.int8)) > 0

def brain_mask(flair):
    thr = np.quantile(flair[flair>0], 0.25) if (flair>0).any() else 0
    bm = binary_fill_holes(flair > thr)
    lab, n = cc_label(bm)
    if n > 1:
        s = np.bincount(lab.ravel()); s[0]=0; bm = lab==s.argmax()
    return bm

def block_any_half(m):
    d,h,w = m.shape
    return m[:d//2*2,:h//2*2,:w//2*2].reshape(d//2,2,h//2,2,w//2,2).max((1,3,5))

def block_mean_half(x):
    d,h,w = x.shape
    return x[:d//2*2,:h//2*2,:w//2*2].reshape(d//2,2,h//2,2,w//2,2).mean((1,3,5))

def auc_region(score, shell, region):
    y = shell[region].astype(np.int8); s = score[region]
    if y.sum() in (0,len(y)): return np.nan
    return float(roc_auc_score(y,s))

def sobel_mag(vol):
    g = [sobel(vol, axis=a, mode='reflect') for a in range(3)]
    return np.sqrt(g[0]**2+g[1]**2+g[2]**2).astype(np.float32)

def wavelet_energy(vol):
    c = pywt.dwtn(vol, 'db1')
    return sum(c[k]**2 for k in c if k!='aaa').astype(np.float32)

def local_contrast_map(vol, win=5):
    """Local contrast map: standard deviation within a sliding window 
    (win x win x win) centered at each voxel, calculated using fast uniform 
    filters. std_local = sqrt(E[x^2] - E[x]^2)."""
    vol = vol.astype(np.float32)
    mean_x  = uniform_filter(vol,    size=win, mode='reflect')
    mean_x2 = uniform_filter(vol**2, size=win, mode='reflect')
    var = np.clip(mean_x2 - mean_x**2, 0, None)
    return np.sqrt(var).astype(np.float32)


def margin_sharpness(flair, shell, win=5):
    """sigma = mean FLAIR gradient on the shell, normalised voxel-wise by the 
    LOCAL contrast (std within a win x win x win window around each voxel) 
    rather than global image deviation."""
    g = sobel_mag(flair)
    local_c = local_contrast_map(flair, win=win)
    eps = 1e-6
    ratio = g[shell] / (local_c[shell] + eps)
    return float(ratio.mean())


def analyze(flair, seg, neigh_iter=8):
    sl = tuple(slice(0, s - s%2) for s in flair.shape)
    flair = flair[sl].astype(np.float32); seg = seg[sl].astype(np.int32)
    mask = seg > 0
    if mask.sum() < 20: return None
    shell = make_shell(mask)
    brain = brain_mask(flair)
    neigh = binary_dilation(mask, np.ones((3,3,3),bool), iterations=neigh_iter) & brain

    sigma = margin_sharpness(flair, shell)

    Ew = wavelet_energy(flair); Es = block_mean_half(sobel_mag(flair))
    shell_h, neigh_h = block_any_half(shell), block_any_half(neigh)
    shp = tuple(min(Ew.shape[i], Es.shape[i], shell_h.shape[i]) for i in range(3))
    cut = tuple(slice(0,s) for s in shp)
    return {
        'sigma': sigma,
        'AUC_wavelet': auc_region(Ew[cut], shell_h[cut], neigh_h[cut]),
        'AUC_sobel':   auc_region(Es[cut], shell_h[cut], neigh_h[cut]),
        'tumor_vox':   int(mask.sum()),
    }


def cohort_report(rows):
    from scipy.stats import spearmanr
    import numpy as np
    sig = np.array([r['sigma'] for r in rows])
    for key in ('AUC_wavelet', 'AUC_sobel'):
        auc = np.array([r[key] for r in rows])
        ok = ~np.isnan(auc)
        if ok.sum() >= 4:
            rho, p = spearmanr(sig[ok], auc[ok])
            print(f"  {key}: Spearman rho(sigma, AUC) = {rho:+.3f}  (p={p:.3g}, n={ok.sum()})")
            
    # Stratify into Sharp vs Diffuse using median sigma
    med = np.median(sig)
    sharp = [r for r in rows if r['sigma'] >= med]
    diff  = [r for r in rows if r['sigma'] <  med]
    for grp, name in ((sharp, 'SHARP (high sigma)'), (diff, 'DIFFUSE (low sigma)')):
        aw = np.nanmean([r['AUC_wavelet'] for r in grp])
        asb= np.nanmean([r['AUC_sobel'] for r in grp])
        print(f"  {name:<22} AUC_wavelet={aw:.3f}  AUC_sobel={asb:.3f}  (n={len(grp)})")


# ---- Validation: Synthetic cohort with CONTINUOUS diffuseness --------------
def phantom(size=72, transition=2, seed=0):
    rng = np.random.default_rng(seed)
    zz,yy,xx = np.mgrid[0:size,0:size,0:size].astype(np.float32)
    c = np.array([size/2]*3) + rng.uniform(-4,4,3)
    r = np.sqrt(((zz-c[0])**2+(yy-c[1])**2+(xx-c[2])**2))
    R = size*0.3; brain = r < size*0.46
    vol = np.where(brain, 0.5+0.1*rng.standard_normal((size,)*3), 0).astype(np.float32)
    ramp = np.clip(1.0-(r-(R-transition))/(2*transition+1e-3),0,1)  # transition width
    vol += 0.6*ramp*brain
    return vol.astype(np.float32), (r<R).astype(np.int32)


if __name__ == "__main__":
    # Cohort: diffuseness ranging from very sharp (transition=1) to very diffuse (=12)
    rows = []
    for i, t in enumerate([1,1,2,3,4,6,8,10,12,12]):
        vol, seg = phantom(72, transition=t, seed=i)
        r = analyze(vol, seg)
        if r: rows.append(r)
        
    print("Validation on synthetic cohort (increasing diffuseness):\n")
    for r in rows:
        print(f"  sigma={r['sigma']:.3f}  AUC_wavelet={r['AUC_wavelet']:.3f}  "
              f"AUC_sobel={r['AUC_sobel']:.3f}")
              
    print("\nH2 Summary:")
    cohort_report(rows)
