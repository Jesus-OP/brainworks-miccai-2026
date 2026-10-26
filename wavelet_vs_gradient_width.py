"""
wavelet_vs_gradient_width.py
============================

HONEST Comparison: Transition width measured via GRADIENT vs. WAVELET.

- Gradient: 10-90% width of the normalised intensity profile (the validated metric).
- Wavelet : FWHM of the wavelet detail energy profile (level 1) vs distance.
            (This is the wavelet edge response, a direct analogue to the gradient).

Evaluated on: (1) tracking the TRUE width, (2) contrast invariance.
"""
import numpy as np
import pywt
from scipy.ndimage import (distance_transform_edt, sobel, gaussian_filter, zoom)
from scipy.stats import spearmanr


def signed_distance(mask, spacing=(1,1,1)):
    mask = np.asarray(mask) > 0
    return (distance_transform_edt(~mask, sampling=spacing)
            - distance_transform_edt(mask, sampling=spacing))


def grad_mag(vol, spacing=(1,1,1)):
    g = [sobel(vol, axis=a, mode='reflect') / spacing[a] for a in range(3)]
    return np.sqrt(sum(gi**2 for gi in g)).astype(np.float32)


def wavelet_edge_energy(vol):
    """Level 1 detail energy (wavelet edge response), at full resolution."""
    c = pywt.dwtn(vol, 'db2')
    E = np.zeros_like(c['aaa'], dtype=np.float32)
    for k in c:
        if k != 'aaa':
            E += np.asarray(c[k], np.float32)**2
    f = tuple(vol.shape[i]/E.shape[i] for i in range(3))
    E = zoom(E, f, order=1)
    sl = tuple(slice(0, vol.shape[i]) for i in range(3))
    out = np.zeros(vol.shape, np.float32); out[sl] = E[sl]
    return out


def _profile(signal, d, dmax=12, binmm=1.0):
    edges = np.arange(-dmax, dmax+binmm, binmm)
    ctr = 0.5*(edges[:-1]+edges[1:])
    P = np.full(len(ctr), np.nan)
    for k in range(len(ctr)):
        sel = (d >= edges[k]) & (d < edges[k+1])
        if sel.sum() >= 10:
            P[k] = np.median(signal[sel])
    return ctr, P


def width_gradient(flair, mask, spacing, dmax=12):
    d = signed_distance(mask, spacing)
    reg = np.abs(d) <= dmax
    ctr, I = _profile(flair[reg], d[reg], dmax)
    ok = ~np.isnan(I); c, Iv = ctr[ok], I[ok]
    if len(c) < 5: return np.nan
    
    I_in = np.median(Iv[c <= c.min()+2]); I_out = np.median(Iv[c >= c.max()-2])
    if abs(I_in-I_out) < 1e-6: return np.nan
    
    norm = (Iv-I_out)/(I_in-I_out)
    
    def cross(L):
        for k in range(len(norm)-1):
            a,b = norm[k],norm[k+1]
            if (a-L)*(b-L) <= 0 and a != b:
                return c[k]+(L-a)/(b-a)*(c[k+1]-c[k])
        return np.nan
        
    d90, d10 = cross(0.9), cross(0.1)
    return abs(d10-d90) if np.isfinite(d90) and np.isfinite(d10) else np.nan


def width_wavelet(flair, mask, spacing, dmax=12):
    d = signed_distance(mask, spacing)
    E = wavelet_edge_energy(gaussian_filter(flair, 0.5))
    reg = np.abs(d) <= dmax
    ctr, P = _profile(E[reg], d[reg], dmax)
    ok = ~np.isnan(P); c, Pv = ctr[ok], P[ok]
    if len(c) < 5 or Pv.max() <= 0: return np.nan
    
    Pv = gaussian_filter(Pv, 0.8)
    half = 0.5*Pv.max()
    above = c[Pv >= half]
    return float(above.max()-above.min()) if above.size >= 2 else np.nan


def phantom(size=80, transition=2, contrast=0.6, seed=0):
    rng = np.random.default_rng(seed)
    zz,yy,xx = np.mgrid[0:size,0:size,0:size].astype(np.float32)
    c = np.array([size/2]*3)+rng.uniform(-3,3,3)
    r = np.sqrt(((zz-c[0])**2+(yy-c[1])**2+(xx-c[2])**2))
    R=size*0.3; brain=r<size*0.46
    vol=np.where(brain,0.5+0.05*rng.standard_normal((size,)*3),0).astype(np.float32)
    t=transition; ramp=np.clip(1.0-(r-(R-t))/(2*t+1e-3),0,1)
    vol+=contrast*ramp*brain
    return vol.astype(np.float32),(r<R).astype(np.int32)


if __name__ == "__main__":
    print("(1) Does it track the TRUE width? (transition ±mm)\n")
    print(f"{'true':<8}{'GRADIENT (10-90)':<20}{'WAVELET (FWHM)'}")
    print("-" * 46)
    truth, wg, ww = [], [], []
    for t in [1,2,3,4,6,8]:
        g = np.mean([width_gradient(*phantom(80,t,0.6,s),(1,1,1)) for s in range(4)])
        w = np.mean([width_wavelet(*phantom(80,t,0.6,s),(1,1,1)) for s in range(4)])
        truth.append(t); wg.append(g); ww.append(w)
        print(f"{t:<8}{g:<20.2f}{w:.2f}")
        
    print(f"\nrho w/ width -> gradient {spearmanr(truth,wg)[0]:+.3f} | "
          f"wavelet {spearmanr(truth,ww)[0]:+.3f}")

    print("\n(2) Contrast invariance (CV% at fixed width, lower=better):")
    for name, fn in (('gradient', width_gradient), ('wavelet', width_wavelet)):
        cvs=[]
        for t in [2,4,6]:
            vals=[np.mean([fn(*phantom(80,t,cont,s),(1,1,1)) for s in range(3)])
                  for cont in (0.3,0.6,1.2)]
            cvs.append(100*np.nanstd(vals)/(abs(np.nanmean(vals))+1e-8))
        print(f"  {name:<10} Mean CV = {np.mean(cvs):.1f}%")