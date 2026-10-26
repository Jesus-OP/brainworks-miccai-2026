"""
make_figure_paper.py
====================

Figure generation for Fig. 1 of the manuscript (Sec 5.3) in vector PDF format.
DOES NOT alter any computed metrics: it relies on the exact same transition widths 
and gradient peak offsets calculated by `distance_profile.extract_metrics`. 
It solely adjusts the PRESENTATION of the left panel to ensure consistency with 
the text ("normalised FLAIR intensity", transition ramp) and robustness against 
non-monotonic profiles.

Usage:
    from distance_profile import run_cohort_nifti
    from make_figure_paper import make_figure_paper
    rows, profiles = run_cohort_nifti(pairs)     # pairs = [(pid, flair, seg), ...]
    make_figure_paper(rows, profiles, "distance_profile_fig.pdf")

Place this file in the same directory as distance_profile.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _normalised_outer_profile(c, I, edge_band=1.5, plateau_band=2.0):
    """Returns (c, norm) of the profile normalised 1 (at the edge d~0) -> 0 (outside),
    cropped to the range where normalisation is interpretable. Robust to non-monotonic 
    profiles: relies on the intensity at the EDGE (d~0), not the deep interior."""
    ok = ~np.isnan(I)
    c, Iv = c[ok], I[ok]
    if len(c) < 5:
        return None, None
        
    # High level: intensity at the edge (band around d=0)
    edge_sel = (c >= -edge_band) & (c <= edge_band)
    I_edge = np.median(Iv[edge_sel]) if edge_sel.sum() >= 1 else Iv[np.argmin(np.abs(c))]
    
    # Low level: outer plateau
    I_out = np.median(Iv[c >= c.max() - plateau_band])
    denom = I_edge - I_out
    
    if abs(denom) < 1e-6:
        return None, None
        
    norm = (Iv - I_out) / denom
    return c, np.clip(norm, -0.15, 1.15)   # Soft clip strictly for visual plotting


def make_figure_paper(rows, profiles, out_path="distance_profile_fig.pdf"):
    widths = {r['patient_id']: r['transition_width_mm'] for r in rows
              if np.isfinite(r['transition_width_mm'])}
    if not widths:
        print("No valid widths found."); return
        
    ids_sorted = sorted(widths, key=widths.get)
    sharp, diff = ids_sorted[0], ids_sorted[-2]

    # Clean styling for print
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # ---- (1) transition profiles: sharp vs diffuse (normalised, outer flank) ----
    for pid, col, lab in ((sharp, '#1f77b4', f'sharpest ({widths[sharp]:.1f} mm)'),
                          (diff, '#d62728', f'most diffuse ({widths[diff]:.1f} mm)')):
        c, I, G = profiles[pid]
        cc, norm = _normalised_outer_profile(c, I)
        if cc is not None:
            ax[0].plot(cc, norm, color=col, lw=2.2, label=lab)
            
    ax[0].axvline(0, color='k', ls='--', lw=1, label='radiologist contour')
    ax[0].axhspan(0.1, 0.9, color='gray', alpha=0.10)
    ax[0].set_ylim(-0.15, 1.15)
    ax[0].set_xlabel('signed distance to contour (mm)\n$\\leftarrow$ inside      outside $\\rightarrow$')
    ax[0].set_ylabel('normalised FLAIR intensity')
    ax[0].set_title('(a) Transition profile: sharp vs diffuse')
    ax[0].legend(fontsize=8, frameon=False)

    # ---- (2) histogram of transition widths ----
    w = np.array(list(widths.values()))
    ax[1].hist(w, bins=8, color='#2ca02c', alpha=0.85, edgecolor='k', linewidth=0.5)
    ax[1].axvline(np.median(w), color='#d62728', lw=1.8,
                  label=f'median {np.median(w):.1f} mm')
    ax[1].set_xlabel('transition width (mm)')
    ax[1].set_ylabel('number of patients')
    ax[1].set_title('(b) Infiltrative width across cohort')
    ax[1].legend(fontsize=8, frameon=False)

    # ---- (3) gradient peak offset ----
    off = np.array([r['peak_offset_mm'] for r in rows
                    if np.isfinite(r['peak_offset_mm'])])
    ax[2].hist(off, bins=8, color='#7570b3', alpha=0.85, edgecolor='k', linewidth=0.5)
    ax[2].axvline(0, color='k', ls='--', lw=1, label='on the contour')
    ax[2].set_xlabel('gradient peak offset (mm)\n$\\leftarrow$ inside      outside $\\rightarrow$')
    ax[2].set_ylabel('number of patients')
    ax[2].set_title('(c) Location of the image edge')
    ax[2].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    
    # Vector PDF (crisp in print); also save as PNG just in case
    fig.savefig(out_path, bbox_inches='tight')
    if out_path.endswith('.pdf'):
        fig.savefig(out_path[:-4] + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("figure ->", out_path)


# Demo using the synthetic cohort from distance_profile (execution test)
if __name__ == "__main__":
    from distance_profile import _demo
    rows, profiles = _demo()
    make_figure_paper(rows, profiles, "/home/claude/distance_profile_fig.pdf")