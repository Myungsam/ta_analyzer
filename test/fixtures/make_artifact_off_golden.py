"""
Generate ``artifact_off_golden.npz``: reference outputs of the fit engine
*before* the coherent-artifact terms were added to ``fit_global_analysis``
/ ``fit_single_trace`` (test/test_artifact_fit.py, AC4).

Provenance
----------
Generated once from the UNMODIFIED v1.11.0 engine (commit ``e194431``,
checked out as a separate git worktree) with::

    python test/fixtures/make_artifact_off_golden.py <worktree>/ta_core.py

NEVER re-run this script after the engine has been edited: the npz is the
frozen "before" state that proves the artifact-Off path is unchanged.

Data: seeded synthetic only (``_art_synth(seed=0)``), never measurement data.
``_art_synth`` is also imported by test/test_artifact_fit.py.
"""
import importlib.util
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'artifact_off_golden.npz')

TAU_TRUE = (0.3, 50.0)
T0_TRUE = 0.05
FWHM_TRUE = 0.12


def _art_synth(tc, seed=0, noise=0.005, art_fwhm=FWHM_TRUE, art_scale=1.0):
    """Two IRF-convolved exponentials plus an XPM-like (G′, G″) artifact.

    ``tc`` is the ta_core module.  Returns (wl, t, D, Bp, Bpp)."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(450.0, 650.0, 60)
    t = np.concatenate([np.arange(-1.0, 2.0, 0.02),
                        np.geomspace(2.0, 500.0, 40)])
    dads1 = np.exp(-((wl - 520.0) / 40.0) ** 2)
    dads2 = -0.5 * np.exp(-((wl - 600.0) / 30.0) ** 2)
    C = np.column_stack([tc.exp_irf_conv(t, tau, T0_TRUE, FWHM_TRUE)
                         for tau in TAU_TRUE])
    D = np.column_stack([dads1, dads2]) @ C.T
    Bp = art_scale * 0.8 * np.sin(2 * np.pi * wl / 80.0)
    Bpp = art_scale * 0.5 * np.cos(2 * np.pi * wl / 60.0)
    G = tc.artifact_basis(t, T0_TRUE, art_fwhm, 'xpm')
    D = D + np.column_stack([Bp, Bpp]) @ G.T
    D = D + rng.normal(0.0, noise, D.shape)
    return wl, t, D, Bp, Bpp


def golden_cases(tc):
    """{case: (callable(tc, **extra) -> result dict)} on _art_synth(seed=0).

    ``extra`` lets the test append artifact kwargs to the very same call."""
    wl, t, D, _, _ = _art_synth(tc)
    tau0 = np.array([0.5, 30.0])
    rows = [10, 30, 50]

    def ga(method, has_inf):
        return lambda tc_, **kw: tc_.fit_global_analysis(
            D, t, tau0, 0.0, 0.1, [False, False], False, False, has_inf,
            method=method, **kw)

    return {
        'ga_trf': ga('trf', True),
        'ga_nm': ga('nm', False),
        'ga_fixed': lambda tc_, **kw: tc_.fit_global_analysis(
            D, t, np.array(TAU_TRUE), T0_TRUE, FWHM_TRUE, [True, True],
            True, True, True, method='trf', **kw),
        'kfit_single': lambda tc_, **kw: tc_.fit_single_trace(
            t, D[30], tau_init=tau0, tau_fixed=[False, False],
            t0_init=0.0, t0_fixed=False, fwhm_init=0.1, fwhm_fixed=True,
            has_inf=True, **kw),
        'kfit_global': lambda tc_, **kw: tc_.fit_global_analysis(
            D[rows], t, tau0, 0.0, 0.1, [False, False], False, False, False,
            method='trf', **kw),
    }


def flatten(case, r):
    """Result dict -> {f'{case}__{key}': array} for every numeric output."""
    out = {}
    for k in ('tau', 'beta', 't0', 'fwhm', 'A', 'fit', 'residual'):
        if k in r:
            out[f'{case}__{k}'] = np.asarray(r[k], float)
    info = r['info']
    for k in ('rss', 'fval', 'rms', 'initialLoss'):
        if k in info:
            out[f'{case}__info_{k}'] = np.asarray(info[k], float)
    st = info.get('stats') or {}
    for k, v in st.items():
        out[f'{case}__stats_{k}'] = np.asarray(v, float)
    return out


def _load_core(path):
    spec = importlib.util.spec_from_file_location('ta_core_golden', path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: make_artifact_off_golden.py <path/to/ta_core.py>')
    tc = _load_core(sys.argv[1])
    arrays = {}
    for case, fn in golden_cases(tc).items():
        arrays.update(flatten(case, fn(tc)))
        print('done', case)
    np.savez(OUT, **arrays)
    print(f'wrote {OUT} ({len(arrays)} arrays) from {sys.argv[1]}')
