"""
Tests for the coherent-artifact terms in the fit engine
(``ta_core.fit_global_analysis`` / ``ta_core.fit_single_trace``).

Verifies (plan .omc/plans/artifact-ga-kfit-plan.md §4, AC1–AC5):
    AC1) GA artifact columns: shapes, fit = exp part + artifact_fit,
         'full' has 3 columns, bad mode raises, TF backend == NumPy.
    AC2) Single trace: shapes, ∞ column kept in A, ignored cases
         (stretched + skip, window starting at t0 + 4σ) give zero B and the
         Off τ; same for GA.  Off gives B None.
    AC3) Separate artifact FWHM: free recovers it, fixed stays exact.
    AC4) Off path equals the frozen pre-change golden (≤ 1e-12).
    AC5) xpm reduces the fast-τ and t0 bias and recovers B′/B″.

Synthetic data only (test/fixtures/make_artifact_off_golden._art_synth).
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
for _p in (_PROJ, os.path.join(_HERE, 'fixtures')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ta_core  # noqa: E402
import make_artifact_off_golden as gold  # noqa: E402

GOLDEN = os.path.join(_HERE, 'fixtures', 'artifact_off_golden.npz')
TAU0 = np.array([0.5, 30.0])
SIG_TRUE = gold.FWHM_TRUE / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _ga(D, t, artifact='off', has_inf=False, **kw):
    return ta_core.fit_global_analysis(
        D, t, TAU0, 0.0, 0.1, [False, False], False, False, has_inf,
        method='trf', artifact=artifact, **kw)


def test_ac1_ga_columns():
    wl, t, D, _, _ = gold._art_synth(ta_core)
    for mode, n in (('xpm', 2), ('full', 3)):
        r = _ga(D, t, mode, has_inf=True)
        assert r['B'].shape == (wl.size, n), r['B'].shape
        assert r['A'].shape == (wl.size, 3), r['A'].shape
        assert r['artifact_fit'].shape == D.shape
        assert r['info']['artifact'] == mode
        assert r['info']['artifact_ignored'] is None
        Cexp = ta_core.build_ga_basis(t, r['tau'], r['t0'], r['fwhm'], True)
        err = np.abs(r['fit'] - r['A'] @ Cexp.T - r['artifact_fit']).max()
        assert err < 1e-10, err
        st = r['info']['stats']
        assert st['A_std'].shape == r['A'].shape
        assert st['B_std'].shape == r['B'].shape
    try:
        _ga(D, t, 'bogus')
    except ValueError:
        pass
    else:
        raise AssertionError('bad artifact mode did not raise')

    # Backend path at fixed nonlinear parameters (x0.size == 0)
    import ta_device
    kw = dict(artifact='xpm', method='trf')
    args = (D, t, np.array(gold.TAU_TRUE), gold.T0_TRUE, gold.FWHM_TRUE,
            [True, True], True, True, False)
    ref = ta_core.fit_global_analysis(*args, **kw)
    # The backend branch itself (NumPy-backed TFBackend) always runs
    r = ta_core.fit_global_analysis(
        *args, backend=ta_device.TFBackend('/CPU:0', 'numpy'), **kw)
    assert np.abs(r['B'] - ref['B']).max() < 1e-10
    if ta_device.tf_available():
        be = ta_device.TFBackend('/CPU:0', 'tf')
        r = ta_core.fit_global_analysis(*args, backend=be, **kw)
        assert np.abs(r['B'] - ref['B']).max() < 1e-10
        assert np.abs(r['fit'] - ref['fit']).max() < 1e-10
        tf_msg = 'TF-CPU == NumPy'
    else:
        tf_msg = 'SKIP TF check (TensorFlow not available)'
    print(f'PASS AC1 GA columns xpm/full, bad mode raises; {tf_msg}')


def test_ac2_single_and_ignored():
    wl, t, D, _, _ = gold._art_synth(ta_core)
    y = D[30]
    kw = dict(tau_init=TAU0, tau_fixed=[False, False], t0_init=gold.T0_TRUE,
              t0_fixed=False, fwhm_init=gold.FWHM_TRUE, fwhm_fixed=True)
    r = ta_core.fit_single_trace(t, y, has_inf=True, artifact='xpm', **kw)
    assert r['B'].shape == (2,) and r['artifact_fit'].shape == t.shape
    assert r['A'].shape == (3,), r['A'].shape
    st = r['info']['stats']
    assert st['A_std'].shape == r['A'][None].shape, st['A_std'].shape
    assert st['B_std'].shape == r['B'][None].shape
    # ∞ column stays last in A: compare with the explicit exp/∞ basis
    Cexp = ta_core.build_ga_basis(t, r['tau'], r['t0'], r['fwhm'], True)
    err = np.abs(r['fit'] - Cexp @ r['A'] - r['artifact_fit']).max()
    assert err < 1e-10, err
    off = ta_core.fit_single_trace(t, y, has_inf=True, **kw)
    assert off['B'] is None and off['artifact_fit'] is None
    assert off['info']['artifact'] == 'off'

    # Ignored: stretched + skip, and a window starting at t0 + 4σ
    t_win = t >= gold.T0_TRUE + 4 * SIG_TRUE
    cases = [
        ('skip-mask', t, y, dict(stretch_on=[False, True], irf_mode='skip')),
        ('outside-window', t[t_win], y[t_win], {}),
    ]
    for reason, tt, yy, extra in cases:
        on = ta_core.fit_single_trace(tt, yy, artifact='xpm', **kw, **extra)
        ref = ta_core.fit_single_trace(tt, yy, **kw, **extra)
        assert on['info']['artifact_ignored'] == reason, on['info']
        assert on['B'].shape == (2,) and not on['B'].any()
        assert on['artifact_fit'].shape == tt.shape
        assert not on['artifact_fit'].any()
        assert np.abs(on['tau'] - ref['tau']).max() <= 1e-12
        assert np.isnan(on['info']['stats']['B_std']).all()

    gkw = dict(method='trf')
    for reason, sl, extra in (
            ('skip-mask', slice(None),
             dict(stretch_on=[False, True], irf_mode='skip')),
            ('outside-window', t_win, {})):
        args = (D[:, sl], t[sl], TAU0, gold.T0_TRUE, gold.FWHM_TRUE,
                [False, False], False, True, False)
        on = ta_core.fit_global_analysis(*args, artifact='xpm', **gkw,
                                         **extra)
        ref = ta_core.fit_global_analysis(*args, **gkw, **extra)
        assert on['info']['artifact_ignored'] == reason, on['info']
        assert on['B'].shape == (wl.size, 2) and not on['B'].any()
        assert on['artifact_fit'].shape == D[:, sl].shape
        assert not on['artifact_fit'].any()
        assert np.abs(on['tau'] - ref['tau']).max() <= 1e-12
        assert ref['B'] is None
    print('PASS AC2 single-trace shapes, ∞ last, ignored skip-mask / '
          'outside-window (single + GA)')


def test_ac3_separate_fwhm():
    wl, t, D, _, _ = gold._art_synth(ta_core, art_fwhm=0.20)
    r = _ga(D, t, 'xpm', art_fwhm=0.15, art_fwhm_fixed=False)
    st = r['info']['stats']
    assert abs(r['art_fwhm'] - 0.20) < 0.03, r['art_fwhm']
    assert np.isfinite(st['art_fwhm_std']) and st['art_fwhm_std'] > 0
    assert st['B_std'].shape == r['B'].shape
    assert np.all(np.isfinite(st['tau_std'])) and np.isfinite(st['t0_std'])
    fx = _ga(D, t, 'xpm', art_fwhm=0.25, art_fwhm_fixed=True)
    assert fx['art_fwhm'] == 0.25
    assert np.isnan(fx['info']['stats']['art_fwhm_std'])
    # Single trace: same parameterisation
    s = ta_core.fit_single_trace(
        t, D[30], tau_init=TAU0, tau_fixed=[False, False], t0_init=0.0,
        t0_fixed=False, fwhm_init=0.1, fwhm_fixed=True, artifact='xpm',
        art_fwhm=0.25, art_fwhm_fixed=True)
    assert s['art_fwhm'] == 0.25
    assert np.isnan(s['info']['stats']['art_fwhm_std'])
    print(f"PASS AC3 FWHM_art free {r['art_fwhm']:.4f} ± "
          f"{st['art_fwhm_std']:.4f} (true 0.20); fixed stays 0.25")


def test_ac4_off_golden():
    g = np.load(GOLDEN)
    worst = 0.0
    for case, fn in gold.golden_cases(ta_core).items():
        for extra in ({}, {'artifact': 'off'}):
            r = fn(ta_core, **extra)
            assert r['B'] is None and r['artifact_fit'] is None
            got = gold.flatten(case, r)
            keys = [k for k in g.files if k.startswith(case + '__')]
            assert keys
            for k in keys:
                a, b = got[k], g[k]
                assert a.shape == b.shape, (k, a.shape, b.shape)
                assert np.array_equal(np.isnan(a), np.isnan(b)), k
                fin = ~np.isnan(b)
                d = float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0
                assert d <= 1e-12, (k, d)
                worst = max(worst, d)
    print(f'PASS AC4 Off == golden for {len(g.files)} arrays '
          f'(max abs diff {worst:.1e})')


def test_ac5_bias():
    wl, t, D, Bp, Bpp = gold._art_synth(ta_core, seed=0)
    off = _ga(D, t, 'off')
    xpm = _ga(D, t, 'xpm')
    tau_true = gold.TAU_TRUE[0]
    e_off = abs(off['tau'][0] - tau_true)
    e_x = abs(xpm['tau'][0] - tau_true)
    t0_off = abs(off['t0'] - gold.T0_TRUE)
    t0_x = abs(xpm['t0'] - gold.T0_TRUE)
    r1 = np.corrcoef(xpm['B'][:, 0], Bp)[0, 1]
    r2 = np.corrcoef(xpm['B'][:, 1], Bpp)[0, 1]
    msg = (f'τ1 err xpm {e_x:.4g} vs off {e_off:.4g} '
           f'(ratio {e_x / e_off:.3f}); t0 err xpm {t0_x:.4g} vs off '
           f'{t0_off:.4g}; corr B′ {r1:.4f}, B″ {r2:.4f}')
    assert e_x <= 0.7 * e_off, 'hard ceiling: ' + msg
    assert e_x <= 0.5 * e_off, 'FAIL (0.5–0.7 band): ' + msg
    assert t0_x < t0_off, msg
    assert r1 > 0.9 and r2 > 0.9, msg
    print('PASS AC5 ' + msg)


if __name__ == '__main__':
    test_ac1_ga_columns()
    test_ac2_single_and_ignored()
    test_ac3_separate_fwhm()
    test_ac4_off_golden()
    test_ac5_bias()
    print('\nAll artifact fit tests passed.')
