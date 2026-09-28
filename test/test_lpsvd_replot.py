"""
Tests for the standalone ``Data/lpsvd_replot.py`` curator.

Verifies (UI-free):
    1) The bundle loader resolves the 4 sidecar files and reads them.
    2) ``recompute_fft`` reproduces the original ``_freq.csv`` magnitudes
       when *all* modes are selected (no float drift beyond 1e-9).
    3) ``combine_time`` recovers the original ``fit`` column when *all*
       modes are selected.
    4) ``save_curated_bundle`` writes the four expected files and the
       sums inside ``_time.csv`` are self-consistent.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import importlib.util
import sys
import tempfile

import numpy as np
from PyQt5 import QtWidgets


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ)


def _load_replot_module():
    """Import ``Data/lpsvd_replot.py`` by path (it lives outside the
    Python package root)."""
    path = os.path.join(_PROJ, 'Data', 'lpsvd_replot.py')
    spec = importlib.util.spec_from_file_location('lpsvd_replot', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_bundle(tmpdir):
    """Run a real LPSVD analysis through the Coherence dialog and write
    the CSV bundle into ``tmpdir/bundle.csv`` (creating the 4 sidecars
    next to it).  Returns the dialog's ``_last_lp`` so the test can
    cross-check exact numbers."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import ta_main
    import ta_coherence
    from ta_coherence import CoherenceDialog

    rng = np.random.default_rng(31)
    wl = np.linspace(500.0, 650.0, 25)
    t = np.linspace(-0.2, 5.0, 281)
    env = np.exp(-((wl - 575.0) / 22.0) ** 2)
    osc = (np.exp(-0.15 * t) * np.cos(2 * np.pi * 1.6 * t)
           + 0.5 * np.exp(-0.20 * t) * np.cos(2 * np.pi * 3.3 * t + 0.3)
           + 0.25 * np.exp(-0.10 * t) * np.cos(2 * np.pi * 5.0 * t - 0.4))
    A = np.outer(env, osc) + 0.004 * rng.standard_normal((25, 281))
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'replot_smoke.csv')

    dlg = CoherenceDialog(win, win)
    dlg.rb_use_data.setChecked(True)
    dlg.ed_t_min.setValue(0.4)
    dlg.ed_t_max.setValue(4.8)
    dlg.rb_lps_lor.setChecked(True)
    dlg.ed_lps_modes.setValue(3)
    dlg.chk_polyfit.setChecked(True)
    dlg.ed_poly_deg.setValue(4)
    dlg.chk_detrend.setChecked(True)
    dlg.chk_center.setChecked(True)
    dlg.chk_window.setChecked(True)
    dlg.dd_win_type.setCurrentIndex(dlg.dd_win_type.findData('kaiser'))
    dlg.ed_win_beta.setValue(7.0)
    dlg.cmb_wl_mode.setCurrentIndex(dlg.cmb_wl_mode.findData('mean'))
    dlg.do_run_lpsvd()
    assert dlg._last_lp is not None

    out_csv = os.path.join(tmpdir, 'bundle.csv')
    orig_ask = ta_coherence.ask_save_path
    orig_info = ta_coherence.info_box
    ta_coherence.ask_save_path = lambda *a, **kw: (out_csv, '')
    ta_coherence.info_box = lambda *a, **kw: None
    try:
        dlg._export_lpsvd()
    finally:
        ta_coherence.ask_save_path = orig_ask
        ta_coherence.info_box = orig_info
    last = dlg._last_lp
    dlg.close()
    return out_csv, last


def test_load_bundle_round_trip():
    mod = _load_replot_module()
    with tempfile.TemporaryDirectory() as tmp:
        out_csv, last_lp = _make_bundle(tmp)
        b = mod.load_bundle(out_csv)

        assert b['win_type'] == 'kaiser'
        assert abs(b['beta'] - 7.0) < 1e-12
        # N_pad in meta should agree with the live analysis result.
        assert b['N_pad'] == last_lp['params']['fft_N_pad']
        assert b['modes_data'].shape[0] == last_lp['params']['num_modes']

        # When ALL modes are selected the recomputed FFT magnitudes
        # must match the saved _freq.csv exactly (modulo float drift).
        all_sel = list(range(b['modes_data'].shape[0]))
        f_hz, f_orig, f_sel, _ = mod.recompute_fft(b, all_sel)

        # Cross-check against the saved freq.csv on positive freqs.
        freq_hdr = b['freq_hdr']; F = b['freq_data']
        wn_saved = F[:, freq_hdr.index('wavenumber_cm-1')]
        orig_saved = F[:, freq_hdr.index('orig_mag')]
        fit_saved = F[:, freq_hdr.index('fit_mag')]

        fcm = f_hz / mod.C_CM
        pos = fcm > 0
        # The saved CSV writes only the positive-frequency slice in the
        # same order recompute_fft returns it after the [pos] mask.
        np.testing.assert_allclose(fcm[pos], wn_saved, atol=1e-9)
        np.testing.assert_allclose(np.abs(f_orig)[pos], orig_saved,
                                   rtol=1e-7, atol=1e-9)
        # |sum(complex)| reconstructed by replot ==
        # |FFT(total_centered)| originally saved.  These differ from
        # Σ |mode_i_mag| (this is the "fit ≠ sum-of-magnitudes" point
        # the user already understands).
        np.testing.assert_allclose(np.abs(f_sel)[pos], fit_saved,
                                   rtol=1e-7, atol=1e-9)
    print('[load+fft]  OK')


def test_combine_time_recovers_full_fit():
    mod = _load_replot_module()
    with tempfile.TemporaryDirectory() as tmp:
        out_csv, _ = _make_bundle(tmp)
        b = mod.load_bundle(out_csv)
        all_sel = list(range(b['modes_data'].shape[0]))
        t, data, bg, fit_full, residual, modes_t = mod.combine_time(
            b, all_sel)
        # Cross-check against the saved ``fit`` column.
        T = b['time_data']; hdr = b['time_hdr']
        fit_saved = T[:, hdr.index('fit')]
        residual_saved = T[:, hdr.index('residual')]
        # %.10g rounding in the saved CSV limits agreement to ~1e-9
        # relative; the algorithm itself is bit-exact in memory.
        np.testing.assert_allclose(fit_full, fit_saved,
                                   rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(residual, residual_saved,
                                   rtol=1e-7, atol=1e-9)
    print('[combine]   OK')


def test_save_curated_bundle_round_trip():
    mod = _load_replot_module()
    with tempfile.TemporaryDirectory() as tmp:
        out_csv, _ = _make_bundle(tmp)
        b = mod.load_bundle(out_csv)
        # Pick a 2-out-of-N subset
        n = b['modes_data'].shape[0]
        sel = [0, n - 1] if n >= 2 else [0]
        cur = os.path.join(tmp, 'curated.csv')
        paths = mod.save_curated_bundle(b, sel, cur)
        assert len(paths) == 4
        for p in paths:
            assert os.path.isfile(p), f'missing {p}'

        # Reload the curated bundle and sanity-check internal consistency
        b2 = mod.load_bundle(paths[0])
        T = b2['time_data']; hdr = b2['time_hdr']
        # mode columns exactly match the originals we extracted from b.
        for i in sel:
            col = f'mode{i + 1}'
            assert col in hdr
            saved_now = T[:, hdr.index(col)]
            saved_src = b['time_data'][:, b['time_hdr'].index(col)]
            np.testing.assert_allclose(saved_now, saved_src,
                                       rtol=1e-7, atol=1e-9)
        # fit_selected = sum(selected mode_i) + bg_total
        fit_sel = T[:, hdr.index('fit_selected')]
        bg = T[:, hdr.index('bg_total')]
        sum_modes = np.zeros_like(fit_sel)
        for i in sel:
            sum_modes += T[:, hdr.index(f'mode{i + 1}')]
        np.testing.assert_allclose(fit_sel, sum_modes + bg,
                                   rtol=1e-7, atol=1e-9)
        # The curated meta records which modes were kept.
        with open(paths[3], encoding='utf-8') as fh:
            text = fh.read()
        assert '[curated_selection]' in text
        for i in sel:
            assert str(i + 1) in text
    print('[save]      OK')


def main():
    test_load_bundle_round_trip()
    test_combine_time_recovers_full_fit()
    test_save_curated_bundle_round_trip()
    print('\nAll LPSVD replot tests passed.')


if __name__ == '__main__':
    main()
