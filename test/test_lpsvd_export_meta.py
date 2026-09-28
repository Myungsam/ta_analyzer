"""
Tests for the extended LPSVD CSV-bundle export.

Verifies that the new ``_meta.txt`` sidecar records all processing
conditions (time window, polynomial baseline + coefficients, detrend /
centering, fit model, FFT window settings) and that the ``_time.csv``
table now carries explicit baseline columns.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import tempfile
import numpy as np
from PyQt5 import QtWidgets


def _build_dataset():
    rng = np.random.default_rng(11)
    wl = np.linspace(500.0, 650.0, 30)
    t = np.linspace(-0.2, 5.0, 301)
    env = np.exp(-((wl - 575.0) / 25.0) ** 2)
    osc = (np.exp(-0.15 * t) * np.cos(2 * np.pi * 1.8 * t)
           + 0.5 * np.exp(-0.20 * t) * np.cos(2 * np.pi * 3.6 * t + 0.3))
    A = np.outer(env, osc) + 0.005 * rng.standard_normal((30, 301))
    return wl, t, A


def _parse_kv(meta_text):
    out = {}
    section = None
    for line in meta_text.splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('[') and s.endswith(']'):
            section = s[1:-1]
            continue
        if '=' in s:
            k, v = s.split('=', 1)
            out[f'{section}.{k.strip()}'] = v.strip()
    return out


def test_lpsvd_export_meta_records_all_conditions():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    import ta_main
    from ta_coherence import CoherenceDialog

    wl, t, A = _build_dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'meta_export_test.csv')

    dlg = CoherenceDialog(win, win)
    dlg.rb_use_data.setChecked(True)
    dlg.ed_t_min.setValue(0.4)
    dlg.ed_t_max.setValue(4.8)
    dlg.rb_lps_lor.setChecked(True)
    dlg.ed_lps_modes.setValue(2)
    dlg.chk_polyfit.setChecked(True)
    dlg.ed_poly_deg.setValue(5)
    dlg.chk_detrend.setChecked(True)
    dlg.chk_center.setChecked(True)
    dlg.ed_append.setValue(0)
    dlg.chk_window.setChecked(True)
    dlg.dd_win_type.setCurrentIndex(dlg.dd_win_type.findData('kaiser'))
    dlg.ed_win_beta.setValue(7.5)
    dlg.cmb_wl_mode.setCurrentIndex(dlg.cmb_wl_mode.findData('mean'))
    dlg.do_run_lpsvd()
    assert dlg._last_lp is not None

    # Verify the result dict carries the new fields.
    res = dlg._last_lp
    assert 'params' in res
    assert 'bg_poly' in res and 'bg_detrend' in res
    assert res['params']['polyfit_deg_used'] == 5
    assert res['params']['window_type'] == 'kaiser'
    assert abs(res['params']['beta'] - 7.5) < 1e-12
    assert res['params']['model'] == 'lorentzian'
    assert res['params']['num_modes'] == 2

    # Drive the export through the actual handler.  ask_save_path is
    # interactive, so monkey-patch it to return a temp path.
    import ta_coherence
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, 'bundle.csv')
        orig_ask = ta_coherence.ask_save_path
        orig_info = ta_coherence.info_box
        ta_coherence.ask_save_path = lambda *a, **kw: (out_csv, '')
        ta_coherence.info_box = lambda *a, **kw: None
        try:
            dlg._export_lpsvd()
        finally:
            ta_coherence.ask_save_path = orig_ask
            ta_coherence.info_box = orig_info

        modes_path = os.path.join(tmp, 'bundle_modes.csv')
        time_path = os.path.join(tmp, 'bundle_time.csv')
        freq_path = os.path.join(tmp, 'bundle_freq.csv')
        meta_path = os.path.join(tmp, 'bundle_meta.txt')
        for p in (modes_path, time_path, freq_path, meta_path):
            assert os.path.isfile(p), f'missing {p}'

        # 1) _time.csv carries the new baseline columns.
        with open(time_path, encoding='utf-8') as fh:
            header = fh.readline().strip().split(',')
        for col in ('t', 'data', 'fit', 'residual', 'window',
                    'bg_total', 'bg_poly', 'bg_detrend', 'bg_center',
                    'y_centered', 'fit_centered',
                    'mode1', 'mode2'):
            assert col in header, f'time.csv missing {col}'

        # bg_poly + bg_detrend + bg_center should sum to bg_total
        data = np.loadtxt(time_path, delimiter=',', skiprows=1)
        cols = {c: data[:, i] for i, c in enumerate(header)}
        np.testing.assert_allclose(
            cols['bg_poly'] + cols['bg_detrend'] + cols['bg_center'],
            cols['bg_total'], atol=1e-10,
            err_msg='bg_poly + bg_detrend + bg_center must equal bg_total')
        # fit_centered + bg_total should equal fit (= total_with_bg)
        np.testing.assert_allclose(
            cols['fit_centered'] + cols['bg_total'], cols['fit'],
            atol=1e-10)

        # 2) _meta.txt contains every condition we promised to record.
        with open(meta_path, encoding='utf-8') as fh:
            txt = fh.read()
        kv = _parse_kv(txt)
        # time window
        assert kv['time_window.t_start_used'].startswith('0.4')
        assert float(kv['time_window.t_end_used']) <= 4.8 + 1e-9
        assert int(kv['time_window.N_window']) > 100
        # polyfit details
        assert kv['processing.polyfit.enabled'] == 'True'
        assert kv['processing.polyfit.degree_used'] == '5'
        assert 'coefficients' in kv.get(
            'processing.polyfit.coefficients', '') or \
            kv.get('processing.polyfit.coefficients', '').count(',') >= 1
        assert 't^5' in kv.get('processing.polyfit.expression', ''), (
            'polynomial expression should reflect the degree')
        # detrend & center
        assert kv['processing.detrend.linear_detrend'] == 'True'
        assert kv['processing.center.mean_center'] == 'True'
        # fit model
        assert kv['fit_model.model'] == 'lorentzian'
        assert kv['fit_model.num_modes'] == '2'
        # FFT window
        assert kv['fft_window.window_type'] == 'kaiser'
        assert kv['fft_window.kaiser_beta'].startswith('7.5')
        assert kv['fft_window.apply_window_to_fft'] == 'True'
        assert int(kv['fft_window.fft_N_pad']) >= 8192

    dlg.close()
    print('[meta]  OK')


def test_lpsvd_export_meta_when_polyfit_disabled():
    """If the user turns off polynomial subtraction the metadata must
    still be coherent (coefficients line says "none", degree_used = 0)."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    import ta_main
    from ta_coherence import CoherenceDialog
    import ta_coherence

    wl, t, A = _build_dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'no_poly_test.csv')

    dlg = CoherenceDialog(win, win)
    dlg.rb_use_data.setChecked(True)
    dlg.ed_t_min.setValue(0.5)
    dlg.ed_t_max.setValue(4.5)
    dlg.rb_lps_gauss.setChecked(True)
    dlg.ed_lps_modes.setValue(2)
    dlg.chk_polyfit.setChecked(False)
    dlg.chk_detrend.setChecked(False)
    dlg.chk_center.setChecked(True)
    dlg.chk_window.setChecked(False)
    dlg.dd_win_type.setCurrentIndex(dlg.dd_win_type.findData('gaussian'))
    dlg.ed_win_sigma.setValue(1.5)
    dlg.cmb_wl_mode.setCurrentIndex(dlg.cmb_wl_mode.findData('mean'))
    dlg.do_run_lpsvd()
    assert dlg._last_lp is not None
    assert dlg._last_lp['params']['do_polyfit'] is False
    assert dlg._last_lp['params']['polyfit_deg_used'] == 0
    assert dlg._last_lp['params']['model'] == 'gaussian'

    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, 'b.csv')
        orig_ask = ta_coherence.ask_save_path
        orig_info = ta_coherence.info_box
        ta_coherence.ask_save_path = lambda *a, **kw: (out_csv, '')
        ta_coherence.info_box = lambda *a, **kw: None
        try:
            dlg._export_lpsvd()
        finally:
            ta_coherence.ask_save_path = orig_ask
            ta_coherence.info_box = orig_info

        with open(os.path.join(tmp, 'b_meta.txt'), encoding='utf-8') as fh:
            txt = fh.read()
        assert 'enabled         = False' in txt
        assert 'coefficients = (none' in txt
        assert 'model        = gaussian' in txt
        # apply_window_to_fft is False here
        assert 'apply_window_to_fft= False' in txt
        # Time-domain CSV should have an all-zero bg_poly column.
        time_path = os.path.join(tmp, 'b_time.csv')
        with open(time_path, encoding='utf-8') as fh:
            header = fh.readline().strip().split(',')
        data = np.loadtxt(time_path, delimiter=',', skiprows=1)
        col_bg_poly = data[:, header.index('bg_poly')]
        assert np.allclose(col_bg_poly, 0.0), (
            'bg_poly column must be zero when polyfit is disabled')

    dlg.close()
    print('[meta.no-poly] OK')


def main():
    test_lpsvd_export_meta_records_all_conditions()
    test_lpsvd_export_meta_when_polyfit_disabled()
    print('\nAll LPSVD meta-export tests passed.')


if __name__ == '__main__':
    main()
