"""
Tests for the right-panel "Export pins" buttons.

Verifies:
    1) ``export_spec_overlays`` writes a CSV whose first column is
       ``wavelength_nm`` and whose remaining columns hold ``deltaA[:, i]``
       at each pinned delay (one column per pin) with headers tagged by
       the snap-to-grid delay value + unit.
    2) ``export_kin_overlays`` writes a CSV whose first column is
       ``delay_<unit>`` and whose remaining columns hold ``deltaA[i, :]``
       at each pinned wavelength (one column per pin).
    3) Both refuse to open a save dialog when no pins exist.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import tempfile

import numpy as np
from PyQt5 import QtWidgets


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


def _make_window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import ta_main
    rng = np.random.default_rng(7)
    wl = np.linspace(400.0, 700.0, 35)
    t = np.linspace(-0.5, 5.0, 60)
    env = np.exp(-((wl - 525.0) / 60.0) ** 2)
    osc = np.exp(-0.4 * np.clip(t, 0, None))
    A = np.outer(env, osc) + 0.005 * rng.standard_normal((35, 60))
    w = ta_main.TAAnalyzer()
    w.set_loaded_data(wl, t, A, 'pin_export_test.csv')
    return w


def test_export_spec_overlays_round_trip():
    import ta_main
    w = _make_window()
    # Pin two delays that are intentionally NOT on the recorded grid;
    # the export must snap each to the nearest measured column.
    w.specOverlays = [0.42, 2.31]
    expected_idx = [int(np.argmin(np.abs(w.delay - v)))
                    for v in w.specOverlays]

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'spec.csv')
        orig_ask = ta_main.ask_save_path
        orig_info = ta_main.info_box
        ta_main.ask_save_path = lambda *a, **kw: (out, '')
        ta_main.info_box = lambda *a, **kw: None
        try:
            w.export_spec_overlays()
        finally:
            ta_main.ask_save_path = orig_ask
            ta_main.info_box = orig_info

        assert os.path.isfile(out)
        with open(out, encoding='utf-8') as fh:
            header = fh.readline().strip().split(',')
        assert header[0] == 'wavelength_nm'
        assert len(header) == 1 + len(expected_idx)
        # Headers reference the snapped delay value, not the click value.
        for col, idx in zip(header[1:], expected_idx):
            snapped = w.delay[idx]
            assert col.startswith('dA_t='), f'bad column header {col!r}'
            assert f'{snapped:.6g}' in col, (
                f'header {col!r} must mention snapped value {snapped:.6g}')
            assert col.endswith('ps'), f'missing unit in {col!r}'

        data = np.loadtxt(out, delimiter=',', skiprows=1)
        np.testing.assert_allclose(data[:, 0], w.wavelength,
                                   rtol=1e-4, atol=1e-4)
        for col_i, idx in enumerate(expected_idx, start=1):
            np.testing.assert_allclose(data[:, col_i],
                                       w.deltaA[:, idx],
                                       rtol=1e-4, atol=1e-4)
    w.close()
    print('[spec]  OK')


def test_export_kin_overlays_round_trip():
    import ta_main
    w = _make_window()
    w.kinOverlays = [432.1, 575.6]
    expected_idx = [int(np.argmin(np.abs(w.wavelength - v)))
                    for v in w.kinOverlays]

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'kin.csv')
        orig_ask = ta_main.ask_save_path
        orig_info = ta_main.info_box
        ta_main.ask_save_path = lambda *a, **kw: (out, '')
        ta_main.info_box = lambda *a, **kw: None
        try:
            w.export_kin_overlays()
        finally:
            ta_main.ask_save_path = orig_ask
            ta_main.info_box = orig_info

        assert os.path.isfile(out)
        with open(out, encoding='utf-8') as fh:
            header = fh.readline().strip().split(',')
        assert header[0] == 'delay_ps'
        assert len(header) == 1 + len(expected_idx)
        for col, idx in zip(header[1:], expected_idx):
            snapped_nm = w.wavelength[idx]
            assert col.startswith('dA_wl='), f'bad header {col!r}'
            assert col.endswith('nm'), f'header {col!r} must end with nm'
            assert f'{snapped_nm:.4g}' in col, (
                f'header {col!r} missing snapped wavelength '
                f'{snapped_nm:.4g}')

        data = np.loadtxt(out, delimiter=',', skiprows=1)
        np.testing.assert_allclose(data[:, 0], w.delay,
                                   rtol=1e-4, atol=1e-4)
        for col_i, idx in enumerate(expected_idx, start=1):
            np.testing.assert_allclose(data[:, col_i],
                                       w.deltaA[idx, :],
                                       rtol=1e-4, atol=1e-4)
    w.close()
    print('[kin]   OK')


def test_export_refuses_empty():
    import ta_main
    w = _make_window()
    w.specOverlays = []
    w.kinOverlays = []
    called = {'asked': False}

    def _ask(*a, **kw):
        called['asked'] = True
        return ('/should/not/run.csv', '')

    orig_ask = ta_main.ask_save_path
    orig_warn = ta_main.warn_box
    ta_main.ask_save_path = _ask
    ta_main.warn_box = lambda *a, **kw: None
    try:
        w.export_spec_overlays()
        w.export_kin_overlays()
    finally:
        ta_main.ask_save_path = orig_ask
        ta_main.warn_box = orig_warn
    assert called['asked'] is False, \
        'No pins -> no save dialog should be opened'

    w.close()
    print('[empty] OK')


def test_export_spec_dedups_collisions():
    """Two pins that snap to the same recorded delay column should
    collapse to a single output column (no duplicate header)."""
    import ta_main
    w = _make_window()
    # Both values land on the same nearest delay sample.
    target = float(w.delay[10])
    w.specOverlays = [target - 1e-6, target + 1e-6]

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'spec_dup.csv')
        orig_ask = ta_main.ask_save_path
        orig_info = ta_main.info_box
        ta_main.ask_save_path = lambda *a, **kw: (out, '')
        ta_main.info_box = lambda *a, **kw: None
        try:
            w.export_spec_overlays()
        finally:
            ta_main.ask_save_path = orig_ask
            ta_main.info_box = orig_info

        with open(out, encoding='utf-8') as fh:
            header = fh.readline().strip().split(',')
        # 1 wavelength col + 1 data col (de-duped from 2 pins)
        assert len(header) == 2, \
            f'collisions should de-dup; got header={header}'
    w.close()
    print('[dedup] OK')


def main():
    test_export_spec_overlays_round_trip()
    test_export_kin_overlays_round_trip()
    test_export_refuses_empty()
    test_export_spec_dedups_collisions()
    print('\nAll pin-export tests passed.')


if __name__ == '__main__':
    main()
