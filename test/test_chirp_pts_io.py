"""
Tests for the Chirp dialog Save/Load Points buttons.

Verifies:
    1) ``save_pts`` writes a 2-column CSV with the expected header and
       rows sorted ascending by wavelength.
    2) ``load_pts`` reads such a CSV back, replacing ``app.chirp_pts``,
       and that the result is sorted ascending by wavelength even when
       the input file is unsorted.
    3) Empty / malformed inputs are rejected without crashing.
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


def _make_app_and_dialog():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import ta_main
    from ta_chirp import ChirpDialog

    rng = np.random.default_rng(2)
    wl = np.linspace(450.0, 650.0, 40)
    t = np.linspace(-0.5, 5.0, 120)
    env = np.exp(-((wl - 550.0) / 60.0) ** 2)
    osc = np.exp(-0.4 * np.clip(t, 0, None))
    A = np.outer(env, osc) + 0.005 * rng.standard_normal((40, 120))

    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'chirp_pts_io_test.csv')
    dlg = ChirpDialog(win, win)
    return win, dlg


def test_save_pts_sorts_and_writes_header():
    import ta_widgets
    win, dlg = _make_app_and_dialog()
    # Unsorted points on purpose
    win.chirp_pts = np.array([
        [600.0, 0.42],
        [475.0, 0.10],
        [550.0, 0.25],
        [525.0, 0.18],
        [625.0, 0.55],
    ])

    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, 'cp.csv')
        orig_ask = ta_widgets.ask_save_path
        orig_info = ta_widgets.info_box
        # Patch the names imported into ta_chirp's namespace.
        import ta_chirp
        ta_chirp.ask_save_path = lambda *a, **kw: (out_csv, '')
        ta_chirp.info_box = lambda *a, **kw: None
        try:
            dlg.save_pts()
        finally:
            ta_chirp.ask_save_path = orig_ask
            ta_chirp.info_box = orig_info

        assert os.path.isfile(out_csv), 'save_pts did not write the file'
        with open(out_csv, encoding='utf-8') as fh:
            header = fh.readline().strip()
        # Header includes both columns and the dataset's time unit.
        assert header.startswith('wavelength_nm,t0_'), \
            f'unexpected header: {header!r}'
        unit_tag = header.split(',')[1].split('t0_')[1]
        assert unit_tag in ('ps', 'us'), f'unexpected unit tag: {unit_tag}'

        data = np.loadtxt(out_csv, delimiter=',', skiprows=1)
        assert data.shape == (5, 2)
        # Ascending order by wavelength
        assert np.all(np.diff(data[:, 0]) > 0), \
            'rows are not sorted by wavelength'
        # Same set of (wavelength, t0) pairs as the source.
        src = win.chirp_pts[np.argsort(win.chirp_pts[:, 0])]
        np.testing.assert_allclose(data, src, rtol=1e-4, atol=1e-4)

    dlg.close()
    print('[save]  OK')


def test_load_pts_round_trip_and_resort():
    import ta_widgets
    win, dlg = _make_app_and_dialog()
    pts_src = np.array([
        [600.0, 0.42],
        [475.0, 0.10],
        [550.0, 0.25],
        [525.0, 0.18],
        [625.0, 0.55],
    ])

    with tempfile.TemporaryDirectory() as tmp:
        # Write an UNsorted CSV by hand to verify load_pts re-sorts.
        path_unsorted = os.path.join(tmp, 'unsorted.csv')
        with open(path_unsorted, 'w', encoding='utf-8') as fh:
            fh.write('wavelength_nm,t0_ps\n')
            for row in pts_src:
                fh.write(f'{row[0]:.6g},{row[1]:.6g}\n')

        import ta_chirp
        orig_open = ta_chirp.ask_open_paths
        orig_info = ta_chirp.info_box
        ta_chirp.ask_open_paths = lambda *a, **kw: path_unsorted
        ta_chirp.info_box = lambda *a, **kw: None
        try:
            dlg.load_pts()
        finally:
            ta_chirp.ask_open_paths = orig_open
            ta_chirp.info_box = orig_info

        loaded = win.chirp_pts
        assert loaded.shape == (5, 2)
        assert np.all(np.diff(loaded[:, 0]) > 0), \
            'load_pts must return wavelengths in ascending order'
        # Compare to the wavelength-sorted source set
        src_sorted = pts_src[np.argsort(pts_src[:, 0])]
        np.testing.assert_allclose(loaded, src_sorted,
                                   rtol=1e-4, atol=1e-4)

        # Headerless variant must still parse.
        path_no_hdr = os.path.join(tmp, 'no_hdr.csv')
        np.savetxt(path_no_hdr, src_sorted, delimiter=',', fmt='%.6g')
        ta_chirp.ask_open_paths = lambda *a, **kw: path_no_hdr
        ta_chirp.info_box = lambda *a, **kw: None
        try:
            dlg.load_pts()
        finally:
            ta_chirp.ask_open_paths = orig_open
            ta_chirp.info_box = orig_info
        np.testing.assert_allclose(win.chirp_pts, src_sorted,
                                   rtol=1e-4, atol=1e-4)

    dlg.close()
    print('[load]  OK')


def test_save_pts_rejects_empty():
    import ta_chirp
    win, dlg = _make_app_and_dialog()
    win.chirp_pts = np.zeros((0, 2))
    called = {'saved': False}

    def _ask(*a, **kw):
        called['saved'] = True
        return ('/should/not/be/used.csv', '')

    orig_ask = ta_chirp.ask_save_path
    orig_warn = ta_chirp.warn_box
    ta_chirp.ask_save_path = _ask
    ta_chirp.warn_box = lambda *a, **kw: None
    try:
        dlg.save_pts()
    finally:
        ta_chirp.ask_save_path = orig_ask
        ta_chirp.warn_box = orig_warn
    assert called['saved'] is False, \
        'save_pts should warn and bail before asking for a path'

    dlg.close()
    print('[empty] OK')


def main():
    test_save_pts_sorts_and_writes_header()
    test_load_pts_round_trip_and_resort()
    test_save_pts_rejects_empty()
    print('\nAll chirp points I/O tests passed.')


if __name__ == '__main__':
    main()
