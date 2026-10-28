"""
Regression tests for the Origin (.opju) export staging.

Verifies:

  1. Each feeding export stages a CSV for its builder slot (2D original,
     2D chirp-corrected only after chirp correction, chirp points, pins,
     DADS/EADS + kinetics/populations, GA fit/residual); .xlsx / .tsv
     exports are staged as comma CSV; re-exporting replaces the entry
  2. Every staged CSV is accepted by the builder's own sheet preparation
     (add_function/ta_opju_builder prep_* — no Origin needed)
  3. opju dataset info window: status table, summary, 저장 / 취소;
     저장 runs the builder in a thread (fake builder) and writes the file
  4. Loading a new TA data set clears the staging and its temp folder
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

import ta_core  # noqa: E402
import ta_main  # noqa: E402
import ta_ga  # noqa: E402
import ta_chirp  # noqa: E402
import ta_opju_export as tox  # noqa: E402


class _Patch:
    """Route every save dialog to the next path; silence message boxes."""

    def __init__(self, paths):
        self.paths = iter(paths)
        self.saved = []

    def __enter__(self):
        ask = lambda *a, **k: (next(self.paths), '')      # noqa: E731
        quiet = lambda *a, **k: None                      # noqa: E731
        self.saved = [(m, n, getattr(m, n)) for m in
                      (ta_main, ta_ga, ta_chirp, tox)
                      for n in ('ask_save_path', 'info_box', 'warn_box')
                      if hasattr(m, n)]
        for m, n, _ in self.saved:
            setattr(m, n, ask if n == 'ask_save_path' else quiet)
        self._qfd = QtWidgets.QFileDialog.getSaveFileName
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(ask)
        return self

    def __exit__(self, *exc):
        for m, n, f in self.saved:
            setattr(m, n, f)
        QtWidgets.QFileDialog.getSaveFileName = self._qfd


def _dataset():
    rng = np.random.default_rng(4)
    wl = np.linspace(420.0, 680.0, 66)
    t = np.r_[np.linspace(-1, 2, 25), np.logspace(0.4, 2.5, 50)]
    D = (np.outer(np.exp(-((wl - 500) / 50) ** 2), ta_core.exp_irf_conv(t, 2.0, 0, .15))
         - 0.6 * np.outer(np.exp(-((wl - 600) / 50) ** 2),
                          ta_core.exp_irf_conv(t, 40.0, 0, .15)))
    return wl, t, D + 0.002 * rng.normal(size=D.shape)


def _run_ga(win):
    ga = ta_ga.GlobalAnalysisDialog(win, win)
    ga.dd_N.setCurrentText('2')
    ga.tau_table.cellChanged.disconnect()
    for r, v in enumerate([1.0, 30.0]):
        ga.tau_table.item(r, 0).setText(str(v))
    ga.tau_table.cellChanged.connect(ga._on_tau_edit)
    ga.read_table()
    ga.do_run()
    QtWidgets.QApplication.processEvents()
    return ga


def test_staging_and_dialog():
    wl, t, D = _dataset()
    builder = tox.load_builder()
    with tempfile.TemporaryDirectory() as tmp:
        P = lambda n: os.path.join(tmp, n)                # noqa: E731
        win = ta_main.TAAnalyzer()
        win.set_loaded_data(wl, t, D, 'Loaded: opju.csv', source_dir=tmp)
        stage = win.opju_stage
        assert stage.entries == {}

        with _Patch([P('a_original.xlsx'), P('a_2D.csv'), P('a_2D_chirp.tsv'),
                     P('spec.csv'), P('kin.dat'), P('chirp_points.csv'),
                     P('DADS_spectra.csv'), P('EADS_spectra.tsv'),
                     P('GA_fit.csv'), P('GA_residual.csv'),
                     P('kin2.csv')]):
            win.export_2d_data_original()                  # .xlsx → matrix
            win.export_2d_data()                           # no chirp: skip
            assert 'chirp_cropped' not in stage.entries
            win.chirp_applied = True
            win.export_2d_data()                           # .tsv → matrix
            win.set_sel_t(t[40]); win.pin_spec()
            win.set_sel_wl(500.0); win.pin_kin()
            win.export_spec_overlays()
            win.export_kin_overlays()                      # .dat, commas
            win.chirp_pts = np.array([[450.0, 0.1], [550.0, 0.2],
                                      [650.0, 0.3], [500.0, 0.15]])
            cd = ta_chirp.ChirpDialog(win, win)
            cd.save_pts()
            ga = _run_ga(win)
            ga.export_dads()
            ga.export_eads()                               # tab-delimited
            ga.export_matrix_2d('fit', 'GA fit', 'fit')
            ga.export_matrix_2d('residual', 'GA residual', 'residual')
            win.set_sel_wl(600.0); win.pin_kin()
            win.export_kin_overlays()                      # replaces kinetics
        cd.close()

        assert set(stage.entries) == set(tox.SLOT_KEYS), set(stage.entries)
        assert stage.entries['pinned_kinetics']['source'] == 'kin2.csv'
        assert stage.entries['cropped_original']['source'] == 'a_original.xlsx'
        assert stage.entries['chirp_cropped']['note'].startswith('chirp')
        files = stage.files()
        for k, p in files.items():
            assert os.path.dirname(p) != tmp and os.path.isfile(p), k
            assert '\t' not in open(p, encoding='utf-8-sig').readline(), k

        # 2. Builder's own preparation accepts every staged CSV
        jobs = builder.sheet_jobs(files, {'smooth_window': 15,
                                          'smooth_poly': 2})
        for sheet, path, prep in jobs:
            arr, labels, note = prep(path)
            assert arr.ndim == 2 and arr.shape[0] > 0, sheet
        arr, _, _ = builder.prep_kinetics(files['pinned_kinetics'])
        assert arr.shape[1] == 3                            # delay + 2 pins
        arr, _, _ = builder.prep_time_series(files['EADS_populations'])
        assert arr.shape[1] == 3

        # 3. opju dataset info window
        dlg = tox.OpjuDatasetDialog(win, win)
        assert dlg.windowTitle() == 'opju dataset info'
        assert [dlg.tbl.item(r, 1).text() for r in range(len(tox.SLOTS))] \
            == ['Saved'] * len(tox.SLOTS)
        assert dlg.btn_save.text() == '저장' and dlg.btn_cancel.text() == '취소'
        assert dlg.btn_save.isEnabled()
        assert f'{len(tox.SLOTS)} / {len(tox.SLOTS)}' in dlg.lbl_summary.text()

        calls = []

        def fake_build(folder, out, files=None, progress_cb=None):
            progress_cb(1, 2, 'sheet'); progress_cb(2, 2, 'saved')
            # inputs are a private snapshot of the staging
            calls.append({k: (open(p, 'rb').read() if p else None)
                          for k, p in files.items()})
            snaps.append(os.path.dirname(files['GA_fit']))
            stage.clear()                 # new data loaded mid-build
            assert all(os.path.isfile(p) for p in files.values())
            open(out, 'wb').write(b'opju')
            return out, ['[OK] fake']

        snaps = []
        expected = {k: open(p, 'rb').read() for k, p in files.items()}

        fake = types.SimpleNamespace(running_origin=lambda: [],
                                     build=fake_build)
        orig_load = tox.load_builder
        tox.load_builder = lambda: fake
        try:
            with _Patch([P('result')]):                    # no extension
                dlg.do_save()
                dlg._thread.wait(10000)
                for _ in range(20):
                    QtWidgets.QApplication.processEvents()
        finally:
            tox.load_builder = orig_load
        assert dlg.out_path == P('result.opju') and os.path.isfile(P('result.opju'))
        assert calls[0] == expected
        assert not os.path.exists(snaps[0])          # snapshot removed
        assert dlg.result() == QtWidgets.QDialog.Accepted
        assert stage.entries == {}                   # cleared mid-build

        # Nothing staged → 저장 explains instead of building
        dlg_e = tox.OpjuDatasetDialog(win, win)
        dlg_e.btn_save.setEnabled(True)
        shown = []
        orig_info = tox.info_box
        tox.info_box = lambda *a, **k: shown.append(a[1])
        try:
            dlg_e.do_save()
        finally:
            tox.info_box = orig_info
        assert shown == ['Nothing staged'] and dlg_e._thread is None
        dlg_e.reject()

        # A failing builder reports and releases the dialog
        stage.stage_matrix('GA_fit', wl, t, D, 'GA_fit.csv')
        dlg_f = tox.OpjuDatasetDialog(win, win)
        boom = types.SimpleNamespace(
            running_origin=lambda: [],
            build=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('')))
        tox.load_builder = lambda: boom
        warned = []
        try:
            with _Patch([P('fail.opju')]):
                tox.warn_box = lambda *a, **k: warned.append(a[2])
                dlg_f.do_save()
                dlg_f._thread.wait(10000)
                for _ in range(20):
                    QtWidgets.QApplication.processEvents()
        finally:
            tox.load_builder = orig_load
        assert warned == ['RuntimeError'] and not dlg_f.is_building()
        dlg_f.reject()

        # Origin running → refused before asking for a path
        dlg2 = tox.OpjuDatasetDialog(win, win)
        tox.load_builder = lambda: types.SimpleNamespace(
            running_origin=lambda: ['Origin95_64.exe'], build=None)
        try:
            with _Patch([]):
                dlg2.do_save()                              # next() would fail
        finally:
            tox.load_builder = orig_load
        assert dlg2._thread is None
        dlg2.reject()

        # Main-window button path; closing is refused while building
        win.open_opju_window()
        win.opju_fig._thread = object()
        ev = types.SimpleNamespace(ignore=lambda: shown.append('ignored'),
                                   accept=lambda: None)
        with _Patch([]):
            win.closeEvent(ev)
        assert shown[-1] == 'ignored'
        win.opju_fig._thread = None
        assert win.opju_fig is not None
        win.opju_fig.reject()

        # 4. New data clears everything
        stage.stage_matrix('GA_fit', wl, t, D, 'GA_fit.csv')
        folder = os.path.dirname(stage.files()['GA_fit'])
        win.set_loaded_data(wl, t, D, 'Loaded: other.csv', source_dir=tmp)
        assert stage.entries == {} and not os.path.exists(folder)
        assert all(v is None for v in stage.files().values())
        dlg3 = tox.OpjuDatasetDialog(win, win)
        assert not dlg3.btn_save.isEnabled()
        assert dlg3.tbl.item(0, 1).text() == '—'
        dlg3.reject()
        QtWidgets.QApplication.processEvents()
        ga.close()
        win.close()
    print('[ok] opju staging, builder prep, dataset info window')


TESTS = [test_staging_and_dialog]


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    for fn in TESTS:
        fn()
    print(f'\nAll {len(TESTS)} opju export checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
