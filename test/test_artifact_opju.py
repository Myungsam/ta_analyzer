"""
Origin (.opju) export of the GA coherent-artifact spectra (plan
.omc/plans/artifact-ga-kfit-plan.md §0 D1).

Verifies:

  1. An artifact-on GA "Export DADS" stages the optional DADS_artifact
     slot (files() / snapshot() include it only while staged)
  2. builder.sheet_jobs appends the DADS_artifact job; prep_wav_series
     accepts the CSV and passes the G-headers through as legends;
     the build adds a TA_DADS-template graph for it
  3. opju dataset info window shows the extra row; "n / n" stays at
     len(SLOTS)
  4. A later artifact-off export unstages it
  5. sheet_jobs without the key equals the original job list

Data: seeded synthetic only, never measurement data.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import shutil
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')

from PyQt5 import QtWidgets

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
for _p in (_PROJ, os.path.join(_HERE, 'fixtures')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Keep a reference: an unreferenced QApplication is collected at once.
qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

import ta_core  # noqa: E402
import ta_main  # noqa: E402
import ta_ga  # noqa: E402
import ta_opju_export as tox  # noqa: E402
import make_artifact_off_golden as gold  # noqa: E402

ORIGINAL_SHEETS = ['Original_Data', 'Data_Chirp_Corrected', 'Chirp_line',
                   'Kinetics', 'Spectra', 'EADS', 'DADS', 'DADS_kinetics',
                   'EADS_populations', 'Global_Fit', 'Residual']
CTX = {'smooth_window': 15, 'smooth_poly': 2}


def _ga(win, mode):
    ga = ta_ga.GlobalAnalysisDialog(win, win)
    ga.dd_N.setCurrentText('2')
    ga.tau_table.cellChanged.disconnect()
    for r, v in enumerate([0.5, 30.0]):
        ga.tau_table.item(r, 0).setText(str(v))
    ga.tau_table.cellChanged.connect(ga._on_tau_edit)
    ga.ed_t0.setValue(0.0)
    ga.cb_t0_fix.setChecked(False)
    ga.ed_fw.setValue(0.1)
    ga.cb_fw_fix.setChecked(False)
    ga.dd_artifact.setCurrentIndex(ga.dd_artifact.findData(mode))
    ga.do_run()
    QtWidgets.QApplication.processEvents()
    return ga


def _export_dads(ga, path):
    saved = (ta_ga.ask_save_path, ta_ga.info_box, ta_ga.warn_box)
    ta_ga.ask_save_path = lambda *a, **k: (path, '')
    ta_ga.info_box = ta_ga.warn_box = lambda *a, **k: None
    try:
        ga.export_dads()
    finally:
        ta_ga.ask_save_path, ta_ga.info_box, ta_ga.warn_box = saved


def artifact_on_export(tmp):
    """Run an artifact-on GA export into ``tmp``; returns (win, ga)."""
    wl, t, D, _, _ = gold._art_synth(ta_core, seed=0)
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, D, 'Loaded: artifact opju', source_dir=tmp)
    ga = _ga(win, 'xpm')
    assert ga._has_art_result()
    _export_dads(ga, os.path.join(tmp, 'DADS_spectra.csv'))
    return win, ga


def test_artifact_opju():
    builder = tox.load_builder()
    assert tox.SLOT_KEYS == [s[0] for s in tox.SLOTS]
    assert 'DADS_artifact' not in tox.SLOT_KEYS
    with tempfile.TemporaryDirectory() as tmp:
        win, ga = artifact_on_export(tmp)
        stage = win.opju_stage

        # 1. staged
        assert {'DADS', 'DADS_kinetics', 'DADS_artifact'} <= set(stage.entries)
        e = stage.entries['DADS_artifact']
        assert e['source'] == 'DADS_artifact_spectra.csv', e
        files = stage.files()
        assert os.path.isfile(files['DADS_artifact'])
        assert set(files) == set(tox.SLOT_KEYS) | {'DADS_artifact'}
        snap, sfiles = stage.snapshot()
        assert os.path.isfile(sfiles['DADS_artifact'])
        shutil.rmtree(snap, ignore_errors=True)

        # 2. builder jobs + prep
        jobs = builder.sheet_jobs(files, CTX)
        assert [j[0] for j in jobs] == ORIGINAL_SHEETS + ['DADS_artifact']
        sheet, path, prep = jobs[-1]
        assert path == files['DADS_artifact'] and prep is builder.prep_wav_series
        arr, labels, _ = prep(path)
        assert arr.shape == (len(win.wavelength), 3), arr.shape
        assert labels == [(1, None, 'Wavelength', 'nm'),
                          (2, u'G1 (σG′)', None, None),
                          (3, u'G2 (σ²G″)', None, None)], labels
        assert builder.OPTIONAL_GRAPHS['DADS_artifact'] == \
            ('TA_DADS', 'DADS_artifact', 'line')
        assert ('TA_DADS', 'DADS', 'line') in builder.GRAPHS
        # artifact amplitudes get a data-driven Y range (not DADS' -35~25)
        a = builder.auto_axes('TA_DADS', arr, 2, 3, None, 'DADS_artifact')
        assert a['y'] != builder.AXES['TA_DADS']['y']
        assert builder.auto_axes('TA_DADS', arr, 2, 3) == builder.AXES['TA_DADS']

        # 3. dataset dialog: extra row, n / n unchanged
        dlg = tox.OpjuDatasetDialog(win, win)
        n = len(tox.SLOTS)
        assert dlg.tbl.rowCount() == n + 1
        assert dlg.tbl.item(n, 0).text() == 'DADS_artifact'
        assert dlg.tbl.item(n, 1).text() == 'Saved'
        n_staged = sum(k in stage.entries for k in tox.SLOT_KEYS)
        assert f'{n_staged} / {n} sheets' in dlg.lbl_summary.text(), \
            dlg.lbl_summary.text()

        # 4. artifact-off export unstages it
        ga.close()
        ga_off = _ga(win, 'off')
        assert not ga_off._has_art_result()
        _export_dads(ga_off, os.path.join(tmp, 'DADS_off.csv'))
        assert 'DADS_artifact' not in stage.entries
        assert stage.entries['DADS']['source'] == 'DADS_off.csv'
        files_off = stage.files()
        assert set(files_off) == set(tox.SLOT_KEYS)
        snap, sfiles_off = stage.snapshot()
        assert set(sfiles_off) == set(tox.SLOT_KEYS)
        shutil.rmtree(snap, ignore_errors=True)
        dlg.refresh()
        assert dlg.tbl.rowCount() == n
        assert f'/ {n} sheets' in dlg.lbl_summary.text()

        # 5. without the key: exactly the original jobs
        jobs_off = builder.sheet_jobs(files_off, CTX)
        assert [j[0] for j in jobs_off] == ORIGINAL_SHEETS
        assert [j[0] for j in builder.sheet_jobs(
            {k: None for k in tox.SLOT_KEYS}, CTX)] == ORIGINAL_SHEETS
        assert ORIGINAL_SHEETS == builder.SHEETS
        explicit_none = dict(files_off, DADS_artifact=None)
        assert [j[0] for j in builder.sheet_jobs(explicit_none, CTX)] \
            == ORIGINAL_SHEETS

        dlg.reject()
        ga_off.close()
        win.close()
        QtWidgets.QApplication.processEvents()
        stage.clear()
    print('[ok] DADS_artifact staging, builder jobs, dialog row, unstage')


TESTS = [test_artifact_opju]


def main():
    for fn in TESTS:
        fn()
    print(f'\nAll {len(TESTS)} artifact opju checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
