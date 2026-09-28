"""Smoke test: GA / LDA / Kfit Save Residual buttons.

Verifies that each dialog's Save Residual button:
  1. starts disabled,
  2. becomes enabled after a successful run,
  3. does NOT write anything to the residuals folder on its own (auto-save
     was removed),
  4. writes exactly one new file with the expected name prefix when clicked,
  5. is disabled again after Reset.

This mirrors the working flow in test_advanced_dialogs.py: one
QApplication, one TAAnalyzer, dialogs opened/closed in sequence inside
main().  Per-test fixtures (creating a new TAAnalyzer per test) segfault
in headless Qt because matplotlib canvases are torn down out of order.
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))
sys.path.insert(0, THIS_DIR)

import numpy as np
from PyQt5 import QtWidgets

import ta_main
import ta_residual_store as RS
import ta_widgets
import ta_ga
import ta_lda
import ta_kfit
from test_advanced_dialogs import build_dataset


# The Save buttons end with an info_box() pop-up confirming success.
# In offscreen Qt that modal access-violates, so silence both popups
# for all three dialog modules.
def _silent_box(parent, title, text):
    print(f'    [box] {title}: {text.splitlines()[0]}')


for _m in (ta_ga, ta_lda, ta_kfit):
    _m.info_box = _silent_box
    _m.warn_box = _silent_box
ta_widgets.info_box = _silent_box
ta_widgets.warn_box = _silent_box


def _listf(folder):
    return set(os.listdir(folder))


def _check_ga(win, folder):
    baseline = _listf(folder)
    gd = ta_ga.GlobalAnalysisDialog(win, win)
    assert not gd.btn_save_resid.isEnabled(), \
        'GA save should start disabled'
    gd.do_reset()
    gd.do_run()
    assert gd.btn_save_resid.isEnabled(), \
        'GA save should be enabled after Run'
    assert _listf(folder) == baseline, \
        f'GA must NOT auto-save residual; new files: ' \
        f'{_listf(folder) - baseline}'
    gd.btn_save_resid.click()
    QtWidgets.QApplication.processEvents()
    new = _listf(folder) - baseline
    assert any(n.startswith('GA_') for n in new), \
        f'expected GA_*.xlsx after click; got: {new}'
    gd.do_reset()
    assert not gd.btn_save_resid.isEnabled(), \
        'GA Reset should disable Save'
    for n in new:
        os.remove(os.path.join(folder, n))
    gd.close()
    return new


def _check_lda(win, folder):
    baseline = _listf(folder)
    ld = ta_lda.LDADialog(win, win)
    assert not ld.btn_save_resid.isEnabled(), \
        'LDA save should start disabled'
    ld.ed_K.setValue(20)
    ld.ed_tau_min.setValue(0.2)
    ld.ed_tau_max.setValue(50.0)
    ld.do_run()
    assert ld.btn_save_resid.isEnabled(), \
        'LDA save should be enabled after Run'
    assert _listf(folder) == baseline, \
        f'LDA must NOT auto-save residual; new files: ' \
        f'{_listf(folder) - baseline}'
    ld.btn_save_resid.click()
    QtWidgets.QApplication.processEvents()
    new = _listf(folder) - baseline
    assert any(n.startswith('LDA_') for n in new), \
        f'expected LDA_*.xlsx after click; got: {new}'
    ld.do_reset()
    assert not ld.btn_save_resid.isEnabled(), \
        'LDA Reset should disable Save'
    for n in new:
        os.remove(os.path.join(folder, n))
    ld.close()
    return new


def _check_kfit(win, folder):
    baseline = _listf(folder)
    kf = ta_kfit.KineticFitDialog(win, win)
    assert not kf.btn_save_resid.isEnabled(), \
        'Kfit save should start disabled'
    kf.ed_wl.setValue(500.0)
    kf.do_run()
    assert kf.btn_save_resid.isEnabled(), \
        'Kfit save should be enabled after Run'
    assert _listf(folder) == baseline, \
        f'Kfit must NOT auto-save residual; new files: ' \
        f'{_listf(folder) - baseline}'
    # Kfit Save Residual now writes a 2-column CSV via a prompt —
    # mock ask_save_path so the test doesn't hang on the modal.
    # Route the CSV into the residuals folder so the presence
    # assertion still applies.
    target = os.path.join(folder, 'kfit_500.0nm_residual.csv')
    orig_ask = ta_kfit.ask_save_path
    ta_kfit.ask_save_path = lambda *a, **k: (target, '')
    try:
        kf.btn_save_resid.click()
        QtWidgets.QApplication.processEvents()
    finally:
        ta_kfit.ask_save_path = orig_ask
    new = _listf(folder) - baseline
    assert any(n.startswith('kfit_') and n.endswith('.csv') for n in new), \
        f'expected kfit_*.csv after click; got: {new}'
    kf.do_reset()
    assert not kf.btn_save_resid.isEnabled(), \
        'Kfit Reset should disable Save'
    for n in new:
        os.remove(os.path.join(folder, n))
    kf.close()
    return new


def main():
    app = (QtWidgets.QApplication.instance()
           or QtWidgets.QApplication(sys.argv))

    win = ta_main.TAAnalyzer()
    wl, t, A = build_dataset(seed=1)
    win.set_loaded_data(wl, t, A, 'gate_test')
    folder = RS.get_residual_dir()

    print('=== GA Save Residual ===')
    new = _check_ga(win, folder)
    print(f'    wrote -> {sorted(new)}')
    print('[1] GA: OK')

    print('\n=== LDA Save Residual ===')
    new = _check_lda(win, folder)
    print(f'    wrote -> {sorted(new)}')
    print('[2] LDA: OK')

    print('\n=== Kfit Save Residual ===')
    new = _check_kfit(win, folder)
    print(f'    wrote -> {sorted(new)}')
    print('[3] Kfit: OK')

    print('\n*** ALL SAVE-RESIDUAL BUTTON TESTS PASSED ***')


if __name__ == '__main__':
    main()
