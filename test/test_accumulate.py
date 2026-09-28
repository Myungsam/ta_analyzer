"""Smoke test for ta_accumulate.AccumulationDialog.

Verifies:
    - Folder scan finds all TA-extension files
    - Unparseable files are marked with error but not crashed on
    - Shape-mismatched files are flagged
    - _on_accumulate averages only checked, error-free files
    - The averaged data is dispatched to app.set_loaded_data
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use Agg for matplotlib to avoid X server / GPU dependency in tests.
import matplotlib
matplotlib.use('Agg')

from PyQt5 import QtWidgets  # noqa: E402
import ta_core  # noqa: E402


def _make_data(wl, t, seed):
    rng = np.random.default_rng(seed)
    base = np.exp(-t[None, :] / 5.0) * np.exp(-((wl[:, None] - 550) / 40) ** 2)
    return base + 0.02 * rng.standard_normal(base.shape)


class _FakeApp:
    def __init__(self):
        self.last = None

    def t_unit_ax(self): return 'ps'
    def t_unit_txt(self): return 'ps'

    def set_loaded_data(self, wl, t, A, desc, source_dir=''):
        self.last = (wl.copy(), t.copy(), A.copy(), desc)
        self.last_source_dir = source_dir


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    tmp = tempfile.mkdtemp(prefix='ta_accum_')

    wl = np.linspace(400, 700, 24)
    t = np.linspace(-1.0, 20.0, 32)
    for k in range(3):
        A = _make_data(wl, t, seed=k)
        ta_core.write_data_file(
            os.path.join(tmp, f'shot_{k+1}.csv'), wl, t, A, ',')

    # Mismatched-shape file: same layout, different wavelength count
    wl2 = np.linspace(400, 700, 30)
    A2 = _make_data(wl2, t, seed=99)
    ta_core.write_data_file(os.path.join(tmp, 'z_oddshape.csv'),
                            wl2, t, A2, ',')

    # Garbage file that must not crash the parser
    with open(os.path.join(tmp, 'notes.txt'), 'w', encoding='utf-8') as f:
        f.write('this is not a TA file at all')

    from ta_accumulate import AccumulationDialog

    # Skip actual matplotlib rendering — headless matplotlib inside Qt
    # can still stumble on certain build combos; the numerical logic is
    # what we're validating.
    AccumulationDialog._draw_preview = lambda self, row: None
    AccumulationDialog._clear_preview = lambda self, msg='': None

    fake = _FakeApp()
    dlg = AccumulationDialog(None, fake, initial_folder=tmp)

    names = {fi['name']: fi for fi in dlg.files}
    assert 'shot_1.csv' in names
    assert 'shot_2.csv' in names
    assert 'shot_3.csv' in names
    # notes.txt has a TA extension (.txt) so it's listed, but it must be
    # flagged as an error and NOT auto-included.
    if 'notes.txt' in names:
        assert names['notes.txt']['error'] is not None
        assert names['notes.txt']['include'] is False
    # z_oddshape.csv is in the list but flagged as error
    assert names['z_oddshape.csv']['error'] is not None
    assert names['z_oddshape.csv']['include'] is False

    # All 3 valid files should be pre-checked
    incl = [fi for fi in dlg.files if fi['include']]
    assert len(incl) == 3, f'expected 3 checked, got {len(incl)}'

    # Uncheck shot_1 and verify accumulation excludes it
    names['shot_1.csv']['include'] = False
    dlg._on_accumulate()
    assert fake.last is not None
    wl_out, t_out, A_out, desc = fake.last
    assert wl_out.shape == wl.shape
    assert t_out.shape == t.shape
    assert A_out.shape == (len(wl), len(t))
    expected = np.mean(
        [_make_data(wl, t, seed=1), _make_data(wl, t, seed=2)], axis=0)
    assert np.allclose(A_out, expected), 'accumulated mean mismatch'
    assert 'Accumulated 2 files' in desc

    # Re-check everything and verify all-3 average
    fake.last = None
    dlg._set_all(True)
    # Corrupt file should still stay unchecked
    assert names['z_oddshape.csv']['include'] is False
    dlg._on_accumulate()
    assert fake.last is not None
    _, _, A_all, desc_all = fake.last
    expected_all = np.mean(
        [_make_data(wl, t, seed=0), _make_data(wl, t, seed=1),
         _make_data(wl, t, seed=2)], axis=0)
    assert np.allclose(A_all, expected_all)
    assert 'Accumulated 3 files' in desc_all

    print('test_accumulate: OK — parsed', len(dlg.files),
          'files, avg 2/3 & 3/3 verified')


if __name__ == '__main__':
    main()
