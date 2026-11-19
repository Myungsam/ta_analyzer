"""
Tests for editing chirp points from the points table.

Verifies:
    1) Remove Selected deletes exactly the selected rows (any index, several
       at once) and is disabled when nothing is selected.
    2) Selected points are highlighted on the 2D map.
    3) The Delete key on the table removes the selection.
    4) Remove Last / Clear still work, and the point editing / file buttons
       sit directly below the points table (not in the bottom row).
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtTest import QTest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


def _setup():
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import ta_main
    from ta_chirp import ChirpDialog
    rng = np.random.default_rng(3)
    wl = np.linspace(450.0, 650.0, 40)
    t = np.linspace(-0.5, 5.0, 120)
    A = (np.outer(np.exp(-((wl - 550.0) / 60.0) ** 2),
                  np.exp(-0.4 * np.clip(t, 0, None)))
         + 0.005 * rng.standard_normal((40, 120)))
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, A, 'chirp_pts_edit_test.csv')
    dlg = ChirpDialog(win, win)
    return qapp, win, dlg


PTS = np.array([[460.0, 0.10], [500.0, 0.05], [540.0, 0.02],
                [580.0, 0.00], [620.0, -0.01]])


def _select(dlg, rows):
    dlg.tbl.clearSelection()
    sm = dlg.tbl.selectionModel()
    for r in rows:
        sm.select(dlg.tbl.model().index(r, 0),
                  QtCore.QItemSelectionModel.Select
                  | QtCore.QItemSelectionModel.Rows)


def test_remove_selected():
    qapp, win, dlg = _setup()
    win.chirp_pts = PTS.copy()
    dlg.redraw()
    assert dlg.tbl.rowCount() == 5
    assert not dlg.btn_del_sel.isEnabled()
    _select(dlg, [1, 3])
    assert dlg.btn_del_sel.isEnabled()
    assert dlg._sel_scatter is not None
    assert dlg._sel_scatter.get_offsets().shape[0] == 2
    dlg.btn_del_sel.click()
    np.testing.assert_allclose(win.chirp_pts, PTS[[0, 2, 4]])
    assert dlg.tbl.rowCount() == 3
    assert not dlg.btn_del_sel.isEnabled()
    assert dlg._sel_scatter is None
    # Nothing selected -> no-op
    dlg.remove_selected()
    assert win.chirp_pts.shape[0] == 3
    # Middle row via the Delete key
    dlg.show()
    qapp.processEvents()
    _select(dlg, [1])
    dlg.tbl.setFocus()
    QTest.keyClick(dlg.tbl, QtCore.Qt.Key_Delete)
    qapp.processEvents()
    np.testing.assert_allclose(win.chirp_pts, PTS[[0, 4]])
    # Remove Last / Clear unchanged
    dlg.btn_undo.click()
    np.testing.assert_allclose(win.chirp_pts, PTS[[0]])
    dlg.btn_clr.click()
    assert win.chirp_pts.shape[0] == 0
    dlg.close()
    print('[1] Remove Selected / Del key / Remove Last / Clear OK')


def test_buttons_below_table():
    qapp, win, dlg = _setup()
    dlg.show()
    qapp.processEvents()
    tbl_geo = dlg.tbl.geometry()
    tbl_bottom = dlg.tbl.mapTo(dlg, QtCore.QPoint(0, tbl_geo.height())).y()
    tbl_left = dlg.tbl.mapTo(dlg, QtCore.QPoint(0, 0)).x()
    apply_y = dlg.btn_apply.mapTo(dlg, QtCore.QPoint(0, 0)).y()
    kin_top = dlg.canvas_kin.mapTo(dlg, QtCore.QPoint(0, 0)).y()
    for b in (dlg.btn_del_sel, dlg.btn_undo, dlg.btn_clr,
              dlg.btn_save_pts, dlg.btn_load_pts):
        p = b.mapTo(dlg, QtCore.QPoint(0, 0))
        assert tbl_bottom <= p.y() < kin_top, (b.text(), p.y())
        assert p.x() >= tbl_left - 2, (b.text(), p.x())
        assert p.y() < apply_y
    dlg.close()
    print('[2] point buttons sit between the table and the kinetics plot')


def test_remove_after_sorted_load():
    """Load Points sorts by λ; the table must follow the sorted array so a
    selected row removes the point it shows."""
    import tempfile
    import ta_chirp
    qapp, win, dlg = _setup()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'unsorted.csv')
        np.savetxt(path, PTS[[3, 0, 4, 1, 2]], delimiter=',',
                   header='wavelength_nm,t0_ps', comments='')
        orig_open, orig_info = ta_chirp.ask_open_paths, ta_chirp.info_box
        ta_chirp.ask_open_paths = lambda *a, **kw: path
        ta_chirp.info_box = lambda *a, **kw: None
        try:
            dlg.load_pts()
        finally:
            ta_chirp.ask_open_paths, ta_chirp.info_box = orig_open, orig_info
    np.testing.assert_allclose(win.chirp_pts, PTS)
    _select(dlg, [2])
    assert dlg.tbl.item(2, 0).text() == '540.00'
    dlg.remove_selected()
    np.testing.assert_allclose(win.chirp_pts, PTS[[0, 1, 3, 4]])
    shown = [float(dlg.tbl.item(r, 0).text()) for r in range(4)]
    np.testing.assert_allclose(shown, PTS[[0, 1, 3, 4], 0])
    dlg.close()
    print('[3] remove after sorted load maps rows correctly')


if __name__ == '__main__':
    test_remove_after_sorted_load()
    test_remove_selected()
    test_buttons_below_table()
    print('\nAll chirp point edit tests passed.')
