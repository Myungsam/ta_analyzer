"""Smoke-test the data extraction half of ta_load_custom.CustomLoadDialog
against the user's sample file.  Uses a real PyQt5 QApplication but never
shows the dialog so this can run headlessly."""
import os
import sys
import numpy as np

# Run Qt offscreen so this works without a display server.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets, QtCore

# Create the QApplication BEFORE importing the dialog module, otherwise
# any widget construction at import time would fail.
_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from ta_load_custom import CustomLoadDialog, _detect_delimiter, _col_letter
import ta_core

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(THIS_DIR, '_5_2nd', '_TA_spectra_Accumulated.csv')


def test_col_letter():
    assert _col_letter(0) == 'A'
    assert _col_letter(25) == 'Z'
    assert _col_letter(26) == 'AA'
    assert _col_letter(701) == 'ZZ'
    print('col-letter mapping OK')


def test_delimiter_detection():
    with open(TEST_FILE, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read(8192)
    delim = _detect_delimiter(text)
    print(f'detected delimiter: {repr(delim)}')
    assert delim == '\t', f'Expected tab, got {repr(delim)}'


def test_full_dialog_with_real_file():
    """Open the dialog (offscreen), feed the sample file, auto-detect, and
    verify the extracted (wl, t, A) match parse_data_file's output."""
    dlg = CustomLoadDialog(file_path=TEST_FILE)
    print(f'numeric_data shape after load: {dlg.numeric_data.shape}')

    # Auto-detect should fill in all three ranges and set stage=DONE.
    dlg._on_auto_detect()
    print(f'auto-detect ranges:')
    print(f'  X: {dlg.x_range}')
    print(f'  Y: {dlg.y_range}')
    print(f'  Z: {dlg.z_range}')
    assert dlg.stage == CustomLoadDialog.STAGE_DONE
    assert dlg.btn_ok.isEnabled()

    wl_got, t_got, A_got = dlg._extract()
    wl_ref, t_ref, A_ref = ta_core.parse_data_file(TEST_FILE)
    print(f'extracted: wl={wl_got.shape}, t={t_got.shape}, A={A_got.shape}')
    print(f'reference: wl={wl_ref.shape}, t={t_ref.shape}, A={A_ref.shape}')

    assert wl_got.shape == wl_ref.shape, 'wl shape mismatch'
    assert t_got.shape == t_ref.shape, 't shape mismatch'
    assert A_got.shape == A_ref.shape, 'A shape mismatch'
    assert np.allclose(wl_got, wl_ref, equal_nan=True), 'wl values mismatch'
    assert np.allclose(t_got, t_ref, equal_nan=True), 't values mismatch'
    nan_both = np.isnan(A_got) & np.isnan(A_ref)
    diff = np.where(~nan_both, np.abs(A_got - A_ref), 0.0)
    max_diff = float(np.nanmax(diff))
    print(f'max |A_got - A_ref|: {max_diff}')
    assert max_diff < 1e-6, 'A values diverge from parse_data_file reference'
    print('auto-detect equivalence: PASS')

    dlg.close()


def test_manual_selection_sequence():
    """Simulate the click sequence: confirm X (column), confirm Y (row),
    confirm Z, then accept.  Verifies the per-stage validation accepts
    valid inputs and produces the right matrix."""
    dlg = CustomLoadDialog(file_path=TEST_FILE)
    n_rows, n_cols = dlg.numeric_data.shape
    num = dlg.numeric_data

    # Find columns/rows that hold the real data (this file is Format B).
    # wl column = first column where row 1 has a number (col 0 is metadata text).
    # t column start = wl_col + 1 (skip the numeric "corner" placeholder
    # at row 0).
    wl_col = next(j for j in range(n_cols) if np.isfinite(num[1, j]))
    t_col_start = wl_col + 1
    t_col_end = n_cols - 1
    while t_col_end > t_col_start and not np.isfinite(num[0, t_col_end]):
        t_col_end -= 1

    # Find the last wavelength row that is numeric.
    wl_row_end = n_rows - 1
    while wl_row_end > 1 and not np.isfinite(num[wl_row_end, wl_col]):
        wl_row_end -= 1

    # ---- STAGE X: select wavelength column ----
    sel_x = QtWidgets.QTableWidgetSelectionRange(
        1, wl_col, wl_row_end, wl_col)
    dlg.table.clearSelection()
    dlg.table.setRangeSelected(sel_x, True)
    dlg._on_confirm_selection()
    assert dlg.stage == CustomLoadDialog.STAGE_Y, \
        f'expected stage Y after X confirm, got {dlg.stage}'
    assert dlg.x_range == (1, wl_col, wl_row_end, wl_col)

    # ---- STAGE Y: select delay row ----
    sel_y = QtWidgets.QTableWidgetSelectionRange(
        0, t_col_start, 0, t_col_end)
    dlg.table.clearSelection()
    dlg.table.setRangeSelected(sel_y, True)
    dlg._on_confirm_selection()
    assert dlg.stage == CustomLoadDialog.STAGE_Z, \
        f'expected stage Z after Y confirm, got {dlg.stage}'

    # ---- STAGE Z: select the matrix ----
    sel_z = QtWidgets.QTableWidgetSelectionRange(
        1, t_col_start, wl_row_end, t_col_end)
    dlg.table.clearSelection()
    dlg.table.setRangeSelected(sel_z, True)
    dlg._on_confirm_selection()
    assert dlg.stage == CustomLoadDialog.STAGE_DONE, \
        f'expected DONE after Z confirm, got {dlg.stage}'

    wl_got, t_got, A_got = dlg._extract()
    wl_ref, t_ref, A_ref = ta_core.parse_data_file(TEST_FILE)
    assert wl_got.shape == wl_ref.shape
    assert t_got.shape == t_ref.shape
    assert A_got.shape == A_ref.shape
    print('manual sequence (X col, Y row, Z block): PASS')

    dlg.close()


def test_dimension_mismatch_rejected():
    """If the user marks a Z block with wrong dimensions, the dialog must
    refuse to advance and keep stage=Z."""
    dlg = CustomLoadDialog(file_path=TEST_FILE)
    n_rows, n_cols = dlg.numeric_data.shape

    # Set X and Y manually (bypass UI) so we can drive stage straight to Z
    dlg.x_range = (1, 0, 5, 0)         # 5 wavelengths
    dlg.y_range = (0, 1, 0, 10)        # 10 delays
    dlg.stage = CustomLoadDialog.STAGE_Z

    # Confirm Z with the wrong shape (3 x 7 instead of 5 x 10 / 10 x 5)
    bad = QtWidgets.QTableWidgetSelectionRange(1, 1, 3, 7)
    dlg.table.clearSelection()
    dlg.table.setRangeSelected(bad, True)

    # Replace warn_box temporarily so it doesn't pop a real message box
    import ta_load_custom as _mod
    _warned = []
    orig = _mod.warn_box
    _mod.warn_box = lambda *a, **k: _warned.append(a)
    try:
        dlg._on_confirm_selection()
    finally:
        _mod.warn_box = orig

    assert dlg.stage == CustomLoadDialog.STAGE_Z, \
        f'stage should stay at Z, got {dlg.stage}'
    assert dlg.z_range is None
    assert _warned, 'expected a warning for dimension mismatch'
    print('dimension-mismatch rejection: PASS')

    dlg.close()


def test_1d_constraint_rejects_2d():
    """Picking a 2D block for X should be rejected."""
    dlg = CustomLoadDialog(file_path=TEST_FILE)
    # Force stage to X
    dlg.stage = CustomLoadDialog.STAGE_X

    # Select a 3 x 4 block — clearly not 1D
    bad = QtWidgets.QTableWidgetSelectionRange(1, 0, 3, 3)
    dlg.table.clearSelection()
    dlg.table.setRangeSelected(bad, True)

    import ta_load_custom as _mod
    _warned = []
    orig = _mod.warn_box
    _mod.warn_box = lambda *a, **k: _warned.append(a)
    try:
        dlg._on_confirm_selection()
    finally:
        _mod.warn_box = orig

    assert dlg.stage == CustomLoadDialog.STAGE_X
    assert dlg.x_range is None
    assert _warned
    print('2D rejection for X stage: PASS')
    dlg.close()


def test_z_transpose_orientation():
    """If the user selects Z as (Y_len x X_len) instead of (X_len x Y_len),
    _extract should auto-transpose so the result is (M wavelengths x N
    delays)."""
    dlg = CustomLoadDialog()   # no file
    # Inject a tiny synthetic numeric matrix so we don't need a file.
    dlg.numeric_data = np.array([
        [np.nan, 400.0, 410.0],   # row 0
        [10.0,   1.0,   4.0],     # row 1
        [20.0,   2.0,   5.0],     # row 2
        [30.0,   3.0,   6.0],     # row 3
    ])
    dlg.x_range = (0, 1, 0, 2)       # X (wavelengths) along row 0, 2 pts
    dlg.y_range = (1, 0, 3, 0)       # Y (delays) along col 0, 3 pts
    dlg.z_range = (1, 1, 3, 2)       # 3x2 block: needs transpose
    dlg.stage = CustomLoadDialog.STAGE_DONE
    wl, t, A = dlg._extract()
    assert A.shape == (2, 3), f'A shape {A.shape}'
    np.testing.assert_allclose(A, [[1, 2, 3], [4, 5, 6]])
    print('Z transpose orientation: PASS')
    dlg.close()


def _make_dlg_with_grid(grid):
    """Create a CustomLoadDialog with no file but a custom numeric grid
    (numpy 2D array of floats; NaN = blank).  Also populates the visible
    table so cursor moves are valid."""
    from PyQt5 import QtCore
    dlg = CustomLoadDialog()
    grid = np.asarray(grid, dtype=float)
    dlg.numeric_data = grid
    n_rows, n_cols = grid.shape
    dlg.table.setRowCount(n_rows)
    dlg.table.setColumnCount(n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            v = grid[i, j]
            text = '' if not np.isfinite(v) else f'{v:g}'
            item = QtWidgets.QTableWidgetItem(text)
            dlg.table.setItem(i, j, item)
    return dlg


def test_data_edge_target_basic():
    """Walk through the four Excel rules on a small 2D grid."""
    NaN = np.nan
    grid = np.array([
        [1.0, 2.0, 3.0, NaN, 5.0, 6.0, NaN, 8.0],   # row 0
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],   # row 1 (all blank)
        [1.0, 2.0, NaN, NaN, NaN, NaN, NaN, 8.0],   # row 2
    ])
    dlg = _make_dlg_with_grid(grid)
    KR = QtCore.Qt.Key_Right
    KL = QtCore.Qt.Key_Left
    KD = QtCore.Qt.Key_Down
    KU = QtCore.Qt.Key_Up

    # On data, right neighbor is data: go to last in run before NaN.
    # Row 0 starts data 0..2, then NaN at 3. From (0,0) RIGHT -> (0,2).
    assert dlg._data_edge_target(0, 0, KR) == (0, 2)
    # From (0,2) RIGHT (data; next is NaN) -> next data block start = (0,4).
    assert dlg._data_edge_target(0, 2, KR) == (0, 4)
    # From (0,4) RIGHT -> last in run before NaN at col 6 -> (0,5).
    assert dlg._data_edge_target(0, 4, KR) == (0, 5)
    # From (0,5) RIGHT -> next data after NaN -> (0,7) (last column data).
    assert dlg._data_edge_target(0, 5, KR) == (0, 7)
    # From (0,7) RIGHT -> at right edge, nothing more -> stay (0,7).
    assert dlg._data_edge_target(0, 7, KR) == (0, 7)

    # From (0,7) LEFT (data; left neighbor (0,6) is NaN) -> next data left
    # is (0,5).
    assert dlg._data_edge_target(0, 7, KL) == (0, 5)

    # From a blank cell: go to next data in direction.
    # (1,0) is blank; RIGHT scans row 1 for data, none -> edge (1, n_cols-1).
    assert dlg._data_edge_target(1, 0, KR) == (1, grid.shape[1] - 1)

    # Vertical: from (0,0) DOWN -> data; (1,0) is NaN -> next data at (2,0).
    assert dlg._data_edge_target(0, 0, KD) == (2, 0)
    # From (2,0) DOWN -> nothing more -> stay (2, 0) since (3,0) out of bounds.
    assert dlg._data_edge_target(2, 0, KD) == (2, 0)
    # From (2,7) UP -> data; (1,7) is NaN -> next data up at (0,7).
    assert dlg._data_edge_target(2, 7, KU) == (0, 7)

    print('data-edge target rules: PASS')


def test_extend_to_data_edge():
    """Verify that Ctrl+Shift+Right from anchor (0,0) selects A1:<edge>."""
    grid = np.array([
        [1.0, 2.0, 3.0, np.nan, 5.0],
        [4.0, 5.0, 6.0, np.nan, 7.0],
    ])
    dlg = _make_dlg_with_grid(grid)
    dlg._sel_anchor = (0, 0)
    dlg.table.setCurrentCell(0, 0)
    dlg._extend_to_data_edge(QtCore.Qt.Key_Right)
    ranges = dlg.table.selectedRanges()
    assert len(ranges) == 1
    r1, c1 = ranges[0].topRow(), ranges[0].leftColumn()
    r2, c2 = ranges[0].bottomRow(), ranges[0].rightColumn()
    # From (0,0) RIGHT should land on (0,2) (last data before NaN at col 3)
    assert (r1, c1, r2, c2) == (0, 0, 0, 2), \
        f'expected A1:C1, got ({r1},{c1})..({r2},{c2})'
    assert dlg.table.currentRow() == 0
    assert dlg.table.currentColumn() == 2
    print('Ctrl+Shift+Right extend from (0,0): PASS')

    # Continue: Ctrl+Shift+Right again should now jump past NaN to col 4
    dlg._extend_to_data_edge(QtCore.Qt.Key_Right)
    rng = dlg.table.selectedRanges()[0]
    assert (rng.topRow(), rng.leftColumn(),
            rng.bottomRow(), rng.rightColumn()) == (0, 0, 0, 4)
    print('Ctrl+Shift+Right second press: PASS')

    # Now extend down from anchor (0,0); current is now (0,4) so first
    # do Ctrl+Shift+Down: anchor stays at (0,0), current goes down. The
    # grid has only 2 rows, so target is (1, 4).
    dlg._extend_to_data_edge(QtCore.Qt.Key_Down)
    rng = dlg.table.selectedRanges()[0]
    # Selection bbox: anchor (0,0) to target (1,4)
    assert (rng.topRow(), rng.leftColumn(),
            rng.bottomRow(), rng.rightColumn()) == (0, 0, 1, 4)
    print('Ctrl+Shift+Down after horizontal extend: PASS')


def test_extend_on_real_file_full_column():
    """The user's typical workflow: click A2, then Ctrl+Shift+Down to
    select the whole wavelength column."""
    dlg = CustomLoadDialog(file_path=TEST_FILE)
    n_rows, n_cols = dlg.numeric_data.shape
    # Wavelength column = first column where row 1 is numeric (col 1 here)
    wl_col = next(j for j in range(n_cols)
                  if np.isfinite(dlg.numeric_data[1, j]))
    # Simulate plain click at (1, wl_col) -> sets anchor & current
    dlg._on_table_cell_pressed(1, wl_col)
    dlg.table.setCurrentCell(1, wl_col)
    # Ctrl+Shift+Down
    dlg._extend_to_data_edge(QtCore.Qt.Key_Down)
    rng = dlg.table.selectedRanges()[0]
    bottom = rng.bottomRow()
    # Should land on the last numeric row (n_rows-1 since the file's wl
    # column is fully numeric).
    assert bottom == n_rows - 1, \
        f'expected to reach last row {n_rows-1}, got {bottom}'
    assert rng.topRow() == 1
    assert rng.leftColumn() == wl_col and rng.rightColumn() == wl_col
    print('Ctrl+Shift+Down across full wavelength column: PASS')
    dlg.close()


if __name__ == '__main__':
    test_col_letter()
    test_delimiter_detection()
    test_full_dialog_with_real_file()
    test_manual_selection_sequence()
    test_dimension_mismatch_rejected()
    test_1d_constraint_rejects_2d()
    test_z_transpose_orientation()
    test_data_edge_target_basic()
    test_extend_to_data_edge()
    test_extend_on_real_file_full_column()
    print('\nAll tests passed.')
