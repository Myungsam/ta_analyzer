"""
TA Analyzer — Origin (.opju) export.

Every CSV export that feeds an opju sheet also stages a CSV copy in a
private temp folder (``OpjuStaging``).  "Export OPJU..." in the main
window opens the **opju dataset info** window (``OpjuDatasetDialog``):
it lists which sheets are staged, and "저장" hands the staged files to
``add_function/ta_opju_builder.build`` (Origin 2018 COM) to write the
.opju; "취소" closes it and keeps the staging.  Loading a new TA data
set clears the staging (``set_loaded_data``).
"""
from __future__ import annotations

import atexit
import csv
import datetime
import os
import shutil
import sys
import tempfile
import traceback

from PyQt5 import QtWidgets, QtCore

from ta_widgets import info_box, warn_box, ask_save_path
import ta_core

# (builder key, opju sheet(s), export that stages it)
SLOTS = [
    ('cropped_original', 'Original_Data', 'Export 2D data_original'),
    ('chirp_cropped', 'Data_Chirp_Corrected',
     'Export 2D Data (after chirp correction)'),
    ('chirp', 'Chirp_line', 'Chirp Correction → Save points'),
    ('pinned_kinetics', 'Kinetics', 'Kinetics panel → Export pins'),
    ('pinned_spectra', 'Spectra', 'Spectrum panel → Export pins'),
    ('EADS', 'EADS', 'Global Analysis → Export EADS'),
    ('DADS', 'DADS', 'Global Analysis → Export DADS'),
    ('DADS_kinetics', 'DADS_kinetics', 'Global Analysis → Export DADS'),
    ('EADS_populations', 'EADS_populations',
     'Global Analysis → Export EADS'),
    ('GA_fit', 'Global_Fit', 'Global Analysis → Export Fit'),
    ('GA_residual', 'Residual', 'Global Analysis → Export Resid.'),
]
SLOT_KEYS = [s[0] for s in SLOTS]
OPJU_NAME = 'Transient absorption_filled.opju'


def _builder_dir() -> str:
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'add_function')


def load_builder():
    """Import add_function/ta_opju_builder (needs comtypes).  In the
    frozen exe the modules are bundled at the top level instead."""
    d = _builder_dir()
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
    import ta_opju_builder
    return ta_opju_builder


class OpjuStaging:
    """Latest staged CSV per builder key, kept in a private temp folder
    that is emptied by ``clear()`` and removed when the program exits."""

    def __init__(self):
        self._dir = None
        self.entries: dict[str, dict] = {}
        atexit.register(self._remove_dir)

    def _folder(self) -> str:
        if self._dir is None or not os.path.isdir(self._dir):
            self._dir = tempfile.mkdtemp(prefix='ta_opju_')
        return self._dir

    def _record(self, key, dst, source, note):
        if key not in SLOT_KEYS:
            raise KeyError(key)
        self.entries[key] = {
            'path': dst, 'source': source, 'note': note,
            'time': datetime.datetime.now().strftime('%H:%M:%S')}

    def stage_file(self, key, src, note=''):
        """Copy an exported delimited file as ``<key>.csv`` (comma).
        Tab-separated content (some exports use tabs for non-.csv
        names) is converted; the delimiter is read from the header."""
        dst = os.path.join(self._folder(), f'{key}.csv')
        with open(src, newline='', encoding='utf-8-sig') as fi:
            tabbed = '\t' in fi.readline()
        if not tabbed:
            shutil.copyfile(src, dst)
        else:
            with open(src, newline='', encoding='utf-8-sig') as fi, \
                    open(dst, 'w', newline='', encoding='utf-8') as fo:
                csv.writer(fo).writerows(csv.reader(fi, delimiter='\t'))
        self._record(key, dst, os.path.basename(src), note)

    def stage_matrix(self, key, wavelength, delay, data, source, note=''):
        """Write a 2D ΔA matrix (Format A CSV) — used when the export
        itself went to .xlsx / .tsv."""
        dst = os.path.join(self._folder(), f'{key}.csv')
        ta_core.write_data_file(dst, wavelength, delay, data, ',')
        self._record(key, dst, source, note)

    def snapshot(self):
        """Copy the staged CSVs into a new folder owned by one build, so
        loading new data or re-exporting during the build cannot delete
        or swap its inputs.  Returns (folder, files mapping)."""
        snap = tempfile.mkdtemp(prefix='ta_opju_build_')
        files = {}
        for k in SLOT_KEYS:
            e = self.entries.get(k)
            if e is None:
                files[k] = None
                continue
            files[k] = os.path.join(snap, f'{k}.csv')
            shutil.copyfile(e['path'], files[k])
        return snap, files

    def files(self) -> dict:
        """Builder ``files`` mapping: every key, None when not staged."""
        return {k: (self.entries[k]['path'] if k in self.entries else None)
                for k in SLOT_KEYS}

    def clear(self):
        self.entries.clear()
        self._remove_dir()

    def _remove_dir(self):
        if self._dir and os.path.isdir(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)
        if not (self._dir and os.path.isdir(self._dir)):
            self._dir = None              # else atexit tries again


def stage(app, key, path, note=''):
    """Stage an export; never lets a staging problem break the export."""
    try:
        app.opju_stage.stage_file(key, path, note)
    except Exception:
        traceback.print_exc()


def stage_matrix(app, key, wavelength, delay, data, source, note=''):
    try:
        app.opju_stage.stage_matrix(key, wavelength, delay, data, source,
                                    note)
    except Exception:
        traceback.print_exc()


class _BuildThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    done = QtCore.pyqtSignal(str, list)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, builder, folder, out, snap, files):
        super().__init__()
        self.builder, self.folder, self.out = builder, folder, out
        self.snap, self.files = snap, files

    def run(self):
        com = None
        try:
            import comtypes
            comtypes.CoInitialize()
            com = comtypes
            out, log = self.builder.build(
                self.folder, self.out, files=self.files,
                progress_cb=lambda s, t, m: self.progress.emit(s, t, m))
            self.done.emit(out, list(log))
        except Exception as e:
            self.failed.emit(f'{str(e) or type(e).__name__}\n\n'
                             f'{traceback.format_exc()}')
        finally:
            if com is not None:
                com.CoUninitialize()
            shutil.rmtree(self.snap, ignore_errors=True)


class OpjuDatasetDialog(QtWidgets.QDialog):
    """opju dataset info: staged sheets + 저장 / 취소."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._thread = None
        self.out_path = None
        self.setWindowTitle('opju dataset info')
        self.resize(900, 520)
        lay = QtWidgets.QVBoxLayout(self)

        lay.addWidget(QtWidgets.QLabel(
            'Data staged for the Origin project (latest export of each '
            'item). Staging is cleared when a new TA data set is loaded.'))
        self.tbl = QtWidgets.QTableWidget(len(SLOTS), 5)
        self.tbl.setHorizontalHeaderLabels(
            ['opju sheet', 'Status', 'Exported file', 'Time', 'Source / details'])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.tbl, stretch=1)

        self.lbl_summary = QtWidgets.QLabel()
        lay.addWidget(self.lbl_summary)
        self.prog = QtWidgets.QProgressBar()
        self.prog.setVisible(False)
        lay.addWidget(self.prog)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setVisible(False)
        lay.addWidget(self.log)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.btn_save = QtWidgets.QPushButton('저장')
        self.btn_cancel = QtWidgets.QPushButton('취소')
        self.btn_save.clicked.connect(self.do_save)
        self.btn_cancel.clicked.connect(self.reject)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_cancel)
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        ent = self.app.opju_stage.entries
        for r, (key, sheet, how) in enumerate(SLOTS):
            e = ent.get(key)
            cells = ([sheet, 'Saved', e['source'], e['time'],
                      e['note'] or how] if e else
                     [sheet, '—', '', '', f'not staged — {how}'])
            for c, txt in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(txt)
                if c == 1:
                    it.setForeground(QtCore.Qt.darkGreen if e
                                     else QtCore.Qt.gray)
                self.tbl.setItem(r, c, it)
        self.tbl.resizeColumnsToContents()
        n = len(ent)
        self.lbl_summary.setText(
            f'{n} / {len(SLOTS)} sheets staged.  Sheets without data are '
            'created empty (with an empty graph frame).')
        self.btn_save.setEnabled(n > 0 and self._thread is None)

    # ---- 저장 ----
    def is_building(self) -> bool:
        return self._thread is not None

    def do_save(self):
        self.refresh()                    # e.g. new data loaded meanwhile
        if not self.app.opju_stage.entries:
            info_box(self, 'Nothing staged',
                     'No exported data is staged yet.  Export CSVs first '
                     '(see the table for which export feeds each sheet).')
            return
        try:
            builder = load_builder()
        except Exception as e:
            warn_box(self, 'opju export unavailable',
                     'The Origin export module could not be loaded '
                     '(needs the comtypes package and add_function/).\n\n'
                     f'{e}')
            return
        busy = builder.running_origin()
        if busy:
            warn_box(self, 'Origin is running',
                     f'Origin is running ({", ".join(busy)}).\n'
                     'Save your work, close every Origin window and try '
                     'again.')
            return
        path, _ = ask_save_path(
            self, 'Save Origin project',
            self.app.default_save_path(OPJU_NAME),
            'Origin project (*.opju)')
        if not path:
            return
        if not path.lower().endswith('.opju'):
            path += '.opju'
            # the dialog only confirmed the name without the extension
            if os.path.exists(path) and QtWidgets.QMessageBox.question(
                    self, 'File exists',
                    f'{os.path.basename(path)} already exists.\n'
                    'Overwrite it?') != QtWidgets.QMessageBox.Yes:
                return
        try:
            snap, files = self.app.opju_stage.snapshot()
        except OSError as e:
            warn_box(self, 'opju export failed',
                     f'Could not prepare the staged data:\n{e}')
            return
        self._thread = _BuildThread(builder, os.path.dirname(path), path,
                                    snap, files)
        self._thread.progress.connect(self._on_progress)
        self._thread.done.connect(self._on_done)
        self._thread.failed.connect(self._on_failed)
        self.btn_save.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.prog.setVisible(True)
        self.prog.setValue(0)
        self.log.setVisible(True)
        self.log.clear()
        self._thread.start()

    def _on_progress(self, step, total, msg):
        self.prog.setMaximum(total)
        self.prog.setValue(step)
        self.log.appendPlainText(f'[{step}/{total}] {msg}')

    def _finish(self):
        self._thread.wait()
        self._thread = None
        self.btn_cancel.setEnabled(True)
        self.refresh()

    def _on_done(self, out, log):
        self._finish()
        self.out_path = out
        self.log.appendPlainText('\n'.join(log))
        info_box(self, 'opju saved', f'Saved Origin project:\n{out}')
        self.accept()

    def _on_failed(self, msg):
        self._finish()
        self.log.appendPlainText('\n[error] ' + msg)
        first = next((ln for ln in msg.splitlines() if ln.strip()),
                     'unknown error')
        warn_box(self, 'opju export failed', first)

    def reject(self):
        if self._thread is not None:      # never leave Origin half-built
            return
        super().reject()

    def closeEvent(self, ev):
        if self._thread is not None:
            ev.ignore()
            return
        super().closeEvent(ev)
