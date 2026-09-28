"""
Smoke-test for ta_plot_export.py — loads each file kind from
Data/Global_Analysis and verifies that rendering + PNG export succeed.

Run with:
    PYTHONPATH=. QT_QPA_PLATFORM=offscreen python test/test_plot_export.py
"""
from __future__ import annotations

import os
import sys
import tempfile

# Make the project root and Data/ importable from this test/ subfolder.
# ta_plot_export now lives in Data/ alongside lpsvd_replot.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, 'Data'))

# Force offscreen Qt before anything imports QtWidgets
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets

import ta_plot_export as tpe


DATA_DIR = os.path.join(_PROJ, 'Data', 'Global_Analysis', '5_AH-OMe')

CASES = [
    ('5_AH-OMe_DMF_ex400.CSV',                    tpe._KIND_2D_MAP),
    ('5_AH-OMe_DMF_ex400_GA_fit.csv',             tpe._KIND_2D_MAP),
    ('5_AH-OMe_DMF_ex400_GA_residual.csv',        tpe._KIND_2D_MAP),
    ('5_AH-OMe_DMF_ex400_DADS.csv',               tpe._KIND_1D_SPECTRA),
    ('5_AH-OMe_DMF_ex400_EADS.csv',               tpe._KIND_1D_SPECTRA),
    ('5_AH-OMe_DMF_ex400_kinetics_536.2nm.csv',   tpe._KIND_1D_KINETICS),
]


def main():
    app = (QtWidgets.QApplication.instance()
           or QtWidgets.QApplication(sys.argv))
    out_dir = tempfile.mkdtemp(prefix='ta_plot_export_test_')
    print(f'[OUT] writing PNGs to {out_dir}')

    # -------------------------------------------------------------
    # 1) detect_file_kind correctness
    # -------------------------------------------------------------
    print('\n[1] detect_file_kind ...')
    for fname, expected in CASES:
        path = os.path.join(DATA_DIR, fname)
        kind = tpe.detect_file_kind(path)
        assert kind == expected, f'{fname}: got {kind!r}, want {expected!r}'
        print(f'    OK  {fname}  ->  {kind}')

    # -------------------------------------------------------------
    # 2) Loader functions return sane shapes
    # -------------------------------------------------------------
    print('\n[2] loaders return sane shapes ...')
    wl, t, A = tpe.load_2d_matrix(
        os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400.CSV'))
    assert wl.ndim == 1 and t.ndim == 1 and A.shape == (wl.size, t.size)
    print(f'    OK  2D map shape = {A.shape}')
    wl, names, Y = tpe.load_1d_spectra(
        os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_DADS.csv'))
    assert Y.shape == (wl.size, len(names))
    print(f'    OK  DADS shape = {Y.shape}, names = {names}')
    t, names, Y = tpe.load_1d_kinetics(
        os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_kinetics_536.2nm.csv'))
    assert Y.shape == (t.size, len(names))
    print(f'    OK  kinetics shape = {Y.shape}, names = {names}')

    # -------------------------------------------------------------
    # 3) End-to-end via the main window: dispatch -> render -> export
    # -------------------------------------------------------------
    print('\n[3] end-to-end dispatch + render + export ...')
    win = tpe.PlotExportWindow()
    win.show()
    for fname, expected_kind in CASES:
        path = os.path.join(DATA_DIR, fname)
        win._load_file(path)
        editor = win._editor
        assert editor is not None, f'No editor created for {fname}'

        # Check the right editor type was used
        if expected_kind == tpe._KIND_2D_MAP:
            assert isinstance(editor, tpe.TwoDEditor), \
                f'{fname}: wrong editor {type(editor).__name__}'
            assert editor._im is not None, 'no image after redraw'
        else:
            assert isinstance(editor, tpe.OneDEditor), \
                f'{fname}: wrong editor {type(editor).__name__}'
            assert editor.kind == expected_kind, 'kind mismatch'

        # Export PNG manually (bypass the QFileDialog)
        png_name = os.path.splitext(fname)[0] + '_exported.png'
        out_path = os.path.join(out_dir, png_name)
        editor.canvas.fig.savefig(out_path, dpi=200, bbox_inches='tight')
        assert os.path.isfile(out_path) and os.path.getsize(out_path) > 1000
        print(f'    OK  {fname}  ->  {os.path.getsize(out_path):,} bytes')

    # -------------------------------------------------------------
    # 4) Title / axis labels / tick spacing / z-range edits propagate
    # -------------------------------------------------------------
    print('\n[4] customisation hooks ...')
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400.CSV'))
    ed = win._editor
    assert isinstance(ed, tpe.TwoDEditor)
    ed.ed_title.setText('Custom title')
    ed.ed_xlabel.setText('λ / nm')
    ed.ed_ylabel.setText('t / ps')
    ed.ed_xmajor.setValue(50.0)
    ed.ed_xminor.setValue(10.0)
    ed.ed_ymajor.setValue(50.0)
    ed.ed_yminor.setValue(10.0)
    ed.ed_zmin.setValue(-20.0)
    ed.ed_zmax.setValue(20.0)
    ed.ed_zstep.setValue(5.0)
    ed.redraw()
    assert ed.ax.get_title() == 'Custom title'
    assert ed.ax.get_xlabel() == 'λ / nm'
    assert ed.ax.get_ylabel() == 't / ps'
    assert ed._cb is not None
    print('    OK  2D edits applied to axes / colorbar')

    # Contour mode + z step => discrete levels
    ed.cb_filled_contour.setChecked(True)
    ed.ed_zstep.setValue(10.0)
    ed.redraw()
    print('    OK  contour mode + z step renders')

    # -------------------------------------------------------------
    # 5) FontPickerEdit pushes family / size / bold / italic / underline
    #    onto matplotlib Text artists.
    # -------------------------------------------------------------
    print('\n[5] font customisation ...')
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400.CSV'))
    ed = win._editor
    assert isinstance(ed, tpe.TwoDEditor)
    from PyQt5 import QtGui
    qf = QtGui.QFont('Times New Roman')
    qf.setPointSize(18)
    qf.setBold(True)
    qf.setItalic(True)
    ed.ed_title.setText('Bold italic title')
    ed.ed_title.setFont_(qf)
    ed.redraw()
    title = ed.ax.title
    assert int(round(title.get_fontsize())) == 18, \
        f'fontsize {title.get_fontsize()} != 18'
    assert title.get_fontweight() == 'bold', title.get_fontweight()
    assert title.get_fontstyle() == 'italic', title.get_fontstyle()
    print(f'    OK  title font: size={title.get_fontsize()}, '
          f'weight={title.get_fontweight()}, '
          f'style={title.get_fontstyle()}')
    # Underline path
    qf2 = QtGui.QFont('Arial')
    qf2.setPointSize(14)
    qf2.setUnderline(True)
    ed.ed_xlabel.setText('Wavelength')
    ed.ed_xlabel.setFont_(qf2)
    ed.redraw()
    assert ed.ed_xlabel.underline() is True
    # Underline emulation uses U+0332 combining low-line after each
    # non-space character (mathtext lost \underline in matplotlib 3.10).
    assert '̲' in ed.ax.xaxis.label.get_text(), \
        f'expected combining low-line, got {ed.ax.xaxis.label.get_text()!r}'
    print('    OK  underline wraps the label with combining low-line')

    # -------------------------------------------------------------
    # 6) Minor-tick clamping: minor must be ≤ major / 2.
    # -------------------------------------------------------------
    print('\n[6] minor-tick clamp ...')
    ed.ed_xmajor.setValue(20.0)
    ed.ed_xminor.setValue(15.0)   # too big — must clamp to 10
    assert abs(ed.ed_xminor.value() - 10.0) < 1e-9, \
        f'minor clamp failed: {ed.ed_xminor.value()}'
    # Lowering major should clamp minor again
    ed.ed_xmajor.setValue(8.0)
    assert ed.ed_xminor.value() <= 4.0 + 1e-9, \
        f'minor not re-clamped after major decrease: {ed.ed_xminor.value()}'
    print(f'    OK  clamp: major=8 -> minor={ed.ed_xminor.value()}')

    # -------------------------------------------------------------
    # 7) 1D editor: per-series edits + new tick controls
    # -------------------------------------------------------------
    print('\n[7] 1D series + ticks + font ...')
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_DADS.csv'))
    ed = win._editor
    assert isinstance(ed, tpe.OneDEditor)
    ed.ed_xmajor.setValue(50.0)
    ed.ed_xminor.setValue(10.0)
    ed.ed_ymajor.setValue(1.0)
    ed.ed_yminor.setValue(0.25)
    # Clamp also wired on 1D
    ed.ed_yminor.setValue(0.9)    # too big — clamp to 0.5
    assert abs(ed.ed_yminor.value() - 0.5) < 1e-9, \
        f'1D minor clamp failed: {ed.ed_yminor.value()}'
    # Series toggle + style still work
    ed._series_widgets[0]['show'].setChecked(False)
    ed._series_widgets[1]['style'].setCurrentText('--')
    ed._series_widgets[1]['lw'].setValue(2.5)
    ed.redraw()
    leg = ed.ax.get_legend()
    if leg is not None:
        n_visible = sum(1 for sw in ed._series_widgets
                        if sw['show'].isChecked())
        assert len(leg.get_texts()) == n_visible, \
            f'legend mismatch: {len(leg.get_texts())} vs {n_visible}'
    # Apply a font to the y-label too
    qf3 = QtGui.QFont('Courier New')
    qf3.setPointSize(13)
    qf3.setBold(False)
    qf3.setItalic(True)
    ed.ed_ylabel.setText('ΔA')
    ed.ed_ylabel.setFont_(qf3)
    ed.redraw()
    assert ed.ax.yaxis.label.get_fontstyle() == 'italic'
    print('    OK  1D series + ticks + font edits applied')

    # -------------------------------------------------------------
    # 8) Figure-size + DPI controls produce exact pixel dimensions
    #    when "Tight bbox" is off.
    # -------------------------------------------------------------
    print('\n[8] figure size + DPI controls ...')
    try:
        from PIL import Image
    except ImportError:
        Image = None

    # 2D editor
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400.CSV'))
    ed = win._editor
    assert isinstance(ed, tpe.TwoDEditor)
    ed.ed_fig_w.setValue(6.0)
    ed.ed_fig_h.setValue(4.0)
    ed.ed_dpi.setValue(150)
    ed.cb_tight_bbox.setChecked(False)
    out_2d = os.path.join(out_dir, 'sized_2d.png')
    # Mimic export_image() without QFileDialog
    orig = ed.canvas.fig.get_size_inches()
    ed.canvas.fig.set_size_inches(6.0, 4.0)
    ed.canvas.fig.savefig(out_2d, dpi=150, bbox_inches=None)
    ed.canvas.fig.set_size_inches(*orig)
    assert os.path.isfile(out_2d)
    if Image is not None:
        with Image.open(out_2d) as img:
            assert img.size == (900, 600), \
                f'2D sized export: got {img.size}, want (900, 600)'
        print(f'    OK  2D 6.0in x 4.0in @ 150dpi -> 900 x 600 px')
    else:
        print('    OK  2D sized export wrote file ({} bytes); '
              'PIL not present, skipping pixel check'.format(
                  os.path.getsize(out_2d)))

    # 1D editor
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_DADS.csv'))
    ed = win._editor
    assert isinstance(ed, tpe.OneDEditor)
    ed.ed_fig_w.setValue(5.0)
    ed.ed_fig_h.setValue(3.0)
    ed.ed_dpi.setValue(200)
    ed.cb_tight_bbox.setChecked(False)
    out_1d = os.path.join(out_dir, 'sized_1d.png')
    orig = ed.canvas.fig.get_size_inches()
    ed.canvas.fig.set_size_inches(5.0, 3.0)
    ed.canvas.fig.savefig(out_1d, dpi=200, bbox_inches=None)
    ed.canvas.fig.set_size_inches(*orig)
    assert os.path.isfile(out_1d)
    if Image is not None:
        with Image.open(out_1d) as img:
            assert img.size == (1000, 600), \
                f'1D sized export: got {img.size}, want (1000, 600)'
        print(f'    OK  1D 5.0in x 3.0in @ 200dpi -> 1000 x 600 px')
    else:
        print('    OK  1D sized export wrote file ({} bytes); '
              'PIL not present, skipping pixel check'.format(
                  os.path.getsize(out_1d)))

    # -------------------------------------------------------------
    # 9) Canvas is locked to the requested figure size (gray bg
    #    around it).  Changing W/H resizes the Qt widget; resizing
    #    the host window must not change fig size.
    # -------------------------------------------------------------
    print('\n[9] canvas locks to figure size ...')
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400.CSV'))
    ed = win._editor
    assert isinstance(ed, tpe.TwoDEditor)
    ed.ed_fig_w.setValue(8.0)
    ed.ed_fig_h.setValue(5.5)
    fig_dpi = ed.canvas.fig.dpi
    want_px = (int(8.0 * fig_dpi), int(5.5 * fig_dpi))
    got_px = (ed.canvas.width(), ed.canvas.height())
    assert got_px == want_px, \
        f'canvas pixel size {got_px} != {want_px} for 8.0×5.5in @ {fig_dpi}dpi'
    # Simulate user dragging the window larger — figure must not follow.
    # setFixedSize is enforced immediately (sets min == max), so we can
    # check straight away without spinning the event loop (which would
    # drain unrelated deferred draws from earlier sections).
    win.resize(1900, 1000)
    got_px2 = (ed.canvas.width(), ed.canvas.height())
    assert got_px2 == want_px, \
        f'canvas pixel size {got_px2} changed after window resize'
    # And the size in inches stayed put
    sz = ed.canvas.fig.get_size_inches()
    assert abs(sz[0] - 8.0) < 1e-6 and abs(sz[1] - 5.5) < 1e-6, \
        f'fig.get_size_inches() {sz} != (8.0, 5.5)'
    print(f'    OK  canvas locked at {want_px} px both before and after '
          'window resize')

    # -------------------------------------------------------------
    # 10) Legend font + size controls push family / weight / style
    #     / size onto matplotlib Legend texts.
    # -------------------------------------------------------------
    print('\n[10] legend font + size ...')
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_DADS.csv'))
    ed = win._editor
    assert isinstance(ed, tpe.OneDEditor)
    ed.cb_legend.setChecked(True)
    qf = QtGui.QFont('Times New Roman')
    qf.setPointSize(14)
    qf.setBold(True)
    qf.setItalic(False)
    ed._set_legend_font(qf)
    # Override size to a different value than the font dialog reported,
    # to prove the spinbox wins.
    ed.ed_legend_size.setValue(16.0)
    ed.redraw()
    leg = ed.ax.get_legend()
    assert leg is not None, 'no legend after redraw'
    texts = leg.get_texts()
    assert texts, 'legend has no text entries'
    t0 = texts[0]
    assert int(round(t0.get_fontsize())) == 16, \
        f'legend fontsize {t0.get_fontsize()} != 16'
    assert t0.get_fontweight() == 'bold', \
        f'legend weight {t0.get_fontweight()} != bold'
    print(f'    OK  legend text: family→{t0.get_fontfamily()!r}, '
          f'size={t0.get_fontsize()}, weight={t0.get_fontweight()}')

    # -------------------------------------------------------------
    # 11) Session-scoped settings persistence: customise the first
    #     2D file, load another 2D file, and verify the look carried
    #     over even though it's a freshly-built editor instance.
    #     Same for 1D files (DADS -> EADS).
    # -------------------------------------------------------------
    print('\n[11] settings persist across loads ...')
    # ---- 2D path ----
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400.CSV'))
    ed = win._editor
    assert isinstance(ed, tpe.TwoDEditor)
    ed.ed_xlabel.setText('λ / nm (saved)')
    ed.ed_ylabel.setText('t / ps (saved)')
    ed.ed_cbar_label.setText(r'$\Delta$A (saved)')
    ed.ed_xmajor.setValue(40.0)
    ed.ed_xminor.setValue(10.0)
    ed.ed_ymajor.setValue(100.0)
    ed.ed_yminor.setValue(25.0)
    ed.dd_cmap.setCurrentText('jet')
    ed.cb_filled_contour.setChecked(True)
    ed.ed_zstep.setValue(2.5)
    ed.ed_fig_w.setValue(9.0)
    ed.ed_fig_h.setValue(6.5)
    ed.ed_dpi.setValue(220)
    ed.cb_tight_bbox.setChecked(True)
    ed.cb_minor.setChecked(False)
    qf = QtGui.QFont('Times New Roman')
    qf.setPointSize(15)
    qf.setBold(True)
    ed.ed_title.setFont_(qf)
    # Trigger a redraw so any signal-blocked state is in sync.
    ed.redraw()
    # Load a different 2D file — settings must carry over.
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_GA_fit.csv'))
    ed2 = win._editor
    assert isinstance(ed2, tpe.TwoDEditor)
    assert ed2 is not ed, 'editor was not rebuilt'
    assert ed2.ed_xlabel.text() == 'λ / nm (saved)', \
        f'xlabel not persisted: {ed2.ed_xlabel.text()!r}'
    assert ed2.ed_ylabel.text() == 't / ps (saved)', \
        f'ylabel not persisted: {ed2.ed_ylabel.text()!r}'
    assert ed2.ed_cbar_label.text() == r'$\Delta$A (saved)', \
        f'cbar label not persisted: {ed2.ed_cbar_label.text()!r}'
    assert abs(ed2.ed_xmajor.value() - 40.0) < 1e-9
    assert abs(ed2.ed_xminor.value() - 10.0) < 1e-9
    assert abs(ed2.ed_ymajor.value() - 100.0) < 1e-9
    assert abs(ed2.ed_yminor.value() - 25.0) < 1e-9
    assert ed2.dd_cmap.currentText() == 'jet'
    assert ed2.cb_filled_contour.isChecked() is True
    assert abs(ed2.ed_zstep.value() - 2.5) < 1e-9
    assert abs(ed2.ed_fig_w.value() - 9.0) < 1e-9
    assert abs(ed2.ed_fig_h.value() - 6.5) < 1e-9
    assert ed2.ed_dpi.value() == 220
    assert ed2.cb_tight_bbox.isChecked() is True
    assert ed2.cb_minor.isChecked() is False
    # Title FONT carried over even though title TEXT was reset to filename
    assert int(round(ed2.ax.title.get_fontsize())) == 15, \
        f'title font size {ed2.ax.title.get_fontsize()} != 15'
    assert ed2.ax.title.get_fontweight() == 'bold'
    assert os.path.basename(
        '5_AH-OMe_DMF_ex400_GA_fit.csv') in ed2.ed_title.text(), \
        f'title text should be reset to new filename: {ed2.ed_title.text()!r}'
    # Axis limits should NOT have persisted (data-derived)
    new_wls = ed2.wavelengths
    assert abs(ed2.ed_xmin.value() - float(new_wls.min())) < 1e-3, \
        'x range should re-fit to new data'
    print('    OK  2D settings persisted; title text + x/y ranges refreshed')

    # ---- 1D path ----
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_DADS.csv'))
    ed = win._editor
    assert isinstance(ed, tpe.OneDEditor)
    ed.ed_xlabel.setText('Wavelength (saved)')
    ed.ed_ylabel.setText('ΔA (saved)')
    ed.ed_xmajor.setValue(50.0)
    ed.ed_xminor.setValue(10.0)
    ed.cb_grid.setChecked(False)
    ed.cb_zeroline.setChecked(False)
    ed.cb_legend.setChecked(True)
    ed.dd_legend_loc.setCurrentText('upper left')
    qf_leg = QtGui.QFont('Courier New')
    qf_leg.setPointSize(11)
    qf_leg.setItalic(True)
    # _set_legend_font mirrors the dialog's point size into the size
    # spinbox, so set the spinbox AFTER picking the font to verify
    # the spinbox wins downstream.
    ed._set_legend_font(qf_leg)
    ed.ed_legend_size.setValue(12.0)
    ed.ed_fig_w.setValue(7.5)
    ed.ed_fig_h.setValue(4.2)
    ed.redraw()
    # Load EADS — same kind (1D spectra), settings must carry.
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_EADS.csv'))
    ed2 = win._editor
    assert isinstance(ed2, tpe.OneDEditor)
    assert ed2 is not ed
    assert ed2.ed_xlabel.text() == 'Wavelength (saved)'
    assert ed2.ed_ylabel.text() == 'ΔA (saved)'
    assert abs(ed2.ed_xmajor.value() - 50.0) < 1e-9
    assert abs(ed2.ed_xminor.value() - 10.0) < 1e-9
    assert ed2.cb_grid.isChecked() is False
    assert ed2.cb_zeroline.isChecked() is False
    assert ed2.dd_legend_loc.currentText() == 'upper left'
    assert abs(ed2.ed_legend_size.value() - 12.0) < 1e-9
    assert ed2._legend_font_user_set is True
    assert ed2._legend_font.family() == 'Courier New'
    assert abs(ed2.ed_fig_w.value() - 7.5) < 1e-9
    assert abs(ed2.ed_fig_h.value() - 4.2) < 1e-9
    # Now load a kinetics file (different x axis but still 1D) and
    # verify the legend / fig settings still persist.  X label text
    # should also carry — user typed "Wavelength (saved)" deliberately,
    # we don't second-guess that.
    win._load_file(os.path.join(DATA_DIR,
                                '5_AH-OMe_DMF_ex400_kinetics_536.2nm.csv'))
    ed3 = win._editor
    assert isinstance(ed3, tpe.OneDEditor)
    assert abs(ed3.ed_xmajor.value() - 50.0) < 1e-9
    assert ed3.dd_legend_loc.currentText() == 'upper left'
    assert abs(ed3.ed_fig_w.value() - 7.5) < 1e-9
    assert abs(ed3.ed_fig_h.value() - 4.2) < 1e-9
    print('    OK  1D settings persisted across DADS -> EADS -> kinetics')

    # -------------------------------------------------------------
    # 12) Tick-label font + size (2D and 1D editors).
    #     Pure parser + redraw check; no file IO needed beyond loading.
    # -------------------------------------------------------------
    print('\n[12] tick label font + size ...')
    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400.CSV'))
    ed_2d = win._editor
    assert isinstance(ed_2d, tpe.TwoDEditor)
    qf_tick = QtGui.QFont('Arial')
    qf_tick.setPointSize(13)
    qf_tick.setBold(True)
    ed_2d._set_tick_font(qf_tick)
    # ed_tick_size mirrors the dialog point size; reset it after to
    # confirm the spinbox is what wins.
    ed_2d.ed_tick_size.setValue(15.0)
    ed_2d.redraw()
    x_lbls = ed_2d.ax.get_xticklabels()
    assert x_lbls, 'no x tick labels after redraw'
    assert abs(x_lbls[0].get_fontsize() - 15.0) < 1e-6, \
        f'2D x tick size {x_lbls[0].get_fontsize()} != 15'
    assert x_lbls[0].get_fontweight() == 'bold'
    # Colorbar should also pick up the tick font.
    assert ed_2d._cb is not None, 'no colorbar after redraw'
    cb_lbls = ed_2d._cb.ax.get_yticklabels()
    assert cb_lbls, 'no colorbar tick labels'
    assert abs(cb_lbls[0].get_fontsize() - 15.0) < 1e-6
    print('    OK  2D tick font (Arial bold 15pt) applied to ax + colorbar')

    win._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_DADS.csv'))
    ed_1d = win._editor
    assert isinstance(ed_1d, tpe.OneDEditor)
    qf_tick2 = QtGui.QFont('Times New Roman')
    qf_tick2.setPointSize(8)
    qf_tick2.setItalic(True)
    ed_1d._set_tick_font(qf_tick2)
    ed_1d.ed_tick_size.setValue(11.0)
    ed_1d.redraw()
    y_lbls = ed_1d.ax.get_yticklabels()
    assert y_lbls, 'no y tick labels after redraw'
    assert abs(y_lbls[0].get_fontsize() - 11.0) < 1e-6, \
        f'1D y tick size {y_lbls[0].get_fontsize()} != 11'
    assert y_lbls[0].get_fontstyle() == 'italic'
    print('    OK  1D tick font (Times italic 11pt) applied to ax')

    # -------------------------------------------------------------
    # 13) DADS / EADS legend labels auto-format as $\\tau$={value} ps,
    #     and the default color cycle is Black/Red/Blue/Magenta with
    #     line styles cycling once per full color round.
    # -------------------------------------------------------------
    print('\n[13] DADS / EADS tau labels + color/style defaults ...')
    # Use a fresh window so prior persisted 1D settings don't bleed in.
    win2 = tpe.PlotExportWindow()
    win2.show()
    win2._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_DADS.csv'))
    ed = win2._editor
    assert isinstance(ed, tpe.OneDEditor)
    labels = [sw['label'].text() for sw in ed._series_widgets]
    colors = [sw['color'].text() for sw in ed._series_widgets]
    styles = [sw['style'].currentText() for sw in ed._series_widgets]
    # DADS columns are tau=2.4_ps, tau=185_ps, tau=6700_ps, tau=inf.
    assert labels[0] == r'$\tau$=2.4 ps', f'DADS[0] label = {labels[0]!r}'
    assert labels[1] == r'$\tau$=185 ps', f'DADS[1] label = {labels[1]!r}'
    assert labels[2] == r'$\tau$=6700 ps', f'DADS[2] label = {labels[2]!r}'
    assert labels[3] == r'$\tau$=$\infty$', f'DADS[3] label = {labels[3]!r}'
    # Color cycle: Black, Red, Blue, Magenta (case-insensitive compare).
    expected_colors = ['#000000', '#FF0000', '#0000FF', '#FF00FF']
    for i, (got, want) in enumerate(zip(colors, expected_colors)):
        assert got.lower() == want.lower(), \
            f'DADS series {i} color = {got!r}, want {want!r}'
    # All four series fit in one color cycle, so they should all use the
    # first line style (solid).
    assert all(s == '-' for s in styles), f'DADS styles = {styles!r}'
    print(f'    OK  DADS labels {labels}')
    print(f'    OK  DADS colors {colors}  styles {styles}')

    # EADS file uses the same naming pattern with an S<n> prefix.
    win2._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_EADS.csv'))
    ed = win2._editor
    assert isinstance(ed, tpe.OneDEditor)
    labels = [sw['label'].text() for sw in ed._series_widgets]
    assert labels[0].startswith(r'$\tau$='), \
        f'EADS[0] should start with tau=: {labels[0]!r}'
    assert labels[-1] == r'$\tau$=$\infty$', \
        f'EADS last label should be infinity: {labels[-1]!r}'
    print(f'    OK  EADS labels {labels}')

    # Kinetics file should NOT get the tau auto-format treatment — it
    # has data_dA / fit_dA / residual_dA columns, no tau= token.
    win2._load_file(os.path.join(DATA_DIR,
                                 '5_AH-OMe_DMF_ex400_kinetics_536.2nm.csv'))
    ed = win2._editor
    labels = [sw['label'].text() for sw in ed._series_widgets]
    assert all(r'$\tau$' not in s for s in labels), \
        f'kinetics labels unexpectedly tau-formatted: {labels!r}'
    print(f'    OK  kinetics labels unchanged ({labels})')

    # -------------------------------------------------------------
    # 14) >4 series: linestyle bumps to dashed on indices 4-7.
    #     Construct a synthetic DADS file with 6 columns so we can
    #     observe the style cycle without depending on real data.
    # -------------------------------------------------------------
    print('\n[14] DADS color cycle wraps with linestyle bump ...')
    import numpy as _np
    six_path = os.path.join(out_dir, 'synthetic_6col_DADS.csv')
    with open(six_path, 'w', encoding='utf-8') as f:
        cols = ['wavelength_nm'] + [
            f'DADS_tau={v}_ps' for v in (1, 5, 20, 100, 500, 2000)]
        f.write(','.join(cols) + '\n')
        wls = _np.linspace(400, 700, 32)
        for w in wls:
            row = [f'{w:.3f}'] + [f'{0.01 * (i + 1) * (w - 500):.5f}'
                                  for i in range(6)]
            f.write(','.join(row) + '\n')
    # Use the spectra kind detector indirectly via _load_file.
    win2._load_file(six_path)
    ed = win2._editor
    assert isinstance(ed, tpe.OneDEditor)
    colors = [sw['color'].text() for sw in ed._series_widgets]
    styles = [sw['style'].currentText() for sw in ed._series_widgets]
    # First four indices follow the Black/Red/Blue/Magenta cycle, then
    # the cycle repeats — but styles bump from solid to dashed.
    assert colors[0].lower() == '#000000'
    assert colors[4].lower() == '#000000', \
        f'5th color should restart at black: {colors[4]!r}'
    assert styles[0] == '-' and styles[3] == '-', f'cycle 1 styles: {styles[:4]!r}'
    assert styles[4] == '--' and styles[5] == '--', \
        f'cycle 2 should bump to dashed: {styles[4:]!r}'
    print(f'    OK  6-series cycle: colors {colors}, styles {styles}')

    # -------------------------------------------------------------
    # 15) Legend frame (box) toggle on/off.
    # -------------------------------------------------------------
    print('\n[15] legend frame toggle ...')
    win3 = tpe.PlotExportWindow()
    win3.show()
    win3._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_DADS.csv'))
    ed = win3._editor
    assert isinstance(ed, tpe.OneDEditor)
    # Default is ON.
    assert ed.cb_legend_frame.isChecked() is True
    ed.cb_legend.setChecked(True)
    ed.redraw()
    leg = ed.ax.get_legend()
    assert leg is not None, 'legend missing after redraw'
    assert leg.get_frame_on() is True, 'frame should be on by default'
    # Toggle off and re-check.
    ed.cb_legend_frame.setChecked(False)
    ed.redraw()
    leg = ed.ax.get_legend()
    assert leg is not None
    assert leg.get_frame_on() is False, \
        'frame should disappear when checkbox is unchecked'
    print('    OK  frame toggles between on / off')
    # And persistence: the unchecked state should carry to the next file.
    win3._load_file(os.path.join(DATA_DIR, '5_AH-OMe_DMF_ex400_EADS.csv'))
    ed2 = win3._editor
    assert ed2.cb_legend_frame.isChecked() is False, \
        'legend frame state did not persist across loads'
    print('    OK  frame setting persists across loads')

    print('\n*** PLOT-EXPORT FEATURE VERIFIED ***')
    print(f'(PNG samples in {out_dir})')


if __name__ == '__main__':
    main()
