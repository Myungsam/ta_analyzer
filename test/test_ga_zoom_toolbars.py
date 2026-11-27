"""
Global Analysis: the DADS row, the EADS row and the kinetics trace each
have a matplotlib navigation toolbar (zoom / pan / home / back / forward /
save), like the map row.  After a re-plot the zoom history is reset, so
Home returns to the new plot's limits rather than stale ones.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402
from matplotlib.backends.backend_qt5agg import (  # noqa: E402
    NavigationToolbar2QT,
)

import ta_core  # noqa: E402
import ta_main  # noqa: E402
import ta_ga  # noqa: E402


def _dataset():
    rng = np.random.default_rng(2)
    wl = np.linspace(450.0, 650.0, 60)
    t = np.r_[np.linspace(-1, 2, 30), np.logspace(0.4, 2.7, 70)]
    s1 = np.exp(-((wl - 500) / 40) ** 2)
    s2 = -0.6 * np.exp(-((wl - 600) / 40) ** 2)
    D = (np.outer(s1, ta_core.exp_irf_conv(t, 2.0, 0.0, 0.15))
         + np.outer(s2, ta_core.exp_irf_conv(t, 40.0, 0.0, 0.15)))
    return wl, t, D + 0.003 * rng.normal(size=D.shape)


def _actions(tb):
    return {a.text() for a in tb.actions() if a.text()}


def test_toolbars():
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)  # noqa: F841
    wl, t, D = _dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, D, 'Loaded: zoom toolbars')
    ga = ta_ga.GlobalAnalysisDialog(win, win)
    canvases = (ga.canvas_maps, ga.canvas_dads, ga.canvas_eads,
                ga.canvas_kin_fit)
    for cv in canvases:
        tb = getattr(cv, 'toolbar', None)
        assert isinstance(tb, NavigationToolbar2QT), cv
        assert {'Home', 'Back', 'Forward', 'Pan', 'Zoom', 'Save'} \
            <= _actions(tb), _actions(tb)
    # every toolbar sits directly above its own canvas
    for cv in canvases:
        lay = cv.parentWidget().layout()
        assert lay.indexOf(cv.toolbar) == 0 and lay.indexOf(cv) == 1

    ga.dd_N.setCurrentText('2')
    ga.tau_table.cellChanged.disconnect()
    for r, v in enumerate([1.0, 30.0]):
        ga.tau_table.item(r, 0).setText(str(v))
    ga.tau_table.cellChanged.connect(ga._on_tau_edit)
    ga.read_table()
    ga.do_run()
    QtWidgets.QApplication.processEvents()

    # Zoom on the kinetics trace, then Back / Home restore it
    ax = ga.ax_kin_fit
    tb = ga.canvas_kin_fit.toolbar
    full = ax.get_xlim()
    tb.push_current()
    ax.set_xlim(1.0, 5.0)
    tb.push_current()
    tb.back()
    assert np.allclose(ax.get_xlim(), full)
    tb.forward()
    assert np.allclose(ax.get_xlim(), (1.0, 5.0))

    # A re-plot (new λ) clears the zoom history: Home keeps the new limits
    ga.cur_kin_wl = float(wl[40])
    ga.plot_kinetics()
    new_full = ax.get_xlim()
    assert len(tb._nav_stack) == 0
    tb.home()
    assert np.allclose(ax.get_xlim(), new_full)

    # Same for the DADS / EADS rows (component toggle re-plots both)
    for cv, axr in ((ga.canvas_dads, ga.ax_dads), (ga.canvas_eads, ga.ax_eads)):
        cv.toolbar.push_current()
        axr.set_xlim(500.0, 520.0)
        cv.toolbar.push_current()
    ga.cb_comps[0].setChecked(False)
    QtWidgets.QApplication.processEvents()
    for cv, axr in ((ga.canvas_dads, ga.ax_dads), (ga.canvas_eads, ga.ax_eads)):
        assert len(cv.toolbar._nav_stack) == 0
        assert axr.get_xlim()[0] < 500.0           # autoscaled again
    # A map click while Zoom is active is a zoom drag, not a λ pick
    from types import SimpleNamespace
    wl_before = ga.cur_kin_wl
    ev = SimpleNamespace(inaxes=ga.ax_exp, xdata=float(wl[5]))
    ga.canvas_maps.toolbar.zoom()                  # Zoom mode on
    ga.on_map_click(ev)
    assert ga.cur_kin_wl == wl_before
    ga.canvas_maps.toolbar.zoom()                  # Zoom mode off
    ga.on_map_click(ev)
    assert abs(ga.cur_kin_wl - float(wl[5])) < 0.01    # spinbox: 2 dp

    # Resetting the results clears every zoom history
    for cv in (ga.canvas_maps, ga.canvas_dads, ga.canvas_eads,
               ga.canvas_kin_fit):
        cv.toolbar.push_current()
    ga._reset_results('reset')
    for cv in (ga.canvas_maps, ga.canvas_dads, ga.canvas_eads,
               ga.canvas_kin_fit):
        assert len(cv.toolbar._nav_stack) == 0
    ga.close()
    win.close()
    print('[ok] GA zoom toolbars')


def test_sas_tabs():
    """DADS and EADS rows live in two tabs; both are plotted after a fit
    whichever is shown, and the chosen tab is remembered on reopen."""
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)  # noqa: F841
    wl, t, D = _dataset()
    win = ta_main.TAAnalyzer()
    win.set_loaded_data(wl, t, D, 'Loaded: SAS tabs')
    ga = ta_ga.GlobalAnalysisDialog(win, win)
    tabs = ga.tabs_sas
    assert [tabs.tabText(i) for i in range(tabs.count())] == ['DADS', 'EADS']
    assert tabs.currentIndex() == 0
    assert tabs.widget(0).isAncestorOf(ga.canvas_dads)
    assert tabs.widget(1).isAncestorOf(ga.canvas_eads)

    ga.dd_N.setCurrentText('2')
    ga.tau_table.cellChanged.disconnect()
    for r, v in enumerate([1.0, 30.0]):
        ga.tau_table.item(r, 0).setText(str(v))
    ga.tau_table.cellChanged.connect(ga._on_tau_edit)
    ga.read_table()
    ga.do_run()
    QtWidgets.QApplication.processEvents()
    # the hidden EADS tab is plotted too
    assert len(ga.ax_dads.get_lines()) >= 2
    assert len(ga.ax_eads.get_lines()) >= 2
    assert len(ga.ax_eads_t.get_lines()) >= 2

    tabs.setCurrentIndex(1)
    assert tabs.currentWidget().isAncestorOf(ga.canvas_eads)
    assert win.ga_sas_tab == 1
    ga.close()
    ga2 = ta_ga.GlobalAnalysisDialog(win, win)
    assert ga2.tabs_sas.currentIndex() == 1         # remembered
    ga2.close()
    win.close()
    print('[ok] GA DADS / EADS tabs')


if __name__ == '__main__':
    test_toolbars()
    test_sas_tabs()
    print('ALL OK')
