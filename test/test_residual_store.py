"""Tests for ta_residual_store + the FFT/LPSVD path that consumes its files."""
import os
import sys
import tempfile

import numpy as np

# Headless Qt — the Coherence dialog itself is not constructed here, but
# the bare modules import QtWidgets, so a QApplication is needed for any
# downstream test that does build widgets.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# Make project root importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

import ta_residual_store as RS


# --------------------------------------------------------------- helpers
def _make_synthetic_2d(M=12, N=80, seed=0):
    rng = np.random.default_rng(seed)
    wl = np.linspace(400.0, 700.0, M)
    t = np.linspace(-1.0, 50.0, N)
    R = rng.normal(scale=1e-3, size=(M, N))
    # Inject a vibrational coherence so FFT tests have something to find.
    nu_cm = 350.0   # ~10.5 THz
    C_CM = 2.99792458e10  # cm/s
    omega = 2 * np.pi * nu_cm * C_CM * 1e-12   # rad/ps
    decay = np.exp(-t / 8.0) * (t >= 0)
    R += 5e-3 * np.outer(np.cos(2 * np.pi * wl / 600.0),
                          np.sin(omega * t) * decay)
    return wl, t, R


def _make_synthetic_1d(N=160, seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 30.0, N)
    nu_cm = 200.0
    C_CM = 2.99792458e10
    omega = 2 * np.pi * nu_cm * C_CM * 1e-12
    decay = np.exp(-t / 5.0)
    r = decay * np.sin(omega * t) + rng.normal(scale=1e-4, size=N)
    return t, r


# --------------------------------------------------------------- tests
def test_sanitize_basic():
    assert RS.sanitize('Loaded: foo / bar.csv') == 'Loaded_foo_bar.csv'
    assert RS.sanitize('   ') == 'data'
    assert RS.sanitize('αβ.txt') == 'αβ.txt'.replace('α', '_').replace(
        'β', '_').lstrip('_') or 'data'


def test_make_filename():
    name = RS.make_filename('GA', 'sample/run-1.csv', wavelength=633.21)
    assert name.startswith('GA_sample_run-1.csv_633.2nm_'), name
    assert name.endswith('.xlsx')
    assert RS.make_filename('Kfit', 'sample.csv', wavelength=None) \
        .startswith('Kfit_sample.csv_')


def test_roundtrip_2d_in_tempdir():
    """save_residual_2d → load_residual reproduces wl/t/R bit-for-bit."""
    with tempfile.TemporaryDirectory() as tmp:
        wl, t, R = _make_synthetic_2d()
        # Sprinkle NaNs so we exercise the empty-cell path.
        R[0, 5] = np.nan
        R[7, -1] = np.nan
        path = RS.save_residual_2d(
            'GA', 'unit_test.csv', wl, t, R,
            base_dir=tmp, time_unit='ps',
            extra={'rms': '1.23e-3'})
        assert os.path.isfile(path)

        rec = RS.load_residual(path)
        assert rec['kind'] == '2D'
        assert np.allclose(rec['wl'], wl, rtol=0, atol=1e-9)
        assert np.allclose(rec['t'], t, rtol=0, atol=1e-9)
        # NaN cells round-trip as NaN.
        assert np.isnan(rec['R'][0, 5])
        assert np.isnan(rec['R'][7, -1])
        mask = np.isfinite(R) & np.isfinite(rec['R'])
        assert np.allclose(rec['R'][mask], R[mask], rtol=0, atol=1e-9)
        meta = rec['meta']
        assert meta['analysis'] == 'GA'
        assert meta['dataset'] == 'unit_test.csv'
        assert meta['kind'] == '2D'
        assert meta['rms'] == '1.23e-3'


def test_roundtrip_1d_with_wavelength_in_filename():
    with tempfile.TemporaryDirectory() as tmp:
        t, r = _make_synthetic_1d()
        path = RS.save_residual_1d(
            'Kfit', 'kfit_test.csv', t, r,
            base_dir=tmp, wavelength=633.0, time_unit='ps',
            extra={'n_avg_px': '3'})
        name = os.path.basename(path)
        assert '_633.0nm_' in name, name
        rec = RS.load_residual(path)
        assert rec['kind'] == '1D'
        assert rec['wl'] is None
        assert np.allclose(rec['t'], t)
        assert np.allclose(rec['R'], r)
        assert rec['meta']['analysis'] == 'Kfit'
        assert rec['meta']['wavelength_nm'].startswith('633')


def test_list_files_orders_newest_first():
    with tempfile.TemporaryDirectory() as tmp:
        wl, t, R = _make_synthetic_2d()
        p1 = RS.save_residual_2d('GA', 'first', wl, t, R, base_dir=tmp)
        # Bump the second file's mtime so the sort is unambiguous on
        # filesystems with coarse mtime resolution.
        p2 = RS.save_residual_2d('LDA', 'second', wl, t, R, base_dir=tmp)
        os.utime(p2, (os.path.getmtime(p2) + 5,
                      os.path.getmtime(p2) + 5))
        files = RS.list_residual_files(base_dir=tmp)
        assert len(files) == 2
        assert os.path.samefile(files[0], p2), \
            'newest file should come first'
        assert os.path.samefile(files[1], p1)


def test_dimension_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        wl = np.array([400, 500, 600], dtype=float)
        t = np.array([0, 1, 2, 3], dtype=float)
        R_bad = np.zeros((4, 3))   # transposed
        try:
            RS.save_residual_2d('GA', 'd', wl, t, R_bad, base_dir=tmp)
        except ValueError as e:
            assert 'shape' in str(e).lower()
            return
        raise AssertionError('expected ValueError')


def test_2d_file_drives_fft_via_compute_coherence():
    """The saved 2D residual must be a valid input to compute_coherence.

    This validates the roundtrip wiring: load_residual produces arrays
    in the layout compute_coherence expects, with no shape, dtype, or
    NaN-propagation issues.
    """
    import ta_core
    with tempfile.TemporaryDirectory() as tmp:
        wl, t, R = _make_synthetic_2d()
        path = RS.save_residual_2d(
            'GA', 'fft_test', wl, t, R, base_dir=tmp, time_unit='ps')
        rec = RS.load_residual(path)
        res = ta_core.compute_coherence(
            rec['R'], rec['t'],
            t_min=float(rec['t'][0]),
            t_max=float(rec['t'][-1]),
            apod='hann', zero_pad=2, detrend='linear',
            freq_unit='cm-1', norm_mode='peak',
            time_unit='ps')
        assert 'P' in res and 'freq' in res
        assert res['P'].shape[0] == rec['wl'].size
        assert np.any(np.isfinite(res['P']))


def test_1d_file_drives_lpsvd():
    """The saved 1D residual must run through run_lpsvd_analysis."""
    import ta_lpsvd
    with tempfile.TemporaryDirectory() as tmp:
        t, r = _make_synthetic_1d(N=256)
        path = RS.save_residual_1d(
            'Kfit', 'lpsvd_test', t, r, base_dir=tmp,
            wavelength=500.0, time_unit='ps')
        rec = RS.load_residual(path)
        res = ta_lpsvd.run_lpsvd_analysis(
            rec['t'], rec['R'], time_unit='ps',
            t_start=float(rec['t'][0]),
            t_end=float(rec['t'][-1]),
            do_polyfit=False, polyfit_deg=4,
            do_detrend=True, do_center=True,
            append_count=0, model='lorentzian',
            num_modes=3, window_type='kaiser',
            beta=8.0, gaussian_sigma=1.0,
            apply_window_to_fft=True)
        assert 'modes' in res and len(res['modes']) >= 1


# --------------------------------------------------------------- main
if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'  PASS  {name}')
            except Exception as e:
                print(f'  FAIL  {name}: {e}')
                raise
    print('All tests passed.')
