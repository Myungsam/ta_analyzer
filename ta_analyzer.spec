# PyInstaller spec — TA Analyzer (one-file, windowed)
#
# Build with:
#     python -m PyInstaller ta_analyzer.spec --noconfirm --clean
#
# Output: dist/TA_Analyzer.exe

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Every ta_*.py sits at the project root and gets pulled in by
# ta_main via normal imports — but PyInstaller's static analysis can
# miss ones only referenced through class strings, lazy imports, or
# dialog registration, so we list them explicitly.
LOCAL_MODULES = [
    'ta_accumulate',
    'ta_chirp',
    'ta_coherence',
    'ta_core',
    'ta_device',
    'ta_dialogs_a',
    'ta_ga',
    'ta_kfit',
    'ta_lda',
    'ta_load_custom',
    'ta_lpsvd',
    'ta_mcr',
    'ta_residual_store',
    'ta_solvent_irf',
    'ta_svd',
    'ta_widgets',
]

# Matplotlib's Qt5Agg backend + a handful of format writers used by
# the navigation toolbar's Save button.  Without these hidden imports
# PyInstaller drops them and the app falls back to Agg (no GUI redraw).
HIDDEN = [
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_agg',
    'matplotlib.backends.qt_editor',
    'matplotlib.backends.qt_editor._formlayout',
    'PyQt5.QtPrintSupport',
    'openpyxl',
    'scipy.optimize',
    'scipy.optimize._minpack_py',
    'scipy.optimize._lsq',
    'scipy.optimize._lsq.least_squares',
    'scipy.linalg',
    'scipy.signal',
    'scipy.sparse',
    'scipy.sparse.linalg',
    'scipy.special',
]
HIDDEN += LOCAL_MODULES
HIDDEN += collect_submodules('scipy.optimize')

# Bundle matplotlib's data files (fonts, mpl-data, etc.).
DATAS = []
DATAS += collect_data_files('matplotlib')

# Excludes: TensorFlow / GPU stack aren't installed here and the code
# already falls back to numpy when they're missing.  Prevent
# PyInstaller from picking them up if they ever appear.
EXCLUDES = [
    'tensorflow', 'tensorflow_core', 'tensorboard',
    'pynvml', 'nvidia',
    'tkinter',
    'PIL.ImageQt',
    'PyQt6', 'PySide2', 'PySide6',
    'notebook', 'IPython', 'jupyter',
    'pytest', 'pandas',
]

a = Analysis(
    ['ta_main.py'],
    pathex=[],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TA_Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # avoid UPX; sometimes triggers antivirus
    upx_exclude=[],
    runtime_tmpdir=None, # one-file: default to %TEMP%
    console=False,       # --windowed: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
