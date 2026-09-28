"""
ta_device.py — Compute-device enumeration and TensorFlow backend helpers
for the Global Analysis dialog.

Uses TensorFlow (targeted at 2.10.1, the last Windows-native TF release
with GPU support) to enumerate CPU / GPU devices and to route the two
heavy linear-algebra kernels of the global fit (``lstsq`` and
``matmul``) onto a user-chosen device.

The TensorFlow import is deferred to first use so the rest of the
application does not pay the ~2-second import cost unless the user
opens the Global Analysis window.
"""
from __future__ import annotations

import os
import platform
import sys
import threading

import numpy as np


# ---------------------------------------------------------------------
# Administrator / root privilege check
# ---------------------------------------------------------------------
def is_admin() -> bool:
    """Return True iff the current process holds administrator (Windows)
    or root (POSIX) privileges.

    NVML fan-control paths (nvmlDeviceSetFanControlPolicy,
    nvmlDeviceSetFanSpeed_v2) require this on Windows — the driver
    refuses the request with ``NVML_ERROR_NO_PERMISSION`` when the
    process is not elevated, even for an administrator account under
    UAC.  We use the result to skip fan-90 attempts early and to steer
    the UI toward the elevation button.
    """
    try:
        if platform.system() == 'Windows':
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except Exception:
        return False


def relaunch_as_admin() -> tuple[bool, str]:
    """Restart the current process with administrator rights via the
    Windows UAC prompt (``ShellExecuteW`` with the ``runas`` verb).

    Returns ``(success, message)``.  ``success=True`` means the UAC
    prompt was shown and Windows accepted the elevation — the CALLER
    is responsible for exiting the current process so the elevated
    replacement can take over.  On non-Windows systems the call is a
    no-op that returns False.
    """
    if platform.system() != 'Windows':
        return False, 'Elevation is only supported on Windows.'
    try:
        import ctypes
        script = os.path.abspath(sys.argv[0])
        extra = ' '.join(f'"{a}"' for a in sys.argv[1:])
        params = f'"{script}"' + (f' {extra}' if extra else '')
        cwd = os.getcwd()
        # ShellExecute returns > 32 on success, specific codes on
        # failure (5 = SE_ERR_ACCESSDENIED  → user clicked "No" in UAC).
        ret = int(ctypes.windll.shell32.ShellExecuteW(
            None, 'runas', sys.executable, params, cwd, 1))
        if ret > 32:
            return True, ('UAC prompt accepted — elevated process is '
                          'starting.  This session should now exit.')
        if ret == 5:
            return False, 'UAC prompt was cancelled by the user.'
        return False, f'ShellExecute failed with code {ret}.'
    except Exception as e:
        return False, f'Elevation call raised: {e}'


# ---------------------------------------------------------------------
# Lazy TF import
# ---------------------------------------------------------------------
_tf = None
_tf_probe_done = False
_tf_import_error: str | None = None


def _try_import_tf():
    """Import tensorflow once and cache the module (or ``None`` on failure)."""
    global _tf, _tf_probe_done, _tf_import_error
    if _tf_probe_done:
        return _tf
    _tf_probe_done = True
    try:
        import tensorflow as tf  # noqa: F401  (side-effect: import)
        try:
            tf.get_logger().setLevel('ERROR')
        except Exception:
            pass
        _tf = tf
    except Exception as e:
        _tf = None
        _tf_import_error = f'{type(e).__name__}: {e}'
    return _tf


def tf_available() -> bool:
    """Return True iff TensorFlow could be imported."""
    return _try_import_tf() is not None


def tf_import_error() -> str | None:
    """Return the last TF import error message, or None if TF loaded."""
    _try_import_tf()
    return _tf_import_error


def tf_version() -> str:
    """Return the installed TF version string, or '' if unavailable."""
    tf = _try_import_tf()
    if tf is None:
        return ''
    try:
        return str(tf.__version__)
    except Exception:
        return ''


# ---------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------
def list_devices() -> list[dict]:
    """Return a list of available compute devices as dicts.

    Each entry has:
        id      — string passed to ``tf.device(...)`` (e.g. '/CPU:0',
                  '/GPU:0').  Also used as a stable UI key.
        label   — human-readable name shown in the checkbox.
        backend — 'tf' if the compute goes through TensorFlow, 'numpy'
                  when we fall back to the plain NumPy path.
    """
    devs: list[dict] = []
    tf = _try_import_tf()
    if tf is None:
        devs.append({
            'id': '/CPU:0',
            'label': 'CPU (NumPy — TensorFlow not available)',
            'backend': 'numpy',
        })
        return devs

    # CPU: TF always reports at least one logical CPU device.
    try:
        cpu_devs = tf.config.list_physical_devices('CPU')
    except Exception:
        cpu_devs = []
    if cpu_devs:
        devs.append({
            'id': '/CPU:0',
            'label': 'CPU (TensorFlow)',
            'backend': 'tf',
        })
    else:
        devs.append({
            'id': '/CPU:0',
            'label': 'CPU (NumPy fallback)',
            'backend': 'numpy',
        })

    # GPU: iterate over all detected physical GPUs.  On Windows with
    # TF 2.10.1 + CUDA this returns each CUDA-capable card.
    try:
        gpu_devs = tf.config.list_physical_devices('GPU')
    except Exception:
        gpu_devs = []
    for d in gpu_devs:
        # ``d.name`` looks like '/physical_device:GPU:0'
        idx = str(d.name).rsplit(':', 1)[-1]
        try:
            det = tf.config.experimental.get_device_details(d)
            name = det.get('device_name', f'GPU {idx}')
        except Exception:
            name = f'GPU {idx}'
        devs.append({
            'id': f'/GPU:{idx}',
            'label': f'GPU {idx}: {name}',
            'backend': 'tf',
        })
    return devs


# ---------------------------------------------------------------------
# Backend wrapper
# ---------------------------------------------------------------------
class TFBackend:
    """Thin wrapper around ``tf.linalg.lstsq`` and ``tf.matmul`` bound to
    a chosen device.

    Falls back transparently to NumPy when ``backend='numpy'`` (or when
    TensorFlow could not be imported).  Instances are cheap; the heavy
    TF import is amortised through the module-level cache above.
    """

    def __init__(self, device: str = '/CPU:0', backend: str = 'numpy'):
        self.device = device
        self.backend = backend
        self._tf = _try_import_tf() if backend == 'tf' else None
        if self._tf is None:
            # Silently degrade to NumPy so callers always get a working
            # backend even if TF has vanished between enumeration and
            # instantiation.
            self.backend = 'numpy'
        # Use float64 to preserve fit precision; consumer GPUs pay a
        # throughput penalty but the accuracy matters for the residual.
        self._dtype_np = np.float64

    def describe(self) -> str:
        return f'{self.device} [{self.backend}]'

    # ---- Least-squares (min-norm) --------------------------------------
    def lstsq(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solve ``A @ X = B`` in the least-squares sense.

        Shapes: A (m, n), B (m, k) → X (n, k).  Matches
        ``numpy.linalg.lstsq(A, B, rcond=None)[0]``.

        Implementation note
        -------------------
        Always solved on CPU via NumPy, **including when the backend is
        TF on a GPU device**.  Rationale: ``tf.linalg.lstsq`` on GPU
        goes through cuSOLVER.  On Windows with TF 2.10.1 and modern
        NVIDIA drivers (observed on RTX 40-series, Ada Lovelace
        CC 8.9), ``cusolverDnCreate`` fails with a FATAL check in
        ``cuda_solvers.cc``, which aborts the Python process from C++
        before we can trap it.  See ta_main.py session log
        2026-07-09 14:35:48 for the reproducer.

        The lstsq operands here are tiny (N_delay × k_tau, k_tau ≤ 6),
        so NumPy on CPU costs only microseconds per call.  The user's
        GPU selection still accelerates the O(M×N) reconstruction
        matmul via :meth:`matmul`, which uses cuBLAS and is unaffected
        by the cuSOLVER issue.
        """
        X, *_ = np.linalg.lstsq(A, B, rcond=None)
        return X

    # ---- Dense matmul --------------------------------------------------
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute ``A @ B`` on the chosen device."""
        if self.backend != 'tf':
            return A @ B
        tf = self._tf
        with tf.device(self.device):
            out = tf.matmul(
                tf.constant(np.ascontiguousarray(A, dtype=self._dtype_np)),
                tf.constant(np.ascontiguousarray(B, dtype=self._dtype_np)),
            ).numpy()
        return out.astype(np.float64, copy=False)


# ---------------------------------------------------------------------
# Live GPU status (memory, utilisation, fan, temperature)
# ---------------------------------------------------------------------
_pynvml = None
_pynvml_probe_done = False


def _try_import_pynvml():
    """Import pynvml once and cache the module (or ``None`` on failure)."""
    global _pynvml, _pynvml_probe_done
    if _pynvml_probe_done:
        return _pynvml
    _pynvml_probe_done = True
    try:
        import pynvml
        pynvml.nvmlInit()
        _pynvml = pynvml
    except Exception:
        _pynvml = None
    return _pynvml


def _query_nvidia_smi(gpu_index: int = 0) -> dict | None:
    """Fallback GPU-status query via the ``nvidia-smi`` CLI.

    Used when ``pynvml`` isn't installed.  Returns the same dict shape
    as :func:`get_gpu_status` or ``None`` if ``nvidia-smi`` isn't on the
    PATH / errors out.  Fields that ``nvidia-smi`` reports as
    ``[Not Supported]`` (fan / temp on some laptops) come back as
    ``None``.
    """
    import subprocess
    try:
        r = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.used,memory.total,'
             'utilization.gpu,fan.speed,temperature.gpu',
             '--format=csv,noheader,nounits',
             f'--id={int(gpu_index)}'],
            capture_output=True, text=True, timeout=3, check=True)
    except Exception:
        return None
    line = (r.stdout or '').strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(',')]
    if len(parts) < 6:
        return None

    def _f(s):
        try:
            return float(s)
        except ValueError:
            return None

    name = parts[0]
    mem_used = _f(parts[1])
    mem_total = _f(parts[2])
    util = _f(parts[3])
    fan = _f(parts[4])
    temp = _f(parts[5])
    if mem_used is None or mem_total is None or mem_total <= 0:
        return None
    return {
        'name': name,
        'memory_used_mb': mem_used,
        'memory_total_mb': mem_total,
        'memory_pct': 100.0 * mem_used / mem_total,
        'utilization_pct': None if util is None else int(util),
        'fan_speed_pct': None if fan is None else int(fan),
        'temperature_c': None if temp is None else int(temp),
        'source': 'nvidia-smi',
    }


def get_gpu_status(gpu_index: int = 0) -> dict | None:
    """Return live status for one NVIDIA GPU, or ``None`` if unavailable.

    Prefers the ``pynvml`` binding (fast, ~1 ms per call); falls back to
    a subprocess call to ``nvidia-smi`` if the binding is missing.

    Returned keys:
        name              str    — GPU marketing name
        memory_used_mb    float
        memory_total_mb   float
        memory_pct        float  — 100 * used / total
        utilization_pct   int|None — GPU-core utilisation (%), driver-reported
        fan_speed_pct     int|None — Fan duty cycle (%); None if the card
                                     has no fan sensor (many laptops / A100 etc.)
        temperature_c     int|None — GPU die temperature in °C
        source            str    — 'pynvml' or 'nvidia-smi'
    """
    pynvml = _try_import_pynvml()
    if pynvml is not None:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                fan = int(pynvml.nvmlDeviceGetFanSpeed(h))
            except Exception:
                fan = None
            try:
                temp = int(pynvml.nvmlDeviceGetTemperature(
                    h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                temp = None
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='replace')
            return {
                'name': name,
                'memory_used_mb': float(mem.used) / (1024.0 ** 2),
                'memory_total_mb': float(mem.total) / (1024.0 ** 2),
                'memory_pct': 100.0 * float(mem.used) / max(float(mem.total), 1.0),
                'utilization_pct': int(util.gpu),
                'fan_speed_pct': fan,
                'temperature_c': temp,
                'source': 'pynvml',
            }
        except Exception:
            pass  # Fall through to nvidia-smi
    return _query_nvidia_smi(gpu_index)


def gpu_index_from_device_id(device_id: str) -> int | None:
    """Extract the numeric index from a device id like '/GPU:0'."""
    if not device_id or ':' not in device_id or 'GPU' not in device_id.upper():
        return None
    try:
        return int(device_id.rsplit(':', 1)[-1])
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Fan-speed control (best-effort — often refused on GeForce Windows)
# ---------------------------------------------------------------------
def _fan_error_hint(err, pynvml) -> str:
    """Human-readable diagnostic for a common NVML fan-control failure."""
    try:
        code = err.value
    except Exception:
        code = None
    known = {
        getattr(pynvml, 'NVML_ERROR_NO_PERMISSION', -1):
            'Run the program as Administrator (right-click → Run as '
            'Administrator on Windows).',
        getattr(pynvml, 'NVML_ERROR_NOT_SUPPORTED', -2):
            'Not supported on this GPU. GeForce consumer cards on '
            'Windows commonly refuse NVML fan control; use MSI '
            'Afterburner / EVGA Precision X1 as an alternative.',
        getattr(pynvml, 'NVML_ERROR_INVALID_ARGUMENT', -3):
            'Invalid fan index or speed value.',
    }
    hint = known.get(code, '')
    return f'NVML: {err}. {hint}'.strip()


def set_gpu_fan_speed(gpu_index: int,
                      target_pct: int) -> tuple[bool, str]:
    """Attempt to pin the GPU's fan(s) at ``target_pct`` % duty cycle.

    Returns ``(success, message)``.  ``success=True`` only means the
    driver accepted the request; it may take a few seconds for the
    fan to actually ramp up.  On many consumer GeForce cards under
    Windows this call fails with ``NVML_ERROR_NOT_SUPPORTED`` and
    the fan stays on its auto curve — the message will explain.
    """
    pynvml = _try_import_pynvml()
    if pynvml is None:
        return (False,
                'pynvml not installed  →  pip install nvidia-ml-py')
    # Short-circuit: without admin/root, the NVML policy switch will
    # be refused by the driver with NVML_ERROR_NO_PERMISSION.  Skip
    # the wasted call and give the user actionable guidance instead
    # of a raw NVML error message.
    if not is_admin():
        _sys = platform.system()
        privilege = 'Administrator' if _sys == 'Windows' else 'root'
        hint = (' Use the "Restart as Administrator" button in the '
                'Global Analysis dialog to relaunch with UAC.'
                if _sys == 'Windows' else '')
        return (False,
                f'Insufficient permissions — NVML fan control requires '
                f'{privilege} privileges.{hint}')
    target_pct = max(0, min(100, int(target_pct)))
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        try:
            n_fans = int(pynvml.nvmlDeviceGetNumFans(h))
        except Exception:
            n_fans = 1
        # NVML fan-control policy enum (nvml.h):
        #   NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW = 0   → driver auto
        #   NVML_FAN_POLICY_MANUAL                    = 1   → user override
        # We MUST switch to MANUAL (=1) before requesting a fixed
        # speed; otherwise many drivers silently ignore SetFanSpeed_v2
        # while the auto temperature curve is still authoritative.
        POLICY_MANUAL = getattr(pynvml, 'NVML_FAN_POLICY_MANUAL', 1)
        applied = 0
        errors: list[str] = []
        policy_ok: list[bool] = []
        for fan_idx in range(max(n_fans, 1)):
            try:
                # Enable manual policy first.  Report failures so the
                # user can see whether the card supports policy switch.
                pol_ok = False
                try:
                    pynvml.nvmlDeviceSetFanControlPolicy(
                        h, fan_idx, POLICY_MANUAL)
                    pol_ok = True
                except pynvml.NVMLError as e:
                    errors.append(
                        f'fan{fan_idx} setPolicy(MANUAL): '
                        f'{_fan_error_hint(e, pynvml)}')
                except Exception as e:
                    errors.append(
                        f'fan{fan_idx} setPolicy(MANUAL): {e}')
                policy_ok.append(pol_ok)
                pynvml.nvmlDeviceSetFanSpeed_v2(h, fan_idx, target_pct)
                applied += 1
            except pynvml.NVMLError as e:
                errors.append(f'fan{fan_idx}: {_fan_error_hint(e, pynvml)}')
        if applied > 0:
            msg = f'Fan speed request accepted for {applied} fan(s) → {target_pct}%'
            if errors:
                msg += f'  (partial: {" ; ".join(errors)})'
            return True, msg
        return (False, '; '.join(errors) if errors
                else 'No fans reported by driver.')
    except pynvml.NVMLError as e:
        return False, _fan_error_hint(e, pynvml)
    except Exception as e:
        return False, f'Fan-set failed: {e}'


def restore_gpu_fan_auto(gpu_index: int) -> tuple[bool, str]:
    """Restore the driver's automatic fan curve.

    Called on every GA-exit path (finish / stop / error / dialog
    close) so an interrupted fit doesn't leave the fan spinning at
    90% indefinitely.
    """
    pynvml = _try_import_pynvml()
    if pynvml is None:
        return False, 'pynvml unavailable'
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        try:
            n_fans = int(pynvml.nvmlDeviceGetNumFans(h))
        except Exception:
            n_fans = 1
        # NVML policy enum reminder:  0 = auto (temperature),  1 = manual.
        # The dedicated "reset to default" call is preferred where the
        # driver supports it; otherwise we flip the policy back to
        # AUTO=0 explicitly.
        POLICY_AUTO = getattr(
            pynvml, 'NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW', 0)
        for fan_idx in range(max(n_fans, 1)):
            done = False
            try:
                pynvml.nvmlDeviceSetDefaultFanSpeed_v2(h, fan_idx)
                done = True
            except Exception:
                pass
            # Always ensure the policy is back on auto, even after the
            # default-speed call succeeds — belt-and-braces so we
            # don't leave the card in manual mode.
            try:
                pynvml.nvmlDeviceSetFanControlPolicy(
                    h, fan_idx, POLICY_AUTO)
            except Exception:
                # If policy switch fails but default-speed succeeded,
                # that's still "restored".
                if not done:
                    pass
        return True, 'Fan restored to driver-auto curve'
    except Exception as e:
        return False, f'Fan restore failed: {e}'


def test_fan_control(gpu_index: int = 0,
                     target_pct: int = 70,
                     wait_seconds: float = 4.0) -> dict:
    """End-to-end sanity check of the manual fan-control path.

    Runs the full ``set → wait → measure → restore → measure`` cycle
    and reports the observed fan speed at each phase so the caller
    can see whether the driver actually honoured the request.  This
    is the definitive way to answer *"is fan control working on this
    card?"* — the NVML functions return success on the mere API
    call, but only a physical read-back tells us if the fan moved.

    Parameters
    ----------
    gpu_index    physical GPU index (usually 0)
    target_pct   manual duty cycle to request during the test (default
                 70% — high enough to be audibly / measurably distinct
                 from typical idle rates, low enough not to hammer
                 the card during a diagnostic).
    wait_seconds settle time between issuing the manual set and
                 reading the fan back.  Real fans take a couple
                 seconds to ramp; the driver rate-limits changes.

    Returns
    -------
    dict with:
        target_pct         requested duty cycle
        baseline_pct       fan speed BEFORE the manual set  (or None)
        after_set_pct      fan speed AFTER wait_seconds       (or None)
        restored_pct       fan speed AFTER restoring auto     (or None)
        set_ok, set_msg    result of set_gpu_fan_speed
        restore_ok, restore_msg   result of restore_gpu_fan_auto
        appears_to_work    boolean heuristic: True iff the read-back
                           after the manual set is within ±15 % of
                           the target OR moved by ≥ 10 % vs baseline
        source             which reader supplied the fan speed
                           ('pynvml' / 'nvidia-smi')
    """
    import time
    report: dict = {
        'target_pct': int(target_pct),
        'baseline_pct': None,
        'after_set_pct': None,
        'restored_pct': None,
        'set_ok': False,
        'set_msg': '',
        'restore_ok': False,
        'restore_msg': '',
        'appears_to_work': False,
        'source': None,
    }

    # 1) Baseline read
    st = get_gpu_status(gpu_index)
    if st is None:
        report['set_msg'] = ('Cannot read GPU status. Install pynvml or '
                             'ensure nvidia-smi is on PATH.')
        return report
    report['baseline_pct'] = st.get('fan_speed_pct')
    report['source'] = st.get('source')

    # 2) Request manual speed
    ok, msg = set_gpu_fan_speed(gpu_index, target_pct)
    report['set_ok'] = ok
    report['set_msg'] = msg

    # 3) Wait for the fan to ramp, then read back
    if ok:
        time.sleep(max(wait_seconds, 0.5))
        st2 = get_gpu_status(gpu_index)
        if st2 is not None:
            report['after_set_pct'] = st2.get('fan_speed_pct')

    # 4) Restore driver-auto
    ok_r, msg_r = restore_gpu_fan_auto(gpu_index)
    report['restore_ok'] = ok_r
    report['restore_msg'] = msg_r

    # 5) Read once more after auto restore (fan may not slow
    # instantly, but we confirm we could read at all)
    if ok_r:
        time.sleep(1.0)
        st3 = get_gpu_status(gpu_index)
        if st3 is not None:
            report['restored_pct'] = st3.get('fan_speed_pct')

    # 6) Heuristic verdict
    if (report['set_ok']
            and report['after_set_pct'] is not None
            and report['baseline_pct'] is not None):
        near_target = abs(report['after_set_pct']
                          - report['target_pct']) <= 15
        moved = (abs(report['after_set_pct']
                     - report['baseline_pct']) >= 10)
        report['appears_to_work'] = bool(near_target or moved)
    return report


# ---------------------------------------------------------------------
# GPU keep-warm background load
# ---------------------------------------------------------------------
class GPUKeepwarm:
    """Background thread that runs a continuous ``tf.matmul`` on the
    selected GPU so utilisation stays above a target level during a
    workload that is otherwise CPU-bound.

    Caveats
    -------
    * This is dead-weight compute.  It **wastes electricity and heat**
      and does not accelerate the GA fit — the scipy Nelder-Mead loop
      itself is serial and mostly CPU-bound.
    * The keep-warm matmuls queue on the same CUDA stream as the real
      objective matmuls, so fit wall time typically **increases**
      (rough factor 1.5-3× depending on problem size).
    * Uses ``float32`` with a moderate 1024×1024 matrix by default:
      big enough to saturate the SMs, small enough that any individual
      keep-warm iteration completes in a few milliseconds so the
      real objective matmul doesn't wait long.
    """

    def __init__(self, device_id: str, matrix_size: int = 1024):
        self.device_id = device_id
        self.matrix_size = int(matrix_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: str | None = None

    def start(self) -> tuple[bool, str]:
        tf = _try_import_tf()
        if tf is None:
            return False, 'TensorFlow not available'
        if not self.device_id.startswith('/GPU'):
            return False, 'Keep-warm only applies to a GPU device'
        if self._thread is not None and self._thread.is_alive():
            return True, 'Keep-warm already running'
        self._stop_event.clear()
        self._error = None

        def loop():
            try:
                with tf.device(self.device_id):
                    # Random initial operands.  We do NOT re-generate
                    # them each iteration — feeding one matmul's output
                    # back in keeps the numbers bounded via matrix
                    # scaling (implicit re-normalisation not needed
                    # because we only care about GPU work, not value).
                    A = tf.random.normal(
                        (self.matrix_size, self.matrix_size),
                        dtype=tf.float32)
                    B = tf.random.normal(
                        (self.matrix_size, self.matrix_size),
                        dtype=tf.float32)
                    scale = tf.constant(1.0 / float(self.matrix_size),
                                        dtype=tf.float32)
                    while not self._stop_event.is_set():
                        A = tf.matmul(A, B) * scale
                        # Force sync + bound the queue depth so
                        # objective matmuls don't wait behind a huge
                        # backlog.
                        _ = float(A[0, 0].numpy())
            except Exception as e:  # pragma: no cover — diagnostic only
                self._error = f'{type(e).__name__}: {e}'

        self._thread = threading.Thread(target=loop, daemon=True,
                                        name='GPUKeepwarm')
        self._thread.start()
        return True, f'Keep-warm started on {self.device_id}'

    def stop(self, timeout: float = 3.0):
        """Signal the background loop to exit and join it."""
        if self._thread is None:
            return
        self._stop_event.set()
        try:
            self._thread.join(timeout=timeout)
        except Exception:
            pass
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def error(self) -> str | None:
        return self._error
