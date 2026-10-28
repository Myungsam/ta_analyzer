# -*- coding: utf-8 -*-
"""
Transient Absorption 데이터 정리 프로그램
=========================================
지정한 폴더의 CSV 데이터를 같은 폴더의 'Transient absorption.opju' 템플릿에
규칙에 맞게 채워 넣어 완성된 opju 파일을 생성합니다.

- Origin 프로젝트(.opju) 조작은 OriginLab 'orglab' 컴포넌트(Origin 2021+ 필요)를
  comtypes 를 통해 헤드리스로 사용합니다. (Origin GUI 를 띄우지 않습니다.)
- 이 PC 에는 Origin 2018 / 2024b 가 함께 설치되어 있는데, pip 의 originpro 는
  기본 COM 서버인 Origin 2018 에 붙어 셀 데이터를 읽고 쓰지 못합니다.
  따라서 반드시 orglab(2024b) 을 사용합니다.

필요 패키지:  numpy, comtypes, (선택) scipy   →  pip install numpy comtypes scipy

CSV → Book-Sheet 매핑 규칙  (파일명 '포함' 규칙은 확장자 제외, 대소문자 무시)
    chirp_points.csv                       → Book1 · Chirped line
    파일명에 '_cropped_original' 포함          → Book1 · Data                 (-10 ps 열 제외)
    파일명에 '_chirp_cropped' 포함, '_pinned' 미포함
                                           → Book1 · Data_Chirp_Corrected (-10 ps 열 제외)
    파일명에 '_pinned_spectra' 포함            → Book1 · Spectra for Graph    (스펙트럼 수만큼)
    파일명에 '_pinned_kinetics' 포함           → Book2 · Kinetics             (불완전(NaN) 행 제외)
    (후보가 여러 개면 GUI 는 선택 창, CLI 는 가장 짧은 경로 + 경고)
    GA_fit.csv                             → Book3 · GA_fit
    GA_residual.csv                        → Book3 · GA_residual
    파일명에 'EADS_spectra' 포함, 없으면 EADS.csv → Book3 · EADS
    파일명에 'DADS_spectra' 포함, 없으면 DADS.csv → Book3 · DADS
    (파일명에 'DADS_kinetics' / 'EADS_populations' 포함한 CSV 도 찾지만,
     이 템플릿 채우기 프로그램은 쓰지 않는다 → ta_opju_builder.py 가 시트로 만든다)
"""

import os
import re
import csv
import glob
import math
import threading
import queue
import traceback

import numpy as np
import comtypes
import comtypes.client
from comtypes import npsupport
npsupport.enable()          # numpy 배열을 SAFEARRAY(R8) 로 전달하기 위해 필요

TEMPLATE_NAME = "Transient absorption.opju"
OUTPUT_SUFFIX = "_filled"


# ==========================================================================
#  orglab (OriginLab Automation Server) 래퍼
# ==========================================================================
def find_orglab_tlb():
    """설치된 orglab 타입 라이브러리 중 가장 최신 버전을 찾는다."""
    cands = []
    for base in (r"C:\Program Files\OriginLab", r"C:\Program Files (x86)\OriginLab"):
        cands += glob.glob(os.path.join(base, "*", "Orglab8", "orglab*.tlb"))
        cands += glob.glob(os.path.join(base, "*", "orglab*.tlb"))
    if not cands:
        raise RuntimeError(
            "orglab 타입 라이브러리를 찾을 수 없습니다.\n"
            "Origin 2021 이상이 설치되어 있어야 합니다 (예: Origin 2024b).")

    def ver(p):
        m = re.search(r"orglab(\d+)", os.path.basename(p).lower())
        return int(m.group(1)) if m else 0

    return sorted(cands, key=ver)[-1]


class OrgLab:
    """orglab COM Application 을 감싼 얇은 래퍼."""

    def __init__(self):
        self.tlb = find_orglab_tlb()
        self.mod = comtypes.client.GetModule(self.tlb)
        self.app = comtypes.client.CreateObject(self.mod.Application)
        self.NUM = self.mod.ARRAY1D_NUMERIC

    def load(self, path):
        if not self.app.Load(os.path.abspath(path)):
            raise RuntimeError("opju 파일을 열지 못했습니다: %s" % path)

    def save(self, path):
        if not self.app.Save(os.path.abspath(path)):
            raise RuntimeError("opju 저장에 실패했습니다: %s\n"
                               "(파일이 Origin 에서 열려 있으면 닫아 주세요.)" % path)

    def exit(self):
        try:
            self.app.Exit()
        except Exception:
            pass

    # -- 시트/컬럼 접근 (컬렉션은 반드시 인덱스로 접근한다) ----------------
    def sheet(self, book, sheet):
        wps = self.app.WorksheetPages
        for i in range(wps.Count):
            wp = wps[i]
            if wp.Name == book:
                lys = wp.Layers
                for j in range(lys.Count):
                    if lys[j].Name == sheet:
                        return lys[j]
        raise KeyError("시트를 찾을 수 없습니다: [%s]%s" % (book, sheet))

    def put(self, book, sheet, data2d):
        """2차원 실수 배열을 [book]sheet! 의 (0,0) 부터 열 단위로 기록한다."""
        arr = np.ascontiguousarray(np.asarray(data2d, dtype=np.float64))
        if arr.ndim != 2:
            arr = arr.reshape(-1, 1)
        rng = "[%s]%s!" % (book, sheet)
        if not self.app.PutWorksheet(rng, arr, 0, 0):
            raise RuntimeError("데이터 입력 실패: %s" % rng)

    def graph_count(self):
        try:
            return self.app.GraphPages.Count
        except Exception:
            return -1


# ==========================================================================
#  CSV 읽기 / 라벨 포맷
# ==========================================================================
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def body_to_array(body, cols=None):
    """CSV 본문(문자열)을 실수 ndarray 로. 빈칸/'nan' → NaN, 'inf' → ±inf."""
    def f(x):
        x = x.strip()
        low = x.lower()
        if low in ("", "nan", "-nan"):
            return np.nan
        if low == "inf":
            return np.inf
        if low == "-inf":
            return -np.inf
        try:
            return float(x)
        except ValueError:
            return np.nan

    if cols is None:
        return np.array([[f(v) for v in r] for r in body], dtype=np.float64)
    return np.array([[f(r[c]) for c in cols] for r in body], dtype=np.float64)


# 라벨은 참고 정답의 그래프 범례 스타일에 맞춰 반올림한다.
#   파장 → 정수 nm,  시간상수 → 유효숫자 2자리.
def _round_sig(x, sig=2):
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def fmt_spectra(h):          # 'dA_t=0.5ps' -> '0.5 ps'
    m = re.search(r"=?\s*([-\d.]+)\s*ps", h)
    return ("%s ps" % m.group(1)) if m else h


def fmt_kinetics(h):         # 'dA_wl=389.9nm' -> '390 nm'
    m = re.search(r"=?\s*([-\d.]+)\s*nm", h)
    return ("%g nm" % round(float(m.group(1)))) if m else h


def fmt_tau(h):              # 'EADS_S1_tau=1.679_ps' -> '1.7 ps' ; '..._tau=inf' -> 'inf'
    if "inf" in h.lower():
        return "inf"
    m = re.search(r"tau\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)_?ps", h)
    return ("%g ps" % _round_sig(float(m.group(1)), 2)) if m else h


# ==========================================================================
#  데이터 폴더 안에서 CSV 파일 자동 탐색
# ==========================================================================
# 파일명(확장자 제외)에 '포함' 문자열이 있고 '제외' 문자열이 없으면 매칭 (대소문자 무시).
#   '_chirp_cropped' 는 *_chirp_cropped_pinned_* 파일명에도 들어 있으므로 '_pinned' 를 제외한다.
NAME_RULES = {
    "cropped_original": ("_cropped_original", None),
    "chirp_cropped":    ("_chirp_cropped", "_pinned"),
    "pinned_spectra":   ("_pinned_spectra", None),
    "pinned_kinetics":  ("_pinned_kinetics", None),
    # TA_Analyzer 의 Global Analysis 내보내기 (앞에 시료 이름이 붙을 수 있다)
    "EADS":             ("eads_spectra", None),       # 없으면 EADS.csv
    "DADS":             ("dads_spectra", None),       # 없으면 DADS.csv
    "DADS_kinetics":    ("dads_kinetics", None),
    "EADS_populations": ("eads_populations", None),
}
OPTIONAL = ("DADS_kinetics", "EADS_populations")      # 없어도 오류가 아니다


class SelectionCancelled(Exception):
    """후보 CSV 선택 창에서 사용자가 취소함."""


def pick_shortest(key, candidates):
    """후보가 여러 개일 때의 기본 선택: 경로가 가장 짧은 파일 + 경고 출력."""
    print("[경고] '%s' 후보 CSV %d개 → %s 사용 (나머지: %s)"
          % (key, len(candidates), os.path.basename(candidates[0]),
             ", ".join(os.path.basename(p) for p in candidates[1:])), flush=True)
    return candidates[0]


def discover(folder, choose=pick_shortest, required=True):
    """CSV 경로 사전을 반환. 후보가 여러 개인 항목은 choose(key, 후보목록) 로 결정한다.
    required=False 면 찾지 못한 항목을 오류 대신 None 으로 둔다."""
    base = glob.escape(folder)                 # 폴더 이름의 [ ] 를 glob 문자로 읽지 않게
    def one(pattern):
        hits = glob.glob(os.path.join(base, pattern))
        if not hits:
            if required:
                raise FileNotFoundError("필수 CSV 를 찾을 수 없습니다: %s" % pattern)
            return None
        return sorted(hits, key=len)[0]

    csvs = glob.glob(os.path.join(base, "*.csv"))

    def by_name(key, fallback=None):
        """fallback: 매칭되는 파일이 없을 때 쓰는 옛 이름 (대소문자 무시).
        EADS.csv 와 EADS_spectra.csv 가 함께 있으면 populations 와 짝인 _spectra 를 쓴다."""
        include, exclude = NAME_RULES[key]
        hits = []
        for p in csvs:
            stem = os.path.splitext(os.path.basename(p))[0].lower()
            if include in stem and not (exclude and exclude in stem):
                hits.append(p)
        if not hits and fallback:
            hits = [p for p in csvs if os.path.basename(p).lower() == fallback]
        if not hits:
            if not required or key in OPTIONAL:
                return None
            raise FileNotFoundError("필수 CSV 를 찾을 수 없습니다: "
                                    + ("%s 또는 " % fallback if fallback else "")
                                    + "파일명에 '%s' 포함" % include
                                    + ("" if exclude is None else ", '%s' 미포함" % exclude))
        hits.sort(key=len)
        return hits[0] if len(hits) == 1 else choose(key, hits)

    return {
        "chirp":            one("chirp_points.csv"),
        "cropped_original": by_name("cropped_original"),
        "chirp_cropped":    by_name("chirp_cropped"),
        "pinned_spectra":   by_name("pinned_spectra"),
        "pinned_kinetics":  by_name("pinned_kinetics"),
        "GA_fit":           one("GA_fit.csv"),
        "GA_residual":      one("GA_residual.csv"),
        "EADS":             by_name("EADS", "eads.csv"),
        "DADS":             by_name("DADS", "dads.csv"),
        "DADS_kinetics":    by_name("DADS_kinetics"),
        "EADS_populations": by_name("EADS_populations"),
    }


# ==========================================================================
#  스무딩 (Spectra 의 'Smoothed Y' 열용, 근사)
# ==========================================================================
def _smooth(y, win, poly):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 3:
        return y.copy()
    w = min(win, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return y.copy()
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, w, min(poly, w - 1), mode="interp")
    except Exception:
        k = np.ones(w) / w
        return np.convolve(y, k, mode="same")


# ==========================================================================
#  각 시트 채우기
# ==========================================================================
def _set_rows_all(sh, n):
    cols = sh.Columns
    for k in range(cols.Count):
        cols[k].Rows = int(n)


def _fill_chirp(ol, path, ctx):
    _, body = read_csv(path)
    arr = body_to_array(body)                          # (n, 2)
    sh = ol.sheet("Book1", "Chirped line")
    ol.put("Book1", "Chirped line", arr)
    _set_rows_all(sh, arr.shape[0])
    return "%d점 (wavelength_nm, t0_ps)" % arr.shape[0]


def _fill_2d(ol, book, sheet, path, drop_minus10):
    header, body = read_csv(path)
    ncol = len(header)
    keep = list(range(ncol))
    if drop_minus10:
        keep = [0] + [j for j in range(1, ncol) if header[j].strip() != "-10"]
    arr = body_to_array(body, cols=keep)
    sh = ol.sheet(book, sheet)
    ol.put(book, sheet, arr)
    for out_idx, j in enumerate(keep):                 # col0=파장(빈 코멘트), 이후=지연시간
        sh.Columns[out_idx].Comments = "" if out_idx == 0 else header[j].strip()
    _set_rows_all(sh, arr.shape[0])
    return "%d×%d (%s)" % (arr.shape[0], arr.shape[1],
                           "-10 ps 제외" if drop_minus10 else "전체 열")


def _fill_data(ol, path, ctx):
    return _fill_2d(ol, "Book1", "Data", path, True)


def _fill_data_chirp(ol, path, ctx):
    return _fill_2d(ol, "Book1", "Data_Chirp_Corrected", path, True)


def _fill_ga_fit(ol, path, ctx):
    return _fill_2d(ol, "Book3", "GA_fit", path, False)


def _fill_ga_res(ol, path, ctx):
    return _fill_2d(ol, "Book3", "GA_residual", path, False)


def _fill_spectra(ol, path, ctx):
    header, body = read_csv(path)
    arr = body_to_array(body)                          # (n, 1+N)
    n, ncsv = arr.shape
    N = ncsv - 1
    wl, raw = arr[:, 0], arr[:, 1:]
    sh = ol.sheet("Book1", "Spectra for Graph")
    Nt = (sh.Columns.Count - 1) // 2                   # 템플릿의 raw/smoothed 개수
    Nuse = min(N, Nt)
    sm = np.column_stack([_smooth(raw[:, k], ctx["smooth_window"], ctx["smooth_poly"])
                          for k in range(Nuse)])
    block = np.column_stack([wl, raw[:, :Nuse], sm])
    ol.put("Book1", "Spectra for Graph", block)
    labels = [fmt_spectra(header[k + 1]) for k in range(Nuse)]
    sh.Columns[0].Comments = ""
    for k in range(Nuse):
        sh.Columns[1 + k].Comments = labels[k]                 # raw
        sc = sh.Columns[1 + Nuse + k]
        sc.Comments = labels[k]                                # smoothed
        sc.LongName = "Smoothed Y%d" % (k + 1)
    _set_rows_all(sh, n)
    note = "%d 스펙트럼 (raw + smoothed 근사)" % Nuse
    if N != Nt:
        note += " ; 주의: CSV %d개 vs 템플릿 %d개" % (N, Nt)
    return note


def _fill_kinetics(ol, path, ctx):
    header, body = read_csv(path)
    arr = body_to_array(body)                          # (n, 1+N)
    n0, ncsv = arr.shape
    N = ncsv - 1
    sh = ol.sheet("Book2", "Kinetics")
    Nt = sh.Columns.Count - 1
    Nuse = min(N, Nt)
    block = arr[:, :1 + Nuse]
    # kinetics 데이터가 불완전(NaN)한 행 제거 → -10 ps 기준행과 말단 NaN 행이 빠지며
    # 참고 정답과 동일해진다.
    good = ~np.isnan(block[:, 1:1 + Nuse]).any(axis=1)
    block = block[good]
    n = block.shape[0]
    ol.put("Book2", "Kinetics", block)
    sh.Columns[0].Comments = ""
    for k in range(Nuse):
        sh.Columns[1 + k].Comments = fmt_kinetics(header[k + 1])
    _set_rows_all(sh, n)
    note = "%d kinetics × %d점 (불완전행 %d개 제외)" % (Nuse, n, n0 - n)
    if N != Nt:
        note += " ; 주의: CSV %d개 vs 템플릿 %d개" % (N, Nt)
    return note


def _fill_wav_series(ol, book, sheet, path, fmt_hdr):
    header, body = read_csv(path)
    arr = body_to_array(body)
    n, ncsv = arr.shape
    sh = ol.sheet(book, sheet)
    ol.put(book, sheet, arr)
    sh.Columns[0].LongName = "Wavelength"
    sh.Columns[0].Units = "nm"
    sh.Columns[0].Comments = ""
    for k in range(ncsv - 1):
        sh.Columns[1 + k].Comments = fmt_hdr(header[k + 1])
    _set_rows_all(sh, n)
    return "%d×%d" % (n, ncsv)


def _fill_eads(ol, path, ctx):
    return _fill_wav_series(ol, "Book3", "EADS", path, fmt_tau)


def _fill_dads(ol, path, ctx):
    return _fill_wav_series(ol, "Book3", "DADS", path, fmt_tau)


# ==========================================================================
#  전체 처리
# ==========================================================================
def fill_all(folder, out_path=None, template=None, progress_cb=None,
             smooth_window=15, smooth_poly=2, files=None):
    """folder 안의 CSV 로 템플릿 opju 를 채워 out_path 로 저장. (out_path 반환)
    files 를 주면 탐색 대신 그 CSV 경로 사전(discover 결과)을 사용한다."""
    folder = os.path.abspath(folder)
    if template is None:
        template = os.path.join(folder, TEMPLATE_NAME)
    if not os.path.isfile(template):
        raise FileNotFoundError("템플릿 opju 가 폴더에 없습니다:\n%s" % template)
    if out_path is None:
        stem = os.path.splitext(TEMPLATE_NAME)[0]
        out_path = os.path.join(folder, stem + OUTPUT_SUFFIX + ".opju")

    if files is None:
        files = discover(folder)
    steps = [
        ("Chirped line", _fill_chirp,      files["chirp"]),
        ("Data",         _fill_data,       files["cropped_original"]),
        ("Data_Chirp",   _fill_data_chirp, files["chirp_cropped"]),
        ("Spectra",      _fill_spectra,    files["pinned_spectra"]),
        ("Kinetics",     _fill_kinetics,   files["pinned_kinetics"]),
        ("GA_fit",       _fill_ga_fit,     files["GA_fit"]),
        ("GA_residual",  _fill_ga_res,     files["GA_residual"]),
        ("EADS",         _fill_eads,       files["EADS"]),
        ("DADS",         _fill_dads,       files["DADS"]),
    ]
    total = len(steps) + 2
    ctx = {"smooth_window": smooth_window, "smooth_poly": smooth_poly}
    ol = OrgLab()
    log = []
    try:
        if progress_cb:
            progress_cb(0, total, "템플릿 여는 중: %s" % os.path.basename(template))
        ol.load(template)
        for i, (name, fn, path) in enumerate(steps, start=1):
            if progress_cb:
                progress_cb(i, total, "%s ← %s" % (name, os.path.basename(path)))
            log.append("[OK] %-13s %s" % (name, fn(ol, path, ctx)))
        if progress_cb:
            progress_cb(total - 1, total, "저장 중: %s" % os.path.basename(out_path))
        ol.save(out_path)
        if progress_cb:
            progress_cb(total, total, "완료 (그래프 %d개 유지)" % ol.graph_count())
    finally:
        ol.exit()
    return out_path, log


# ==========================================================================
#  GUI (Tkinter)
# ==========================================================================
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("Transient Absorption · opju 데이터 정리")
    root.geometry("720x470")
    root.minsize(640, 420)

    msg_q = queue.Queue()
    state = {"out_path": None, "running": False}

    # ---- 상단: 폴더 경로 지정 ------------------------------------------
    top = ttk.LabelFrame(root, text="① 데이터 폴더 지정")
    top.pack(fill="x", padx=12, pady=(12, 6))

    folder_var = tk.StringVar()
    row = ttk.Frame(top); row.pack(fill="x", padx=10, pady=10)
    ttk.Label(row, text="폴더:").pack(side="left")
    folder_entry = ttk.Entry(row, textvariable=folder_var)
    folder_entry.pack(side="left", fill="x", expand=True, padx=6)

    tmpl_var = tk.StringVar(value="템플릿(opju): —")

    def refresh_template(*_):
        f = folder_var.get().strip()
        t = os.path.join(f, TEMPLATE_NAME) if f else ""
        if f and os.path.isfile(t):
            tmpl_var.set("템플릿(opju): %s  ✔" % TEMPLATE_NAME)
        elif f:
            tmpl_var.set("템플릿(opju): '%s' 없음  ✘" % TEMPLATE_NAME)
        else:
            tmpl_var.set("템플릿(opju): —")

    folder_var.trace_add("write", refresh_template)

    def browse():
        d = filedialog.askdirectory(title="CSV 와 템플릿 opju 가 있는 폴더 선택")
        if d:
            folder_var.set(d)

    ttk.Button(row, text="찾아보기…", command=browse).pack(side="left")
    ttk.Label(top, textvariable=tmpl_var, foreground="#555").pack(anchor="w", padx=10, pady=(0, 8))

    run_btn = ttk.Button(root, text="데이터 대체 실행 ▶")
    run_btn.pack(pady=6)

    # ---- 로그 ----------------------------------------------------------
    logf = ttk.LabelFrame(root, text="처리 로그")
    logf.pack(fill="both", expand=True, padx=12, pady=6)
    log_txt = tk.Text(logf, height=9, wrap="word", state="disabled",
                      font=("Consolas", 9))
    log_txt.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
    sb = ttk.Scrollbar(logf, command=log_txt.yview); sb.pack(side="right", fill="y", pady=8)
    log_txt.config(yscrollcommand=sb.set)

    def log_line(s):
        log_txt.config(state="normal")
        log_txt.insert("end", s + "\n")
        log_txt.see("end")
        log_txt.config(state="disabled")

    # ---- 하단: 진행 바 + 결과 열기 버튼 --------------------------------
    bottom = ttk.LabelFrame(root, text="② 진행 상황")
    bottom.pack(fill="x", padx=12, pady=(6, 12))
    brow = ttk.Frame(bottom); brow.pack(fill="x", padx=10, pady=10)

    prog = ttk.Progressbar(brow, mode="determinate", maximum=100)
    prog.pack(side="left", fill="x", expand=True)
    status_var = tk.StringVar(value="대기 중")
    open_btn = ttk.Button(brow, text="결과 opju 열기 ↗", state="disabled")
    open_btn.pack(side="right", padx=(10, 0))
    ttk.Label(bottom, textvariable=status_var, foreground="#555").pack(anchor="w", padx=10, pady=(0, 8))

    # ---- 동작 ----------------------------------------------------------
    def worker(folder, files):
        comtypes.CoInitialize()                    # 이 스레드에서 COM 초기화
        try:
            def cb(step, tot, m):
                msg_q.put(("progress", step, tot, m))
            out, log = fill_all(folder, progress_cb=cb, files=files)
            msg_q.put(("done", out, log))
        except Exception as e:
            msg_q.put(("error", "%s\n\n%s" % (e, traceback.format_exc())))
        finally:
            comtypes.CoUninitialize()

    def start():
        folder = folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("경로 확인", "유효한 데이터 폴더를 선택하세요.")
            return
        if not os.path.isfile(os.path.join(folder, TEMPLATE_NAME)):
            messagebox.showwarning("템플릿 없음",
                                   "폴더 안에 '%s' 가 없습니다." % TEMPLATE_NAME)
            return
        # CSV 탐색과 후보 선택은 메인 스레드에서 끝내고, 워커에는 확정된 경로만 넘긴다.
        try:
            files = discover(folder, choose=ask_choice)
        except SelectionCancelled:
            status_var.set("취소됨")
            return
        except FileNotFoundError as e:
            messagebox.showwarning("CSV 없음", str(e))
            return
        state["running"] = True
        state["out_path"] = None
        run_btn.config(state="disabled")
        open_btn.config(state="disabled")
        prog.config(value=0)
        log_txt.config(state="normal"); log_txt.delete("1.0", "end"); log_txt.config(state="disabled")
        status_var.set("처리 시작…")
        threading.Thread(target=worker, args=(folder, files), daemon=True).start()

    def ask_choice(key, candidates):
        """후보 CSV 가 여러 개인 항목을 사용자가 고르게 한다 (모달 창)."""
        result = {}
        dlg = tk.Toplevel(root)
        dlg.title("CSV 선택")
        dlg.transient(root)
        dlg.grab_set()
        ttk.Label(dlg, text="'%s' 에 해당하는 CSV 가 %d개입니다. 사용할 파일을 선택하세요."
                  % (key, len(candidates))).pack(anchor="w", padx=12, pady=(12, 6))
        lb = tk.Listbox(dlg, width=80, height=min(len(candidates), 10), exportselection=False)
        for p in candidates:
            lb.insert("end", os.path.basename(p))
        lb.selection_set(0)
        lb.pack(fill="both", expand=True, padx=12)

        def ok(*_):
            sel = lb.curselection()
            if sel:
                result["path"] = candidates[sel[0]]
            dlg.destroy()

        brow = ttk.Frame(dlg); brow.pack(pady=10)
        ttk.Button(brow, text="선택", command=ok).pack(side="left", padx=4)
        ttk.Button(brow, text="취소", command=dlg.destroy).pack(side="left", padx=4)
        lb.bind("<Double-Button-1>", ok)
        root.wait_window(dlg)
        if "path" not in result:
            raise SelectionCancelled()
        return result["path"]

    def open_result():
        p = state["out_path"]
        if p and os.path.isfile(p):
            try:
                os.startfile(p)                    # 연결된 Origin 으로 열기
            except Exception as e:
                messagebox.showerror("열기 실패", str(e))

    run_btn.config(command=start)
    open_btn.config(command=open_result)

    def poll():
        try:
            while True:
                item = msg_q.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _, step, tot, m = item
                    prog.config(value=step * 100.0 / tot)
                    status_var.set("[%d/%d] %s" % (step, tot, m))
                    log_line("[%d/%d] %s" % (step, tot, m))
                elif kind == "done":
                    _, out, log = item
                    for ln in log:
                        log_line(ln)
                    prog.config(value=100)
                    state["out_path"] = out
                    state["running"] = False
                    status_var.set("완료: %s" % os.path.basename(out))
                    log_line("\n생성됨: %s" % out)
                    run_btn.config(state="normal")
                    open_btn.config(state="normal")
                elif kind == "error":
                    _, msg = item
                    state["running"] = False
                    status_var.set("오류 발생")
                    log_line("\n[오류] " + msg)
                    run_btn.config(state="normal")
                    messagebox.showerror("처리 오류", msg.splitlines()[0])
        except queue.Empty:
            pass
        root.after(80, poll)

    root.after(80, poll)
    root.mainloop()


# ==========================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] not in ("--gui", "-g"):
        # CLI:  python ta_opju_filler.py <folder>
        def cb(s, t, m):
            print("  [%d/%d] %s" % (s, t, m), flush=True)
        out, log = fill_all(sys.argv[1], progress_cb=cb)
        print("\n결과:", out)
        print("\n".join(log))
    else:
        launch_gui()
