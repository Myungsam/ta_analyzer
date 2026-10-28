# -*- coding: utf-8 -*-
"""
Transient Absorption 새 opju 생성 프로그램 (Origin 2018 용)
==========================================================
템플릿 opju 를 읽지 않고, 데이터 폴더의 CSV 만으로 새 Origin 프로젝트를 만들어
'Transient absorption_filled.opju' 로 저장합니다.

- 데이터 폴더·프로그램 폴더·출력 경로에 한글이나 공백, [ ] % 가 있어도 됩니다.
  (그래프 템플릿은 필요하면 영문 임시 경로에 복사해 씁니다.)

- Origin 조작은 설치된 Origin 2018 의 COM 서버(Origin.Application)를 씁니다.
  결과 파일을 Origin 2018 이 직접 저장하므로 2018 에서 그대로 열립니다.
- 그래프 모양(축·색상표·선 색·글꼴)은 origin_templates/ 의 그래프 템플릿(.otpu)
  에서 가져옵니다. 이 템플릿은 기존 'Transient absorption.opju' 의 그래프에서
  한 번 추출해 프로그램과 함께 둔 것입니다.
- 실행 중인 Origin 이 있으면 시작하지 않습니다. 작업을 저장하고 Origin 을 모두
  닫은 뒤 실행하세요.

필요 패키지:  numpy, comtypes, (선택) scipy

사용법:  python ta_opju_builder.py                       → GUI
         python ta_opju_builder.py <폴더> [출력.opju] [--overwrite]   → CLI

Book1 시트 ← CSV  (파일 탐색 규칙은 ta_opju_filler.discover 와 같음)
    Original_Data         ← 파일명에 '_cropped_original' 포함        (-10 ps 열 제외)
    Data_Chirp_Corrected  ← '_chirp_cropped' 포함, '_pinned' 미포함   (-10 ps 열 제외)
    Chirp_line            ← chirp_points.csv
    Kinetics              ← '_pinned_kinetics' 포함                  (NaN 행 제외)
    Spectra               ← '_pinned_spectra' 포함                   (raw + smoothed)
    EADS / DADS           ← EADS.csv / DADS.csv, 또는 'EADS_spectra' / 'DADS_spectra' 포함
                                                                      (전부 0 인 행 → 빈 셀)
    DADS_kinetics         ← 'DADS_kinetics' 포함     (성분 kinetics C(t), Kinetics 모양 그래프)
    EADS_populations      ← 'EADS_populations' 포함  (순차 모델 population, Kinetics 모양 그래프)
    Global_Fit / Residual ← GA_fit.csv / GA_residual.csv
CSV 가 없는 시트는 이름만 만들고 비워 두며, 그 시트의 그래프는 빈 틀로 만듭니다.
"""

import os
import re
import sys
import shutil
import tempfile
import subprocess
import threading
import queue
import traceback

import numpy as np
import comtypes
import comtypes.client
from comtypes import npsupport
npsupport.enable()

from ta_opju_filler import (read_csv, body_to_array, fmt_spectra, fmt_kinetics,
                            fmt_tau, discover, pick_shortest, SelectionCancelled,
                            _smooth)

OUTPUT_NAME = "Transient absorption_filled.opju"
BOOK = "Book1"
SHEETS = ["Original_Data", "Data_Chirp_Corrected", "Chirp_line", "Kinetics",
          "Spectra", "EADS", "DADS", "DADS_kinetics", "EADS_populations",
          "Global_Fit", "Residual"]
# PyInstaller 로 묶은 실행 파일이면 풀린 임시 폴더(sys._MEIPASS)에서 템플릿을 찾는다.
TEMPLATE_DIR = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
                            "origin_templates")

# 그래프 (기존 템플릿의 Graph1~7 순서 + Kinetics 모양의 Graph8~9): (템플릿, 시트, 종류)
#   line 그래프는 X=1열, Y=시트별 열 범위 / contour 는 1열=파장, 나머지 열=지연시간
GRAPHS = [
    ("TA_2D_Data",     "Data_Chirp_Corrected", "contour"),
    ("TA_Spectra",     "Spectra",              "line"),
    ("TA_Kinetics",    "Kinetics",             "scatter"),
    ("TA_GA_fit",      "Global_Fit",           "contour"),
    ("TA_GA_residual", "Residual",             "contour"),
    ("TA_EADS",        "EADS",                 "line"),
    ("TA_DADS",        "DADS",                 "line"),
    ("TA_Kinetics",    "DADS_kinetics",        "scatter"),
    ("TA_Kinetics",    "EADS_populations",     "scatter"),
]
PLOT_TYPE = {"line": 200, "scatter": 201, "contour": 226}

# Kinetics 템플릿의 범례는 곡선 색에 맞춘 개별 글상자(1~4번 곡선 순서)이고,
# NLFit 시간상수 글상자(1.1 ps / 37 ps / Long)는 피팅을 하지 않으므로 지운다.
KINETICS_LABELS = ["Text", "Text2", "Text6", "Text7"]
KINETICS_FIT_TEXT = ["Text1", "Text4", "Text5"]

# 템플릿 그래프의 축 범위·색상 단계 (기존 템플릿 opju 에서 한 번 읽어 둔 값).
# 템플릿은 새 데이터에 맞춰 축을 다시 잡으므로, 그린 뒤에 이 값으로 되돌린다.
# 단, 아래 두 가지는 그래프 X·Y 범위 안에 보이는 데이터로 다시 정한다 (auto_axes).
#   컨투어 색상 단계  → ±(|값| 최대의 올림 정수)
#   Spectra·Kinetics Y → 최소의 내림 정수 ~ 최대의 올림 정수
AXES = {
    "TA_2D_Data":     dict(x=(375, 725, 50), y=(-1, 450, 1), z=(-30, 30, 4, 9)),
    "TA_Spectra":     dict(x=(375, 725, 50), y=(-7.5, 30, 5)),
    "TA_Kinetics":    dict(x=(0.3, 500, 1), y=(-10, 30, 10)),
    "TA_GA_fit":      dict(x=(375, 725, 50), y=(0.3, 450, 1), z=(-30, 30, 4, 9)),
    "TA_GA_residual": dict(x=(375, 725, 50), y=(0.3, 450, 1), z=(-20, 20, 4, 9)),
    "TA_EADS":        dict(x=(375, 725, 50), y=(-15, 30, 10)),
    "TA_DADS":        dict(x=(375, 725, 50), y=(-35, 25, 10)),
}
AUTO_Y = ("TA_Spectra", "TA_Kinetics")

# 축 제목 (x, y). None 은 템플릿 그대로. 컨투어 템플릿은 제목을 데이터에서 가져오고,
# Spectra 템플릿의 X 제목은 원래 'Delay (ps)' 로 잘못 들어 있어 고친다.
CONTOUR_TITLES = ("Wavelength (nm)", "Delay (ps)")
TITLES = {
    "TA_2D_Data":     CONTOUR_TITLES,
    "TA_GA_fit":      CONTOUR_TITLES,
    "TA_GA_residual": CONTOUR_TITLES,
    "TA_Spectra":     ("Wavelength (nm)", None),
}
# 같은 템플릿을 다른 값에 쓰는 시트는 Y 제목을 바꾼다 (Kinetics 템플릿은 ΔA).
SHEET_TITLES = {
    "DADS_kinetics":    (None, "Amplitude"),
    "EADS_populations": (None, "Population"),
}


# ==========================================================================
#  Origin 2018 COM 래퍼
# ==========================================================================
# OriginLab 실행 파일 이름 (Origin95_64.exe, Origin64.exe, 짧은 이름 ORIGIN~1.EXE).
# EA 게임 런처 'Origin.exe' 와 구분하려고 숫자가 붙은 이름만 본다.
ORIGINLAB_EXE = re.compile(r"^(origin\d+(_64)?|origin~\d+)\.exe$", re.I)


def running_origin():
    """실행 중인 OriginLab Origin 프로세스 이미지 이름 목록.
    확인 자체가 실패하면 안전을 위해 실행 중인 것으로 본다."""
    try:
        out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True,
                             text=True, errors="replace", check=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.CalledProcessError):
        return ["(프로세스 확인 실패)"]
    names = [ln.split('","')[0].strip('"') for ln in out.splitlines() if ln]
    return sorted({n for n in names if ORIGINLAB_EXE.match(n)})


class Origin2018:
    def __init__(self):
        try:
            self.app = comtypes.client.CreateObject("Origin.Application")
        except OSError as e:
            raise RuntimeError("Origin 을 시작하지 못했습니다. 이 PC 에 Origin(2018 이상)이 설치되어 "
                               "있는지 확인하세요.\n(%s)" % e)

    def lt(self, script):
        if not self.app.Execute(script):
            raise RuntimeError("Origin 명령 실패: %s" % script)

    def set_text(self, target, text):
        """target(예: 'xb', 'Text2') 의 글을 바꾼다. 글은 Origin 문자열 변수로 넘겨 LabTalk 이
        한글·%·$( 같은 글자를 해석하지 않게 한다."""
        self.app.LTStr["ta_text$"] = str(text)
        self.lt("%s.text$ = ta_text$;" % target)

    def exit(self):
        try:
            self.app.IsModified = False         # 중간에 실패해도 저장 확인 없이 종료되게
        except Exception:
            pass
        try:
            self.app.Exit()
        except Exception:
            pass

    def new_book(self, sheets):
        self.app.NewProject()
        self.lt('newbook name:=%s sheet:=1 option:=lsname;' % BOOK)
        self.lt('win -a %s; page.active=1; wks.name$=%s;' % (BOOK, sheets[0]))
        for s in sheets[1:]:
            self.lt('newsheet book:=%s name:=%s;' % (BOOK, s))

    def put(self, sheet, arr):
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        if not self.app.PutWorksheet("[%s]%s" % (BOOK, sheet), arr, 0, 0):
            raise RuntimeError("데이터 입력 실패: %s" % sheet)
        self._mark_missing(sheet, arr)

    def _mark_missing(self, sheet, arr):
        """PutWorksheet 로 쓴 NaN 은 Origin 의 결측값(--)이 아니어서, 레이어의 마지막 곡선이
        빈 구간 경계에서 세로선으로 그려진다. NaN 구간을 LabTalk 의 0/0(결측값)으로 바꾼다."""
        cmds = []
        for c in range(arr.shape[1]):
            nan = np.isnan(arr[:, c])
            if not nan.any():
                continue
            edges = np.flatnonzero(np.diff(np.r_[0, nan.astype(int), 0]))
            cmds.append("range rr=[%s]%s!col(%d);" % (BOOK, sheet, c + 1))
            cmds += ["for(ii=%d;ii<=%d;ii++) rr[ii]=0/0;" % (a + 1, b)
                     for a, b in zip(edges[::2], edges[1::2])]
        for i in range(0, len(cmds), 200):         # 한 번에 보내는 스크립트 길이를 제한
            chunk = cmds[i:i + 200]
            if not chunk[0].startswith("range"):    # 이어지는 묶음도 대상 열을 다시 지정
                head = next(x for x in reversed(cmds[:i]) if x.startswith("range"))
                chunk = [head] + chunk
            self.lt(" ".join(chunk))

    def get(self, sheet, shape):
        """시트 값을 shape 크기로 읽는다. Origin 이 끝쪽 빈 행·열을 잘라 돌려주면 NaN 으로 채운다."""
        back = np.asarray(self.app.GetWorksheet("[%s]%s" % (BOOK, sheet), 0, 0, -1, -1, 0),
                          dtype=np.float64)
        out = np.full(shape, np.nan)
        if back.ndim == 2:
            r, c = min(shape[0], back.shape[0]), min(shape[1], back.shape[1])
            out[:r, :c] = back[:r, :c]
        return out

    def label(self, sheet, labels):
        """[(열번호(1부터), comment, lname, units), ...]. None 인 항목은 건드리지 않는다.
        COM 속성으로 직접 넣으므로 한글·특수문자도 그대로 들어간다."""
        cols = self.app.FindWorksheet("[%s]%s" % (BOOK, sheet)).Columns
        for col, comment, lname, units in labels:
            c = cols(col - 1)
            if comment is not None:
                c.Comments = comment
            if lname is not None:
                c.LongName = lname
            if units is not None:
                c.Units = units

    def graph_count(self):
        return self.app.GraphPages.Count

    def save(self, path):
        if not self.app.Save(os.path.abspath(path)):
            raise RuntimeError("opju 저장에 실패했습니다: %s" % path)


# ==========================================================================
#  CSV → 시트 데이터 (기존 ta_opju_filler 와 같은 가공 규칙)
#  각 함수는 (배열, [(열번호, comment, lname, units), ...], 메모) 를 반환
# ==========================================================================
def _read_numeric(path, cols=None):
    """CSV 를 (헤더, 2차원 배열)로. 데이터 행이나 Y 열이 없으면 알기 쉬운 오류를 낸다."""
    header, body = read_csv(path)
    arr = body_to_array(body, cols)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        raise ValueError("CSV 에 데이터가 없거나 열이 부족합니다: %s" % os.path.basename(path))
    return header, arr


def _is_minus10(h):
    try:
        return float(h) == -10.0                # '-10', '-10.0', ' -10 ' 모두
    except ValueError:
        return False


def prep_2d(path, drop_minus10):
    header, _ = read_csv(path)
    keep = list(range(len(header)))
    if drop_minus10:
        keep = [0] + [j for j in range(1, len(header)) if not _is_minus10(header[j])]
    _, arr = _read_numeric(path, keep)
    # 지연시간을 Comments 에 둔다 (컨투어의 시간축이 이 값을 쓴다)
    labels = [(i + 1, header[j].strip(), None, None) for i, j in enumerate(keep) if i > 0]
    return arr, labels, "%d×%d%s" % (arr.shape + (" (-10 ps 제외)" if drop_minus10 else "",))


def prep_chirp(path):
    _, arr = _read_numeric(path)
    return arr, [], "%d점" % arr.shape[0]


def smooth_segments(y, win, poly):
    """NaN 구간(펌프 산란 제거 영역 등)을 건너뛰고 연속 구간마다 따로 스무딩한다.
    한 번에 스무딩하면 창이 NaN 을 넘나들어 빈 구간이 넓어진다."""
    out = np.full_like(y, np.nan)
    ok = ~np.isnan(y)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        out[a:b] = _smooth(y[a:b], win, poly)
    return out


def prep_spectra(path, win, poly):
    header, arr = _read_numeric(path)
    wl, raw = arr[:, 0], arr[:, 1:]
    N = raw.shape[1]
    sm = np.column_stack([smooth_segments(raw[:, k], win, poly) for k in range(N)])
    labels = []
    for k in range(N):
        lab = fmt_spectra(header[k + 1])
        labels.append((2 + k, lab, None, None))
        labels.append((2 + N + k, lab, "Smoothed Y%d" % (k + 1), None))
    return np.column_stack([wl, raw, sm]), labels, "%d 스펙트럼 (raw + smoothed 근사)" % N


def prep_kinetics(path):
    header, arr = _read_numeric(path)
    # 불완전(NaN)한 행 제외 → -10 ps 기준행과 말단 행이 빠지며 정답과 같아진다.
    # 통째로 NaN 인 열(예: 펌프 산란 구간의 파장)은 판단에서 빼서 시트 전체가 지워지지 않게 한다.
    nan = np.isnan(arr[:, 1:])
    dead = nan.all(axis=0)
    good = ~nan[:, ~dead].any(axis=1)
    labels = [(2 + k, fmt_kinetics(header[k + 1]), None, None) for k in range(arr.shape[1] - 1)]
    note = "%d kinetics × %d점 (불완전행 %d개 제외)" % (arr.shape[1] - 1, good.sum(), (~good).sum())
    if dead.any():
        note += " ; 주의: 값이 없는 열 %s" % ", ".join(labels[k][1] for k in np.flatnonzero(dead))
    return arr[good], labels, note


def prep_wav_series(path):
    header, arr = _read_numeric(path)
    # 펌프 산란 제거 구간은 CSV 에 모든 성분이 정확히 0 인 행으로 들어 있다 → 빈 셀(NaN)로
    # 바꿔 Spectra 와 같은 빈 구간으로 그린다 (정답 opju 와 같음).
    arr[(arr[:, 1:] == 0).all(axis=1), 1:] = np.nan
    labels = [(1, None, "Wavelength", "nm")]
    labels += [(2 + k, fmt_tau(header[k + 1]), None, None) for k in range(arr.shape[1] - 1)]
    return arr, labels, "%d×%d" % arr.shape


def prep_time_series(path):
    """DADS_kinetics / EADS_populations: 1열=delay, 나머지=성분별 값 (머리글의 tau 로 범례)."""
    header, arr = _read_numeric(path)
    arr = arr[~np.isnan(arr[:, 1:]).all(axis=1)]          # 값이 하나도 없는 행은 뺀다
    m = re.match(r"delay_(\w+)", header[0].strip())
    labels = [(1, None, "Delay", m.group(1) if m else None)]
    labels += [(2 + k, fmt_tau(header[k + 1]), None, None) for k in range(arr.shape[1] - 1)]
    return arr, labels, "%d 성분 × %d점" % (arr.shape[1] - 1, arr.shape[0])


def sheet_jobs(files, ctx):
    """시트 이름 → (CSV 경로, 가공 함수). 경로가 None 이면 빈 시트."""
    return [
        ("Original_Data",        files["cropped_original"], lambda p: prep_2d(p, True)),
        ("Data_Chirp_Corrected", files["chirp_cropped"],    lambda p: prep_2d(p, True)),
        ("Chirp_line",           files["chirp"],            prep_chirp),
        ("Kinetics",             files["pinned_kinetics"],  prep_kinetics),
        ("Spectra",              files["pinned_spectra"],
         lambda p: prep_spectra(p, ctx["smooth_window"], ctx["smooth_poly"])),
        ("EADS",                 files["EADS"],             prep_wav_series),
        ("DADS",                 files["DADS"],             prep_wav_series),
        ("DADS_kinetics",        files.get("DADS_kinetics"),    prep_time_series),
        ("EADS_populations",     files.get("EADS_populations"), prep_time_series),
        ("Global_Fit",           files["GA_fit"],           lambda p: prep_2d(p, False)),
        ("Residual",             files["GA_residual"],      lambda p: prep_2d(p, False)),
    ]


def _same(a, b):
    """NaN 위치까지 같은지. Origin 은 빈 셀을 NaN 으로 돌려준다."""
    if a.shape != b.shape:
        return False
    na, nb = np.isnan(a), np.isnan(b)
    return bool((na == nb).all() and np.array_equal(a[~na], b[~nb]))


# ==========================================================================
#  그래프
# ==========================================================================
def _template(name, tdir=TEMPLATE_DIR):
    p = os.path.join(tdir, name + ".otpu")
    if not os.path.isfile(p):
        raise FileNotFoundError("그래프 템플릿이 없습니다: %s" % p)
    return p


def lt_safe_path(p):
    """LabTalk 명령에 그대로 넣어도 되는 경로인지 (영문·숫자 위주, 특수문자 없음).
    Origin 2018 은 한글이나 [ ] % 가 든 경로의 템플릿으로 그래프를 만들지 못한다."""
    return p.isascii() and not any(ch in p for ch in '%$[]<>"\';')


def _short_path(p):
    """Windows 짧은 경로(8.3). 한글 사용자 이름 폴더도 영문으로 바뀐다. 실패하면 None."""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(p, buf, len(buf))
        return buf.value if 0 < n < len(buf) else None
    except Exception:
        return None


def prepare_templates():
    """그래프 템플릿 폴더를 LabTalk 에 안전한 경로로 준비해 그 경로를 돌려준다.
    프로그램 폴더 경로에 한글·특수문자가 있으면(다른 PC, OneDrive 등) 영문 경로에 복사한다.
    복사하면서 OneDrive '온라인 전용' 파일도 내려받아진다."""
    names = [g[0] for g in GRAPHS]
    for n in names:
        _template(n)                                # 빠진 템플릿이 있으면 여기서 알린다
    if lt_safe_path(TEMPLATE_DIR):
        return TEMPLATE_DIR
    tmp = tempfile.gettempdir()
    bases = [tmp, _short_path(tmp), os.environ.get("PUBLIC"), os.environ.get("ProgramData")]
    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        dst = os.path.join(base, "ta_opju_templates")
        if not lt_safe_path(dst):
            continue
        try:
            os.makedirs(dst, exist_ok=True)
            for n in names:
                shutil.copyfile(_template(n), os.path.join(dst, n + ".otpu"))
            return dst
        except OSError:
            continue
    raise RuntimeError("그래프 템플릿을 둘 영문 경로를 찾지 못했습니다.\n"
                       "프로그램을 영문 경로(예: C:\\TA_tool)에 두고 실행하세요.")


def set_titles(org, tmpl, sheet=None):
    x, y = SHEET_TITLES.get(sheet) or TITLES.get(tmpl, (None, None))
    if x is not None:
        org.set_text("xb", x)
    if y is not None:
        org.set_text("yl", y)


def plotted_cols(sheet, kind, ncols):
    """그래프에 쓰는 Y(컨투어는 값) 열 범위 (1부터 센 첫 열, 끝 열)."""
    if sheet == "Spectra" and kind != "contour":    # smoothed 열만 그린다
        n = (ncols - 1) // 2
        return 2 + n, 1 + 2 * n
    return 2, ncols


def nice_step(span, ticks=5):
    """span 을 대략 ticks 칸으로 나누는 1·2·5×10^k 간격."""
    raw = span / float(ticks)
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag


def _inside(v, lim):
    return (v >= lim[0]) & (v <= lim[1])


def auto_axes(tmpl, arr, y1, y2, delays=None):
    """AXES[tmpl] 에 보이는 데이터로 다시 정한 컨투어 단계·Y 범위를 덮어쓴 사본.
    delays: 컨투어 값 열(y1..y2)의 지연시간 (없으면 Y 방향은 거르지 않음)."""
    a = dict(AXES[tmpl])
    vals = arr[_inside(arr[:, 0], a["x"]), y1 - 1:y2]
    if "z" in a and delays is not None and len(delays) == vals.shape[1]:
        vals = vals[:, _inside(np.asarray(delays, dtype=float), a["y"])]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return a
    if "z" in a:
        m = max(int(np.ceil(np.abs(vals).max())), 1)
        a["z"] = (-m, m) + a["z"][2:]
    elif tmpl in AUTO_Y:
        lo, hi = int(np.floor(vals.min())), int(np.ceil(vals.max()))
        if hi <= lo:
            hi = lo + 1
        a["y"] = (lo, hi, nice_step(hi - lo))
    return a


def _delays(comments):
    """컨투어 열 Comments(지연시간 글) → 숫자. 읽을 수 없으면 None."""
    try:
        return [float(c) for c in comments]
    except (TypeError, ValueError):
        return None


def make_graph(org, tmpl, sheet, kind, arr, legends=(), tdir=TEMPLATE_DIR):
    """arr 는 시트에 기록한 값 (None 이면 빈 그래프 틀). legends 는 곡선 순서의 범례 글."""
    t = _template(tmpl, tdir)
    rng = "[%s]%s!" % (BOOK, sheet)
    if arr is None:
        org.lt('win -t plot "%s";' % t)
        if tmpl == "TA_Kinetics":
            fix_kinetics_text(org, [])
        set_titles(org, tmpl, sheet)
        return "빈 틀"
    y1, y2 = plotted_cols(sheet, kind, arr.shape[1])
    if kind == "contour":
        # 1열=파장(X), 나머지 열=지연시간(Y, 각 열 Comments 의 값)
        org.lt('plotvm irng:=%s(2:%d) format:=yacross rowpos:=label label:=C colpos:=xcol '
               'type:=%d ogl:=[<new template:="%s">];' % (rng, y2, PLOT_TYPE[kind], t))
        note = "컨투어 %d열" % (y2 - 1)
    else:
        org.lt('plotxy iy:=%s(1,%d:%d) plot:=%d rescale:=0 ogl:=[<new template:="%s">];'
               % (rng, y1, y2, PLOT_TYPE[kind], t))
        note = "%d개 곡선" % (y2 - y1 + 1)
    a = auto_axes(tmpl, arr, y1, y2, _delays(legends) if kind == "contour" else None)
    restore_axes(org, a)
    set_titles(org, tmpl, sheet)
    if tmpl == "TA_Kinetics":
        fix_kinetics_text(org, legends)
    if "z" in a:
        note += ", 색상 ±%g" % a["z"][1]
    elif tmpl in AUTO_Y:
        note += ", Y %g~%g" % a["y"][:2]
    return note


def fix_kinetics_text(org, legends):
    remove = list(KINETICS_FIT_TEXT)
    if len(legends) <= len(KINETICS_LABELS):
        for name, text in zip(KINETICS_LABELS, legends):
            org.set_text(name, text)
        remove += KINETICS_LABELS[len(legends):]
        auto_legend = False
    else:                                       # 글상자가 모자라면 자동 범례로 바꾼다
        remove += KINETICS_LABELS
        auto_legend = True
    org.lt("".join("label -r %s; " % n for n in remove) + ("legend -r;" if auto_legend else ""))


def restore_axes(org, a):
    """a: AXES 항목 형식의 축 범위·색상 단계."""
    s = ""
    for ax in ("x", "y"):
        lo, hi, inc = a[ax]
        s += "layer.%s.from=%g; layer.%s.to=%g; layer.%s.inc=%g; " % (ax, lo, ax, hi, ax, inc)
    org.lt(s)
    if "z" in a:
        zmin, zmax, major, minor = a["z"]
        org.lt("layer.cmap.zmin=%g; layer.cmap.zmax=%g; layer.cmap.numMajorLevels=%d; "
               "layer.cmap.numMinorLevels=%d; layer.cmap.SetLevels();" % (zmin, zmax, major, minor))


# ==========================================================================
#  전체 처리
# ==========================================================================
def build(folder, out_path=None, files=None, progress_cb=None,
          smooth_window=15, smooth_poly=2):
    """folder 의 CSV 로 새 opju 를 만들어 out_path 에 저장. (out_path, 로그) 반환."""
    folder = os.path.abspath(folder)
    if out_path is None:
        out_path = os.path.join(folder, OUTPUT_NAME)
    if files is None:
        files = discover(folder, required=False)
    if not any(files.values()):
        raise FileNotFoundError("폴더에서 처리할 CSV 를 하나도 찾지 못했습니다:\n%s" % folder)
    busy = running_origin()
    if busy:
        raise RuntimeError("Origin 이 실행 중입니다 (%s).\n작업을 저장하고 Origin 을 모두 닫은 뒤 "
                           "다시 실행하세요." % ", ".join(busy))
    tdir = prepare_templates()
    jobs = sheet_jobs(files, {"smooth_window": smooth_window, "smooth_poly": smooth_poly})
    total = len(jobs) + len(GRAPHS) + 2

    def step(i, msg):
        if progress_cb:
            progress_cb(i, total, msg)

    log, data, comments = [], {}, {}
    org = Origin2018()
    try:
        step(0, "새 프로젝트 만드는 중")
        org.new_book(SHEETS)
        for i, (sheet, path, prep) in enumerate(jobs, start=1):
            if path is None:
                step(i, "%s ← (CSV 없음, 빈 시트)" % sheet)
                log.append("[--] %-21s CSV 없음 → 빈 시트" % sheet)
                data[sheet] = None
                continue
            step(i, "%s ← %s" % (sheet, os.path.basename(path)))
            arr, labels, note = prep(path)
            arr[np.isinf(arr)] = np.nan             # ±inf 는 그릴 수 없으므로 빈 셀로
            org.put(sheet, arr)
            org.label(sheet, labels)
            if not _same(org.get(sheet, arr.shape), arr):
                raise RuntimeError("기록 확인 실패: %s 시트 값이 CSV 와 다릅니다." % sheet)
            data[sheet] = arr
            comments[sheet] = [c for _, c, _, _ in sorted(labels) if c is not None]
            log.append("[OK] %-21s %s ← %s" % (sheet, note, os.path.basename(path)))
        base = len(jobs)
        for k, (tmpl, sheet, kind) in enumerate(GRAPHS, start=1):
            step(base + k, "그래프 %d (%s)" % (k, sheet))
            note = make_graph(org, tmpl, sheet, kind, data[sheet], comments.get(sheet, ()), tdir)
            log.append("[OK] Graph%d %-15s %s" % (k, sheet, note))
        step(total - 1, "저장 중: %s" % os.path.basename(out_path))
        org.save(out_path)
        step(total, "완료 (그래프 %d개)" % org.graph_count())
    finally:
        org.exit()
    return out_path, log


# ==========================================================================
#  GUI (Tkinter)
# ==========================================================================
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("Transient Absorption · 새 opju 만들기 (Origin 2018)")
    root.geometry("720x470")
    root.minsize(640, 420)

    msg_q = queue.Queue()
    state = {"out_path": None, "running": False}

    def on_close():
        # 작업 중에 창을 닫으면 숨은 Origin 이 남아 다음 실행이 막히므로 끝날 때까지 기다린다.
        if state["running"]:
            messagebox.showinfo("작업 중", "opju 를 만드는 중입니다. 끝난 뒤에 닫아 주세요.")
            return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    top = ttk.LabelFrame(root, text="① 데이터 폴더 지정")
    top.pack(fill="x", padx=12, pady=(12, 6))
    folder_var = tk.StringVar()
    row = ttk.Frame(top); row.pack(fill="x", padx=10, pady=10)
    ttk.Label(row, text="폴더:").pack(side="left")
    ttk.Entry(row, textvariable=folder_var).pack(side="left", fill="x", expand=True, padx=6)

    def browse():
        d = filedialog.askdirectory(title="CSV 가 있는 폴더 선택")
        if d:
            folder_var.set(d)

    ttk.Button(row, text="찾아보기…", command=browse).pack(side="left")
    ttk.Label(top, text="결과: 폴더 안의 '%s'" % OUTPUT_NAME,
              foreground="#555").pack(anchor="w", padx=10, pady=(0, 8))

    run_btn = ttk.Button(root, text="새 opju 만들기 ▶")
    run_btn.pack(pady=6)

    logf = ttk.LabelFrame(root, text="처리 로그")
    logf.pack(fill="both", expand=True, padx=12, pady=6)
    log_txt = tk.Text(logf, height=9, wrap="word", state="disabled", font=("Consolas", 9))
    log_txt.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
    sb = ttk.Scrollbar(logf, command=log_txt.yview); sb.pack(side="right", fill="y", pady=8)
    log_txt.config(yscrollcommand=sb.set)

    def log_line(s):
        log_txt.config(state="normal")
        log_txt.insert("end", s + "\n")
        log_txt.see("end")
        log_txt.config(state="disabled")

    bottom = ttk.LabelFrame(root, text="② 진행 상황")
    bottom.pack(fill="x", padx=12, pady=(6, 12))
    brow = ttk.Frame(bottom); brow.pack(fill="x", padx=10, pady=10)
    prog = ttk.Progressbar(brow, mode="determinate", maximum=100)
    prog.pack(side="left", fill="x", expand=True)
    status_var = tk.StringVar(value="대기 중")
    open_btn = ttk.Button(brow, text="결과 opju 열기 ↗", state="disabled")
    open_btn.pack(side="right", padx=(10, 0))
    ttk.Label(bottom, textvariable=status_var, foreground="#555").pack(anchor="w", padx=10, pady=(0, 8))

    def worker(folder, out, files):
        comtypes.CoInitialize()
        try:
            out, log = build(folder, out, files=files,
                             progress_cb=lambda s, t, m: msg_q.put(("progress", s, t, m)))
            msg_q.put(("done", out, log))
        except Exception as e:
            msg_q.put(("error", "%s\n\n%s" % (str(e) or type(e).__name__, traceback.format_exc())))
        finally:
            comtypes.CoUninitialize()

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

        b = ttk.Frame(dlg); b.pack(pady=10)
        ttk.Button(b, text="선택", command=ok).pack(side="left", padx=4)
        ttk.Button(b, text="취소", command=dlg.destroy).pack(side="left", padx=4)
        lb.bind("<Double-Button-1>", ok)
        root.wait_window(dlg)
        if "path" not in result:
            raise SelectionCancelled()
        return result["path"]

    def start():
        folder = folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("경로 확인", "유효한 데이터 폴더를 선택하세요.")
            return
        busy = running_origin()
        if busy:
            messagebox.showwarning("Origin 실행 중",
                                   "Origin 이 실행 중입니다 (%s).\n작업을 저장하고 Origin 을 모두 닫은 뒤 "
                                   "다시 실행하세요." % ", ".join(busy))
            return
        # CSV 탐색과 후보 선택은 메인 스레드에서 끝내고, 워커에는 확정된 경로만 넘긴다.
        try:
            files = discover(folder, choose=ask_choice, required=False)
        except SelectionCancelled:
            status_var.set("취소됨")
            return
        if not any(files.values()):
            messagebox.showwarning("CSV 없음", "이 폴더에서 처리할 CSV 를 찾지 못했습니다.")
            return
        out = os.path.join(folder, OUTPUT_NAME)
        if os.path.exists(out) and not messagebox.askyesno(
                "덮어쓰기 확인", "'%s' 가 이미 있습니다.\n덮어쓸까요?" % OUTPUT_NAME):
            return
        state["running"] = True
        run_btn.config(state="disabled")
        open_btn.config(state="disabled")
        prog.config(value=0)
        log_txt.config(state="normal"); log_txt.delete("1.0", "end"); log_txt.config(state="disabled")
        status_var.set("처리 시작…")
        threading.Thread(target=worker, args=(folder, out, files), daemon=True).start()

    def open_result():
        p = state["out_path"]
        if p and os.path.isfile(p):
            try:
                os.startfile(p)
            except Exception as e:
                messagebox.showerror("열기 실패", str(e))

    run_btn.config(command=start)
    open_btn.config(command=open_result)

    def poll():
        try:
            while True:
                item = msg_q.get_nowait()
                if item[0] == "progress":
                    _, step, tot, m = item
                    prog.config(value=step * 100.0 / tot)
                    status_var.set("[%d/%d] %s" % (step, tot, m))
                    log_line("[%d/%d] %s" % (step, tot, m))
                elif item[0] == "done":
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
                else:
                    _, msg = item
                    state["running"] = False
                    status_var.set("오류 발생")
                    log_line("\n[오류] " + msg)
                    run_btn.config(state="normal")
                    first = next((ln for ln in msg.splitlines() if ln.strip()), "알 수 없는 오류")
                    messagebox.showerror("처리 오류", first)
        except queue.Empty:
            pass
        root.after(80, poll)

    root.after(80, poll)
    root.mainloop()


# ==========================================================================
if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        launch_gui()
        sys.exit(0)
    out = args[1] if len(args) > 1 else os.path.join(args[0], OUTPUT_NAME)
    if os.path.exists(out) and "--overwrite" not in sys.argv:
        if input("'%s' 가 이미 있습니다. 덮어쓸까요? [y/N] " % out).strip().lower() != "y":
            print("취소했습니다.")
            sys.exit(1)
    out, log = build(args[0], out,
                     progress_cb=lambda s, t, m: print("  [%d/%d] %s" % (s, t, m), flush=True))
    print("\n결과:", out)
    print("\n".join(log))
