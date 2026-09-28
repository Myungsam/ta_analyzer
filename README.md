# TA Data Analyzer (Python port)

MATLAB `TRSpecAnalyzer.m`(8,326줄)를 Python/PyQt5로 옮긴 프로그램입니다.
시간분해 transient absorption(TA) 분광 데이터를 불러오고, 보정하고, 시각화하고, 여러 방식으로 분석합니다.

> 전체 로직과 모듈 구조는 [`ARCHITECTURE.md`](ARCHITECTURE.md)에 있습니다.
> 사용법은 `manual/`의 Quick/Detailed 매뉴얼(한/영, md·pdf)을 보세요.

## 실행

```bash
python ta_main.py
```

또는 빌드된 실행 파일 `dist/TA_Analyzer.exe`를 실행합니다.

## 의존성

```bash
pip install PyQt5 numpy scipy matplotlib openpyxl
```

| 패키지 | 필수 여부 | 용도 |
|---|---|---|
| `PyQt5`, `numpy`, `scipy`, `matplotlib` | 필수 | GUI, 수치 계산, 플롯 |
| `openpyxl` | 권장 | Excel(.xlsx) export, residual 파일 저장과 로드 (`residuals/*.xlsx`) |
| `tensorflow` (2.10.x) | 선택 | Global Analysis의 CPU/GPU 장치 선택. 없으면 NumPy로 동작 |
| `pynvml` | 선택 | GPU 상태와 팬 제어. 없으면 `nvidia-smi`로 대체 |

## 빌드 (Windows 단일 실행 파일)

```bash
python -m PyInstaller ta_analyzer.spec --noconfirm --clean
# → dist/TA_Analyzer.exe  (one-file, console 없음)
```

## 모듈 구성

| 파일 | 역할 |
|---|---|
| `ta_main.py` | 메인 윈도우. 앱 상태, 보정 파이프라인, 2D map/spectrum/kinetics 패널, toolbar, crop/보간/t₀ 이동, export |
| `ta_core.py` | GUI와 무관한 수치 커널과 파일 I/O (파싱, chirp, solvent 정렬, 결측 보간, IRF 컨볼루션, GA, LDA, MCR, FFT 등) |
| `ta_widgets.py` | `MplCanvas` 등 Qt/matplotlib 공용 헬퍼 |
| `ta_accumulate.py` | Load Data 방식 선택 다이얼로그, 폴더 기반 **Accumulation** 로더 |
| `ta_load_custom.py` | **Custom** CSV 로더 (X/Y/Z 영역을 직접 지정) |
| `ta_dialogs_a.py` | Background, Crop(+drop 보간), Mask, Load & Average 다이얼로그 |
| `ta_chirp.py` | Chirp correction 다이얼로그 (Sellmeier fit) |
| `ta_solvent_irf.py` | 순수 용매 IRF / coherent-artifact 빼기 다이얼로그 |
| `ta_ga.py` | Global Analysis 다이얼로그 (stretched-exp, 최적화기와 장치 선택) |
| `ta_device.py` | GA용 TensorFlow CPU/GPU backend, GPU 모니터링과 팬 제어 |
| `ta_svd.py` | SVD analysis 다이얼로그 |
| `ta_kfit.py` | Single-trace kinetic fit 다이얼로그 |
| `ta_lda.py` | Lifetime Density Analysis 다이얼로그 |
| `ta_mcr.py` | MCR-ALS 다이얼로그 |
| `ta_coherence.py` | Vibrational coherence 다이얼로그 (FFT map 탭 + LPSVD 탭) |
| `ta_lpsvd.py` | LPSVD(Lorentzian)와 Gaussian 감쇠 진동자 모드 분해 커널 |
| `ta_residual_store.py` | 분석 residual을 `residuals/` 폴더에 저장하고 불러오기 |
| `ta_analyzer.spec` | PyInstaller 빌드 스펙 |

## 핵심 기능

### Data I/O
- **Load Data** 버튼에서 로딩 방식 세 가지 중 하나를 고릅니다.
  - **Standard**: 파일 하나를 자동 파싱합니다. 여러 파일을 고르면 Load & Average 창이 열려 평균을 냅니다.
  - **Custom**: 스프레드시트 형태의 미리보기에서 X(λ), Y(t), Z(ΔA) 영역을 드래그로 지정합니다. 전치는 자동으로 처리하고, 자동 감지 기능도 있습니다.
  - **Accumulation**: 폴더 안의 반복 측정 파일(`.csv/.tsv/.dat/.txt`)을 하나씩 2D map으로 미리 보고, 체크한 파일만 평균합니다.
- CSV / TSV / DAT / TXT 자동 파싱 (Format A: MATLAB 스타일, Format B: 앞 2열이 메타데이터)
- NaN, Inf, 빈 셀 자동 처리
- `numpy.genfromtxt` 기반 빠른 파싱 (2136×192 데이터 약 370 ms)
- 2D ΔA 행렬을 CSV / TSV / Excel(.xlsx)로 export. 보정된 데이터와 로딩 원본(`Export 2D data_original`) 중 선택
- Pinned spectra/kinetics를 CSV로 export
- 저장 다이얼로그는 기본적으로 불러온 데이터가 있는 폴더에서 열립니다

### 시각화
- 2D heatmap (delay 축 linear / log / split 모드, colormap 선택, Z 범위 지정)
- Spectrum / kinetics 패널. **사용자가 확대한 상태를 유지**합니다 (delay나 wavelength를 바꿔도 그대로)
- 패널 우클릭 메뉴: 축별 Auto scale X/Y 토글, Set X/Y range
- Pan/Zoom 후 Home 버튼을 누르면 zoom 캐시가 지워집니다
- Display t_min/t_max로 2D map y축 범위 조절
- Pinned overlays (여러 t/λ를 동시에 표시)
- Delay 인덱스 기반 입력 (실제 측정된 delay 점으로만 이동)
- 시간 단위 ps / µs 전환
- 레이아웃이 바뀌지 않으면 그림을 새로 만들지 않고 값만 갱신하는 빠른 redraw

### 보정

보정은 원본을 바꾸지 않고 매번 raw 데이터에서 다시 계산합니다. 적용 순서는 다음과 같습니다.
**raw → Background → Chirp → Solvent IRF 빼기 → Wavelength mask**

- **Background subtraction**: pre-trigger 구간 앞쪽 N개 delay의 평균을 뺍니다
- **Chirp correction**: 4-parameter Sellmeier fit. 점은 클릭으로 찍거나 ridge(max\|dA/dt\| 또는 max\|dA\|)로 자동 배치합니다. 점 목록은 CSV로 저장하고 불러올 수 있습니다
- **Solvent IRF subtraction**: 순수 용매 측정을 샘플 grid에 맞춰 리샘플링하고, 샘플과 같은 chirp 보정을 적용한 뒤 `ΔA − s·ΔA_solvent`를 계산합니다. scale `s`는 슬라이더로 조절하거나 t≈0 근처에서 자동 추정할 수 있습니다. toolbar 체크박스로 켜고 끕니다
- **Wavelength masking**: 지정한 구간을 NaN 또는 0으로 바꿉니다
- **Crop**: λ, t 범위를 제한합니다. 새로 figure를 만들어 colorbar가 쌓이지 않게 합니다
  - 특정 delay나 wavelength를 빼고 보간할 수 있습니다 (linear / cubic / pchip / akima / bilinear)
- **Set t=0 here**: 크로스헤어 위치를 t=0으로 옮깁니다
- **Reset Corrections**: 보정을 모두 해제합니다. 불러온 용매 데이터는 유지됩니다

### 분석 (More Analysis ▾ 메뉴)

| 분석 | 설명 |
|---|---|
| **SVD** | Singular value spectrum + U/V basis vectors + rank-N 재구성 + residual map |
| **Kinetic Fit** | 단일 trace fit (multi-exp + IRF + 선택적 stretched β). Residual을 CSV로 저장 |
| **LDA** | Tikhonov 정규화 lifetime density (l2 / l2-derivative) + L-curve 도우미 |
| **MCR-ALS** | Non-negativity / unimodality 제약 (SVD / random / custom 초기값) |
| **Coherence** | 탭 ① residual의 2D \|FFT\|² (apod, zero-pad, detrend, 주파수 단위 cm⁻¹/THz/Hz) · 탭 ② **LPSVD** 모드 분해 (Lorentzian / Gaussian 모델, 전처리, FFT window) |

### Global Analysis
- **5컬럼 tau 테이블**: tau_init / Fixed / **Stretched** / **β init** / **β Fixed**
- 성분별 stretched-exponential 토글 (Kohlrausch 함수 + IRF 컨볼루션)
- IRF mode 선택 (numerical convolution / skip)
- τ=∞ offset 성분 선택 가능, IRF t₀와 FWHM 고정 또는 피팅
- Fit 구간(t_min ~ t_max) 직접 지정
- **최적화기 선택**: TRF(`least_squares`, 기본값) / Nelder-Mead(legacy). 비선형 파라미터만 최적화하고 진폭은 선형 최소제곱으로 푸는 VARPRO 방식
- **계산 장치 선택**: TensorFlow가 설치되어 있으면 CPU/GPU를 고를 수 있습니다. GPU 상태 모니터링, 팬 제어, keep-warm 기능 포함
- 실행 중 **Stop** 가능
- 결과: τ, β, IRF t₀, IRF FWHM, RMS, DADS, EADS, fit, residual, 클릭한 파장의 kinetics
- Export: DADS / EADS / fit / residual / kinetics CSV

### Residual 공유 (Coherence 분석용)
- GA와 LDA 창의 **Save residual** 버튼을 누르면 `residuals/<Analysis>_<dataset>_<timestamp>.xlsx`가 저장됩니다 (시트: `Residual2D` 또는 `Residual1D`, `Meta`)
- Coherence 창에서 residual 소스를 고릅니다: raw ΔA / 마지막 GA 결과 / 마지막 LDA 결과 / 파일(`residuals/` 목록 또는 Browse로 xlsx·csv 선택. Kfit residual CSV 포함)

## 검증

`test/` 폴더에 테스트 스크립트 23개가 있습니다. 각 파일을 그대로 실행하면 됩니다 (`pytest` 불필요).

```bash
# 프로젝트 루트에서 실행
set PYTHONPATH=.                 # 일부 테스트는 sys.path를 직접 설정하지 않음
set PYTHONIOENCODING=utf-8       # 한국어 Windows 콘솔(cp949)에서 —, ≈ 같은 문자 출력 오류 방지
python test/test_integration.py
# headless 환경에서는 QT_QPA_PLATFORM=offscreen 추가
```

**최근 실행 결과 (2026-09-28)**: 23개 중 18개 통과
- 실제 측정 파일 경로가 테스트 코드에 고정되어 있어 이 PC에서 실행되지 않는 테스트 4개: `test_real_file`, `test_session_fixes`, `test_crop_input_fix`(`/mnt/user-data/uploads/...`), `test_crop_preview_perf`(`Data/A_MAPbI3/...`)
- 실패 1개: `test_new_features` [4.SpecZoom]. delay를 바꾼 뒤 spectrum zoom이 유지되지 않습니다. 이후 추가된 축별 Auto scale 동작과 테스트의 가정이 맞지 않는 것으로 보이며, 원인은 아직 확인하지 않았습니다

| 영역 | 테스트 |
|---|---|
| 통합 / 실제 데이터 | `test_integration`, `test_real_file`, `test_consistency`, `test_session_fixes`, `test_new_features` |
| 로딩 | `test_accumulate`, `test_custom_load` |
| Crop / 보간 | `test_crop_input_fix`, `test_crop_interpolate`, `test_crop_interpolate_wl`, `test_crop_preview_perf` |
| 보정 | `test_chirp_pts_io`, `test_solvent_irf` |
| 분석 커널 / 다이얼로그 | `test_advanced_analysis`, `test_advanced_dialogs`, `test_lpsvd`, `test_lpsvd_export_meta`, `test_lpsvd_replot` |
| Residual | `test_residual_store`, `test_save_residual_buttons` |
| Export | `test_export_2d_original`, `test_pin_export`, `test_plot_export` |

## Synthetic 데이터 검증 결과

- **Stretched-exp GA**: 합성 데이터 (τ=[0.5, 8 stretched β=0.7, 200] ps) → 회복 [0.504, 7.965, 199.28], β=**0.698** ✓
- **Single-trace fit**: 노이즈 포함 데이터에서 τ=[0.5, 50] → 회복 [0.502, 49.916] ✓
- **Coherence**: 50 cm⁻¹ 입력 → 49 cm⁻¹ 검출 ✓
- **MCR-ALS**: LOF 3.09%, 13 iterations 수렴 ✓
- **SVD**: top-3 components가 에너지의 99.91%를 설명, rank-3 재구성 RMS 0.005 ✓

## 알려진 제약

- 다이얼로그의 `Reset zoom` 버튼은 제거되었습니다. toolbar Home 버튼이 같은 기능을 합니다
- Coherence 분석은 t-grid의 Nyquist 한계를 넘는 주파수를 검출할 수 없습니다 (log 간격 delay grid에서는 보통 약 5 cycles/ps 이하)
- 2D map의 빠른 redraw는 레이아웃이 바뀌면 자동으로 전체 rebuild로 전환됩니다
- 새 데이터를 불러오거나 Crop하면 BG, chirp, mask, t₀ 이동, GA 결과가 초기화됩니다 (예전 grid 기준으로 계산된 값이기 때문)
- 긴 계산(GA, LDA, MCR, LPSVD)은 GUI 스레드에서 `processEvents()`로 진행됩니다. 별도 워커 스레드는 없습니다
