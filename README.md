# TA Analyzer — Transient Absorption 데이터 분석기

**한국어** | [English](README_EN.md)

![TA Analyzer 메인 창](docs/images/main_window.png)

## 1. 개요

TA Analyzer는 펌프-프로브 transient absorption(TA) 측정 데이터 ΔA(λ, t)를 불러와 보정하고, 시각화하고, 분석하는 Windows용 GUI 프로그램입니다. MATLAB `TRSpecAnalyzer.m`을 Python/PyQt5로 옮겼습니다.

주요 기능:
- **로딩**: 단일 파일 자동 인식, 여러 파일 평균, 폴더의 반복 측정 누적(Accumulation), 레이아웃을 직접 지정하는 Custom 로더
- **보정**: Background, Chirp(Sellmeier), 용매 IRF 빼기, 파장 마스크, t=0 이동, Crop과 **파장축 Resampling**(v1.1.0), **Moving average 평활**(v1.2.0), 특정 delay/λ 제거 후 보간
- **분석**: Global Analysis(DADS/EADS, stretched exponential), SVD, 단일 trace Kinetic Fit, LDA, MCR-ALS, Coherence(FFT 맵, LPSVD)
- **내보내기**: 2D 행렬(CSV/TSV/xlsx), 고정(pin)한 스펙트럼과 kinetics, 분석 결과와 residual

## 2. 버전과 받는 법

| 버전 | 태그 | 내용 |
|---|---|---|
| **v1.4.0** (최신) | [`v1.4.0`](https://github.com/Myungsam/ta_analyzer/releases/tag/v1.4.0) | Global Analysis σ·fit 통계, Kinetic Fit에 pin import·global fit·population ratio 추가 |
| v1.3.0 | [`v1.3.0`](https://github.com/Myungsam/ta_analyzer/releases/tag/v1.3.0) | Spectrum/Kinetics 패널에 구간 평균(Integrate: delay ±idx, 파장 BW) 추가 |
| v1.2.0 | [`v1.2.0`](https://github.com/Myungsam/ta_analyzer/releases/tag/v1.2.0) | Crop Data에 Moving average(pixel window, Box / Triangular / Gaussian) 추가 |
| v1.1.0 | [`v1.1.0`](https://github.com/Myungsam/ta_analyzer/releases/tag/v1.1.0) | Crop Data에 파장축 Resampling(Average / Decimate) 추가, 한/영 사용 설명서 |
| v1.0.0 | [`v1.0.0`](https://github.com/Myungsam/ta_analyzer/releases/tag/v1.0.0) | Resampling 이전 버전 |

**실행 파일로 받기 (Python 불필요)**
[Releases](https://github.com/Myungsam/ta_analyzer/releases) 페이지에서 원하는 버전의 `TA_Analyzer.exe`를 받아 실행합니다.

**소스로 받기**
```bash
git clone https://github.com/Myungsam/ta_analyzer.git
cd ta_analyzer
git checkout v1.4.0      # 최신 버전 (main과 같음)
git checkout v1.3.0      # fit 통계·pin import 이전
git checkout v1.2.0      # 구간 평균(Integrate) 이전
git checkout v1.1.0      # Moving average 이전
git checkout v1.0.0      # Resampling 이전
```
git 없이 받으려면 Releases 페이지의 **Source code (zip)** 를 쓰면 됩니다.

## 3. 설치와 실행

**실행 파일**: `TA_Analyzer.exe`를 더블클릭합니다. 한 파일짜리 실행 파일이라 처음 켤 때 몇 초 걸릴 수 있습니다.

**Python으로 실행** (Python 3.10 이상):
```bash
pip install PyQt5 numpy scipy matplotlib openpyxl
python ta_main.py
```

| 패키지 | 필수 여부 | 용도 |
|---|---|---|
| PyQt5, numpy, scipy, matplotlib | 필수 | GUI, 계산, 그래프 |
| openpyxl | 권장 | Excel(.xlsx) 내보내기, residual 파일 |
| tensorflow 2.10.x, pynvml | 선택 | Global Analysis의 GPU 계산과 GPU 상태 표시 |

## 4. 빠른 시작: 표준 워크플로

1. **Load Data...** → *Standard*를 고르고 `_TA_spectra_Accumulated.csv` 같은 파일을 엽니다.
2. **Crop Data...** 에서 λ/t 범위를 정하고, 필요하면 **Resample / smooth λ**를 켜서 파장 점 수를 줄이거나(Average/Decimate) 스펙트럼을 평활합니다(Moving avg, 5.3절).
3. **Background Correction...** 에서 t<0 구간 앞쪽 N개 delay의 평균을 뺍니다.
4. **Chirp Correction...** 에서 2D 맵을 클릭하거나 *Auto-place*로 t₀(λ) 점을 찍고 **Fit & Apply**를 누릅니다.
5. (선택) **Subtract solvent IRF**, **Mask Wavelengths...**
6. **Global Analysis...** 에서 성분 수와 τ 초깃값을 넣고 **Run** → DADS/EADS 확인
7. **Export 2D Data...**, 각 분석 창의 Export 버튼으로 결과를 저장합니다.

2D 맵을 클릭하면 오른쪽 Spectrum(선택한 t)과 Kinetics(선택한 λ) 패널이 바뀝니다. **Pin**을 누르면 여러 개를 겹쳐 볼 수 있고, 아랫줄 **Integrate**를 켜면 delay/파장 구간 평균을 봅니다(5.6절).

## 5. 기능별 사용법

### 5.1 데이터 불러오기
- **Standard**: 파일 하나를 자동 인식합니다. 여러 개를 고르면 *Load & Average* 창이 열려 체크한 파일만 평균합니다.
- **Custom**: 표 미리보기에서 X(파장), Y(delay), Z(ΔA) 영역을 드래그로 지정합니다. 방향이 바뀐 행렬은 자동으로 전치합니다.
- **Accumulation**: 폴더의 반복 측정(`_TA_spectra_Current_Set_*.csv` 등)을 하나씩 2D 맵으로 확인하고, 깨끗한 것만 체크해 평균합니다.

### 5.2 보정
보정은 원본을 바꾸지 않고 매번 다시 계산하며, 순서는 **Background → Chirp → 용매 IRF 빼기 → 마스크**입니다. **Reset Corrections**를 누르면 전부 해제됩니다(불러온 용매 데이터는 유지).
- **Background**: 앞쪽 N개 delay의 평균 스펙트럼을 뺍니다.
- **Chirp**: 점 (λ, t₀)을 찍으면 `t₀(λ) = a·√((bλ²−1)/(cλ²−1)) + d`로 피팅해 각 파장을 시간축으로 이동합니다. 점 목록은 CSV로 저장하고 불러올 수 있습니다.
- **용매 IRF 빼기**: 순수 용매 측정을 불러와 샘플 grid에 맞추고 `ΔA − s·ΔA_solvent`를 계산합니다. scale `s`는 슬라이더나 *Auto*로 정합니다.
- **Mask**: 파장 구간을 NaN 또는 0으로 바꿉니다.
- **Set t=0 here**: 크로스헤어 위치를 t=0으로 옮깁니다.

### 5.3 Crop과 파장축 Resampling / Moving average

![Crop Data 창의 Resampling](docs/images/crop_resample.png)

**Crop Data...** 창에서 λ/t 범위를 자르고, 선택적으로 파장축의 점 수를 줄이거나(Average / Decimate, v1.1.0) 점 수는 그대로 둔 채 평활합니다(Moving avg, v1.2.0). 세 모드 중 하나만 적용됩니다.

1. `λ range`와 `t range`를 입력합니다(*Full λ range* / *Full t range*로 전체 범위).
2. **Resample / smooth λ**를 체크하고 **Δλ (nm)** 에 원하는 간격을 입력합니다.
3. **Average** 또는 **Decimate**를 고릅니다.
4. 오른쪽 Spectrum 패널에서 원본(회색 선)과 결과(검은 점+선)를 비교하고, `1200 → 79 λ points`처럼 표시되는 점 수를 확인합니다.
5. **Apply**를 누릅니다.

**동작 방식**
- 잘라 낸 범위의 첫 파장 λ_min부터 폭 Δλ인 구간(bin) `[λ_min + kΔλ, λ_min + (k+1)Δλ)`으로 나눕니다. 경계에 정확히 걸린 점은 위쪽 구간에 들어갑니다.
- **Average**: 각 구간에 들어간 모든 점의 파장과 ΔA를 평균합니다(NaN은 제외, 전부 NaN이면 NaN). 잡음이 줄어듭니다.
- **Decimate**: 각 구간의 중심에 가장 가까운 원본 점 하나만 남기고 나머지는 버립니다. 값은 평균하지 않은 실제 측정값입니다.
- 원본 점이 하나도 없는 구간(예: 마스크나 측정 공백)은 결과에서 빠집니다.
- 측정 파장 간격은 보통 균일하지 않으므로(예: 0.27–0.38 nm), 결과 파장도 정확히 Δλ 간격이 아니라 "대략 Δλ 간격"입니다.

**주의사항**
- Crop과 마찬가지로 Apply하면 **Background, Chirp, Mask, t=0 이동, Global Analysis 결과가 초기화**됩니다. Resampling을 먼저 하고 보정을 나중에 하세요.
- Δλ가 원본의 평균 파장 간격 이하이면 경고가 뜨고, **resampling 없이 crop만 적용**됩니다.
- 용매 IRF 빼기를 쓰는 경우 용매 데이터도 샘플과 **똑같은 구간 묶음**으로 줄여서 뺍니다.
- Resampling이 켜진 상태에서 delay/λ를 pin해 보간하면 원본부터 다시 만들기 때문에, 이전에 적용한 보간 중 다시 pin하지 않은 것은 사라집니다.
- **Revert to Original** 또는 Full range + Resample 해제 + Apply로 원래 해상도로 돌아갑니다. 상태 표시줄에 `[resampled Δλ=5 nm, avg]`처럼 현재 상태가 표시되고, 내보내는 파일 이름에 `rs5nm`이 붙습니다.

#### Moving average (v1.2.0)

![Crop Data 창의 Moving average](docs/images/crop_moving_average.png)

1. **Resample / smooth λ**를 체크하고 **Moving avg**를 고릅니다(Δλ 칸은 비활성, Window·Kernel 칸이 활성).
2. **Window (px)** 에 창 크기 k를 입력합니다. 홀수 **3–101**만 허용되고, 기본은 5입니다.
3. **Kernel**을 고릅니다.
   - **Box**: 창 안 점들을 모두 같은 가중치로 평균
   - **Triangular**: 가중치 1, 2, …, m, …, 2, 1 (m = (k+1)/2). 가운데 점이 가장 큼
   - **Gaussian**: 가중치 exp(−x²/2σ²), **σ = k/6** (창이 ±3σ를 덮음)
4. Spectrum 패널에서 원본(회색)과 평활 결과(검정 선)를 비교하고, `1200 → 1200 λ points (window 15 px, gaussian)`처럼 점 수가 그대로인지 확인한 뒤 **Apply**를 누릅니다.

**동작 방식**
- 각 파장 점 i의 ΔA를, i를 가운데 둔 k개 pixel의 가중 평균으로 바꿉니다: ΔA′ᵢ = Σ wⱼ ΔAᵢ₊ⱼ / Σ wⱼ. 모든 delay에 같은 방식으로 적용합니다.
- **파장축은 그대로**이고 점 개수도 원본과 같습니다. window는 nm가 아니라 pixel 단위이므로, 파장 간격이 균일하지 않으면 창의 nm 폭이 위치마다 조금씩 다릅니다.
- **가장자리**(앞뒤 k//2개 점)에서는 범위 밖 위치를 빼고 남은 가중치로 다시 나눕니다. 그래서 가장자리 값은 안쪽 이웃 쪽으로 약간 치우칩니다.
- 창 안의 **NaN** 점(마스크 등)은 빼고 다시 나누고, 창 전체가 NaN이면 결과도 NaN입니다.
- 용매 IRF 빼기를 쓰면 용매 데이터도 **같은 kernel**로 평활한 뒤 뺍니다.
- 창 크기가 짝수이거나 3–101 밖이거나, 선택 범위의 점 수보다 작지 않으면 경고 후 **평활 없이 crop만 적용**됩니다.
- Average/Decimate와 같이 Apply하면 보정이 초기화되고, Revert로 원래대로 돌아갑니다. 상태 표시줄에는 `[smoothed 15 px, gaussian]`, 내보내는 파일 이름에는 `_ma15gaussian`이 붙습니다.

**특정 delay/λ 제거 후 보간**: 스펙트럼이나 kinetics에서 튀는 delay/λ를 **Pin (mark for drop)** 으로 표시하면 linear/cubic/pchip/akima/bilinear로 보간해 채웁니다.

### 5.4 분석
| 메뉴 | 내용 |
|---|---|
| **Global Analysis** | 여러 지수 함수 ⊗ Gaussian IRF를 전체 행렬에 피팅합니다(τ 고정, stretched β, τ=∞ 성분, 피팅 구간 지정). 결과: τ ± σ, DADS, EADS, fit, residual, fit 통계(5.7절). TensorFlow가 있으면 GPU 선택 가능 |
| **SVD** | 특이값 스펙트럼, U/V 벡터, rank-N 재구성과 residual |
| **Kinetic Fit** | 한 파장(또는 파장 구간 평균) trace, 또는 main 창에서 pin한 kinetics 여러 개를 multi-exp ⊗ IRF로 피팅(여러 개면 global fit). τ·A ± σ, population ratio, fit 통계(5.7절) |
| **LDA** | Tikhonov 정규화 lifetime density 맵, L-curve로 α 선택 |
| **MCR-ALS** | 순수 스펙트럼 S(λ)와 농도 프로파일 C(t)로 분해 |
| **Coherence** | residual의 2D \|FFT\|² 맵(cm⁻¹/THz), LPSVD 모드 분해 |

GA와 LDA의 **Save residual**은 `residuals/` 폴더에 xlsx로 저장되고, Coherence 창에서 다시 불러올 수 있습니다.

### 5.5 내보내기
- **Export 2D Data...**: 보정된 행렬. 파일 이름에 상태가 붙습니다(예: `_2D_BG5_chirp_cropped_rs2nm.csv`, Moving avg면 `_ma5box`).
- **Export 2D data_original...**: 보정 전 행렬(crop/resampling은 반영됨)
- Spectrum/Kinetics 패널의 **Export pins**, 각 분석 창의 Export 버튼. Integrate로 pin한 열은 구간 평균값이고 헤더에 구간이 붙습니다(`dA_t=30.75ps_avg_idx95-103`, `dA_wl=650nm_BW20nm`).

### 5.6 Spectrum / Kinetics 구간 평균 (Integrate, v1.3.0)

![Integrate](docs/images/pin_gating.png)

Spectrum과 Kinetics 패널 아랫줄의 **Integrate** 체크박스를 켜면 한 점 대신 구간 평균(NaN 제외)을 그립니다.

| 패널 | 입력 | 평균 구간 | Legend 예 |
|---|---|---|---|
| Spectrum | **±idx** (N) | delay index `idx−N … idx+N` (2N+1점, 배열 끝에서는 잘림) | `t = 31.1 ± 4.6 ps (idx 95–103)` — 구간 delay의 평균 ± 표준편차 |
| Kinetics | **BW (nm)** | 선택 파장 λc에서 `|λ − λc| ≤ BW/2`인 픽셀 | `λ = 650.0 nm, BW 20 nm` |

- 오른쪽 표시창(indicator)에 실제로 평균한 구간, 중심값, 점 수가 나옵니다(예: `idx 95–103 (24.47–38.63 ps) · center 30.75 ps · 9 pts`, `640.1–659.9 nm · center 650.0 nm · 62 pts`). 꺼져 있으면 `off`.
- 체크한 상태에서 **Pin**하면 그 pin은 당시 구간을 기억합니다. 체크를 끄고 pin하면 기존처럼 한 점 trace입니다. 같은 delay/λ를 다시 pin하면 구간이 새 값으로 바뀝니다.
- ±idx = 0 또는 BW가 픽셀 간격보다 작으면 한 점과 같습니다. 데이터 자체는 바꾸지 않으며, 보기와 Export pins에만 적용됩니다.

### 5.7 Fit 통계와 Kinetic Fit Import (v1.4.0)

![Kinetic Fit global](docs/images/kfit_global.png)

**Global Analysis 결과 창**에 다음이 추가됩니다.
- 각 time constant의 표준편차: `tau_1 = 2.997 ± 0.0046 ps`. β, IRF t₀, FWHM도 자유 변수이면 ± σ가 붙고, 고정값은 `(fixed)`로 표시됩니다.
- **Fit statistics** 블록: R², Adjusted R², Reduced χ², RSS, RMS residual, AIC / BIC, 데이터 점 수·파라미터 수·자유도

σ는 최적점의 Jacobian으로 만든 공분산 행렬의 1σ(점근 표준오차)입니다. 진폭(DADS)은 τ마다 다시 풀리므로, τ의 σ에는 진폭 불확실성이 이미 반영됩니다. 값이 τ보다 크게 나오면 그 성분이 데이터로 결정되지 않는다는 뜻입니다(성분 수를 줄이세요).

**Kinetic Fit 창**(More Analysis → Kinetic Fit)
- 데이터는 **Import** 버튼을 눌렀을 때만 들어옵니다. 창을 연 직후에는 비어 있고, λ·half-width나 main 창 pin을 바꾼 뒤에는 Import를 다시 눌러야 반영됩니다.
- **Import pinned kinetics** 체크박스(main 창에 pin된 kinetics가 있으면 처음부터 켜져 있음)
  - 켜짐: main 창 Kinetics 패널에 pin한 trace를 모두 가져옵니다. Integrate BW로 pin한 trace는 그 대역 평균 그대로입니다.
  - 꺼짐 또는 pin 없음: 기존처럼 Centre λ ± half-width의 trace 하나를 가져옵니다.
- trace가 2개 이상이면 **global fit**(모든 trace가 τ, β, t₀, FWHM 공유, 진폭은 trace마다), 1개면 **단일 fit**을 합니다.
- 결과: τ ± σ, trace별 A ± σ와 **population ratio** `|Aᵢ| / Σ|Aⱼ| × 100 %`(τ=∞ offset을 켰다면 A∞ 포함), trace별 R², 위와 같은 Fit statistics
- Export: Params(CSV)에 σ·비율·통계가 들어가고(global이면 `T1_`, `T2_` … 접두사), Trace+Fit은 trace별 data/fit/residual 열, Save Residual은 trace마다 `kfit_{λ}nm_<이름>.csv` 파일을 하나씩 씁니다(Coherence 창에서 λ가 인식됨).

![Global Analysis statistics](docs/images/ga_stats.png)

## 6. 입력 파일 형식

CSV / TSV / DAT / TXT를 자동으로 읽습니다(구분자 자동 인식, 빈 칸과 NaN 처리).

**Format A** (첫 행 = delay, 첫 열 = 파장)
```
corner, t1, t2, ..., tN
λ1,     ΔA, ΔA, ..., ΔA
λ2,     ΔA, ...
```
**Format B** (앞 2열이 메타데이터: `row[0][2:]` = delay, `row[i][1]` = 파장)

이 둘에 맞지 않는 파일은 **Load Data → Custom**으로 영역을 직접 지정해 불러오세요. 내보낸 2D 파일은 Format A라 다시 불러올 수 있습니다.

## 7. 문제 해결

| 증상 | 해결 |
|---|---|
| exe 실행이 느리거나 백신이 막음 | 한 파일 실행 파일이라 첫 실행에 압축을 풉니다. 백신 예외에 추가하세요 |
| 파일을 못 읽음 | Custom 로더로 X/Y/Z 영역을 직접 지정하세요 |
| Resampling 후 BG/Chirp가 풀림 | 정상 동작입니다(5.3절). Resampling 후 보정을 다시 적용하세요 |
| "Δλ must be larger..." 경고 | Δλ를 원본 평균 파장 간격보다 크게 입력하세요 |
| "moving-average window must be odd..." 경고 | Window를 홀수(3–101)로, 선택한 파장 범위의 점 수보다 작게 입력하세요 |
| Global Analysis가 수렴하지 않음 | τ 초깃값을 바꾸거나 IRF t₀/FWHM을 고정해 보세요 |
| 확대한 그래프를 원래대로 | 툴바의 Home 버튼 또는 패널 우클릭 → Auto scale |

## 8. 개발자 정보

- 구조: [ARCHITECTURE.md](ARCHITECTURE.md) · 상세 매뉴얼: [`manual/`](manual/) (한/영, md·pdf)
- 테스트: `test/test_*.py`를 각각 실행합니다(pytest 불필요).
  ```bash
  # Git Bash 기준, 프로젝트 루트에서
  export PYTHONPATH=. PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen
  python test/test_crop_resample.py
  python test/test_crop_moving_average.py
  python test/test_pin_gating.py
  python test/test_fit_stats_kfit_import.py
  ```
  측정 데이터 경로가 코드에 고정된 테스트 4개(`test_real_file`, `test_session_fixes`, `test_crop_input_fix`, `test_crop_preview_perf`)는 해당 데이터가 없으면 실행되지 않습니다. `test_new_features`의 zoom 유지 검사는 v1.0.0부터 알려진 실패입니다.
- 빌드: `python -m PyInstaller ta_analyzer.spec --noconfirm --clean` → `dist/TA_Analyzer.exe`
- README 그림 다시 만들기: `python tools/make_screenshots.py` (합성 데이터 사용, `--scenes main,crop,moving,gating,kfit,ga`로 일부만 가능)
- 측정 데이터(`Data/`), 테스트 데이터 폴더, 빌드 결과물은 저장소에 포함하지 않습니다.

## 9. 변경 이력

**v1.4.0** (2026-10-27)
- Global Analysis: τ·β·t₀·FWHM의 1σ 표준편차와 Fit statistics(R², Adjusted R², Reduced χ², RSS, RMS, AIC/BIC, 자유도) 표시
- Kinetic Fit: **Import** 버튼과 **Import pinned kinetics** 체크박스. pin한 kinetics를 가져와 global fit(τ 공유), 1개면 단일 fit
- Kinetic Fit: τ·A ± σ, population ratio |Aᵢ|/Σ|Aⱼ| (%), trace별 R², fit 통계, 내보내기에 σ·비율·통계 포함
- Kinetic Fit 데이터는 Import를 눌렀을 때만 반입(λ 입력을 바꿔도 자동으로 다시 그리지 않음)

**v1.3.0** (2026-10-27)
- Spectrum 패널에 **Integrate ±idx**: delay index 구간 평균 스펙트럼, legend에 구간 delay 평균 ± 표준편차
- Kinetics 패널에 **Integrate BW (nm)**: λc ± BW/2 구간 평균 kinetics
- 각 패널 아랫줄에 체크박스·입력·indicator(평균 구간, 중심값, 점 수) 추가
- Pin이 구간을 기억하고, Export pins는 구간 평균값과 구간 정보를 헤더에 기록
- Pin이 있을 때 Kinetics 제목이 이전 파장으로 남던 문제 수정

**v1.2.0** (2026-09-28)
- Crop Data에 **Moving avg** 모드 추가: pixel window(홀수 3–101), Box / Triangular / Gaussian(σ = k/6) kernel
- 파장축과 점 개수는 그대로, 가장자리·NaN은 가중치 재정규화
- 용매 IRF도 같은 kernel로 평활, 상태 표시줄 `[smoothed k px, kernel]`, 파일 이름 `_ma{k}{kernel}`
- Crop 창 그룹 이름을 "Wavelength resampling / smoothing"으로 변경

**v1.1.0** (2026-09-28)
- Crop Data에 파장축 Resampling 추가: Δλ(nm) 입력, Average(구간 평균) / Decimate(구간 중심에 가장 가까운 점)
- Crop 창 Spectrum 패널에서 resampling 결과 미리보기, 점 수 표시
- 용매 IRF 빼기가 샘플과 같은 구간 묶음을 사용
- 스핀박스 반올림 때문에 끝 파장이 잘리던 문제 수정
- 한국어/영어 사용 설명서(README.md, README_EN.md)와 화면 그림 추가

**v1.0.0**
- Resampling 이전 기준 버전: 로딩 3종, BG/Chirp/용매 IRF/Mask 보정, Crop과 보간, Global Analysis, SVD, Kinetic Fit, LDA, MCR-ALS, Coherence
