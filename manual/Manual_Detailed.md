# TA Analyzer — 세부 매뉴얼 (Detailed Guide)

본 문서는 첨부 `Manual.pptx` 의 5 개 슬라이드 내용을 기준으로,
각 기능의 **목적 · 사전 조건 · 상세 절차 · 옵션 · 확인 사항 · 자주 묻는 오류** 까지
범위를 한정하여 정리한 세부 매뉴얼입니다.

| 슬라이드 | 주제                                | 본문 절 |
| ----- | --------------------------------- | ---- |
| 5     | 프로그램 개요 / 메인 화면                   | §0   |
| 1     | Load Data (Custom) — 영역 직접 지정    | §1   |
| 2     | Background Correction              | §2   |
| 3     | Chirp Correction                   | §3   |
| 4     | Kinetic Fit (single trace)         | §4   |

---

## §0. 프로그램 개요 (Slide 5)

### 0.1 프로그램 실행 방법

1. **Anaconda Prompt** 를 실행한다 (Windows 시작 메뉴 → "Anaconda Prompt").
2. `cd` 명령으로 본 프로젝트의 폴더 경로로 이동한다.
   - 예: `cd C:\Users\watqd\OneDrive\KNU_ULSIL\Python_Code\TA_Analyzer_rev6`
3. `python ta_main.py` 를 입력해 프로그램을 실행한다.

> 단순 `python` 명령이 아니라 **Anaconda Prompt** 에서 실행해야 numpy / scipy / PyQt5 / matplotlib 등 의존 패키지가 정상적으로 로드됩니다.

### 0.2 메인 화면 구성

실행 후 메인 윈도우는 크게 세 영역으로 구성됩니다.

- **좌측 큰 패널** — 2D ΔA map (가로축: wavelength, 세로축: delay time, 색축: ΔA)
- **우측 상단** — 선택된 delay 에서의 **Spectrum (ΔA vs wavelength)**
- **우측 하단** — 선택된 wavelength 에서의 **Kinetics (ΔA vs delay)**

상단에는 두 줄의 **툴바** 가 있으며, 본 매뉴얼에서 다루는 버튼은 아래와 같습니다.

| 버튼                            | 설명                                                       |
| ----------------------------- | -------------------------------------------------------- |
| **Load Data...**              | 표준 포맷 CSV/DAT/TXT 를 자동 인식해서 로드                          |
| **Load Data (Custom)...**     | 자동 인식이 불가한 임의의 CSV 에 대해 X/Y/Z 영역을 사용자가 직접 지정         |
| **Background Correction...** | 초기 delay 평균을 background 로 사용해 보정                     |
| **Chirp Correction...**       | t₀ 점들을 클릭으로 지정 → fit → chirp 보정                       |
| **More Analysis ▾**           | SVD, **Kinetic Fit**, LDA, MCR-ALS, Coherence 메뉴       |

### 0.3 데이터 로딩 흐름

1. **Load Data...** 클릭 → 표준 포맷이면 즉시 로드.
2. 만약 인식이 실패하거나 측정 포맷이 다른 경우, **Load Data (Custom)...** 을 사용 (§1 참고).
3. 로드가 끝나면 메인의 2D map 과 우측의 spectrum / kinetics 패널이 자동으로 갱신됩니다.

> 단일 파일을 선택하면 곧바로 로드되고, 여러 파일을 동시에 선택하면 **Load & Average** 창이 열립니다.

---

## §1. Load Data (Custom) — X / Y / Z 영역 직접 지정 (Slide 1)

### 1.1 사용 목적

측정 장비/파일마다 헤더 구조가 달라 자동 인식기가 실패하는 경우, CSV 를 **Excel 처럼 표 형태로 미리보면서**
파장(X)·시간(Y)·ΔA(Z) 의 위치를 직접 드래그 선택해서 불러오기 위한 기능입니다.

### 1.2 사전 조건

- 파일 형식: **CSV (콤마/탭/세미콜론 구분, 공백 분리도 자동 인식)**
- X(파장) 와 Y(시간) 는 **1D (단일 행 또는 단일 열)** 로 저장되어 있어야 함.
- Z 행렬의 크기는 `len(X) × len(Y)` 또는 그 전치 `len(Y) × len(X)` 이어야 함.

### 1.3 화면 구성

상단에서 아래로:

1. **File 경로 / Browse...** — 다른 파일로 교체 가능.
2. **단계 배너 (파란색)** — 현재 단계 표시 (`STEP 1 of 3 ...`).
3. **Current selection** — 마우스 선택 즉시 갱신되는 현재 선택 정보 (A1:K1, `n rows × m cols`).
4. **표 (Excel 풍)** — 셀 색
   - 흰색: 숫자 셀
   - 노란색: 비숫자(헤더/메타데이터)
   - 회색: 빈 셀
5. **Tip 안내** — 단축키 사용법.
6. **Confirm This Selection / Clear Current Selection / Back to Previous Step / Reset All / Auto-detect** 버튼.
7. **Confirmed selections** 패널 — 지금까지 확정된 X/Y/Z 의 주소 및 크기.
8. **Load / Cancel** — 모두 확정되었을 때 Load 가 활성화.

### 1.4 절차

순서는 **X → Y → Z** 로 고정되어 있으며, 각 단계마다 검증을 통과해야 다음 단계로 진행됩니다.

#### STEP 1: X축(wavelength) 영역 설정
- 표에서 파장 값이 들어있는 **한 행 또는 한 열만** 드래그로 선택.
- **Confirm This Selection** 클릭.
- 검증
  - 2D 영역(행과 열 모두 2 이상)은 거부 → 경고: "X-axis must be a 1D array."
  - 빈 셀·비숫자 포함 시 거부 → 경고: "Non-numeric / Too Short".

#### STEP 2: Y축(delay) 영역 설정
- 시간 값이 들어있는 단일 행 또는 단일 열을 선택 → **Confirm This Selection**.
- 검증 기준은 X 와 동일 (1D · 모든 셀이 유한한 숫자).

#### STEP 3: Z-Matrix(ΔA) 영역 설정
- ΔA 값이 들어있는 직사각형 2D 영역을 드래그로 선택 → **Confirm This Selection**.
- 검증
  - 형태가 `(len(X) × len(Y))` 또는 `(len(Y) × len(X))` 이 아닐 경우 거부 →
    "Z-matrix shape (n × m) does not match the X-axis (X points) and Y-axis (Y points)."
  - 전치된 방향으로 선택해도 OK — 추출 시 자동으로 `(파장 × 시간)` 방향으로 회전됩니다.

#### 종료
- 세 영역이 모두 확정되면 단계 배너가 "All three regions confirmed" 로 바뀌고 **Load** 버튼이 활성화됩니다.
- **Load** 클릭 → 메인 윈도우로 데이터 전달.

### 1.5 보조 기능

| 기능                            | 설명                                                                 |
| ----------------------------- | ------------------------------------------------------------------ |
| **Clear Current Selection**   | 현재 단계의 표 선택만 해제 (확정된 단계는 유지)                                     |
| **Back to Previous Step**     | 한 단계 뒤로 이동하면서 그 단계의 확정값을 비움                                       |
| **Reset All**                 | X/Y/Z 모두 비우고 STEP 1 로 복귀                                          |
| **Auto-detect**               | 내부 자동 파서로 X/Y/Z 영역을 즉시 채움 (잘못 인식되면 수동 보정 후 Load)                |

### 1.6 단축키 (Excel 호환)

본 대화상자의 표에서 다음 키 입력이 지원됩니다.

| 키                                 | 동작                                                              |
| --------------------------------- | --------------------------------------------------------------- |
| **드래그**                           | 일반적인 영역 선택                                                      |
| **Shift + 클릭**                    | 현재 anchor 에서 클릭한 셀까지 영역 확장                                     |
| **Ctrl + ↑ / ↓ / ← / →**          | 커서를 유효 데이터 끝(또는 다음 데이터 블록) 으로 점프                              |
| **Ctrl + Shift + ↑ / ↓ / ← / →**  | anchor 에서 유효 데이터 끝까지 한 번에 영역 확장                                 |

> Ctrl+Shift+화살표 동작 규칙 (Excel 동일)
> - 현재 셀이 데이터이고 다음 셀도 데이터: 빈 셀 직전까지 확장
> - 현재 셀이 데이터이고 다음 셀이 빈 셀: 다음 데이터 블록까지 확장
> - 현재 셀이 빈 셀: 다음 데이터 셀까지 확장
> - 더 이상 데이터가 없으면 표의 가장자리까지

### 1.7 자주 발생하는 경고

| 메시지                                                                       | 원인 / 대처                                                                |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `X/Y-axis must be a 1D array.`                                            | 두 방향 모두 길이가 2 이상인 영역을 선택. **한 행 또는 한 열만** 선택할 것.                  |
| `Non-numeric / Too Short`                                                 | 선택 안에 빈 셀이나 텍스트가 섞여 있음. 범위를 다시 잡거나 헤더 셀을 제외할 것.                    |
| `Z-matrix shape (n × m) does not match the X-axis ... and Y-axis ...`     | Z 의 행/열 수가 X·Y 길이와 다름. **Reset All** 후 STEP 1 부터 다시 진행 권장.        |

---

## §2. Background Correction (Slide 2)

### 2.1 사용 목적

펌프 효과가 도달하기 전 (t < 0) 의 평균 스펙트럼을 background 로 추정해서 모든 delay 에서 빼주는 보정입니다.

### 2.2 사전 조건

- 데이터가 로드된 상태여야 합니다 (메인 윈도우 상단 상태 표시줄에 `N λ × N t` 가 보여야 함).

### 2.3 화면 구성

- **# of initial delay points to average** — 평균에 사용할 초기 delay 점의 개수 (스피너 / 직접 입력).
- 오른쪽 표시: `Using t = ... ... ps` (선택한 개수가 실제로 어떤 시간 구간을 의미하는지 자동 안내).
- 중앙 그래프: 초기 N 개 delay 의 ΔA spectra (얇은 색선) + 평균선 (굵은 검은선) → 평균이 평탄에 가까울수록 background 로 적합.
- 하단: **Apply & Close**, **Cancel**.

### 2.4 절차

1. 툴바의 **Background Correction...** 클릭.
2. **# of initial delay points to average** 값을 조정한다.
   - 값이 너무 작으면 noise 가 background 로 들어가고, 너무 크면 t≈0 부근의 신호가 섞일 수 있음.
   - 그래프 상단의 `Using t = a ... b ps` 가 모두 t < 0 (pump 도달 전) 안에 들어오는 범위가 좋다.
3. 평균선(검은선) 이 0 근처에서 비교적 평탄한지 확인한다.
4. **Apply & Close** 클릭 → 전체 ΔA 행렬에서 이 평균이 차감됩니다.

### 2.5 해제 / 재적용

- 잘못 적용한 경우 메인 툴바의 **Reset Corrections** 로 background 를 포함한 모든 보정을 한 번에 되돌릴 수 있습니다.
- 값만 바꿔 다시 적용하고 싶다면 **Background Correction...** 을 다시 열어 새 값으로 **Apply & Close**.

---

## §3. Chirp Correction (Slide 3)

### 3.1 사용 목적

펌프-프로브 펨토초 측정에서 white-light continuum 의 파장별 군속도 분산 때문에 t = 0 이 파장에 따라 어긋나는 현상을 보정합니다.
사용자가 각 파장에서의 실제 t₀ 를 클릭으로 찍어 주면, 그 점들을 모델 함수로 fitting 한 뒤 모든 파장의 delay 축을 보정합니다.

### 3.2 사전 조건

- 데이터가 로드된 상태.
- t = 0 근처에서 coherent artifact / IRF 가 시각적으로 확인 가능한 신호여야 클릭이 쉽습니다.

### 3.3 화면 구성

좌측에 2D map (chirp 보정 작업용), 우측에 fit 결과 표 및 fit residual 그래프가 표시됩니다.

| 영역                                                | 설명                                                                |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| **View t_min / t_max**                            | map 의 delay 표시 범위. chirp 가 보이는 구간 (예: −0.5 ~ 1.5 ps) 으로 좁히면 클릭 정확도↑ |
| **Auto-place**                                    | ridge 검출 알고리즘으로 t₀ 점들을 자동 배치                                   |
| **Recompute**                                     | ridge 검출 매개변수가 바뀌었을 때 다시 계산                                    |
| **Reset view**                                    | 시간 범위와 fit 결과 그래프를 기본 상태로 복원                                  |
| **Remove Last / Clear Points**                    | 가장 최근 / 모든 클릭점 제거                                                |
| **Fit & Apply**                                   | 현재 클릭점을 fitting → 전체 데이터에 chirp 보정 적용                          |

### 3.4 절차

1. 툴바의 **Chirp Correction...** 클릭.
2. **View t_min**, **View t_max** 를 조정해 chirp 영역에 집중한다.
3. 2D map 위에서 각 파장의 t₀ (coherent artifact 가장자리) 를 차례로 클릭한다.
   - 한 번 클릭할 때마다 검정 점선 형태의 fit curve 가 갱신됨.
   - 잘못 찍었으면 **Remove Last**, 처음부터 다시 하려면 **Clear Points**.
4. 우상단의 fit curve 와 RMS 값을 확인. residual 그래프가 비교적 무작위에 가까울 것.
5. **Fit & Apply** 클릭 → chirp 보정이 메인 데이터에 반영됩니다.

### 3.5 권장 사항

- 점은 **양 끝 + 중앙 + 굴곡 부근** 까지 균등하게 6~10 개 정도를 찍는 것을 권장.
- 한 파장에 점을 두 번 찍으면 자동으로 가장 최근 클릭으로 갱신됩니다.
- Background correction 을 먼저 적용해 두면 ridge 가 더 잘 보이는 경우가 많습니다.

---

## §4. Kinetic Fit (single trace) (Slide 4)

### 4.1 사용 목적

선택한 단일 파장(또는 좁은 평균 구간) 의 ΔA(t) trace 를 **다중 지수 + Gaussian IRF** 모델로 fitting 해서
시간상수 τ 와 진폭 A 를 추출합니다. 필요 시 stretched exponential (β) 도 지원합니다.

### 4.2 사전 조건

- 데이터가 로드된 상태.
- (권장) Background, Chirp 등 필요한 보정을 먼저 적용.

### 4.3 화면 구성

좌측 = Setup 패널, 우측 = trace + fit + residual 그래프.

#### 4.3.1 Setup
- **Centre λ (nm)** — fitting 대상의 중심 파장. 옆의 **Use main λ** 로 메인 화면 crosshair 의 λ 를 가져올 수 있음.
- **± half-width (nm)** — 중심 ± 폭만큼 파장 평균을 적용 (0 이면 단일 픽셀).
- **Components** — 지수 성분 개수 (1~여러 개).
- **τ = ∞ offset** — 무한 수명 성분(상수 offset) 포함 여부.
- **Components — initial values (ps)** 표 — 각 성분마다
  - `τ_init (ps)` 초기값
  - `τ fixed` 체크 시 fitting 에서 고정
  - `Stretched` 체크 시 stretched exp, `β_init` 와 `β fixed` 사용
- **IRF (Gaussian)**
  - `t₀ (ps)` 와 그 `Fixed`
  - `FWHM (ps)` 와 그 `Fixed`
  - `Stretched-IRF mode` — stretched exp 일 때 IRF 처리 방식 (`Skip (mask 3σ around t₀)` 등).
- **Fit window (delay)** — `From` / `To`. `Full` 버튼으로 전체 delay 범위 사용.
- **Run Fit** — fitting 실행. 옆의 **Reset** 으로 초기값으로 복귀.
- **Export** — `Trace+Fit (CSV)`, `Params (CSV)` 로 결과 저장.
- 하단: fit 메시지 (`Fit converged ... iters, RMS = ...`), 각 성분의 추정 τ / β / A / 타입, 그리고 추정된 t₀ / FWHM.

#### 4.3.2 그래프
- 상단: data (검은 점) + fit (빨강 선).
- 하단: residual (data − fit).

### 4.4 절차

1. 툴바의 **More Analysis ▾** → **Kinetic Fit (single trace)…** 클릭.
2. **Fit window (delay)** 의 `From` / `To` 로 fit 범위를 설정.
   - 너무 좁으면 긴 성분이 안 잡히고, 너무 넓으면 baseline drift 가 결과를 흐트릴 수 있음.
3. **Components** 개수를 정하고, 표의 `τ_init` 에 대략적인 초기값을 적는다 (예: 0.5, 3, 50 ps).
4. (선택) `τ = ∞ offset` 체크로 상수 offset 추가, 필요한 성분은 `Stretched` 활성화.
5. **Run Fit** 클릭.
6. 결과 확인
   - 빨간 fit 선이 데이터를 따라가는지.
   - **residual** 패널이 0 주위에 무작위로 분포하는지 (구조가 보이면 모델이 부족 → 성분 추가 / 범위 조정).
   - 하단 텍스트의 RMS 와 추정 τ 가 합리적인 범위인지.
7. 필요한 경우 **Trace+Fit (CSV)** 와 **Params (CSV)** 로 결과를 내보낸다.

### 4.5 흔한 문제

| 증상                                | 점검 / 대처                                                       |
| --------------------------------- | ------------------------------------------------------------- |
| 빨간 fit 선이 데이터에서 크게 벗어남          | `τ_init` 가 실제 값과 너무 다름 → 자릿수 단위로 다시 추정 (ex. 100 ps 대신 1 ps). |
| residual 에 정현파/계단 구조가 보임         | 성분 부족. `Components` 를 늘리거나 `τ = ∞ offset` 사용.                |
| t₀ 가 음수쪽으로 크게 움직임                | Chirp 보정 누락 가능 — §3 먼저 진행.                                  |
| `Run Fit` 후 경고 "n_min ..."        | 성분 수 대비 fit 범위 안 점 개수가 부족 → `From/To` 를 더 넓혀 점 수를 확보.      |

---

## 부록 A. 본 매뉴얼이 다루지 않는 기능

본 문서는 첨부 pptx 의 5 슬라이드 내용에만 한정합니다. 다음은 동일 프로그램에 존재하지만
별도 매뉴얼이 필요한 기능 목록 (참고용):

- Crop Data, Mask Wavelengths
- Subtract solvent IRF
- Set t = 0 here
- Global Analysis
- SVD, LDA, MCR-ALS, Coherence (More Analysis 메뉴)
- Load & Average (멀티 파일 평균)
