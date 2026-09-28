# TA Analyzer — 필수 매뉴얼 (Quick Guide)

본 문서는 첨부된 `Manual.pptx` (5장) 의 내용에 한정하여 작성된 **필수 조작 순서**입니다.
자세한 옵션 설명은 `Manual_Detailed.md` 를 참고하세요.

---

## 0. 프로그램 시작 (Slide 5)

1. **Anaconda Prompt** 를 실행한다.
2. `cd` 명령으로 프로젝트 폴더 경로로 이동한다 (예: `cd C:\Users\watqd\OneDrive\KNU_ULSIL\Python_Code\TA_Analyzer_rev6`).
3. `python ta_main.py` 를 입력해 프로그램을 실행한다.
4. 상단 툴바의 **Load Data...** 또는 **Load Data (Custom)...** 버튼으로 측정 파일(CSV) 을 연다.

> 자동 인식 포맷이면 **Load Data...** 한 번으로 끝.
> 측정 포맷이 다르면 **Load Data (Custom)...** 으로 진행 (1번 항목).

---

## 1. Load Data (Custom) — X / Y / Z 영역 수동 지정 (Slide 1)

1. **X축 영역 설정** — 표에서 wavelength(파장) 이 들어 있는 단일 행 또는 단일 열을 드래그로 선택
2. **Confirm This Selection** 클릭
3. **Y축 영역 설정** — delay(시간) 이 들어 있는 단일 행 또는 단일 열을 드래그로 선택
4. **Confirm This Selection** 클릭
5. **Z-Matrix 영역 설정** — ΔA 데이터의 2D 영역을 드래그로 선택 (크기는 X · Y 와 일치해야 함)
6. **Confirm This Selection** 클릭
7. **Load** 클릭

> Tip: 셀 위에서 **Ctrl + Shift + ↑ / ↓ / ← / →** 로 유효 데이터의 끝까지 한 번에 선택할 수 있습니다 (Excel 동일).

---

## 2. Background Correction (Slide 2)

1. 툴바의 **Background Correction...** 클릭
2. **# of initial delay points to average** 값 설정 (초기 N 개 delay 의 평균을 background 로 사용)
3. **Apply & Close** 클릭

---

## 3. Chirp Correction (Slide 3)

1. 툴바의 **Chirp Correction...** 클릭
2. **View t_min**, **View t_max** 로 chirp 이 보이는 시간 범위를 좁힌다
3. 2D map 을 클릭해 **t₀ 포인트** 들을 추가한다
4. fit curve(점선) 모양을 확인하고 **Fit & Apply** 클릭

---

## 4. Kinetic Fit — single trace (Slide 4)

1. 툴바의 **More Analysis ▾** → **Kinetic Fit (single trace)…** 클릭
2. **Fit window (delay)** 의 From / To 값으로 fit 범위 설정
3. **Components** 개수와 각 성분의 **τ_init** 초기값 확인
4. **Run Fit** 클릭
