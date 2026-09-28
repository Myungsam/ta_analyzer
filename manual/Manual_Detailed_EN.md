# TA Analyzer - Detailed Guide

This document is the detailed manual scoped to the five slides of the attached
`Manual.pptx`. For each topic it covers **purpose · prerequisites · step-by-step ·
options · checks · common warnings**.

| Slide | Topic                                  | Section |
| ----- | -------------------------------------- | ------- |
| 5     | Program overview / main window         | §0      |
| 1     | Load Data (Custom) - manual selection  | §1      |
| 2     | Background Correction                   | §2      |
| 3     | Chirp Correction                        | §3      |
| 4     | Kinetic Fit (single trace)              | §4      |

---

## §0. Program overview (Slide 5)

### 0.1 Launching the program

1. Start **Anaconda Prompt** (Windows Start menu → "Anaconda Prompt").
2. Use `cd` to navigate to this project's folder.
   - Example: `cd C:\Users\watqd\OneDrive\KNU_ULSIL\Python_Code\TA_Analyzer_rev6`
3. Run `python ta_main.py` to launch the program.

> Launching from **Anaconda Prompt** (rather than a plain `python` shell) ensures that numpy / scipy / PyQt5 / matplotlib and the other dependencies load correctly.

### 0.2 Main window layout

After launch the main window has three primary panels.

- **Large left panel** - 2D ΔA map (x: wavelength, y: delay time, color: ΔA)
- **Top right** - **Spectrum** (ΔA vs wavelength) at the selected delay
- **Bottom right** - **Kinetics** (ΔA vs delay) at the selected wavelength

The top of the window has two toolbar rows. The buttons used in this manual are:

| Button                            | Purpose                                                                |
| --------------------------------- | ---------------------------------------------------------------------- |
| **Load Data...**                  | Load a CSV/DAT/TXT in a recognised standard format (auto-detect)      |
| **Load Data (Custom)...**         | Manually pick X / Y / Z regions for non-standard CSV layouts          |
| **Background Correction...**     | Subtract the mean of the first N delay points as background           |
| **Chirp Correction...**           | Click t₀ points → fit → apply chirp correction                        |
| **More Analysis ▾**               | Submenu with SVD, **Kinetic Fit**, LDA, MCR-ALS, Coherence            |

### 0.3 Data-loading flow

1. Click **Load Data...** - if the format is recognised, the file loads immediately.
2. If detection fails or the layout is unusual, use **Load Data (Custom)...** (see §1).
3. Once loaded, the 2D map and the side spectrum / kinetics panels refresh automatically.

> Selecting a single file loads it directly; selecting several files opens the
> **Load & Average** window.

---

## §1. Load Data (Custom) - manual X / Y / Z region selection (Slide 1)

### 1.1 Purpose

For measurement files whose header layout the automatic parser cannot recognise,
this dialog presents the CSV as an **Excel-like spreadsheet preview** so you can
drag-select the locations of wavelength (X), delay (Y), and ΔA (Z) by hand.

### 1.2 Prerequisites

- File format: **CSV (comma / tab / semicolon delimited; whitespace also recognised)**
- X (wavelength) and Y (delay) must be stored as **1D arrays - a single row or single column**.
- The Z matrix must have shape `len(X) × len(Y)` or its transpose `len(Y) × len(X)`.

### 1.3 Dialog layout

From top to bottom:

1. **File path / Browse...** - replace the previewed file.
2. **Stage banner (blue)** - current step indicator (`STEP 1 of 3 ...`).
3. **Current selection** - live readout of what the mouse has selected (`A1:K1, n rows × m cols`).
4. **Table (Excel-style)** - cell color
   - white: numeric cell
   - yellow: non-numeric (header / metadata)
   - gray: blank cell
5. **Shortcut hint line** below the table.
6. Buttons: **Confirm This Selection / Clear Current Selection / Back to Previous Step / Reset All / Auto-detect**.
7. **Confirmed selections** panel - addresses and sizes of the X / Y / Z ranges fixed so far.
8. **Load / Cancel** - **Load** is enabled only after all three regions are confirmed.

### 1.4 Procedure

The order is fixed at **X → Y → Z**, and each step must pass validation before
the dialog advances.

#### STEP 1: select the X-axis (wavelength)
- Drag-select **one row OR one column** containing the wavelengths.
- Click **Confirm This Selection**.
- Validation
  - 2D selections (more than one row AND more than one column) are rejected → "X-axis must be a 1D array."
  - Selections with blank or non-numeric cells are rejected → "Non-numeric / Too Short".

#### STEP 2: select the Y-axis (delay)
- Drag-select a single row or single column that holds the delays → **Confirm This Selection**.
- Same validation criteria as X (1D, every cell is a finite number).

#### STEP 3: select the Z-matrix (ΔA)
- Drag-select the rectangular 2D block of ΔA values → **Confirm This Selection**.
- Validation
  - If the shape is neither `(len(X) × len(Y))` nor `(len(Y) × len(X))`, the
    dialog rejects it → "Z-matrix shape (n × m) does not match the X-axis
    (X points) and Y-axis (Y points)."
  - You may select the transposed orientation - on **Load** the matrix is
    automatically rotated to `(wavelength × delay)`.

#### Finish
- When all three regions are confirmed, the stage banner reads "All three regions confirmed" and the **Load** button enables.
- Click **Load** to push the data into the main window.

### 1.5 Helper actions

| Action                            | Description                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| **Clear Current Selection**       | Clear only the table highlight at the current step (confirmed steps stay) |
| **Back to Previous Step**         | Go back one step and discard that step's confirmed value          |
| **Reset All**                     | Clear X/Y/Z and return to STEP 1                                  |
| **Auto-detect**                   | Fill X/Y/Z with the built-in auto parser (correct manually if wrong, then Load) |

### 1.6 Keyboard shortcuts (Excel-compatible)

The following keys work while focus is on the preview table.

| Key                                  | Action                                                              |
| ------------------------------------ | ------------------------------------------------------------------- |
| **Drag**                             | Standard region selection                                            |
| **Shift + Click**                    | Extend the selection from the anchor to the clicked cell             |
| **Ctrl + ↑ / ↓ / ← / →**             | Jump the cursor to the data edge (or the next data block)            |
| **Ctrl + Shift + ↑ / ↓ / ← / →**     | Extend the selection from the anchor to the data edge                |

> Ctrl + Shift + Arrow rules (Excel-equivalent)
> - Current cell is data and the next neighbor is data: extend to the last cell before a blank.
> - Current cell is data and the next neighbor is blank: extend to the next data block.
> - Current cell is blank: extend to the next data cell.
> - If no further data exists in that direction, extend to the table boundary.

### 1.7 Common warnings

| Message                                                                       | Cause / fix                                                                |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `X/Y-axis must be a 1D array.`                                                | Both row and column lengths are greater than 1. Select **only one row or one column**. |
| `Non-numeric / Too Short`                                                     | The selection contains blank cells or text. Exclude header cells and try again. |
| `Z-matrix shape (n × m) does not match the X-axis ... and Y-axis ...`         | Z's row / column counts differ from X and Y. Use **Reset All** and start from STEP 1. |

---

## §2. Background Correction (Slide 2)

### 2.1 Purpose

Estimate the pre-time-zero (`t < 0`) average spectrum as a background and
subtract it from every delay.

### 2.2 Prerequisites

- A dataset must be loaded (`N λ × N t` should appear in the status line of the main window).

### 2.3 Dialog layout

- **# of initial delay points to average** - number of initial delay points to average (spinner or direct input).
- Side label: `Using t = ... ... ps` - informs which actual time interval the chosen count corresponds to.
- Center graph: ΔA spectra for the first N delays (thin colored lines) plus the average (thick black line). A flat average is desirable.
- Bottom: **Apply & Close**, **Cancel**.

### 2.4 Procedure

1. Click **Background Correction...** on the toolbar.
2. Adjust the **# of initial delay points to average** value.
   - Too small → noise gets baked into the background; too large → real near-time-zero signal contaminates it.
   - Aim for a range where `Using t = a ... b ps` stays entirely in `t < 0` (before the pump arrives).
3. Verify that the average (black line) is relatively flat around zero.
4. Click **Apply & Close** - this average is subtracted from the entire ΔA matrix.

### 2.5 Undo / re-apply

- If applied incorrectly, **Reset Corrections** on the main toolbar reverts background (and every other) correction at once.
- To re-apply with a different value, reopen **Background Correction...** and click **Apply & Close** again.

---

## §3. Chirp Correction (Slide 3)

### 3.1 Purpose

In femtosecond pump-probe measurements, group-velocity dispersion of the
white-light continuum makes t = 0 wavelength-dependent. The user clicks the
true t₀ at multiple wavelengths; the dialog fits those points and corrects
every wavelength's delay axis accordingly.

### 3.2 Prerequisites

- A dataset must be loaded.
- The coherent artifact / IRF feature near t = 0 should be visible so the clicks can be placed accurately.

### 3.3 Dialog layout

The 2D map (for picking) is on the left and the fit-result table / residual graph are on the right.

| Region                                            | Description                                                       |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| **View t_min / t_max**                            | Delay range shown on the map. Narrow to the chirp region (e.g. −0.5 ~ 1.5 ps) for accurate clicks. |
| **Auto-place**                                    | Place t₀ points automatically via ridge detection                  |
| **Recompute**                                     | Recompute when ridge-detection parameters change                   |
| **Reset view**                                    | Restore the time range and the fit-result graph to defaults        |
| **Remove Last / Clear Points**                    | Remove the most recent / all picked points                         |
| **Fit & Apply**                                   | Fit the current points and apply chirp correction to the data      |

### 3.4 Procedure

1. Click **Chirp Correction...** on the toolbar.
2. Adjust **View t_min** and **View t_max** to focus on the chirp.
3. On the 2D map, click each wavelength's t₀ (the leading edge of the coherent artifact) in turn.
   - The dashed fit curve updates after every click.
   - Use **Remove Last** to undo, **Clear Points** to start over.
4. Inspect the fit curve and RMS at the top right. The residual graph should look essentially random.
5. Click **Fit & Apply** - the correction is applied to the main dataset.

### 3.5 Recommendations

- Place points evenly: **both endpoints + center + any inflection regions**, typically 6 to 10 points.
- Clicking twice on the same wavelength replaces the previous click for that wavelength.
- Applying Background Correction first often makes the ridge easier to see.

---

## §4. Kinetic Fit (single trace) (Slide 4)

### 4.1 Purpose

Fit a single-wavelength (or narrow-band-averaged) ΔA(t) trace with a
**sum-of-exponentials + Gaussian IRF** model and extract lifetimes τ and
amplitudes A. Stretched exponentials (β) are also supported.

### 4.2 Prerequisites

- A dataset must be loaded.
- (Recommended) Apply background, chirp, and any other needed corrections first.

### 4.3 Dialog layout

Left = Setup panel, right = trace + fit + residual graphs.

#### 4.3.1 Setup
- **Centre λ (nm)** - center wavelength for the fit. **Use main λ** copies the crosshair λ from the main window.
- **± half-width (nm)** - average over `centre ± half-width` (0 = single pixel).
- **Components** - number of exponential components.
- **τ = ∞ offset** - include a constant (infinite-lifetime) offset.
- **Components - initial values (ps)** table - per component:
  - `τ_init (ps)` initial value
  - `τ fixed` - hold τ constant during the fit
  - `Stretched` - use stretched exponential; configures `β_init` and `β fixed`
- **IRF (Gaussian)**
  - `t₀ (ps)` plus its `Fixed` checkbox
  - `FWHM (ps)` plus its `Fixed` checkbox
  - `Stretched-IRF mode` - dropdown for how the IRF interacts with stretched exponentials (e.g. `Skip (mask 3σ around t₀)`).
- **Fit window (delay)** - `From` / `To`. **Full** uses the entire delay range.
- **Run Fit** - run the optimization. **Reset** restores the initial values.
- **Export** - `Trace+Fit (CSV)` and `Params (CSV)`.
- Bottom info: status line (`Fit converged ... iters, RMS = ...`), per-component τ / β / A / type, plus the estimated t₀ and FWHM.

#### 4.3.2 Graphs
- Top: data (black dots) and fit (red line).
- Bottom: residual (data − fit).

### 4.4 Procedure

1. Open **More Analysis ▾** → **Kinetic Fit (single trace)…** on the toolbar.
2. Set the fit range via the **Fit window (delay)** From / To.
   - Too narrow misses long components; too wide may let baseline drift bias the fit.
3. Choose **Components** and enter approximate initial values for `τ_init` (e.g. 0.5, 3, 50 ps).
4. (Optional) Tick `τ = ∞ offset` for a constant offset; turn on `Stretched` where applicable.
5. Click **Run Fit**.
6. Check the results
   - The red fit line should follow the data.
   - The **residual** panel should be roughly randomly distributed around zero (visible structure → model is too simple; add components or adjust the range).
   - Check that the reported RMS and τ values are physically reasonable.
7. If needed, export with **Trace+Fit (CSV)** and **Params (CSV)**.

### 4.5 Common problems

| Symptom                                | Check / fix                                                       |
| -------------------------------------- | ----------------------------------------------------------------- |
| Red fit clearly deviates from data     | `τ_init` is far from reality → re-estimate by orders of magnitude (e.g. 1 ps instead of 100 ps). |
| Sinusoidal / step structure in residual | Too few components. Increase `Components` or enable `τ = ∞ offset`. |
| t₀ drifts strongly negative            | Chirp correction missing - run §3 first.                          |
| `Run Fit` warns "n_min ..."            | Number of points in fit window is too small for the chosen number of components → widen `From` / `To`. |

---

## Appendix A. Features NOT covered in this manual

This document is scoped to the 5 slides of the attached pptx. The following
features exist in the program but require separate documentation (for reference):

- Crop Data, Mask Wavelengths
- Subtract solvent IRF
- Set t = 0 here
- Global Analysis
- SVD, LDA, MCR-ALS, Coherence (More Analysis menu)
- Load & Average (multi-file averaging)
