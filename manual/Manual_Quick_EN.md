# TA Analyzer - Quick Guide

This document is the **essential operating sequence** limited to the
contents of the attached `Manual.pptx` (5 slides).
For detailed option explanations, see `Manual_Detailed_EN.md`.

---

## 0. Launching the program (Slide 5)

1. Start **Anaconda Prompt**.
2. Use `cd` to navigate to the project folder (e.g. `cd C:\Users\watqd\OneDrive\KNU_ULSIL\Python_Code\TA_Analyzer_rev6`).
3. Run `python ta_main.py` to launch the program.
4. Open a measurement file (CSV) with the **Load Data...** or **Load Data (Custom)...** button on the top toolbar.

> If the file is in a recognized format, **Load Data...** alone is enough.
> If your acquisition format differs, use **Load Data (Custom)...** (Section 1).

---

## 1. Load Data (Custom) - manual X / Y / Z region selection (Slide 1)

1. **Select the X-axis region** - drag-select a single row OR single column that holds the wavelengths
2. Click **Confirm This Selection**
3. **Select the Y-axis region** - drag-select a single row OR single column that holds the delays
4. Click **Confirm This Selection**
5. **Select the Z-Matrix region** - drag-select the 2D block of deltaA values (its shape must match the X and Y lengths)
6. Click **Confirm This Selection**
7. Click **Load**

> Tip: while in the table, **Ctrl + Shift + ↑ / ↓ / ← / →** extends the selection to the end of the contiguous data run, just like Excel.

---

## 2. Background Correction (Slide 2)

1. Click **Background Correction...** on the toolbar
2. Set **# of initial delay points to average** (uses the mean of the first N delays as background)
3. Click **Apply & Close**

---

## 3. Chirp Correction (Slide 3)

1. Click **Chirp Correction...** on the toolbar
2. Narrow the visible delay range with **View t_min** and **View t_max** so the chirp is clearly visible
3. Click the 2D map to add **t₀ points** along the chirp
4. Inspect the dashed fit curve and click **Fit & Apply**

---

## 4. Kinetic Fit - single trace (Slide 4)

1. Open **More Analysis ▾** → **Kinetic Fit (single trace)…** on the toolbar
2. Set the fit range using **Fit window (delay)** From / To
3. Choose the number of **Components** and check each component's **τ_init** initial value
4. Click **Run Fit**
