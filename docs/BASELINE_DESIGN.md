# MODI-SYNTH — Baseline Reader: Design Document

**Project:** Writing to Read: Style-Conditioned Handwriting Generation for Low-Resource Historical Modi Script Recognition
**Team:** Omkar Haryan, Udit Sevak, Tanish Agarwal
**Version:** 1.0 · 3 October 2026 · Status: design approved for implementation, no model trained yet

---

## 0. Summary

We will build a **paragraph-level CTC recogniser (the "Reader")** that reads a whole MoDeTrans patch
(3–4 lines of handwritten Modi) and outputs its Devanagari text. It does not need line segmentation or line-level labels.
The core is a **SPAN-style fully convolutional network** (Coquenet et al., ICDAR 2021). Each row of its output
feature map predicts characters, the rows are concatenated, and a single CTC loss aligns them with the patch transcription.
A **per-row BiLSTM** and a **character-count auxiliary head** are added because the literature shows both help in
low-resource settings. The model is small (6.9M parameters as implemented), so it trains on a free Colab or Kaggle T4.

The baseline has three jobs:
1. Give the reference score that every later experiment must beat. This is **Arm A ("real data only")** of the main study.
2. Freeze a Reader configuration that is then reused **unchanged** for the synthetic-data arms. Only the training data changes.
3. Produce **line-level text** for free: the row-wise CTC alignment tells us which line every character sits on.
   The Writer (generator) needs this later.

Primary metric: **CER** on the MoDeTrans test split (306 patches), plus an **out-of-domain test** on the folder-18 book
(7 letters that share nothing with MoDeTrans).

---

## 1. Where we are (inputs to this design)

| Asset | State | Used by baseline? |
|---|---|---|
| MoDeTrans (2,043 patches, 1268×463, binarised, 3–4 lines each) | split 1,430 / 307 / 306 (`splits.json`, seed 42) | **yes**: train, val, test |
| Text targets `text_train` | NFC; σ → `<FRAC>`; ASCII digits → Devanagari; editorial `?` `[ ]` removed; newlines → space | **yes** |
| Charset | 86 symbols + `<FRAC>`, `<blank>`, `<unk>` (`charset.json`) | **yes** |
| Folder 18: *Illustrative Modi Documents* (1978) | 17 letters, facsimile (odd pages) + printed transcription (even pages); `folder18_labels.csv` | **yes**: OOD test (strict tier = letters 3, 5, 8, 11, 12, 13, 15) |
| MODI-HHDoc pool (3,056 pages, unlabelled) | cleaned, deduplicated | no (Stage 2+) |
| MODI-HChar (575,912 char images) | pretraining only | optional ablation only |

Text length per patch: mean 121 characters, median 98, max 452. Eleven patches contain `<FRAC>`.

---

## 2. Task definition

- **Input:** one grayscale MoDeTrans patch, 1268×463 px, already binarised and deskewed by the dataset authors,
  containing 3–4 handwritten lines with **no line-break annotation** in the transcription.
- **Output:** the `text_train` string (Unicode Devanagari, 86-symbol alphabet plus `<FRAC>`).
- **Unit of recognition:** the whole patch (paragraph-level HTR). Modi is written without word spaces in many
  documents, and the transcriptions add spaces, so the model must also learn to insert them.

> The MoDeTrans paper calls this "transliteration". Technically it is handwritten text recognition with a
> script change: the output is the same Marathi text in a different script. We use the HTR framing and metrics.

---

## 3. What the literature tells us → design decisions

Every decision below traces back to a paper in `FYP-modi-script-work/Research Papers/` or to a source checked in October 2026.

| # | Evidence | Source | Decision |
|---|---|---|---|
| D1 | MoDeTrans patches hold 3–4 lines but the text has no line breaks. A line CRNN fed the whole patch scored only **27.3 BLEU** | C01 (MoScNet), Sec. 3, Tab. 1 | Use a **paragraph-level** model that needs no line labels |
| D2 | SPAN: end-to-end FCN, "does not require line breaks in the transcription labels", trains from scratch, code and weights public | Coquenet et al., ICDAR 2021 (arXiv 2102.08742) | **SPAN-style encoder + row-wise CTC** is the core |
| D3 | VAN generalises best out of domain, but it needs line-level supervision | C03 (CVPR 2025) | VAN rejected for now (no line labels). Revisit once D9 produces lines |
| D4 | CTC beats autoregressive decoders out of domain (40.4 vs 48.7 CER) and uses synthetic data better; CRNN is the best synthetic-data user | C03 | **CTC**, not seq2seq. Matters later for Arms B and C |
| D5 | Cross-language transfer in low-resource HTR comes from **sequence modelling** (CRNN), not from shared CNN features | arXiv 2605.05900 (2026) | Add a **BiLSTM per row**; keep it as a switchable ablation |
| D6 | UniLipi: removing the count token raises CER 6.9 → 9.3; removing both auxiliary tokens → 9.7; CTC beats cross-entropy (6.9 vs 8.3) | C02 (UniLipi), Tab. 4 | Add a **character-count regression head** (MAE, λ = 0.4) |
| D7 | Modi line-level CER: Tesseract 47.3, Kraken 38.7, HTR-VT 26.4, UniLipi 18.6 (2.5M synthetic lines + real) | C02, Tab. 2 | Published reference points; our real-only baseline is reported next to them in the write-up |
| D8 | HTR-VT: ResNet-18 + ViT encoder + CTC, **SAM optimiser + span masking** make transformers work on small data | J09 (Pattern Recognition 2025) | Borrow **span masking** as regularisation; HTR-VT is the line-level comparison model (B2) |
| D9 | SPAN's row-wise CTC output assigns every character to a feature row, i.e. to a text line | SPAN design | Use B1's alignment to **split patch text into lines**, the line data the Writer needs |
| D10 | A small CNN–BiLSTM–CTC fine-tuned on **1–3 pages** of a new manuscript beats Gemini; small adaptable models win on obscure scripts | C07 (FLAME, Sanskrit) | Small CTC Reader, not a VLM. Few-shot adaptation is a later experiment |
| D11 | Transfer learning from a large varied handwriting model gives high accuracy with little data | J05 (DARE) | Pretraining is an **ablation**, not part of the clean baseline |
| D12 | Standard augmentation set (rotation, dilation, erosion, perspective, elastic, shear, Gaussian noise, p = 0.5) | C03, Sec. 3.3 | Adopt as-is (Sec. 5.3) |
| D13 | Multi-source inputs should be **homogenised** to one representation before recognition | J08 (npj Heritage 2025) | Binarise every OOD page the same way MoDeTrans was (Gaussian adaptive threshold + deskew) |
| D14 | Report NFC-normalised CER/WER/chrF++ and **median CER + catastrophic-failure rate** (share of samples with CER > 50 %) | C06 (OCR-VLMs on Devanagari) | Adopted (Sec. 7) |
| D15 | Errors concentrate on look-alike pairs (भ/म, क/फ, ट/ठ/ढ, न/ण), matras and anusvara | C01 Sec. 5.1; J14 (anusvara the largest error class) | Report an **error taxonomy** for these classes |
| D16 | Modi fonts exist (*Marathi CursiveT*, Noto Sans Modi) to render Marathi text as Modi | J14 (IIETA 2024) | Font-rendered synthetic data = **Arm B** of the main study (not baseline) |
| D17 | Large VLMs need 128 H100s (MoScNet) and are contaminated for our test (ModiTrans-12B was trained on all of MoDeTrans) | C01; HF card `historyHulk/ModiTrans-12B-Gemma-Teacher` | No VLM in the baseline. The teacher is never evaluated on MoDeTrans |

Papers read for this design but not driving baseline choices: J03 and J17 (domain adaptation, for Stage 3 with the
HHDoc pool), C08 (MetaWriter writer adaptation), J04 (crossed-out words), J06 and J07 (line segmentation, for Stage 2),
J02 (handwriting synthesis survey, for the Writer), C04, C05, C09, C10, J01, J10, J11, J13, J16 (context).

---

## 4. Model design

### 4.1 Main baseline: B1 = SPAN-style FCN + row BiLSTM + CTC + count head

```
patch (1 × 512 × 1402, grayscale, white bg)
  └─ FCN encoder (SPAN/VAN encoder: conv blocks + depthwise-separable blocks, dropout)
        downsampling ×32 in height, ×8 in width  →  feature map 256 × 16 × 175
  └─ span masking (training only): zero random horizontal spans of the feature map   [D8]
  └─ per-row BiLSTM (2 layers, 256 hidden, shared weights across rows)                 [D5]
  └─ 1×1 conv → 89 classes per position (86 symbols + <FRAC> + <unk> + <blank>)
  └─ flatten rows top-to-bottom → sequence of 16 × 175 = 2,800 frames
  └─ CTC loss against text_train (line breaks not needed)                              [D2]
  + count head: global average pool → MLP → predicted character count, MAE loss        [D6]

L = L_CTC + 0.4 · L_count
```

- **Why the 512 px input height?** The native height is 463. We resize to 512 (×1.106) so that ÷32 gives 16 feature
  rows, i.e. about 4 rows per text line, enough to separate lines. Width scales by the same factor to 1402 and is padded
  to a multiple of 8. Aspect ratio is preserved.
- **CTC frame budget:** 2,800 frames against at most 452 characters, enough even with blanks between repeated letters.
- **Size:** 6.9M parameters in total (encoder + 2-layer row BiLSTM + heads), measured in `notebooks/01_baseline_reader_B1.ipynb`, Cell 9. It fits a T4 (16 GB) at batch 8 with fp16.
- **Decoding:** greedy CTC (the main reported number). Optionally, beam search with a character 6-gram LM trained
  **only on train-split text**, reported as a separate row.

### 4.2 Comparison and ablation models

| ID | Model | Purpose |
|---|---|---|
| **B0** | Plain CRNN (CNN + BiLSTM + CTC) fed the **whole patch as if it were one line** (height collapsed to 1 row), like C01's CRNN baseline (C01 does not report its exact input size; we use 128×1024) | Reproduces C01's 27.3 BLEU, so our pipeline is comparable with the published number. A lower bound |
| **B1** | Sec. 4.1 | **Main baseline = Arm A** |
| B1-noLSTM | B1 without the row BiLSTM (pure SPAN) | Tests D5 |
| B1-noCount | B1 without the count head | Tests D6 |
| B1-noAug | B1 without augmentation | Shows how much augmentation matters on 1,430 samples |
| B1-HChar | B1 encoder pretrained as a 57-class classifier on MODI-HChar, then fine-tuned | Tests D11 cheaply (HChar is pretraining-only, never evaluated) |
| **B2** | Line-level HTR-VT (or CRNN) trained on line images cut from patches, using B1's row alignment as line labels (D9) | Line-level comparison. Also proves the line data is usable before the Writer depends on it |
| B3 (stretch) | LoRA fine-tune of a small pretrained VLM (e.g. Qwen2.5-VL-3B, 4-bit) on the train split | One autoregressive/VLM point for the discussion. Only if time allows; report median CER and catastrophic rate (D14) |

### 4.3 Implementation source

Port the SPAN encoder and row-CTC from the official repository (`github.com/FactoDeepLearning/SPAN`, MIT licence) into
`src/modi/recognition/`. Write our own data module, count head, BiLSTM, span masking and training loop, so all
variants share one code path and one config format. HTR-VT for B2 comes from `github.com/YutingLi0606/HTR-VT`.

---

## 5. Data pipeline

### 5.1 Loading
- Images come from the Hugging Face dataset `historyHulk/MoDeTrans`. Our local parquet has the same SHA-256
  (`b51ea5f4…`), so filenames and our splits match exactly.
- Targets: `text_train` from `metadata/modetrans_text.csv`. Splits: `metadata/splits.json → modetrans.assignment`.
- Tokenisation: one token per Unicode code point, except `<FRAC>`, which is one token. Anything not in
  `charset.json` maps to `<unk>`; the train split has none by construction.

### 5.2 Preprocessing
1. Grayscale, keep polarity as dark ink on white.
2. Resize to height 512 with aspect ratio kept; pad the width to a multiple of 8 with white.
3. Normalise to [0, 1], then (x − 0.5) / 0.5.
4. **OOD pages only (folder 18):** Gaussian adaptive threshold, deskew, then split each facsimile page into horizontal
   bands of about 4 lines using a projection profile, so the input looks like a MoDeTrans patch (D13). Band predictions
   are concatenated top to bottom to give the page text.

### 5.3 Augmentation (training only, each p = 0.5) [D12]
Small rotation (±2°), dilation or erosion (stroke-width change, 3×3 kernel), random perspective (≤ 0.05), elastic
transform (α 30–60, σ 5–7), shear (±10°), Gaussian noise (σ ≤ 0.05), random white margins (0–20 px). **No** flips.
Random resize within height 480–544 so the row count varies.

---

## 6. Training protocol

| Item | Setting |
|---|---|
| Optimiser | AdamW, lr 3e-4, weight decay 1e-2, β = (0.9, 0.999) |
| Schedule | Linear warm-up for 1,000 steps, then cosine decay to 1e-6 |
| Batch | 8 (gradient accumulation 2 → effective 16), fp16 mixed precision |
| Epochs | Up to 400. Early stop if val CER does not improve by ≥ 0.1 within 60 epochs (C03 used 0.1 within 100 over 500 epochs) |
| Checkpointing | Every epoch to Google Drive; keep `last.pt` and `best_val_cer.pt`. Must resume after a Colab disconnect |
| Seeds | 3 seeds (42, 43, 44) for every reported configuration; report mean ± std |
| Model selection | Validation CER only. The **test split is evaluated once per final configuration** and never used for tuning |
| Logging | Config YAML, seed, git commit hash, per-epoch train loss, val CER/WER, learning rate, GPU type → `outputs/logs/<run_id>/` (+ optional W&B) |

**First sanity check before any real run:** overfit 20 training patches to near-zero CER. If the model cannot memorise
20 patches, the pipeline is broken (wrong labels, wrong charset, wrong frame budget). Do not start full training until this passes.

Estimated cost: 1,430 patches / 8 ≈ 180 steps per epoch, about 1 minute on a T4. That's 2–5 hours per run, i.e.
1–2 Colab sessions. Three seeds × 5 B1 variants ≈ 15 runs ≈ 40–60 GPU-hours. Use Kaggle (30 GPU-h/week)
in parallel with Colab.

---

## 7. Evaluation and validation

### 7.1 Test sets

| ID | Set | Size | What it measures |
|---|---|---|---|
| **T1** | MoDeTrans test split | 306 patches | In-domain accuracy. **The headline number** |
| **T2-strict** | Folder 18, letters 3, 5, 8, 11, 12, 13, 15 | 14 facsimile pages | True out-of-domain: different source (printed facsimile of archival letters), no overlap with MoDeTrans |
| T2-partial | Folder 18, letters sharing a writer with a MoDeTrans patch | 10 pages | OOD layout and scan, but seen writer. Reported separately, never pooled with strict |
| — | Folder-18 pages that *are* MoDeTrans sources | 14 pages | **Never evaluated** |

**Folder-18 ground truth must be created first.** The transcriptions are printed Devanagari on the even pages.
Digitise them with a printed-Devanagari OCR (Surya OCR 2, PaddleOCR `devanagari_PP-OCRv5_mobile_rec`, or Tesseract
`mar`), then a team member proofreads every line against the print. Drop the English header, the "क्र. N" line and
the footnotes. The printed line breaks do not match the handwritten ones, so evaluation is **per letter**:
the concatenated prediction for all facsimile pages of a letter against its full transcription.

### 7.2 Metrics (all on NFC-normalised text; `<FRAC>` counts as one character) [D14]

| Metric | Role |
|---|---|
| **CER** (Levenshtein / reference length, micro-averaged over the set) | **Primary** |
| WER (space-separated) | Secondary. Modi word boundaries are editorial, so treat with care |
| chrF++ | Robust character n-gram score; standard for Devanagari OCR |
| BLEU (sacreBLEU, character-tokenised as in C01) | **Only** for comparability with MoScNet's published 27.3 / 51.0 |
| Median CER and catastrophic rate (% patches with CER > 50 %) | Exposes collapse that the mean hides |
| Count-MAE | Quality of the auxiliary head; also a free sanity flag at inference |
| Error taxonomy: substitutions within {भ,म}, {क,फ}, {ट,ठ,ढ}, {न,ण}; matra errors; anusvara drop/insert; conjunct errors | Diagnoses *what* fails [D15] |

### 7.3 Statistics
- Mean ± std over 3 seeds for every configuration.
- **95 % bootstrap confidence interval** (10,000 resamples of test patches) for the headline CER.
- **Paired bootstrap** for every A-vs-B comparison (same test patches, resample jointly). A difference counts as real only if the 95 % CI of the difference excludes 0.

### 7.4 Validation of the pipeline itself
1. Overfit-20 sanity test (Sec. 6).
2. A unit test that `decode(encode(text)) == text` for all 2,043 targets.
3. A unit test that every patch's CTC frame count ≥ its label length + repeated-character count.
4. Visual check: plot 20 val predictions next to their images after every major change.
5. Leakage guard: the dataloader asserts that no train filename appears in val or test (`splits.json`).

### 7.5 Definition of done for the baseline
- B1 converges on all 3 seeds; T1 CER reported with CI.
- B0 reproduces C01's CRNN within about ±5 BLEU (otherwise investigate before trusting anything).
- Ablations B1-noLSTM, B1-noCount and B1-noAug done.
- T2-strict evaluated per letter.
- Line splits exported from B1 alignment for at least the train split (input to B2 and the Writer).
- One results table plus one error-taxonomy figure in `outputs/reports/`.

---

## 8. Alternatives considered (and why not now)

| Alternative | Why not for the baseline | When it comes back |
|---|---|---|
| Segment lines first, then a line CRNN | No line-level labels exist; Kraken's `blla` segmenter would need fine-tuning on Modi | B2 uses lines *derived from* B1, and Stage 2 (HHDoc pages) needs a segmenter anyway |
| VAN / DAN (Coquenet) | VAN needs line-level supervision; DAN is heavier and needs more data | After D9 gives line labels |
| OrigamiNet (learned 2D→1D unfolding) | Comparable to SPAN but less maintained code | Fallback if SPAN does not converge |
| TrOCR / ViT encoder-decoder | Autoregressive, worse OOD (C03); 385M params; Devanagari TrOCR checkpoints are printed/modern | No |
| Fine-tuned VLM (Gemma, Qwen-VL) | Compute; hallucination and repetition; the public Modi teacher is contaminated on MoDeTrans | B3 stretch only |
| Character classifiers (J12, J15) | Isolated characters, no sequence; does not transfer to documents | No |
| Off-the-shelf Kraken / Tesseract | Kraken 38.7 and Tesseract 47.3 CER on Modi lines (C02) | Cited only, as reference numbers |
| Community HF dataset `Visulon/modi-ocr-dataset` (8,189 word crops) | No provenance; first samples match MoDeTrans text → **probable leakage** | Only after an overlap check, if ever |

---

## 9. How the baseline connects to the main study

```
Stage 1 (this doc)   Reader on real data only ──────────────► Arm A score (T1, T2)
Stage 2              B1 alignment → line images + line text ─► Writer training data
Stage 3              Arm B: Reader + font-rendered synthetic lines (Marathi CursiveT / Noto Sans Modi)
                     Arm C: Reader + Writer-generated synthetic lines (same count as B)
                     [Arm D: + HHDoc pseudo-labels, only after the binarisation-aware overlap check]
Hypothesis: for cursive Modi, C < B < A on CER (learned style beats font rendering)
```

The Reader config frozen at the end of Stage 1 is reused unchanged in Arms B, C and D. Only the training data changes.

---

## 10. Compute and data access (Colab / Kaggle)

Nothing large is uploaded from the laptop.

| Data | How Colab gets it | Size |
|---|---|---|
| MoDeTrans | `load_dataset("historyHulk/MoDeTrans")` straight from Hugging Face | 161 MB |
| Our metadata + folder 18 | `colab_pack.zip` (built by `scripts/17_make_colab_pack.py`), uploaded once to Google Drive | 7.6 MB |
| Code | the project's GitHub repository, `git clone` in Colab | < 1 MB |
| MODI-HHDoc (Stage 2+) | direct download from Mendeley inside Colab (`wget`, 7.2 GB .rar, sha256 in the pack manifest) | 7.2 GB |
| Checkpoints and logs | written to Google Drive (`MyDrive/modi/runs/`) | ~100 MB/run |

Setup cells are in the appendix.

---

## 11. Work plan

Work runs as three parallel workstreams — **recognition** (`src/modi/recognition/`), **evaluation** (`src/modi/evaluation/`, `src/modi/segmentation/`) and **synthetic data** (`src/modi/synthetic/`). Each one is developed on its own branch against the three interface formats described in the repository `README.md` (section “Interfaces between workstreams”), so no workstream waits for another. Progress is shown in a weekly demo (Fridays 16, 23, 30 October and 6 November); the first joint comparison of Arm A and Arm B is on 6 November.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| 1,430 training patches is small → overfitting | Augmentation, span masking, dropout, early stopping on val CER; consider SAM optimiser (J09) |
| Row-CTC fails when lines are slanted or touch | Augmentation includes rotation and shear; the input height gives about 4 feature rows per line; fallback is OrigamiNet |
| Colab disconnects mid-run | Checkpoint every epoch to Drive with automatic resume; use Kaggle for long runs |
| Folder-18 OCR of the printed transcription has errors | Proofread 100 % of lines; keep the OCR output and the corrections both versioned |
| B0 fails to reproduce C01's 27.3 BLEU | Their split is unpublished; if we land far off, report it honestly and check the BLEU tokenisation first |
| **924 MoDeTrans images come from HHDoc** (C01), but only 72 are detectable by feature matching because of their binarisation and cleaning | Irrelevant for the baseline (HHDoc unused). **Blocker for Stage 3 Arm D**: run a binarisation-aware overlap search first (`audit_fixes.md`) |
| Reviewers ask "why not a VLM?" | B3 stretch point plus the cost and contamination argument (D17) |

---

## 13. References (local copies in `FYP-modi-script-work/Research Papers/`)

- **C01** Kausadikar et al., *Historic Scripts to Modern Vision: MoDeTrans and MoScNet*, Springer LNCS 2025 (arXiv 2503.13060)
- **C02** *UniLipi: A Unified Multi-Script OCR for Historical Indic Manuscripts*, 2026 (arXiv 2608.28195)
- **C03** *On the Generalization of Handwritten Text Recognition Models*, CVPR 2025
- **C06** *Can OCR-VLMs Read Devanagari? A Stress-Test Benchmark and Post-Correction Study*, 2026
- **C07** *Impact of Iterative Fine-Tuning on Transcription of Sanskrit Manuscripts* (FLAME), 2026
- **J05** *DARE: Handwritten Date Recognition*, IJDAR 2026 · **J08** *Multi-modal ancient scripts recognition*, npj Heritage Science 2025
- **J09** Li et al., *HTR-VT: Handwritten Text Recognition with Vision Transformer*, Pattern Recognition 2025
- **J14** *Modi Document Transcription to Devanagari using LSTM*, IIETA 2024
- Coquenet, Chatelain, Paquet, *SPAN: a Simple Predict & Align Network for Handwritten Paragraph Recognition*, ICDAR 2021 (arXiv 2102.08742)
- *Understanding Cross-Language Transfer Improvements in Low-Resource HTR: The Role of Sequence Modeling*, 2026 (arXiv 2605.05900)
- *PLATTER: A Page-Level Handwritten Text Recognition System for Indic Scripts* (arXiv 2502.06172) · *DohaScript* (arXiv 2602.18089), candidate Devanagari pretraining data
- Datasets: MoDeTrans (HF `historyHulk/MoDeTrans`, MIT); MODI-HHDoc (Mendeley, doi 10.17632/sg337vf6wn.1); MODI-HChar

---

## Appendix A — Colab setup cells

```python
# 1. GPU check + Drive
!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
RUNS = '/content/drive/MyDrive/modi/runs'
```

```python
# 2. Code (public repository, no login needed; replace the placeholder with the repository URL)
!git clone https://github.com/<owner>/<repository>.git /content/code
%cd /content/code
!pip -q install -r requirements-train.txt
```

```python
# 3. Small data pack (uploaded once to MyDrive/modi/colab_pack.zip)
!mkdir -p /content/data && unzip -q -o /content/drive/MyDrive/modi/colab_pack.zip -d /content/data
import json, hashlib, pathlib
man = json.load(open('/content/data/MANIFEST.json'))
bad = [k for k, h in man['files'].items()
       if hashlib.sha256(pathlib.Path('/content/data', k).read_bytes()).hexdigest() != h]
print('pack OK' if not bad else f'CORRUPT: {bad}')
```

```python
# 4. MoDeTrans straight from Hugging Face (no upload)
from datasets import load_dataset
mt = load_dataset('historyHulk/MoDeTrans', split='train')   # 2,043 rows: filename, image, text
splits = json.load(open('/content/data/metadata/splits.json'))['modetrans']['assignment']
assert len(mt) == 2043 and all(r in splits for r in mt['filename'])
```

```python
# 5. (Stage 2+ only) MODI-HHDoc straight from Mendeley, about 3-5 min on Colab's network
!wget -q -O /content/hhdoc.rar "$(python -c "import json;print(json.load(open('/content/data/MANIFEST.json'))['hhdoc_url'])")"
!sha256sum /content/hhdoc.rar        # must equal MANIFEST hhdoc_rar_sha256
!apt-get -qq install unrar >/dev/null; unrar x -idq /content/hhdoc.rar /content/
```

## Appendix B — files this design depends on

`FYP-modi-script-data/processed/`: `splits.json`, `modetrans_text.csv` (`text_train`), `charset.json`,
`folder18_labels.csv`, `modetrans_docs.csv`, `hhdoc_manifest.csv` · Reports: `FYP-modi-script-code/outputs/reports/data_audit.md`
(includes `audit_fixes.md`) · Pipeline: `scripts/run_all.py` (steps 01–16), `scripts/17_make_colab_pack.py`.
