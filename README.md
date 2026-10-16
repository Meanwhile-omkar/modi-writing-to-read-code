# MODI-SYNTH

**Writing to Read: Style-Conditioned Handwriting Generation for Low-Resource Historical Modi Script Recognition**

Modi is a cursive script that was used to write Marathi in administrative documents from the 17th to the mid-20th century. Very few people can read it today, and only about two thousand handwritten samples have expert transcriptions. This project asks a simple question:

> Can a *learned* handwriting generator (the **Writer**) produce synthetic training data that helps a **Reader** more than synthetic data made from fonts?

To answer it fairly we build (1) a baseline Reader trained on real data, (2) an evaluation suite and a leak-free out-of-domain test, and (3) a font-based synthetic-data generator that serves as the control. The learned Writer comes next.

| Arm | Reader is trained on |
|---|---|
| A | real data only |
| B | real data + font-rendered synthetic patches |
| C | real data + Writer-generated synthetic patches |
| D (optional) | + pseudo-labelled unlabelled pages |

Design of the baseline: [`docs/BASELINE_DESIGN.md`](docs/BASELINE_DESIGN.md).

## Status

**Stage 1 (baseline Reader) is complete** — tag `stage1-final-baseline`. The Reader works on text **lines**
(per-strip line finding + synthetic-line pretraining + character LM + a 4-model ensemble) and reaches
**test CER 24.55 %** (95 % CI 23.09–26.15) on the 306 MoDeTrans test patches. Results, recipe and how to reproduce:
[`results/README.md`](results/README.md); the full experimental record: [`research/root-cause/`](research/root-cause/).

## Repository layout

```
configs/paths.yaml       single source of truth for every data path
src/modi/
  data/                  dataset helpers (image statistics, matching)
  preprocess/            binarisation, deskew, quality filters
  segmentation/          text-line / band segmentation
  recognition/           the Reader (CTC models)
  evaluation/            CER / WER / chrF++ / BLEU, statistics, error types
  generation/            the Writer (later stage)
  synthetic/             font-based synthetic patches (Arm B)
  utils/                 shared helpers
scripts/                 numbered data-preparation steps (01-18) and run_all.py
assets/metadata/         small ready-made data files (see below)
notebooks/               Colab / Kaggle notebooks
tests/                   unit tests
outputs/                 reports, figures, logs (generated, mostly not tracked)
```

## Getting started

```bash
git clone <this repository>
cd <repository>
python -m venv .venv
.venv/Scripts/activate            # Windows Git Bash:  source .venv/Scripts/activate
pip install -r requirements.txt   # local (CPU) work
pip install -r requirements-train.txt   # Colab / Kaggle GPU work
python -m pytest                  # runs the unit tests
```

### Data (not stored in this repository)

| Dataset | Where to get it | Licence / terms |
|---|---|---|
| **MoDeTrans** — 2,043 handwritten patches + transcriptions | https://huggingface.co/datasets/historyHulk/MoDeTrans (`load_dataset("historyHulk/MoDeTrans")`) | MIT |
| **MODI-HHDoc** — 3,350 unlabelled page photos | https://data.mendeley.com/datasets/sg337vf6wn/1 | research use only, **no redistribution** |
| **MODI-HChar** — isolated character images | https://data.mendeley.com/datasets/pk2zrt58pp/1 | research use only, **no redistribution** |

Never commit raw dataset files, or any image of MODI-HHDoc, MODI-HChar or the Folder-18 book (their terms forbid redistribution). Figures made from MoDeTrans images (MIT) or from synthetic renders may be committed as part of a run's record. Paths are read from `configs/paths.yaml`; edit that file if your data sits elsewhere.

### Ready-made data files (`assets/metadata/`)

| File | Content |
|---|---|
| `modetrans_text.csv` | expert transcriptions: `text_raw`, `text_norm` and `text_train` (what models learn from) |
| `charset.json` | the 86 output characters + `<FRAC>`, `<blank>`, `<unk>` |
| `splits.json` | train / validation / test assignment of every MoDeTrans patch (seed 42, split by source document) |
| `modetrans_docs.csv` | source-document grouping of the patches |
| `folder18_labels.csv` | page types and test tiers for the 88 pages of the book used as out-of-domain test data |

**Use these files as they are.** Everyone must train and evaluate on the identical split; regenerating them produces a different split and results that cannot be compared. The scripts in `scripts/` document how they were made.

## Interfaces between workstreams

| Interface | Format |
|---|---|
| **Predictions file** | UTF-8 CSV `results/<run_name>/test_predictions.csv` with columns `image_id,reference,prediction`, next to a `config.json` of the run |
| **Book answer key** | CSV with columns `letter_no,tier,facsimile_pages,transcription_pages,text,n_chars` (kept outside git because of the book's terms) |
| **Synthetic patch** | Python function `make_patch(seed: int) -> (image, text)`; image is `numpy.uint8` of shape `(512, 1408)`, dark ink = 0 on white = 255; text only uses symbols from `charset.json`; the same seed always returns the same output |

## Working rules

1. One branch per workstream (`feat/reader`, `feat/eval`, `feat/synth`); open a pull request into `main`.
2. Only change files in your own area; shared files (`configs/`, `requirements*.txt`, this README) change through a reviewed pull request.
3. Raw data is never committed, nor images from HHDoc, HChar or the Folder-18 book; derived files stay small.
4. Splits are made at **source-document level**, never at patch or line level.
5. Every experiment logs its config, random seed and git commit hash, and keeps its failed runs.
6. Report results as they are, with the seed and the number of times the test set was used.
