# Team metadata (derived files, safe to commit)

| File | What it is |
|---|---|
| `splits.json` | which MoDeTrans patch is train / val / test (seed 42); also HHDoc and HChar assignments |
| `modetrans_text.csv` | expert transcriptions: `text_raw` (original), `text_norm`, `text_train` (what models learn) |
| `charset.json` | the 86 symbols + `<FRAC>`, `<blank>`, `<unk>` |
| `modetrans_docs.csv` | patch -> source-document grouping |
| `folder18_labels.csv` | page type, letter number and test tier for the 88 pages of the book (folder 18) |

Images are NOT here. MoDeTrans images come from Hugging Face (`historyHulk/MoDeTrans`);
folder-18 pages come from the HHDoc download or `folder18_pages.zip`.
Regenerate with `python scripts/18_export_team_metadata.py`.
