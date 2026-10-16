# SPAN input audit

The SPAN pipeline edits an image in exactly two ways. Both are rendered here.

| file | what it shows |
|---|---|
| `01_the_one_edit_with_output_grid.png` | the whole patch scaled once to 880x320, with the 10 rows of the model's output grid drawn on it. Nothing is cut, split, reordered or discarded. |
| `02_augmentations_at_published_rate_*.png` | eleven training samples per patch, each augmentation at SPAN's probability of 0.2 |
| `03_stress_test_all_augmentations_on.png` | every augmentation forced on at once, to see the worst case |
| `augmentation_stats.json` | measured over 900 augmented samples: how often an augmentation leaves a page low-contrast |

Why this replaced the line-cutting pipeline of notebooks 02 and 03: that pipeline had to find the
text lines first, and it failed in three separate ways (merged lines crushed into one band, header
lines dropped, ink thrown away) before any of them could be seen. SPAN needs no segmentation, so
that whole class of failure cannot occur.
