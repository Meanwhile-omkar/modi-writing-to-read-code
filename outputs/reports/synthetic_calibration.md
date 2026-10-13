# Synthetic patch calibration (Factory Step 4)

1430 real training patches vs 1430 fake patches (`make_patch(0..)`), both at the Reader input size (512 x 1402).

| statistic | real median | fake median | KS distance (0 = same, 1 = disjoint) |
|---|---|---|---|
| ink share | 0.147 | 0.149 | 0.16 |
| stroke width (px) | 9.11 | 8.49 | 0.11 |
| text lines | 2 | 2 | 0.13 |
| text length | 91.5 | 91.5 | 0.03 |
