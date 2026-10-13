# Floor baselines (Referee Step 3)

Silly predictors on the MoDeTrans validation (307) and test (306) patches, `text_train`, seed 42. CER and WER in %, lower is better; chrF++ higher is better.

| predictor | val CER | val WER | val chrF++ | test CER | test WER | test chrF++ |
|---|---|---|---|---|---|---|
| empty text | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 |
| random training sentence | 95.2 | 116.7 | 9.0 | 95.0 | 116.1 | 8.8 |
| random training text, same length | 83.1 | 106.8 | 13.1 | 83.2 | 107.4 | 13.3 |
| correct letters, shuffled | 84.5 | 99.9 | 15.5 | 84.5 | 99.8 | 15.5 |
| correct text, 10 % damaged | 9.9 | 50.0 | 62.6 | 9.9 | 51.9 | 62.0 |
| correct text, 30 % damaged | 28.8 | 88.6 | 30.0 | 28.8 | 88.9 | 29.9 |
