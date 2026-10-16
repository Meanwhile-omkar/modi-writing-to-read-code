# Floor baselines (Referee Step 3)

Silly predictors on the MoDeTrans validation (307) and test (306) patches, `text_train`. Each random predictor is run with 100 seeds (42-141): mean ± standard deviation. CER and WER in %, lower is better; chrF++ higher is better.

| predictor | val CER | val WER | val chrF++ | test CER | test WER | test chrF++ |
|---|---|---|---|---|---|---|
| empty text | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 |
| random training sentence | 98.6 ± 1.8 | 120.2 ± 1.9 | 9.1 ± 0.2 | 98.1 ± 1.7 | 119.6 ± 1.8 | 9.0 ± 0.2 |
| random training sentence, cut or padded with spaces | 84.2 ± 0.1 | 101.3 ± 0.3 | 8.6 ± 0.2 | 84.3 ± 0.1 | 101.3 ± 0.2 | 8.5 ± 0.2 |
| random training text, same length | 83.1 ± 0.1 | 106.4 ± 0.4 | 13.4 ± 0.1 | 83.1 ± 0.1 | 107.2 ± 0.4 | 13.4 ± 0.1 |
| correct letters, shuffled | 84.6 ± 0.1 | 99.9 ± 0.0 | 15.5 ± 0.0 | 84.6 ± 0.1 | 99.9 ± 0.0 | 15.6 ± 0.0 |
| correct text, 10 % damaged | 9.9 ± 0.0 | 50.3 ± 0.3 | 62.6 ± 0.2 | 9.9 ± 0.0 | 50.5 ± 0.4 | 62.6 ± 0.1 |
| correct text, 30 % damaged | 28.9 ± 0.0 | 88.5 ± 0.4 | 29.9 ± 0.2 | 28.9 ± 0.1 | 88.5 ± 0.3 | 30.0 ± 0.1 |
