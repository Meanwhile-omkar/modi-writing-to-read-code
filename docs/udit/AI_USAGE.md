# Use of AI assistance (Udit Sevak)

The team rules allow AI assistants and ask each of us to say which parts they wrote.

For my parts (Referee and Factory), most of the code, tests and documentation were written by an AI coding
assistant, Claude (Anthropic), used through Claude Code from 12 to 16 October 2026. This covers
`src/modi/synthetic/`, `src/modi/evaluation/`, `src/modi/segmentation/folder18_lines.py`, `scripts/30_*`, the
tests for these, `docs/factory/`, `docs/EVALUATION_PROTOCOL.md`, `docs/referee_report.md`, and first drafts of
this record (`docs/udit/`).

What I did: decided what to build and in which order from the handouts and the lead review; chose between the
options the assistant proposed (for example the leakage test, publishing the synthetic set as a release instead
of committing images, the look-alike rule); checked results and pictures; judged whether the generated Marathi
reads like the old administrative language; ran the Colab check; reviewed and approved every commit; and did the
manual work: proofreading the answer key and counting the lines of the book pages by hand.

Every number in the reports comes from code in the repository and can be rerun. I am responsible for all of it
and should be able to explain each part; where I cannot yet, I will go through it before the viva.
