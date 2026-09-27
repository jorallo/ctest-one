# C-test ONE

Code and data for *An LLM-referenced C-test for supraintelligence*.

C-test ONE generates sequence problems whose difficulty is measured before anyone solves them.
Each item is a sequence produced by a rule. A small open-weight language model, the *reference
machine*, certifies the item by running a description of the rule, and the item's difficulty is
the conditional Levin complexity $\widehat{Kt}^{|}$ of the continuation given the prefix, in bits:
the cost of the cheapest description that makes the reference machine continue the sequence,
plus the logarithm of the tokens it needs. Items are kept only if they are *valid* (the rule
reproduces the sequence), *unquestionable* (no cheaper rival rule continues the shown prefix
differently) and *certified* (the reference machine can run a description of them). The released
battery has 151 generated items, 107 valid, and results for twelve examinee models under three
reference machines: Qwen3-14B, Qwen3-8B and Phi-4-reasoning.

## What is here

```
notebooks/   the pipeline, one notebook per stage (see below)
data/        the battery, the pricing and certification records, the examinees' results,
             the calibration constants, themes, prompts, and Epoch AI's ECI data
stats/       scripts that reproduce the statistics reported in the paper
```

## Reproducing the results from the released data

This needs no GPU and no API keys.

```
pip install -r requirements-analysis.txt
python stats/ctest_stats_scipy.py
python stats/ctest_stats_more.py
jupyter lab notebooks/07_results.ipynb notebooks/08_results_eci.ipynb
```

`07_results` produces the tables and figures for one reference machine (set `MACHINE_ID` and
`MACHINE_TAG`), and `08_results_eci` the comparison with Epoch AI's Capabilities Index. The
scripts in `stats/` reproduce the statistical tests; see `stats/README.md`.

## The pipeline

| notebook | stage | needs | reads | writes |
|---|---|---|---|---|
| `01_calibrator` | constants of each reference machine | GPU | | `ctest_calibration.json` |
| `02_generator` | compose, validate and screen items; equivalents and rivals | Gemini API | calibration, themes | battery, programs, themes, prompts |
| `03_unquestionability` | cost rivals, find the unquestionable prefix $L_u$ | GPU | battery, programs, calibration | `ctest_pricing_<machine>.json` |
| `04_rt_measure` | type descriptions into retrieval and transformation steps | Gemini API | battery, pricing | battery |
| `05_annotator` | certify items and measure $\widehat{Kt}^{\|}$ | GPU, Gemini API | battery, programs, pricing | `ctest_metrics.json` |
| `06_evaluator` | administer items to examinees and score them | APIs; GPU for local examinees | battery, metrics | `ctest_results.json` |
| `07_results` | tables and figures for one machine | nothing | all of the above | figures and LaTeX tables |
| `08_results_eci` | abilities against Epoch AI's ECI | nothing | all of the above, ECI files | figures and LaTeX tables |

Stages 03 to 05 run once per reference machine. The GPU stages load the reference machines in
4-bit, which fits a 16 GB GPU for models up to 14B and needs `bitsandbytes` with CUDA. The
annotator is the slowest stage, at tens of minutes per item at the hard end of the scale.

## Running the notebooks

**Locally.** Install everything with `pip install -r requirements.txt`, copy `.env.example` to
`.env` and fill in the keys you need, and open the notebooks from the repository root with
`jupyter lab`. On its first local run, each notebook copies `data/` into a `work/` folder and runs
there, so the released data are never modified. Set `CTEST_WORK_DIR` to use another folder.

**Colab.** Upload the files in `data/` to `MyDrive/ctest`, mount Drive, and add the keys in the
Secrets panel.

**Kaggle.** Create a dataset with the files in `data/`, attach it to the notebook, and set
`KAGGLE_DATASET` to its name. Add the keys in Add-ons, Secrets. With Kaggle credentials the
notebooks can push their results back as a new version of the dataset.

The keys are `GOOGLE_API_KEY` (Gemini), `OPENAI_API_KEY`, `CLAUDE_API_KEY` (Anthropic) and
`HF_TOKEN` (Hugging Face). Llama and Gemma are gated on Hugging Face: accept their licences
there before loading them.

## Regenerating the battery

The released battery was built over many successive versions of these notebooks, and the
notebooks here are the final versions. They reproduce every table, figure and statistic from the
released data. Running the generator again produces a new battery, not this one, since generation
depends on the composer model's sampling and on the versions used along the way.

## Known issues

These affect the released data and are discussed in the paper where relevant.

- **Items with a missed rival.** In six items the prefix admits a simpler reading that the
  unquestionability check missed, and most examinees give the same wrong answer: 0133, 0110 and
  0127 under Qwen3-14B, 0002 and 0130 under Qwen3-8B, and 0133, 0002, 0045 and 0130 under
  Phi-4-reasoning. They are kept in the battery. Removing them raises every correlation between
  $\widehat{Kt}^{|}$ and accuracy, to $-0.84$ under Qwen3-14B.
- **An arithmetic slip of the generator.** In 0142, certified only under Phi-4-reasoning, the
  generator reversed *autumn* wrongly in 15 of its 72 terms.
- **Asterisks in terms.** In 0089, 0090, 0102, 0120 and 0142 the terms contain `*`, which the
  examinees' answers do not keep, so their scores depend on how the judge treats them.
- **Two values of $\delta$ for Qwen3-8B.** 29 items were costed with an earlier value, 13.3 bits
  (recorded under `_delta_calibration` in `ctest_pricing_qwen3-8b.json`), and the rest with 22.7.
  With 22.7, items 0012 and 0045 become questionable and 0002 needs $L_u = 13$ instead of 5.
- **Examinee budgets.** gpt-5.6-sol, Claude Opus 5, Sonnet 5 and Fable 5.1 had 16000 tokens,
  except on the last 13 items added, where 147 of their replies ran with 8000; none of those
  reached 8000. Each row of `ctest_results.json` records its own `budget`.
- **Fallback.** Examinees named `-fallback` used Anthropic's server-side fallback, which answers a
  refused request with another model. Which model served each reply was not recorded.
- **Calibration.** Items 1 to 146 were generated with the constants in `pooled_before` and items
  147 to 151 with those in `pooled`, both in `ctest_calibration.json`.
- **Typing fields.** The unsuffixed `r_hat` and `t_hat` in the battery match no single machine;
  use `r_hat_<machine>` and `t_hat_<machine>`.
- **The judge.** At least one wrong answer is scored as correct (0003, Gemini 3.8 Flash, first
  attempt). Elsewhere the judge agrees with a mechanical comparison of the terms on nearly every
  reply.

$\widehat{Kt}^{|}$ is an upper bound: it is the cost of the cheapest description the search
found, and a longer search can only lower it.

## Licence

The code is under the MIT licence (`LICENSE`). The data and the documentation are under CC BY 4.0
(`LICENSE-DATA.md`), which allows any use with credit. The ECI files in `data/` are from Epoch AI,
also under CC BY 4.0.

## Citation

See `CITATION.cff`, or use GitHub's "Cite this repository" button.
