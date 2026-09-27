# Statistical analyses

Scripts that reproduce the statistics reported in the paper from the released data. They read
`../data` unless the environment variable `CTEST_DATA_DIR` points elsewhere.

| script | what it computes |
|---|---|
| `ctest_data.py` | shared inputs: the measures of each certified item, each examinee's accuracy under full marking, and the abilities |
| `ctest_stats_scipy.py` | the tests of the statistics appendix, with `scipy.stats` |
| `ctest_stats_plain.py` | the same tests in plain Python, as they were first computed |
| `ctest_stats_more.py` | the further analyses in the other appendices, in named sections |

```
python stats/ctest_stats_scipy.py
python stats/ctest_stats_more.py                  # every section
python stats/ctest_stats_more.py ratio kt_min     # named sections only
```

The sections of `ctest_stats_more.py` are `prior`, `ratio`, `kt_min`, `tau`, `upper`, `seed`,
`budget`, `refusals`, `judge`, `rivals`, `supra` and `misspecified`. The random seeds are fixed, so
each run gives the same result. Bootstrap bounds and permutation p-values can differ slightly
between scripts, since they come from different random resamples.
