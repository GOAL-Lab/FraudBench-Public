"""FraudBench regression-modeling layer (HiBayes / NumPyro).

Implements the three regressions of the paper appendix
(``analysis/regressions/M*.yaml`` are the run configurations):

* :mod:`fraudbench.analysis.regressions.models` -- the ``fraudbench_m1``,
  ``fraudbench_m2`` and ``fraudbench_m3`` builders and the shared M1 linear
  predictor.
* :mod:`fraudbench.analysis.regressions.process` -- row filters, index /
  indicator builders and the outcome selector consumed by the builders.
* :mod:`fraudbench.analysis.regressions.merge_external` -- joins of
  scenario / case / model / human-baseline metadata onto the per-row
  evaluation parquet.
* :mod:`fraudbench.analysis.regressions.predictor` -- NumPy reconstruction
  of the M1 predictor from posterior draws (posterior-predictive
  human-vs-LLM gap).
* :mod:`fraudbench.analysis.regressions.checkers` -- non-interactive
  posterior-predictive checker for batch fits.
"""
