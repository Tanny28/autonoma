"""O1 — Signal extraction layer (three families). Owner: Pranav Shimpi.

  distributional : KS-test, PSI, chi-square, ADWIN
  performance    : rolling accuracy / F1 delta, once delayed labels arrive
  integrity      : null rate, cardinality change, range/unit violation,
                   staleness, schema order

The integrity family is what makes `upstream_pipeline_fault` separable from
genuine drift. Without it the project's central distinction is unmeasurable.
"""
