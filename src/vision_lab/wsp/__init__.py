"""wsp — weight-space programming (WSP) library for the Vision R3 planted-marker study.

Layout
------
metrics       clean confusion / AUROC / AP / R1 Youden freeze rule / exact feasibility test
gates         signed R1/R2 gate arithmetic (1a,1b,2,3,4,5,6,7) + impossibility flags
significance  "Weight-Space Reading" ground-truth rule: significance AND minimum effect
calibration   source-level calibration partition, access ledger, per-pipeline thresholds
ordering      committed k-list ordering rule (r2_methods.ordered_sources)
heads         A / matched-k retrainers (linear numpy, 1-hidden-layer tanh MLP in torch)
edits         weight-edit families (deletion, scaling, direction removal, noise, signed
              sparse) and six function-preserving no-op controls
slots         slot 1 input masking, slot 2 object-patch scoring, slot 3 hidden deletion
encoders      SigLIP2 / DINOv2 token extraction with in-memory aggregate hooks
actuator      "does the edit move what it claims to move" checks
harness       per-attempt evaluation: thresholds, metrics, gates, feasibility, calls, no-ops
ledger        attempt ledger (no fallbacks) and resource guard
synthetic     planted-marker synthetic worlds for tests and known-answer checks

Nothing in this package reads a holdout: every scoring entry point refuses rows whose
split is not 'train' (calibration) or 'dev'.
"""
__version__ = "0.1.0"
