"""Post-hoc analysis tools that may read private hands.

NOT part of the hand-free RL core (``src/megagem/rl/``). Nothing here is imported by
the reward / advantage / scorer / export paths; the structural hand-free
leakage gate (tests/test_rl_scorer.py) is unaffected. One-directional: this
package imports hand-free primitives from ``megagem.rl``, never the reverse.
"""
