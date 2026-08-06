"""Supervised value head — probabilistic gem-valuation forecaster.

The head predicts, per color, the
distribution over the final value-display COUNT n_c (deal-determined, revealed over
the game, chart-independent), which the public value chart maps to a value
distribution. Opponent-INDEPENDENT and ground-truthed, so its gains should transfer.
"""
