"""evalnoise: where does the noise in an eval score come from, and where should the next calls go?"""

from .files import ScoreTable, read_scores
from .study import Component, Plan, Projection, Study, compare, study, to_score

__version__ = "0.1.0"

__all__ = ["Component", "Plan", "Projection", "Study", "ScoreTable", "compare", "read_scores", "study", "to_score", "__version__"]
