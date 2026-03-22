"""Feature engineering for AIMO3.

For this competition, "features" means prompt engineering -
transforming raw LaTeX math problems into effective LLM prompts.
"""

from __future__ import annotations

import re


def clean_problem_text(problem: str) -> str:
    """Clean and normalize LaTeX problem text.

    - Normalize whitespace
    - Ensure LaTeX delimiters are consistent
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", problem).strip()
    return text


def classify_problem_domain(problem: str) -> str:
    """Heuristic classification of math problem domain.

    Returns one of: algebra, combinatorics, geometry, number_theory, unknown
    """
    text_lower = problem.lower()

    geometry_keywords = [
        "triangle", "circle", "circumcircle", "incircle", "angle", "polygon",
        "perpendicular", "parallel", "midpoint", "circumscribed", "inscribed",
        "tangent", "chord", "diameter", "radius", "area", "perimeter",
    ]
    number_theory_keywords = [
        "divisor", "prime", "gcd", "lcm", "modulo", "remainder", "divides",
        "congruent", "factorial", "integer", "coprime",
    ]
    combinatorics_keywords = [
        "tournament", "permutation", "combination", "count", "arrange",
        "sequence", "subset", "bijection", "coloring", "graph",
    ]
    algebra_keywords = [
        "polynomial", "equation", "root", "function", "inequality",
        "sum", "product", "series", "real number",
    ]

    scores = {
        "geometry": sum(1 for kw in geometry_keywords if kw in text_lower),
        "number_theory": sum(1 for kw in number_theory_keywords if kw in text_lower),
        "combinatorics": sum(1 for kw in combinatorics_keywords if kw in text_lower),
        "algebra": sum(1 for kw in algebra_keywords if kw in text_lower),
    }

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"
