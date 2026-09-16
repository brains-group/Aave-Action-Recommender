"""Finite-grid sensitivity search without a monotonicity assumption."""
import math

def first_flip(predict, baseline, groups):
    base = bool(predict(baseline))
    # Use actual distances, not rounded grouping distances.
    points = sorted((math.dist(coeffs, baseline), tuple(coeffs))
                    for group in groups for coeffs in group['combinations'])
    evaluations = 0
    for distance, coeffs in points:
        evaluations += 1
        # Exceptions propagate: an unevaluated point is not evidence of stability.
        if bool(predict(coeffs)) != base:
            return dict(base_prediction=base, change_found=True,
                        min_distance=distance, evaluations=evaluations,
                        method='finite-grid-ascending-v1')
    return dict(base_prediction=base, change_found=False, min_distance=None,
                evaluations=evaluations, method='finite-grid-ascending-v1')
