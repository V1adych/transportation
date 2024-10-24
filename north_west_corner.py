import numpy as np


def north_west_corner(S: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    This method uses north-west corner method to find the initial basic feasible solution
    for the transportation problem with the given supply, demand and cost matrices.

    Args:
        S (np.ndarray): Supply vector of shape (m)
        C (np.ndarray): Cost matrix of shape (m, n)
        D (np.ndarray): Demand vector of shape (n)

    Returns:
        tuple: A tuple containing a boolean value 
        indicating if the solution is feasible and 
        the initial basic feasible solution.
    """
    if np.sum(S) != np.sum(D):
        return False, None
    m = S.size
    n = D.size

    X = np.zeros((m, n))
    i = 0
    j = 0
    S = S.copy()
    D = D.copy()
    while i < m and j < n:
        if S[i] < D[j]:
            X[i][j] = S[i]
            D[j] -= S[i]
            i += 1
        elif S[i] > D[j]:
            X[i][j] = D[j]
            S[i] -= D[j]
            j += 1
        else:
            X[i][j] = S[i]
            i += 1
            j += 1

    return True, X
