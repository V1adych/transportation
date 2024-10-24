import numpy as np


def russel(S: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    This method uses Russel's approximation method to find the initial basic feasible solution
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

    S = S.copy()
    D = D.copy()

    used_cols = np.zeros(n, dtype=np.int32)
    used_rows = np.zeros(m, dtype=np.int32)

    while np.sum(S) > 0 and np.sum(D) > 0:
        penalties = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if used_cols[j] or used_rows[i]:
                    continue
                row_max = np.max(C[i, :])
                col_max = np.max(C[:, j])
                penalties[i, j] = C[i, j] - row_max - col_max
        print(penalties)
        i, j = np.unravel_index(np.argmin(penalties), penalties.shape)

        allocation = min(S[i], D[j])
        X[i, j] = allocation
        S[i] -= allocation
        D[j] -= allocation
        if S[i] == 0:
            used_rows[i] = 1
        if D[j] == 0:
            used_cols[j] = 1

    return True, X
