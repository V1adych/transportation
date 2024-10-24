import numpy as np


def vogel(S: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    This method uses Vogel's approximation method to find the initial basic feasible solution
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
    n = D.size
    m = S.size
    D = D.copy()
    S = S.copy()
    X = np.zeros((m, n))

    while np.any(D > 0) and np.any(S > 0):
        row_penalty = np.zeros(m)
        col_penalty = np.zeros(n)

        for i in range(m):
            if S[i] > 0:
                sorted_row = np.sort(C[i, :][D > 0])
                if len(sorted_row) > 1:
                    row_penalty[i] = sorted_row[1] - sorted_row[0]
                else:
                    row_penalty[i] = sorted_row[0]

        for j in range(n):
            if D[j] > 0:
                sorted_col = np.sort(C[:, j][S > 0])
                if len(sorted_col) > 1:
                    col_penalty[j] = sorted_col[1] - sorted_col[0]
                else:
                    col_penalty[j] = sorted_col[0]

        max_row_penalty = np.max(row_penalty)
        max_col_penalty = np.max(col_penalty)

        if max_row_penalty > max_col_penalty:
            i = np.argmax(row_penalty)
            allowed_indices = np.where(D > 0)[0]
            j = allowed_indices[np.argmin(C[i, allowed_indices])]
        else:
            j = np.argmax(col_penalty)
            allowed_indices = np.where(S > 0)[0]
            i = allowed_indices[np.argmin(C[allowed_indices, j])]

        allocation = min(S[i], D[j])
        X[i, j] = allocation
        S[i] -= allocation
        D[j] -= allocation

    return True, X
