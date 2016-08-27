import bisect
from functools import partial


import numpy as np
from numpy.linalg import norm, svd as _dense_svd
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_array

try:
    from pypropack import svds as svds
except:
    from scipy.sparse.linalg import svds


def _dense_partial_svd(a, k):
    u, sig, v = _dense_svd(a, full_matrices=0)
    return u[:, :k], sig[:k], v[:k, :]


def _svd_truncate(u, sig, vt, r):
    return u[:, :r], sig[:r], vt[:r, :]


def _svd_reconstruct(u, sig, vt, out=None, tmp_out=None):
    tmp = np.multiply(u, sig, out=tmp_out)
    return np.dot(tmp, vt, out=out)


def _svd_choice(n, d):
    ratio = float(d) / float(n)
    vals = [(0, 0.02), (100, 0.06), (200, 0.26),
            (300, 0.28), (400, 0.34), (500, 0.38)]

    i = bisect.bisect_left([r[0] for r in vals], n)
    choice = 'sparse' if ratio < vals[i][1] else 'dense'
    return _svd_choices[choice]

_svd_choices = {
    'dense': _dense_partial_svd,
    'sparse': svds,
    'random': randomized_svd
}


def svt(X, tau, k, svd_type='auto', out=None):
    m, n = X.shape

    if svd_type == 'auto':
        svd_fun = _svd_choice(np.min((m, n)), k)
    else:
        svd_fun = _svd_choices[svd_type]

    u, sig, vt = svd_fun(X, k)

    sig = soft_threshold(sig, tau)
    r = np.sum(sig > 0)

    if r > 0:
        tmp_out = X[:, :r]
        u, sig, vt = _svd_truncate(u, sig, vt, r)
        Z = _svd_reconstruct(u, sig, vt, out=out, tmp_out=tmp_out)
    else:
        out[:] = 0
        Z = out
        u, sig, vt = np.empty((m, 0)), np.empty((0, 0)), np.empty((0, n))
    return (Z, r, (u, sig, vt))


def soft_threshold(v, tau, out=None):
    tmp = np.absolute(v, out=out)
    tmp = np.subtract(tmp, tau, out=out)
    tmp = np.maximum(tmp, 0.0, out=out)
    return np.multiply(np.sign(v), tmp, out=out)


def _fro_norm(M):
    fnorm = norm(M, 'fro')

    def f(R):
        return norm(R, 'fro') / fnorm
    return f


def _pcp_alm(X, gamma=None, maxiter=500, tol=1e-6,
             check_input=True, svd_type='auto'):
    """Principal Component Pursuit

    Finds the Principal Component Pursuit solution.

    Solves the optimization problem::

        (L^*,S^*) = argmin || L ||_* + gamma * || S ||_1
                    (L,S)
                    subject to    L + S = X

    where || . ||_* is the nuclear norm.  Uses an augmented Lagrangian approach

    Parameters
    ----------

    X : array of shape [n_samples, n_features]
        Data matrix.

    gamma : float, 1/sqrt(max(n_samples, n_features)) by default
        The l_1 regularization parameter to be used.

    maxiter : int, 500 by default
        Maximum number of iterations to perform.

    tol : float, 1e-6 by default

    check_input : boolean, optional
        If False, the input array X will not be checked.

    Returns
    -------

    L : array of shape [n_samples, n_features]
        The low rank component

    S : array of shape [n_samples, n_features]
        The sparse component

    (u, sig, vt) : tuple of arrays
        SVD of L.

    n_iter : int
        Number of iterations

    Reference
    ---------

       Candes, Li, Ma, and Wright
       Robust Principal Component Analysis?
       Submitted for publication, December 2009.

    """

    if check_input:
        X = check_array(X, ensure_min_samples=2)

    n = X.shape
    frob_norm = np.linalg.norm(X, 'fro')
    two_norm = np.linalg.norm(X, 2)
    one_norm = np.sum(np.abs(X))
    inf_norm = np.max(np.abs(X))

    mu_inv = 4 * one_norm / np.prod(n)

    # Kicking
    k = np.min([
        np.floor(mu_inv / two_norm),
        np.floor(gamma * mu_inv / inf_norm)
    ])
    Y = k * X
    sv = 10

    # Variable init
    zero_mat = np.zeros(n)
    S = zero_mat.copy()
    L = zero_mat.copy()
    R = X.copy()
    T1 = zero_mat.copy()
    T2 = zero_mat.copy()

    np.multiply(Y, mu_inv, out=T1)
    np.add(T1, X, out=T1)

    for i in range(maxiter):
        # Shrink entries
        np.subtract(T1, L, out=T2)
        S = soft_threshold(T2, gamma * mu_inv, out=S)

        # Shrink singular values
        np.subtract(T1, S, out=T2)
        L, r, (u, sig, vt) = svt(T2, mu_inv, sv, out=L, svd_type=svd_type)

        if r < sv:
            sv = np.min([r + 1, np.min(n)])
        else:
            sv = np.min([r + np.round(0.05 * np.min(n)), np.min(n)])

        np.subtract(X, L, out=R)
        np.subtract(R, S, out=R)
        stopCriterion = np.linalg.norm(R, 'fro') / frob_norm

        # Check convergence
        if stopCriterion < tol:
            break

        # Update dual variable
        np.multiply(R, 1. / mu_inv, out=T2)
        np.add(T2, Y, out=Y)
        Y += R / mu_inv

        np.add(T1, R, out=T1)

    return L, S, (u, sig, vt), i + 1


def _low_rank_project(U, X, rho=1, alpha=1.5, maxiter=100):
    """L1 minimized Low Rank Projection

    Project set of obserations X onto the subspace U using orthongal
    least absolute deviations

    Parameters
    ----------

    U: array of shape [n_samples, ]
        Left subspace of low-rank component

    X: array of shape [n_samples, n_features]
        Observations samples

    rho: float (default=1)
        The augmented Lagrangian parameter

    alpha: float default=1.5)
        The over-relaxation parameter (typical values for alpha
        are between 1.0 and 1.8).

    maxiter: int (default=100)
        Maximum iterations for LAD algorithm.

    Returns
    -------

    L: array of shape (n)
        Low-rank projection.

    S: array of shape (n)
        Sparse residual

    """

    lad = partial(_orthogonal_least_absolute_deviations,
                  U, rho=rho, alpha=alpha, maxiter=maxiter)

    for x in X:
        l, s = lad(x)


def _orthogonal_least_absolute_deviations(A, b, alpha=1.5,
                                          rho=1, mu=10, tau=2, maxiter=1000):
    """Orthogonal Least absolute deviations fitting via ADMM

    Specialized LAD solving with an orthogonal matrix input.  Solves the
    optimization problem via ADMM:

        (x^*, z^*) = argmin || z ||_1
                       z
                     subject to   A x - z = b

    The solution is returned in the vector x and the residual is in z.

    Parameters
    ----------

    A: array of shape (m, n)
        Orthogonal data matrix.

    b: array of shape (n)
        Target values

    rho: float
        The augmented Lagrangian parameter

    alpha: float
        The over-relaxation parameter (typical values for alpha
        are between 1.0 and 1.8).

    Returns
    -------

    x: array of shape (n)
        Solution

    z: array of shape (n)
        Residual

    Reference
    ---------

       S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein.  Distributed
       optimization and statistical learning via the alternating direction
       method ofmultipliers.  Foundations and Trends in Machine Learning,
       3(1):1–124, 2011.

       https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html

    """

    # Global constants and defaults

    _ABSTOL = 1e-4
    _RELTOL = 1e-2

    m, n = A.shape

    # ADMM solver
    x = np.zeros(n, 1)
    z = np.zeros(m, 1)
    z_old = np.zeros(m, 1)
    u = np.zeros(m, 1)

    eps_pri_factor = np.sqrt(m) * _ABSTOL + _RELTOL
    eps_dual_factor = np.sqrt(n) * _ABSTOL + _RELTOL

    for k in range(maxiter):
        np.copyto(z_old, z)

        # ADMM steps
        x = np.dot(A.T, b + z - u)
        Ax = np.dot(A, x)
        Ax_hat = alpha * Ax + (1 - alpha) * (z + b)
        z = soft_threshold(Ax_hat - b + u, 1 / rho, out=z)
        u += Ax_hat - z - b

        # Primal/Dual residuals
        r_norm = norm(Ax - z - b)
        s_norm = rho * norm(z - z_old)

        tmp = np.max([norm(x), norm(z), norm(b)])
        eps_pri = eps_pri_factor * tmp
        eps_dual = eps_dual_factor * rho * norm(np.dot(A, u))

        if (r_norm < eps_pri) and (s_norm < eps_dual):
            break

        # Update rho
        if r_norm > mu * s_norm:
            rho = tau * rho
            u = u / tau
        elif s_norm > mu * r_norm:
            rho = rho / tau
            u = u * tau

    return x, z
