pyrpca
======


Python implementations of RPCA

Usage
-----

```
import numpy as np
import pyrpca

n = 50
r = 2
np.random.seed(123)
base = 100 + np.cumsum(np.random.randn(n, r), axis=0)
scales = np.abs(np.random.randn(n, r))
L = np.dot(base, scales.T)
S = np.round(0.25 * np.random.randn(n, n))
M = L + S

L_hat, S_hat, r, niter = pyrpca.rpca_alm(M)
np.max(np.abs(S - S_hat))
np.max(np.abs(L - L_hat))

_, s, _ = np.linalg.svd(L, full_matrices=False)
print s[s > 1e-11]

_, s_hat, _ = np.linalg.svd(L_hat, full_matrices=False)
print s_hat[s_hat > 1e-11]

```


Requirements
------------

+ Numpy
+ PyPROPACK (https://github.com/jakevdp/pypropack)


Authors
-------

`pyrpca` was written by `Alex Papanicolaou <alex.papanic@gmail.com>`_.


Reference
---------

Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal component analysis?. Journal of the ACM (JACM), 58(3), 11.
[http://statweb.stanford.edu/~candes/papers/RobustPCA.pdf](http://statweb.stanford.edu/~candes/papers/RobustPCA.pdf)