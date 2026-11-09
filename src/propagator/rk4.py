# This .py file has the runge-kutta methods to solve differential equations

import numpy as np

# Computes terms for obtaining the runge-kutta-nymstrom k1, k2, k3 and k4 parameters and x[n+1], x'[n+1]
def rkn_terms(fun, zn, xn, xprimen, h):
    k1 = fun(zn, xn, xprimen)
    k2 = fun(zn + (h/2), xn + xprimen*(h/2) + (h**2/8)*k1, xprimen + k1*(h/2))
    k3 = fun(zn + (h/2), xn + xprimen*(h/2) + (h**2/8)*k1, xprimen + k2*(h/2))
    k4 = fun(zn + h, xn + xprimen*h + (h**2/2)*k3, xprimen + k3*h)  

    xn_plus1 = xn + h*xprimen + (h**2/6)*(k1+k2+k3)

    xprimen_plus1 = xprimen + (h/6)*(k1+ 2*k2 + 2*k3 + k4)

    return xn_plus1, xprimen_plus1


# Runge-Kutta-Nymstrom method for x'' = f(z,x,x'), useful for when there is magnetic field (momentum constant)
# Check https://iopscience.iop.org/article/10.1088/1748-0221/4/04/P04001/pdf
def rkn(fun, z0, z_end, x0, xprime0, h):
    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}")

    n = max(1, int(np.ceil(abs(z_end - z0) / h)))
    vec_z = np.linspace(z0, z_end, n + 1)
    h_real = vec_z[1] - vec_z[0]    # negativo si se propaga hacia atrás

    vec_x = np.empty((n + 1, *np.shape(x0)))
    vec_xprime = np.empty_like(vec_x)
    vec_x[0], vec_xprime[0] = x0, xprime0

    for k in range(n):
        vec_x[k + 1], vec_xprime[k + 1] = rkn_terms(fun, vec_z[k], vec_x[k], vec_xprime[k], h_real)

    return vec_z, vec_x, vec_xprime


