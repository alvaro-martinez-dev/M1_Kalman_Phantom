# This .py file has the runge-kutta methods to solve differential equations

import numpy as np

# ---------------------------------------------------------------------------------------------
# ----- Runge-Kutta-Nymstrom method for x'' = f(z, x, x'), with x' = dx/dz --------------------
# ----- Check https://iopscience.iop.org/article/10.1088/1748-0221/4/04/P04001/pdf ------------
# ---------------------------------------------------------------------------------------------

# Check https://iopscience.iop.org/article/10.1088/1748-0221/4/04/P04001/pdf
# Computes terms for obtaining the runge-kutta-nymstrom k1, k2, k3 and k4 parameters and x[n+1], x'[n+1]

def rkn4_error(h, k1, k2, k3, k4): # estimated local error for x[z_n] taken from the article
    # ki: rkn k terms
    # h: step length

    return h**2 * np.abs(k1 - k2 - k3 + k4)

def rkn4_terms(fun, zn, xn, xprimen, h):
    # fun: x'' = f(z, x, x')
    # zn: z_n point
    # xn: x[z_n] point associated to z_n
    # xprimen: xprime_n[z_n] point associated to z_n
    # h: step length

    k1 = fun(zn, xn, xprimen)
    k2 = fun(zn + (h/2), xn + xprimen*(h/2) + (h**2/8)*k1, xprimen + k1*(h/2))
    k3 = fun(zn + (h/2), xn + xprimen*(h/2) + (h**2/8)*k1, xprimen + k2*(h/2))
    k4 = fun(zn + h, xn + xprimen*h + (h**2/2)*k3, xprimen + k3*h)  

    xn_plus1 = xn + h*xprimen + (h**2/6)*(k1+k2+k3)

    xprimen_plus1 = xprimen + (h/6)*(k1+ 2*k2 + 2*k3 + k4)

    local_err = rkn4_error(h, k1, k2, k3, k4)

    return xn_plus1, xprimen_plus1, local_err


# Runge-Kutta-Nymstrom method for x'' = f(z,x,x'), useful for when there is magnetic field (momentum constant)
# Check https://iopscience.iop.org/article/10.1088/1748-0221/4/04/P04001/pdf

# Runge-Kutta-Nymstrom method with fixed step (adaptative=False) or adaptive step (adaptative=True)
def rkn4(fun, z0, z_end, x0, xprime0, h, adaptative=False, tol=None,
        h_min=1e-6, h_max=np.inf, safety=0.9, fac_min=0.2, fac_max=5.0):
    # fun: x'' = f(z, x, x')
    # z0: z_0 point
    # z_end: z_final point
    # x0: x[z0] point associated to z_n
    # xprime0: xprime[z0] point associated to z_n
    # h: step length if adaptative=False, initial guess of the step length if adaptative=True
    # adaptative: if True, the step length is changed so that the local error is <= tol
    # tol: maximum local error allowed in each step, same units as x (only if adaptative=True)
    # h_min, h_max: limits for the step length (only if adaptative=True)
    # safety: factor < 1 that keeps the next step slightly below the optimal one (only if adaptative=True)
    # fac_min, fac_max: limits for how much h can change in one step (only if adaptative=True)

    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}")

    # ----- Fixed step -----
    if not adaptative:
        n = max(1, int(np.ceil(abs(z_end - z0) / h)))
        vec_z = np.linspace(z0, z_end, n + 1)
        h_real = vec_z[1] - vec_z[0]    # negative when propagating backwards

        vec_x = np.empty((n + 1, *np.shape(x0)))
        vec_xprime = np.empty_like(vec_x)
        vec_x[0], vec_xprime[0] = x0, xprime0

        for k in range(n):
            vec_x[k + 1], vec_xprime[k + 1], _ = rkn4_terms(fun, vec_z[k], vec_x[k], vec_xprime[k], h_real)

        return vec_z, vec_x, vec_xprime

    # ----- Adaptive step -----
    if tol is None or tol <= 0:
        raise ValueError(f"tol must be > 0 when adaptative=True, got {tol}")

    direction = np.sign(z_end - z0)    # -1 when propagating backwards
    h_next = min(h, h_max)
    z, x, xprime = z0, np.asarray(x0, dtype=float), np.asarray(xprime0, dtype=float)
    vec_z, vec_x, vec_xprime = [z], [x], [xprime]

    while direction * (z_end - z) > 0:
        remaining = abs(z_end - z)
        h_try = min(h_next, remaining)    # do not go past z_end

        x_new, xprime_new, local_err = rkn4_terms(fun, z, x, xprime, direction * h_try)
        err = np.max(local_err)    # worst component, in case x is a vector
        if np.isnan(err):    # reject the step and reduce h instead of looping forever
            err = np.inf

        if err <= tol:    # accept the step; if not, it is repeated from the same z with a smaller h
            z = z_end if h_try == remaining else z + direction * h_try
            x, xprime = x_new, xprime_new
            vec_z.append(z)
            vec_x.append(x)
            vec_xprime.append(xprime)
        elif h_try <= h_min:
            raise RuntimeError(f"Tolerance not reached at z={z}: err={err}, h={h_try}")

        # local_err scales as h^4, hence the exponent 1/4
        factor = fac_max if err == 0 else np.clip(safety * (tol / err)**0.25, fac_min, fac_max)
        h_next = np.clip(h_try * factor, h_min, h_max)

    return np.array(vec_z), np.array(vec_x), np.array(vec_xprime)





# ------------------------------------------------------------------------------------------------------------
# ----- Runge-Kutta method for x' = f(z, x), with x' = dx/dz -------------------------------------------------
# ----- Check https://web.mit.edu/10.001/Web/Course_Notes/Differential_Equations_Notes/node5.html ------------
# ------------------------------------------------------------------------------------------------------------

def rk4_terms(fun, zn, xn, h):
    k1 = fun(zn, xn)
    k2 = fun(zn + h/2, xn + (h/2)*k1)
    k3 = fun(zn + h/2, xn + (h/2)*k2)
    k4 = fun(zn + h, xn + h*k3)

    return xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


# One Dormand-Prince step. k1 is passed in because it is reused from the previous step
def rk45_terms(fun, zn, xn, h, k1):
    k2 = fun(zn + h/5, xn + h*(k1/5))
    k3 = fun(zn + 3*h/10, xn + h*(3/40*k1 + 9/40*k2))
    k4 = fun(zn + 4*h/5, xn + h*(44/45*k1 - 56/15*k2 + 32/9*k3))
    k5 = fun(zn + 8*h/9, xn + h*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4))
    k6 = fun(zn + h, xn + h*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5))

    xn_plus1 = xn + h*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
    k7 = fun(zn + h, xn_plus1)

    local_err = np.abs(h*(71/57600*k1 - 71/16695*k3 + 71/1920*k4 - 17253/339200*k5 + 22/525*k6 - k7/40))

    return xn_plus1, local_err, k7


# RK4 with fixed step (adaptative=False) or RK45 with adaptive step (adaptative=True) for x' = f(z, x)
def rk4(fun, z0, z_end, x0, h, adaptative=False, tol=None,
        h_min=1e-6, h_max=np.inf, safety=0.9, fac_min=0.2, fac_max=5.0):
    # h: step length if adaptative=False, initial guess if adaptative=True
    # tol, h_min, h_max, safety, fac_min, fac_max: same meaning as in rkn

    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}")

    # ----- Fixed step: RK4 -----
    if not adaptative:
        n = max(1, int(np.ceil(abs(z_end - z0) / h)))
        vec_z = np.linspace(z0, z_end, n + 1)
        h_real = vec_z[1] - vec_z[0]    # negative when propagating backwards

        vec_x = np.empty((n + 1, *np.shape(x0)))
        vec_x[0] = x0

        for k in range(n):
            vec_x[k + 1] = rk4_terms(fun, vec_z[k], vec_x[k], h_real)

        return vec_z, vec_x

    # ----- Adaptive step: RK45 -----
    if tol is None or tol <= 0:
        raise ValueError(f"tol must be > 0 when adaptative=True, got {tol}")

    direction = np.sign(z_end - z0)
    h_next = min(h, h_max)
    z, x = z0, np.asarray(x0, dtype=float)
    vec_z, vec_x = [z], [x]
    k1 = fun(z, x)

    while direction * (z_end - z) > 0:
        remaining = abs(z_end - z)
        h_try = min(h_next, remaining)

        x_new, local_err, k7 = rk45_terms(fun, z, x, direction * h_try, k1)
        err = np.max(local_err)
        if np.isnan(err):    # reject the step and reduce h instead of looping forever
            err = np.inf

        if err <= tol:
            z = z_end if h_try == remaining else z + direction * h_try
            x, k1 = x_new, k7    # k7 of this step is k1 of the next one
            vec_z.append(z)
            vec_x.append(x)
        elif h_try <= h_min:
            raise RuntimeError(f"Tolerance not reached at z={z}: err={err}, h={h_try}")

        # local_err scales as h^5, hence the exponent 1/5
        factor = fac_max if err == 0 else np.clip(safety * (tol / err)**0.2, fac_min, fac_max)
        h_next = np.clip(h_try * factor, h_min, h_max)

    return np.array(vec_z), np.array(vec_x)
