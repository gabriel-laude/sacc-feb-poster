import numpy as np
from scipy import linalg
import quasinewton, path_integral


def optimiser(xguess, pes, beta, N, gtol=1e-8):
    f = len(xguess[0])
    if f == 2:
        oldN = len(xguess)
        if N != oldN:
            from modules import interpolate
            xguess = interpolate(xguess, f, N).flatten()
        traj=path_integral.adiabatic(pes, 1, f=2)
        x, value = quasinewton.lbfgs(traj.both_trial, xguess.ravel(), gtol=gtol, tau=beta*pes.hbar, maxiter=1000)

        return x.reshape((len(xguess), f))
