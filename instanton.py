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



if 1:
    from potentials import CreaghWhelan
    pes=CreaghWhelan(n=2, k=0.5, c=0.3, gamma=0.5, rotate=True)
    f=2
    beta=60
    N=64
    xmin=np.array([-1,0])
    x_comp = np.linspace(xmin[0], -xmin[0], N)
    y_comp = np.linspace(xmin[1], xmin[1], N)
    xguess=np.zeros((N,f))
    xguess[:,0]=x_comp
    xguess[:,1]=y_comp
    pes.x0=xmin
    x_opt=optimiser(xguess, pes, beta, N)
    pes.plot(trajectory=x_opt)
    print("the file is working....")
