import numpy as np
from scipy import linalg
import quasinewton, path_integral
from modules import information
from matplotlib import pyplot as plt

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

def splitting(x_opt, xmin, beta, pes, vib_ind=[0,0]):
    """
    vib_ind is [n_1, n_2].
    """
    N=len(x_opt)
    tau=np.linspace(0, beta*pes.hbar, int(N))
    f=2
    V, dVdx, d2Vdx2 = information(x_opt, pes, f=f)
    traj=path_integral.adiabatic(pes, 1, f=2)
    print('action: ', traj.S(x_opt, tau[-1]))

    # Build Y
    Y=np.zeros((N*f, N*f))
    evals_l, Ul = linalg.eigh(pes.hessian(x_opt[0]))
    evals_r, Ur = linalg.eigh(pes.hessian(x_opt[-1]))
    omega_l, omega_r = np.sqrt(evals_l), np.sqrt(evals_r)
    Xl=Xr=np.zeros((f,f))
    Xl[0,0], Xl[1,1] = omega_l[0], omega_l[1]
    Xr[0,0], Xr[1,1] = omega_r[0], omega_r[1]
    Y[:f,:f]=np.linalg.multi_dot([Ul, Xl, Ul.T])
    Y[(N-1)*f:,(N-1)*f:]=np.linalg.multi_dot([Ur, Xr, Ur.T])

    # alpha factors
    alpha_left = omega_l / pes.hbar
    alpha_right = omega_r / pes.hbar

    # build A
    A = traj.d2Sdx2(x_opt, tau[-1]) + Y
    

    return 0



def plot_instanton(N, beta, d=0): # for now d does nothing
    beta=60
    from potentials import CreaghWhelan
    pes=CreaghWhelan(n=2, k=0.5, c=0.3, gamma=0., rotate=True)
    f=2
    N=int(N)
    xmin=np.array([-1,0])
    x_comp = np.linspace(xmin[0], -xmin[0], N)
    y_comp = np.linspace(xmin[1], xmin[1], N)
    xguess=np.zeros((N,f))
    xguess[:,0]=x_comp
    xguess[:,1]=y_comp
    pes.x0=xmin
    x_opt=optimiser(xguess, pes, beta, N)
    
    
    pes.plot(trajectory=x_opt, show=False)
    # generate splitting results and put into plot
    data = [['(0,0)', '(1,0)', '(0,1)', '(1,1)', '(2,0)', '(0,2)'],
            ['1','2','3', '4'],
            [1,2,3]]
    
    columns=[r'$n_1,n_2$', r'$\theta_n^\mathrm{inst}$', r'$\Delta_n^\mathrm{inst}$', r'$\Delta_\mathrm{DVR}$' ]
    
    #rows=data[0][:2]
    #cellText=[['1', '2', '3'], ['1', '2', '3']]
    rows=data[0]
    cellText=[data[1] for i in range(len(rows))]
    
    print(len(cellText))
    table=plt.table(colLabels=columns, cellText=cellText,
                    loc='right', bbox=[1.1, 0.0, 0.9, 1])
    #plt.subplots_adjust(bottom=0.2)
    #pes.plot(trajectory=x_opt, show=False)
    




if 0:
    from potentials import CreaghWhelan
    pes=CreaghWhelan(n=2, k=0.5, c=0.3, gamma=0.0, rotate=True)
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
    splitting(x_opt, xmin, beta, pes)
