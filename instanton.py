import numpy as np
from scipy import linalg
import quasinewton, path_integral
from modules import information
from matplotlib import pyplot as plt

def optimiser(xguess, pes, beta, N, gtol=1e-6):
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
    path_min_l = np.full((N, f), -pes.x0)
    path_min_r = np.full((N, f), pes.x0)
    Vl, dVldx, d2Vldx2 = information(path_min_l, pes, f=f)
    Vr, dVrdx, d2Vrdx2 = information(path_min_r, pes, f=f)

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
    B=linalg.inv(A)
    evals_A = linalg.eigvalsh(A)

    # build A0
    Yl=Yr=Y
    Yl[(N-1)*f:, (N-1)*f:] = Y[:f, :f]
    Yr[:f, :f] = Y[(N-1)*f:, (N-1)*f:]
    A0l=traj.d2Sdx2(path_min_l, tau[-1]) + Yl
    A0r=traj.d2Sdx2(path_min_r, tau[-1]) + Yr

    # determine ground state splitting
    d=0 # for now
    S=traj.S(x_opt, tau[-1])
    _, logdet_Al = np.linalg.slogdet(A0l)
    _, logdet_Ar = np.linalg.slogdet(A0r)
    logdet_A = np.sum(np.log(evals_A[1:]))
    Phi = np.sqrt(beta*pes.hbar/N * np.exp(logdet_A - 0.5*(logdet_Al + logdet_Ar)))
    theta0=np.sqrt(S / (2*np.pi*pes.hbar)) / Phi * np.exp(-S / pes.hbar)
    delta0=2*pes.hbar*theta0
    if 0:
        print('determinants: ', linalg.det(A0l), linalg.det(A0r), np.prod(evals_A[1:]))
        print("Phi is: ", Phi) 
        print('theta0, delta0: ', theta0, 2*pes.hbar*theta0)

    # linear instanton, so we make it far far simpler
    xi=x_opt[0] - pes.x0
    xf=x_opt[-1] + pes.x0
    ratio_parallel=2*np.sqrt(alpha_left[1] * alpha_right[1]) * np.exp(beta*pes.hbar*omega_l[1]) * xi[0] * xf[0]
    ratio_perp=2*np.sqrt(alpha_left[0] * alpha_right[0]) * np.exp(beta*pes.hbar*omega_l[0]) * B[1,-1] * pes.hbar
    
    return theta0*pes.hbar, pes.hbar*theta0*ratio_perp, pes.hbar*theta0*np.abs(ratio_parallel), 0, 0, 0



def plot_instanton(N, beta, d=0): # for now d does nothing
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
    
    pes.plot(trajectory=x_opt, show=False, npts=20)
    theta0, theta10, theta01, theta11, theta20, theta02 = splitting(x_opt, xmin, beta, pes)
    d00=d10=d01=d11=d20=d02=0
    
    # generate splitting results and put into plot
    data = [['(0,0)', "{:.2e}".format(theta0), "{:.2e}".format(2*np.sqrt(theta0**2 + d00**2)), 'tbd'],
            ['(1,0)', "{:.2e}".format(theta10), "{:.2e}".format(2*np.sqrt(theta10**2 + d10**2)), 'tbd'],
            ['(0,1)', "{:.2e}".format(theta01), "{:.2e}".format(2*np.sqrt(theta01**2 + d01**2)), 'tbd'],
            ['(1,1)', "{:.2e}".format(theta11), "{:.2e}".format(2*np.sqrt(theta11**2 + d11**2)), 'tbd'],
            ['(2,0)', "{:.2e}".format(theta20), "{:.2e}".format(2*np.sqrt(theta20**2 + d20**2)), 'tbd']]
            #['(0,2)', "{:.2e}".format(theta02), "{:.2e}".format(2*np.sqrt(theta02**2 + d02**2)), 'tbd']]
    columns=[r'$n_1,n_2$', r'$\hbar\theta_{n_1, n_2}^\mathrm{inst}$', r'$\Delta_{n_1, n_2}^\mathrm{inst}$', r'$\Delta^\mathrm{DVR}_{n_1, n_2}$' ]
    if d00 == 0:
        data=np.array(data)
        data[:,-1] = ["2.47e-8", "3.45e-8", "4.06e-6", "5.05e-6", "3.22e-7"] 
    
    table=plt.table(colLabels=columns, cellText=data,
                    loc='right', bbox=[1.1, 0.0, 0.9, 1])



if 0:
    from potentials import CreaghWhelan
    pes=CreaghWhelan(n=2, k=0.5, c=0.3, gamma=0.0, rotate=True)
    f=2
    beta=60
    N=256
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
