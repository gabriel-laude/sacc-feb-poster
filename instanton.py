import numpy as np
from scipy import linalg
import quasinewton, path_integral
from modules import information
from matplotlib import pyplot as plt

def optimiser(xguess, pes, beta, N, gtol=1e-8, f=2):
    oldN = len(xguess)
    if N != oldN:
        from modules import interpolate
        xguess = interpolate(xguess, f, int(N)).flatten()
    traj=path_integral.adiabatic(pes, 1, f=f)
    x, value = quasinewton.lbfgs(traj.both_trial, xguess.ravel(), gtol=gtol, tau=beta*pes.hbar, maxiter=1000, verbosity=0)

    if f==2:
        return x.reshape((len(xguess), f))
    else:
        return x


def momentum_integrand(x, pes):
    p=np.sqrt(2*pes.mass*pes.potential(x))
    f=1/p*pes.gradient(x)

    return f



def splitting(x_opt, xmin, beta, pes, vib_ind=[0,0], f=2):
    """
    vib_ind is [n_1, n_2].
    """
    # instanton
    N=len(x_opt)
    tau=np.linspace(0, beta*pes.hbar, int(N))
    V, dVdx, d2Vdx2 = information(x_opt, pes, f=f)
    traj=path_integral.adiabatic(pes, 1, f=f)
    print('S/ħ: ', traj.S(x_opt, tau[-1]) / pes.hbar)
    
    # minima 
    if f == 2:
        path_min_l = np.full((N, f), -pes.x0)
        path_min_r = np.full((N, f), pes.x0)
    
    if f == 1:
        path_min_l = np.linspace(-pes.x0, -pes.x0, N)
        path_min_r = np.linspace(pes.x0, pes.x0, N)
    
    Vl, dVldx, d2Vldx2 = information(path_min_l, pes, f=f)
    Vr, dVrdx, d2Vrdx2 = information(path_min_r, pes, f=f)

    # Build Y
    Y=np.zeros((N*f, N*f))

    if f == 1:
        omega_l, omega_r = np.sqrt(pes.hessian(pes.x0)), np.sqrt(pes.hessian(-pes.x0))
        Y[0,0]=omega_l
        Y[-1,-1]=omega_r
    if f == 2:
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
    if f==2:
        Yl[(N-1)*f:, (N-1)*f:] = Y[:f, :f]
        Yr[:f, :f] = Y[(N-1)*f:, (N-1)*f:]
    if f==1:
        Yl[-1,-1]=Y[0,0]
        Yr[0,0]=Y[-1,-1]
    
    A0l=traj.d2Sdx2(path_min_l, tau[-1]) + Yl
    A0r=traj.d2Sdx2(path_min_r, tau[-1]) + Yr

    # determine ground state splitting
    S=traj.S(x_opt, tau[-1])
    _, logdet_Al = np.linalg.slogdet(A0l)
    _, logdet_Ar = np.linalg.slogdet(A0r)
    logdet_A = np.sum(np.log(evals_A[1:]))
    Phi = np.sqrt(beta*pes.hbar/N * np.exp(logdet_A - 0.5*(logdet_Al + logdet_Ar)))
    theta0=np.sqrt(S / (2*np.pi*pes.hbar)) / Phi * np.exp(-S / pes.hbar)
    
    if 0:
        delta0=2*pes.hbar*theta0
        print('determinants: ', linalg.det(A0l), linalg.det(A0r), np.prod(evals_A[1:]))
        print('logdets: ', logdet_Al, logdet_Ar, logdet_A)
        print('exp logdets: ', np.exp(logdet_A - 0.5*(logdet_Al + logdet_Ar)))
        print('tau_N: ', beta/N*pes.hbar)
        print('beta: ', beta)
        print("Phi is: ", Phi) 
        print('theta0, delta0: ', theta0, 2*pes.hbar*theta0)

    # linear instanton; thus appropriate measures taken
    integral=True
    if f==2 or (f==1 and not integral):
        xi=x_opt[0] - pes.x0
        xf=x_opt[-1] + pes.x0
        #print('xi: ', xi)
        #print('xf: ', xf)
    if integral and f!=2: # momentum is integral
        from scipy import integrate
        ind=np.argmax(V)
        integrand_l=np.array([momentum_integrand(xi, pes) for xi in x_opt[:ind]])
        integrand_r=np.array([momentum_integrand(xi, pes) for xi in x_opt[ind:]])
        integral_l=integrate.simps(integrand_l, tau[:ind])
        integral_r=integrate.simps(integrand_r, tau[ind:])
        xi=np.exp(integral_l)*np.sqrt(2*pes.potential(x_opt[ind])) / omega_l
        xf=np.exp(-integral_r)*np.sqrt(2*pes.potential(x_opt[ind])) / omega_r
        #print('xi, xf: ', xi, xf)
    
    Bl=linalg.inv(A0l)
    Br=linalg.inv(A0r)
    if 0:
        ratio_parallel=2*np.sqrt(alpha_left[1] * alpha_right[1]) * np.exp(beta*pes.hbar*omega_l[1]) * xi[0] * xf[0]
        ratio_perp=2*np.sqrt(alpha_left[0] * alpha_right[0]) * np.exp(beta*pes.hbar*omega_l[0]) * B[1,-1] * pes.hbar
        ratio20=4*alpha_left[0]*alpha_right[0]*np.exp(2 * beta*pes.hbar*omega_l[0])*pes.hbar**2*B[1,-1]**2

    if f == 1:
        #ratio1=xi * xf / (np.sqrt(Bl[0,-2]*Br[0,-2])*pes.hbar)
        ratio1=np.abs(2*np.sqrt(alpha_left*alpha_right)*np.exp(beta*pes.hbar*(omega_l+omega_r)/2)*xi*xf)
        ratio2=ratio1**2/2
        #print('ground state, ratio1, ratio2: ', theta0*2, ratio1, ratio2)
        return theta0, theta0*ratio1, theta0*ratio2

    if 1 and f==2:
        #ratio_parallel= xi[0] * xf[0] / (np.sqrt(Bl[0,-2]*Br[0,-2])*pes.hbar)
        ratio_parallel=2*np.sqrt(alpha_left[1] * alpha_right[1]) * np.exp(beta*pes.hbar*(omega_l[1]+omega_r[1])/2) * xi[0] * xf[0]
        ratio_perp=B[1,-1] / np.sqrt(Bl[1,-1] * Br[1,-1])
        ratio11=B[1,-1]*xi[0]*xf[0] / (pes.hbar*Bl[0,-2]*Bl[1,-1]) #+ 2*Bl[0,-1]**2)
        ratio20=B[1,-1]**2 / np.sqrt(Bl[1,-1]**2*Br[1,-1]**2)
        return theta0*pes.hbar, pes.hbar*theta0*ratio_perp, pes.hbar*theta0*np.abs(ratio_parallel), np.abs(ratio11)*theta0*pes.hbar, np.abs(ratio20)*theta0*pes.hbar, 0



def plot_instanton(N, beta, b=0): # for now d does nothing
    from potentials import CreaghWhelan
    #pes=CreaghWhelan(n=2, k=0.5, c=0.3, gamma=0., rotate=True)
    pes=CreaghWhelan(n=2, k=0.2, c=2, b=b, rotate=False, gamma=0)
    f=2
    N=int(N)
    alpha=np.sqrt(1.0)
    xmin=np.array([-np.sqrt(alpha),0])
    x_comp = np.linspace(xmin[0], -xmin[0], N)
    y_comp = np.linspace(xmin[1], xmin[1], N)
    xguess=np.zeros((N,f))
    xguess[:,0]=x_comp
    xguess[:,1]=y_comp
    pes.x0=xmin
    #print('check potential at minima: ', pes.potential(pes.x0))
    x_opt=optimiser(xguess, pes, beta, N, gtol=1e-8)
    
    pes.plot(trajectory=x_opt, show=False, npts=20)
    theta0, theta10, theta01, theta11, theta20, theta02 = splitting(x_opt, xmin, beta, pes)
    omega_l=np.sqrt(linalg.eigvalsh(pes.hessian(pes.x0)))
    omega_r=np.sqrt(linalg.eigvalsh(pes.hessian(-pes.x0)))
    #print('hessian at minima: ', pes.hessian(pes.x0))
    #print('try eigvals again: ', np.sqrt(linalg.eigvalsh(pes.hessian(pes.x0))))
    print('frequencies (left, right): ', omega_l, omega_r)
    #print('barrier height: ', pes.potential(np.array([0,0])))
    
    # all energies
    E0l=0.5*pes.hbar*(np.sum(omega_l))
    E0r=0.5*pes.hbar*(np.sum(omega_r))
    E10l=(3/2*omega_l[0] + 1/2*(omega_l[1]))*pes.hbar
    E10r=(3/2*omega_r[0] + 1/2*(omega_r[1]))*pes.hbar
    E01l=(3/2*omega_l[1] + 1/2*(omega_l[0]))*pes.hbar
    E01r=(3/2*omega_r[1] + 1/2*(omega_r[0]))*pes.hbar
    E11l=(3/2*pes.hbar*(np.sum(omega_l)))
    E11r=(3/2*pes.hbar*(np.sum(omega_r)))
    E20l=(5/2*omega_l[0] + 1/2*(omega_l[1]))*pes.hbar
    E20r=(5/2*omega_r[0] + 1/2*(omega_r[1]))*pes.hbar
    

    d00=0.5*np.abs(E0l-E0r)
    d10=0.5*np.abs(E10l-E10r)
    d01=0.5*np.abs(E01l-E01r)
    d11=0.5*np.abs(E11l-E11r) 
    d20=0.5*np.abs(E20l-E20r) 
    #print(d00, d10, d01, d11, d20)
    
    #print('harmonic energies, E0l, E01, E11: ', E0l, E01l, E11l)
    #print('harmonic energies, E0l, E10, E20: ', E0l, E10l, E20l)
    #print('more harmonic: ', 1.5*omega_l[0], 1.5*omega_r[0], omega_l[0]-omega_r[0])
    
    #d00=d10=d01=d11=d20=d02=0
    
    # generate splitting results and put into plot
    data = [['(0,0)', "{:.2e}".format(d00), "{:.2e}".format(theta0), "{:.2e}".format(2*np.sqrt(theta0**2 + d00**2)), 'tbd'],
            ['(1,0)',"{:.2e}".format(d10),  "{:.2e}".format(theta10), "{:.2e}".format(2*np.sqrt(theta10**2 + d10**2)), 'tbd'],
            ['(0,1)', "{:.2e}".format(d01), "{:.2e}".format(theta01), "{:.2e}".format(2*np.sqrt(theta01**2 + d01**2)), 'tbd'],
            ['(1,1)', "{:.2e}".format(d11), "{:.2e}".format(theta11), "{:.2e}".format(2*np.sqrt(theta11**2 + d11**2)), 'tbd'],
            ['(2,0)',"{:.2e}".format(d20),  "{:.2e}".format(theta20), "{:.2e}".format(2*np.sqrt(theta20**2 + d20**2)), 'tbd']]
            #['(0,2)', "{:.2e}".format(theta02), "{:.2e}".format(2*np.sqrt(theta02**2 + d02**2)), 'tbd']]
    columns=[r'$n_1,n_2$', r'$d_{n_1,n_2}$', r'$\hbar\theta_{n_1, n_2}^\mathrm{inst}$', r'$\Delta_{n_1, n_2}^\mathrm{inst}$', r'$\Delta^\mathrm{DVR}_{n_1, n_2}$' ]
    data=np.array(data)
    print('b is: ', (b))
    if b == 1e-9:
        data[:,-1] = ["4.58e-8", "2.28e-7", "7.82e-6", "3.55e-5", "1.21e-6"] 
    if b == 1e-7:
        data[:,-1] = ["4.77e-8", "2.28e-7", "7.82e-6", "3.55e-5", "1.21e-6"] 
    if b>= 9.5e-6 and b <= 1.05e-5:
        data[:,-1] = ["1.33e-6", "1.38e-6", "8.59e-6", "3.56e-5", "1.95e-6"] 
    if b == 1e-3:
        data[:,-1] = ["1.33e-4", "1.36e-4", "3.56e-4", "3.43e-4", "1.52e-4"] 
    
    table=plt.table(colLabels=columns, cellText=data,
                    loc='right', bbox=[1.1, 0.0, 0.9, 1])


def plot_1d(N=32, beta=30, b=0, V0=2): # for now d does nothing
    from potentials import AsymDW
    x0=5*np.sqrt(V0)
    a_pass=b/x0**2
    b_pass=b/x0
    
    pes=AsymDW(V0, x0, a_pass, b_pass)
    
    if 1:
        tau=np.linspace(0, beta*pes.hbar, int(N))
        omega0=np.sqrt(pes.hessian(pes.x0))
        xguess=np.array([-pes.x0*np.tanh(omega0/2*(tau[i]-tau[int(N)//2])) for i in range(int(N))])
    
    if N >= 32:
        maxlog = int(np.log2(N))
        logvals=range(4, maxlog)
        allx=[]
        #Nvals=[2**i for i in logvals]
        for i, val in enumerate(logvals):
            N=2**val
            if i == 0:
                x_opt=optimiser(xguess, pes, beta, N, f=1, gtol=1e-8)
            else:
                x_opt=optimiser(allx[i-1].flatten(), pes, beta, N, f=1, gtol=1e-8)
            allx.append(x_opt)
    else:
        x_opt=optimiser(xguess, pes, beta, N, f=1, gtol=1e-8)
    
    
    if 1:
        plt.plot(x_opt, pes.potential(x_opt), 'bo-')
        plt.yticks(np.arange(0, V0+0.5, step=0.5))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$V(x)$')
    
    
    if 0:
        import nbinteract as nbi
        nbi.scatter(x_opt, pes.potential(x_opt))
    theta0, theta1, theta2 = splitting(x_opt, x0, beta, pes, f=1)
    omega_l, omega_r = np.sqrt(pes.hessian(-pes.x0)), np.sqrt(pes.hessian(pes.x0)) 
    d0=1./4 * np.abs(omega_l - omega_r)
    d1=3./4 * np.abs(omega_l - omega_r)
    d2=5./4 * np.abs(omega_l - omega_r)
    print('ω_l, ω_r: ', omega_l, omega_r)
    #print('d_0, d_1, d_2: ', d0, d1, d2)

    # generate splitting results and put into plot
    data = [['0', "{:.2e}".format(d0), "{:.2e}".format(theta0), "{:.2e}".format(2*np.sqrt(theta0**2 + d0**2)), 'tbd'],
            ['1', "{:.2e}".format(d1), "{:.2e}".format(theta1), "{:.2e}".format(2*np.sqrt(theta1**2 + d1**2)), 'tbd'],
            ['2', "{:.2e}".format(d2), "{:.2e}".format(theta2), "{:.2e}".format(2*np.sqrt(theta2**2 + d2**2)), 'tbd']]
    columns=[r'$n$', r'$d_n$', r'$\hbar\theta_{n}^\mathrm{inst}$', r'$\Delta_{n}^\mathrm{inst}$', r'$\Delta^\mathrm{DVR}_{n}$' ]
    if d0 == 0:
        data=np.array(data)
        #data[:,-1] = ["4.58e-8", "22.8e-7", "7.82e-6", "3.55e-5", "1.21e-6"] 
    
    table=plt.table(colLabels=columns, cellText=data,
                    loc='right', bbox=[1.1, 0.0, 0.9, 1])


###############################################################################
# tests go here
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


if 0:
    from potentials import AsymDW
    V0=2
    x0=5*np.sqrt(V0)
    a=1e-7/x0**2
    b=1e-7/x0
    a=b=0
    pes=AsymDW(V0, x0, a, b)
    N=128
    beta=30
    
    if 0: 
        xguess=np.linspace(-pes.x0, pes.x0, N)
    if 1:
        tau=np.linspace(0, beta*pes.hbar, N)
        omega0=np.sqrt(pes.hessian(pes.x0))
        xguess=np.array([-pes.x0*np.tanh(omega0/2*(tau[i]-tau[N//2])) for i in range(N)])
    
    x_opt=optimiser(xguess, pes, beta, N, f=1, gtol=1e-7)

    #print(x_opt)
    #pes.plot(trajectory=x_opt)
    plt.show()
    splitting(x_opt, x0, beta, pes, f=1)
