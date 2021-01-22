"""open path integrals in imaginary time
not cyclic so just 1/2 potentials on ends
pathway defined by N+1 points

this is the main use of MPI in this project
run a test using 
$ mpiexec -n 4 python path_integral.py

"""
from scipy import linalg
import numpy as np
import util

class ZeroPES(object):
        def potential(x): return 0
        def gradient(x): return np.zeros_like(x)
        def hessian(x): return np.zeros((x.size,x.size))

class Beads(object):
        """Base class for path-integral and ring-polymer methods
        with optional MPI for single bead calculations.
        Could be extended to memory functionality to increase efficiency.
        """
        def __init__(self, PES, mpi=False):
                """If PES is None, return zeros"""
                self.PES = ZeroPES() if PES is None else PES
                self.mpi = mpi
                if self.mpi:
                        from mpi4py import MPI
                        self.size = MPI.COMM_WORLD.Get_size()
                        self.rank = MPI.COMM_WORLD.Get_rank()
                        self.name = MPI.Get_processor_name()
                        print("Hello from process %i of %i on %s in path_integral.Beads.__init__" % (self.rank, self.size, self.name))

        def compute(self, func, x, shape):
                """Return array([func(xi) for xi in x])
                May use MPI to do this computation
                """
                if self.mpi:
                        from mpi4py import MPI
                        f = np.zeros(shape)
                        index = util.share(len(x), self.size)
                        for i in range(index[self.rank],index[self.rank+1]):
                                f[i] = func(x[i])
                        f = MPI.COMM_WORLD.allreduce(f)
                else:
                        f = np.array([func(xi) for xi in x])
                return f

        def potentials(self, x):
                return self.compute(self.PES.potential, x, len(x))
        def gradients(self, x):
                return self.compute(self.PES.gradient, x, x.shape)
        def hessians(self, x):
                return self.compute(self.PES.hessian, x, (len(x),)+2*(x.size//len(x),))

class adiabatic(Beads):
        def __init__(self, PES, mass, f=1):
                Beads.__init__(self, PES)
                self.mass = mass
                self.PES=PES 
                self.omega0=np.sqrt(PES.hessian(PES.x0))
                self.f = f

        def beads(self, x, t):
                N = len(x) - 1
                dt = float(t) / N
                return N, dt

        def f_trial(self, x, t):
                f = self.f 
                if self.f == 1:
                    #omega0=np.sqrt(self.PES.hessian(x[0]))
                    #print omega0
                    omega0 = self.omega0
                    return self.S(x, t) + 0.5*omega0*((x[0] - self.PES.x0)**2 + (x[-1] + self.PES.x0)**2) # correction for asymmetry required! 
        
                if self.f > 1:
                    x = x.reshape((len(x)/f,f))
                    S = self.S(x, t)
                    xmin=np.array([-1,0]) # of course not always true!
                    
                    # build X
                    evals_l=evals_r=linalg.eigvalsh(self.PES.hessian(xmin))
                    evals_l, U_l=linalg.eigh(self.PES.hessian(xmin))
                    evals_r, U_r=linalg.eigh(self.PES.hessian(-xmin))
                    X_l=np.zeros((2,2))
                    omega0=np.sqrt(evals_l)
                    X_l[0,0]=omega0[0]
                    X_l[1,1]=omega0[1]
                    X_r=X_l

                    # extra factors generate now!
                    x_i=x[0] - xmin
                    x_f=x[-1] + xmin
                    f_l = np.linalg.multi_dot([x_i.T, U_l, X_l, U_l.T, x_i])
                    f_r = np.linalg.multi_dot([x_f.T, U_r, X_r, U_r.T, x_f])

                    return S + 0.5*(f_l + f_r)
                    


        def df_trial(self, x, t):
                f = self.f # pure laziness right here
                x = x.reshape((len(x)/f,f))
                res=self.dSdx(x, t)
                xmin=np.array([-1,0])
                N=len(x)
                if f == 1:
                    res[0] += self.omega0*(x[0] - self.PES.x0)
                    res[-1] += self.omega0*(x[-1] + self.PES.x0)
                
                else:
                    # build X
                    evals_l, U_l=linalg.eigh(self.PES.hessian(xmin))
                    evals_r, U_r=linalg.eigh(self.PES.hessian(-xmin))
                    X_l=np.zeros((2,2))
                    omega0=np.sqrt(evals_l)
                    X_l[0,0]=omega0[0]
                    X_l[1,1]=omega0[1]
                    X_r=X_l
                    
                    # additions to gradient
                    res=res.ravel()
                    #x=res.ravel()
                    #print res.shape, res[:2] , res[:2] + np.array([0,1])
                    #print np.linalg.multi_dot([U_l, X_l, U_l.T, x[:f]])
                    x_i=x[0] - xmin
                    x_f=x[-1] + xmin
                    res[:f] += np.linalg.multi_dot([U_l, X_l, U_l.T, x_i])
                    res[(N-1)*f:] += np.linalg.multi_dot([U_r, X_r, U_r.T, x_f])
                    
                    #print res[(N-1)*f:]

                return res.ravel()
        
        def both_trial(self, x, t):
                return self.f_trial(x, t), self.df_trial(x, t)
        def T(self, x, dt):
                """Return kinetic energy"""
                return np.sum(self.mass*(x[1:] - x[:-1])**2)/(2*dt)

        def dTdx(self, x, dt):
                res = np.empty_like(x)
                res[0] = self.mass * (x[0] - x[1]) / dt
                res[1:-1] = self.mass * (2*x[1:-1] - x[2:] - x[:-2]) / dt
                res[-1] = self.mass * (x[-1] - x[-2]) / dt
                return res

        def S(self, x, t):
                V = self.potentials(x) # shape is (N)
                N, dt = self.beads(x, t)
                return self.T(x, dt) + (V[0]/2 + np.sum(V[1:-1]) + V[-1]/2)*dt

        def dSdx(self, x, t):
                dVdx = self.gradients(x) # shape is (N) or (N,f)
                N, dt = self.beads(x, t)
                res = self.dTdx(x, dt)
                res[0] += dVdx[0]*dt/2
                res[1:-1] += dVdx[1:-1]*dt
                res[-1] += dVdx[-1]*dt/2
                return res

        def d2Sdx2(self, x, t):
                d2Vdx2 = self.hessians(x) # shape is (N) or (N,f,f)
                if d2Vdx2.ndim == 1:
                        f = 1
                elif d2Vdx2.ndim == 3:
                        f = d2Vdx2.shape[2]
                else:
                        assert False
                N, dt = self.beads(x, t)
                res = np.zeros(((N+1)*f,(N+1)*f))
                for i in range(N):
                        res[i*f:(i+1)*f,(i+1)*f:(i+2)*f] = - self.mass * np.identity(f) / dt
                        res[(i+1)*f:(i+2)*f,i*f:(i+1)*f] = - self.mass * np.identity(f) / dt
                res[0:f,0:f] = self.mass * np.identity(f) / dt + d2Vdx2[0]*dt/2
                for i in range(1,N):
                        res[i*f:(i+1)*f,i*f:(i+1)*f] = 2 * self.mass * np.identity(f) / dt + d2Vdx2[i]*dt
                res[-f:,-f:] = self.mass * np.identity(f) / dt + d2Vdx2[-1]*dt/2
                return res

        def dSdt(self, x, t):
                V = self.potentials(x) # shape is (N)
                N, dt = self.beads(x, t)
                return - self.T(x, dt) / t + (V[0]/2 + np.sum(V[1:-1]) + V[-1]/2)/N

        def d2Sdt2(self, x, t):
                N, dt = self.beads(x, t)
                return 2 * self.T(x, dt) / t**2

        def d2Sdxdt(self, x, t):
                dVdx = self.gradients(x) # shape is (N) or (N,f)
                N, dt = self.beads(x, t)
                res = - self.dTdx(x, dt) / t
                res[0] += dVdx[0]/(2*N)
                res[1:-1] += dVdx[1:-1]/N
                res[-1] += dVdx[-1]/(2*N)
                return res


############################
