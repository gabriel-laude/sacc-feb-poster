import math
import numpy as np
from matplotlib import pyplot as plt

class CreaghWhelan(object):
        def __init__(self, k=0.1, n=4, c=2, assym=False, rotate=False, gamma=0.75):
                self.k = k
                self.c = c
                self.mass = 1 # * 0.1 
                self.hbar = 0.1 # determine the barrier height before adjusting. should not make this too big!
                self.n = n
                self.assym = assym
                self.rotate=rotate
                self.gamma=gamma
                if assym: self.a,self.b=1./3, 1./3 # Allow user to change this?
                else: self.a,self.b=0,0
        def potential(self, x):
                x, y = x[...,0], x[...,1]
                if self.rotate:
                    return (x**2 - 1)**self.n *(self.a*x**2+self.b*x+1) + self.c*x**2*y**2 + self.k*(y + self.gamma*(x**2 - 1))**2

                else:
                    return (x**2 - 1)**self.n *(self.a*x**2+self.b*x+1) + self.c*x**2*y**2 + self.k*y**2
                

        def gradient(self, x):
                g = np.empty_like(x)
                x, y = x[...,0], x[...,1]
                #g[...,0] = 2*self.n*x*(x**2 - 1)**(self.n-1) + 2*self.c*x*y**2 + 2*self.k*(y+self.gamma*(x**2 - 1))*(2*self.gamma*x)
                #g[...,1] = 2*self.c*x**2*y + 2*self.k*y # old

                g[..., 0] = (self.a * x ** 2 + self.b * x + 1) * 2 * self.n * x * (x ** 2 - 1) ** (self.n - 1) + (
                                        2 * self.a * x + self.b) * (x ** 2 - 1) ** self.n + 2 * self.c * x * y ** 2 + 2*self.k*(y+self.gamma*(x**2 - 1))*(2*self.gamma*x) # asymmetric
                g[...,1] = 2*self.c*x**2*y + 2*self.k*(y+self.gamma*(x**2 - 1))
                return g
        def force(self, x):
                return - self.gradient(x)
        def hessian(self, x):
                H = np.empty((2,2))
                x, y = x[...,0], x[...,1]
                #H[0,0] = 2*self.n*(x**2-1)**(self.n-1) + 4*self.n*(self.n-1)*x**2*(x**2-1)**(self.n-2) + 2*self.c*y**2
                H[0, 0] = (2 * self.n * (x ** 2 - 1) ** (self.n - 1) + 4 * self.n * (self.n - 1) * x ** 2 * (x ** 2 - 1) ** (
                                        self.n - 2)) * (self.a * x ** 2 + self.b * x + 1) + 2 * self.c * y ** 2 + 2 * self.a * (
                                                          x ** 2 - 1) ** self.n + 2 * x * (2 * self.a * x + self.b) * (self.n - 1) * (x ** 2 - 1) ** (
                                                          self.n - 1) + (2*self.a*x + self.b)*(2*x*self.n)*(x**2-1)**(self.n-1) + 4*self.k*self.gamma*(y+self.gamma*(3*x**2-1))
                H[0,1] = H[1,0] = 4*self.c*x*y + 4*self.k*self.gamma*x
                H[1,1] = 2*self.c*x**2 + 2*self.k
                
                if 0: # not rotated
                    H[0, 0] = (2 * self.n * (x ** 2 - 1) ** (self.n - 1) + 4 * self.n * (self.n - 1) * x ** 2 * (x ** 2 - 1) ** (
                                        self.n - 2)) * (self.a * x ** 2 + self.b * x + 1) + 2 * self.c * y ** 2 + 2 * self.a * (
                                                          x ** 2 - 1) ** self.n + 2 * x * (2 * self.a * x + self.b) * (self.n - 1) * (x ** 2 - 1) ** (
                                                          self.n - 1) + (2*self.a*x + self.b)*(2*x*self.n)*(x**2-1)**(self.n-1) #+ 4*self.k*self.gamma*(3*x**2-1)
                    H[0,1] = H[1,0] = 4*self.c*x*y
                    H[1,1] = 2*self.c*x**2 + 2*self.k
                
                return H

        def plot(self, trajectory=None, npts=100, show=False):
                x=np.linspace(-1.6, 1.6, npts)
                y=np.linspace(-1.6, 1.6, npts)
                V=np.empty((npts,npts))
                for i in range(npts):
                    for j in range(npts):
                        V[i,j]=self.potential(np.array([x[i], y[j]]))

                # contour plot generation
                ax=plt.gca()
                
                X, Y=np.meshgrid(x, y)
                CS=ax.contour(X, Y, V.T, np.linspace(0,2,10))
                ax.set_xbound(-2,2)
                ax.set_ybound(-2.0,2.0)
                
                if trajectory is not None: # plot trajectories
                    xt=np.array(trajectory)
                    xt,yt=np.split(xt, 2, -1)
                    #print(x.shape)
                    #print('updated....again...')
                    ax.plot(xt, yt,'o-', ms=8)

                if show: 
                    plt.show()

####################################
class AsymDW(object):
    
        def __init__(self, V0, x0, a, b):
            self.V0 = V0                        # barrier height
            self.x0 = x0                        # minima
            self.hbar = 1
            self.mass = 1
            self.a = a                          # for imaginary part
            self.b = b
            self.c = 1
            self.Vc = V0
            #self.max = opt.fminbound(lambda x: - self.V0*((x/self.x0)**2-1)**2 * (a*x**2+b*x+c), -self.x0, self.x0)
            #self.V0corr = self.V0**2/self.max  # correction factor to get actual barrier height

        def potential(self, x):
            return self.Vc * (x**2/self.x0**2 - 1)**2 * (self.a*x**2+self.b*x+self.c)

        def gradient(self, x):
            return self.Vc * 4*x/self.x0**2*(x**2/self.x0**2-1) * (self.a*x**2+self.b*x+self.c) + self.Vc*(x**2/self.x0**2-1)**2 * (2*self.a*x+self.b)

        def force(self, x):
            return -self.gradient(x)

        def hessian(self, x):
            return 4*self.Vc*(3*x**2/self.x0**4-1/self.x0**2) * (self.a*x**2+self.b*x+self.c) + 8*self.Vc*x/self.x0**2*(x**2/self.x0**2-1)*(2*self.a*x+self.b) + self.Vc*2*self.a*(x**2/self.x0**2 - 1)**2

        def plot(self, xmin = -5, xmax = 5, trajectory = None, npoints = 100, label = None, marker = 'o-', color='k'):
            x = np.linspace(xmin, xmax, 100)
            V = [self.potential(xi) for xi in x]
            if trajectory is not None:
                if label is None: label = 'Trajectory'
                plt.plot(trajectory[0], trajectory[1], marker, label = label, color=color)
            plt.plot(x, V, marker, label=label, color=color)
            plt.xlabel('$x$')
            plt.ylabel('$V$')
            plt.legend()#; plt.show()


###############################
# test potentials
if __name__ == "__main__": 
    pes=CreaghWhelan(n=2, k=0.5, c=0.3, gamma=0.5, rotate=True)
    pes.plot(show=True)
