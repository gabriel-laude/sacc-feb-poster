import math
import numpy
from scipy import linalg


##########################

def lbfgs(func, x0, M=3, maxiter=100, gtol=1e-5, dguess=1.0, maxstep=0.3, maxerise=1.0, path=False, verbosity=0, tau=None):
        """Optimize x until |G|<gtol using func(x) which returns potential and gradient.
        Start with dguess along diagonal of Hessian and don't use linesearch.
        Store M copies of previous steps and gradients, where 3<=M<=7 is recommended.
        If path is True, xt is returned, otherwise only the final position is returned."""
        xk = numpy.asarray(x0, numpy.float).copy()
        xt = [xk.copy()]
        wss = numpy.empty((M,)+xk.shape) # work array stores last M search steps (in circular order controlled by parameter point).
        wgd = numpy.empty((M,)+xk.shape) # work array stores last M gradient differences (in circular order controlled by parameter point).
        rho = numpy.empty(M)
        alpha = numpy.empty(M) # used in the formula that computes H*G

        # initialize
        if tau is not None: f, g = func(xk, tau)
        else: f, g = func(xk) 
        point = 0 # ranges from 0 to M-1
        if verbosity >= 1: print("iter %i, f=%.8g, |grad|=%.1e" % (0, f, numpy.linalg.norm(g)))

        diag = numpy.ones_like(xk) * dguess
        wss[0] = -g*diag
        wg = -g*diag
        normg = numpy.linalg.norm(g)

        # main iteration loop
        for iter in range(1,maxiter+1):
                bound = iter - 1
                if iter == 1:
                        step = min(1.0/normg, normg)
                else:
                        if iter > M: bound = M
                        ys = numpy.dot(wgd[npt],wss[npt])
                        yy = numpy.dot(wgd[npt],wgd[npt])
                        if yy==0.0: yy=1.0
                        diag[:] = ys/yy

                        # compute -H*g
                        cp = point
                        if point==0: cp=M
                        rho[cp-1] = 1.0/ys
                        wg = - g
                        cp = point
                        for i in range(bound):
                                cp -= 1
                                if cp == -1: cp=M-1
                                sq = numpy.dot(wss[cp],wg)
                                alpha[cp] = rho[cp]*sq
                                wg += -alpha[cp]*wgd[cp]
                        wg *= diag
                        for i in range(bound):
                                yr = numpy.dot(wgd[cp],wg)
                                beta = rho[cp]*yr
                                beta = alpha[cp] - beta
                                wg += beta * wss[cp]
                                cp += 1
                                if cp == M: cp = 0
                        step = 1.0

                # store the new search direction
                wss[point] = wg.copy()

                # instead of line search
                normg = numpy.linalg.norm(g)
                normwg = numpy.linalg.norm(wg)
                overlap = 0.0
                if normg*normwg > 0.0: overlap = numpy.dot(g,wg) / (normg*normwg)
                if overlap > 0.0: wss[point] = - wg.copy()
                wg = g.copy()
                slength = numpy.linalg.norm(wss[point])
                if step*slength > maxstep: step = maxstep/slength

                # cycle to determine the step
                ndecrease=0
                while True:
                        xnew = xk + step * wss[point]
                        if tau is not None: fnew, gnew = func(xnew, tau)
                        else:fnew, gnew = func(xnew)
                        if verbosity >= 1: print("iter %i, f=%.8g, |grad|=%.1e, step=%.1e" % (iter, fnew, numpy.linalg.norm(gnew), step*slength))
#                       if fnew==0.0: fnew=1.0e-100 # to prevent divide by zero
                        if fnew-f <= maxerise*abs(fnew):
                                xk=xnew.copy()
                                f=fnew
                                g=gnew.copy()
                                break
                        else:
                                if ndecrease > 5:
                                        print("WARNING: lbfgs failed after 5 decreases")
                                        if path: return xt, f
                                        else: return xk, f
                                else:
                                        ndecrease += 1
                                        step /= 10.0
                xt.append(xk.copy())

                # compute the new step and gradient change
                npt = point
                wss[npt] *= step
                wgd[npt] = g - wg
                point += 1
                if point == M: point = 0

                # termination test
                normg = numpy.linalg.norm(g)
                if normg < gtol:
                        #if verbosity >= 0: print("lbfgs converged after %i iterations" % iter)
                        if path: return xt, f
                        else: return xk, f

        print("WARNING: lbfgs failed to converge")
        if path: return xt, f
        else: return xk, f

##########################

