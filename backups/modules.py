import numpy as np

def information(x, pes, f, pot_only=False, grad_only=False, hess_only=False):
    """
    Quickly generate V, G, H data.
    """
    if f == 1:
        pass
    else:
        N = len(x) - 1
        x = x.reshape(N + 1, f)
    V = np.array([pes.potential(xi) for xi in x])
    dVdx = np.array([pes.gradient(xi) for xi in x])
    d2Vdx2 = np.array([pes.hessian(xi) for xi in x])

    if pot_only:
        return V
    elif grad_only:
        return dVdx
    elif hess_only:
        return d2Vdx2
    else:
        return V, dVdx, d2Vdx2

def find_nearest(array, value): # Courtesy of StackExchange user unutbu
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    print('Value of upper bound of integral x(tau): ', array[idx])
    return idx

def dvr_calc(pes, basis=300, n=0, mpmath=False):
    """
    Calculate tunnelling splitting from DVR (1D).

    n=0,1,2....
    """
    from polylib import DVR
    from mpmath import mp, mpf, matrix
    mp.dps = 30
    omega=np.sqrt(pes.hessian(pes.x0))
    dvr = DVR.Hermite(basis, x0=pes.x0, mass=pes.mass, omega=omega / np.sqrt(pes.mass), hbar=pes.hbar)
    V = np.array(list(map(pes.potential, dvr.x)))
    E = dvr.calculate(V, eigvals_only=True)

    if mpmath: return -mpf(E[2*n])+mpf(E[2*n+1])
    else: return -E[2*n]+E[2*n+1]


def input_open(filein):
    with open(filein, 'r') as infile:
        lines = infile.readlines()
        x = []
        for line in lines:
            xi = line.split()
            if len(xi) == 1: xi = float(xi[0])
            else: xi = np.array([float(i) for i in xi])
            x.append(xi)
        return np.array(x)

def output_gen(x, fileout):
    with open(fileout, 'w') as outfile:
        try: # For now 2D and 1D only. TODO: make multidimensional. should be easy.
            x = ['{0} {1}\n'.format(xi[0], xi[1]) for xi in x]
        except IndexError:
            x = ['{0}\n'.format(xi[0]) for xi in x]
        outfile.writelines(x)

def interpolate(x, f, N):
    '''
    Interpolation for increasing bead numbers
    '''
    from scipy import interpolate
    oldN = len(x)
    coord = np.linspace(0, 1, oldN)
    coord_new = np.linspace(0, 1, N)
    if f != 1: 
        interp = [interpolate.interp1d(coord, x[:,j], kind='cubic') for j in range(f)]
        x = np.array([interp[j](coord_new) for j in range(f)]).T
    else: 
        interp = interpolate.interp1d(coord, x, kind='cubic')
        x = np.array([interp(coord_new)]).T
  
    return x


