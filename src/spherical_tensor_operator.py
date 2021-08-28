import py3nj
import numpy as np

def selection(l1, l2, l3, m1, m2, m3):
    return (np.abs(l1 - l2) <= l3) and (l3 <= l1 + l2) and np.isclose(m1+m2+m3, 0)

def wigner3j(l1, l2, l3, m1, m2, m3):
    to_int = lambda x: int(round(2*x))
    if selection(to_int(l1), to_int(l2), to_int(l3), to_int(m1), to_int(m2), to_int(m3)):
        result = py3nj.wigner3j(to_int(l1), to_int(l2), to_int(l3), to_int(m1), to_int(m2), to_int(m3))
    else:
        result = 0
    return result

def Xelem_theta_0(M, l1, m1, l2, m2):
    return (-1)**int(m1) * np.sqrt((2*l1+1)*(2*l2+1)) * \
           wigner3j(l1, 1, l2, 0, 0, 0) * wigner3j(l1, 1, l2, -m1, M, m2)

def X_theta_0(M, lmax):
    dim = int(lmax+1)**2
    Xmat = np.zeros((dim, dim))

    for ix in range(dim):
        for iy in range(dim):
            l1 = int(np.sqrt(ix))
            m1 = ix - l1**2 - l1
            l2 = int(np.sqrt(iy))
            m2 = iy - l2**2 - l2

            Xmat[ix, iy] = Xelem_theta_0(M, l1, m1, l2, m2)

    return Xmat

def L2_theta_0(lmax):
    dim = int(lmax+1)**2
    L2mat = np.zeros(dim)

    for ix in range(dim):
        l = int(np.sqrt(ix))
        L2mat[ix] = l * (l+1)

    return np.diag(L2mat)

def Xelem_theta_pi(M, l1, m1, l2, m2):
    return (-1)**int(l1+l2+1) * (-1)**int(m1+0.5) * \
           wigner3j(l1, 1, l2, -m1, M, m2) * wigner3j(l1, 1, l2, -0.5, 0, 0.5)

def X_theta_pi(M, lmax):
    dim = int((lmax+0.5)*(lmax+1.5))
    Xmat = np.zeros((dim, dim))

    for ix in range(dim):
        for iy in np.arange(dim):
            l1 = int(np.sqrt(ix+0.25)-0.5)+0.5
            m1 = ix - (l1-0.5)*(l1+0.5) - l1
            l2 = int(np.sqrt(iy+0.25)-0.5)+0.5
            m2 = iy - (l2-0.5)*(l2+0.5) - l2

            Xmat[ix, iy] = Xelem_theta_pi(M, l1, m1, l2, m2)

    return Xmat

def L2_theta_pi(lmax):
    dim = int((lmax+0.5)*(lmax+1.5))
    L2mat = np.zeros(dim)

    for ix in range(dim):
        l = int(np.sqrt(ix+0.25)-0.5)+0.5
        L2mat[ix] = l * (l+1)

    return np.diag(L2mat)


