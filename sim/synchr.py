from math import sqrt as msqrt
from numexpr import evaluate
from numpy import newaxis, diff

import sim.const as c


def Psyn(eps, gam, B):
    def G_fun(y):
        return evaluate(
            "(1.808*y**(1./3.))/(sqrt(1.+3.4*y**(2./3.)))*(1.+2.21*y**(2./3.)+0.347*y**(4./3.))/(1.+1.353*y**(2./3.)+0.217*y**(4./3.)) * exp(-y)")

    const_Psyn1 = (msqrt(3.) * c.q ** 3) / (c.mcc * c.h)  # CGS
    const_Psyn2 = (3. * c.q * c.h * c.c) / (4. * c.pi * c.mcc)  # CGS
    return const_Psyn1 * B[:, newaxis, newaxis] * \
           G_fun(eps / (const_Psyn2 * B[:, newaxis, newaxis] * gam[:, newaxis] ** 2))

def Psyn2(eps, gam, B):
    def G_fun(y):
        return evaluate(
            "(1.808*y**(1./3.))/(sqrt(1.+3.4*y**(2./3.)))*(1.+2.21*y**(2./3.)+0.347*y**(4./3.))/(1.+1.353*y**(2./3.)+0.217*y**(4./3.)) * exp(-y)")

    const_Psyn1 = (msqrt(3.) * c.q ** 3) / (c.mcc * c.h)  # CGS
    const_Psyn2 = (3. * c.q * c.h * c.c) / (4. * c.pi * c.mcc)  # CGS
    return const_Psyn1 * B * G_fun(eps / (const_Psyn2 * B * gam ** 2))

def j_approx(eps, gam, B):
    return Psyn(eps, gam, B) /(4. * c.pi)

def loss_rate(gam, B):
    const_synLR = c.sig_T * c.c / (c.mcc * 6. * c.pi)
    return const_synLR * B[:, newaxis] ** 2 * (gam - 1.) ** 2
#
def SSA_approx(eps, gam, Dgam, B):
    const_SSA = c.c ** 2 * c.h ** 3 / (8. * c.pi * c.mcc)  # CGS
    derivative = (diff(Psyn(eps, gam, B) * gam[:, newaxis] ** 2, axis=0) / Dgam[:, newaxis])
    return derivative * const_SSA / (eps**2) / (gam[:, newaxis] ** 2)
