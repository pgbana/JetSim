from numexpr import evaluate
from numpy import newaxis, trapz, linspace, where
import sim.const as c


def loss_rate_approx(gam, eps):
    const_ICloss = 12. * c.c * c.sig_T / c.mcc
    GAM = 4. * gam[:, newaxis] * eps / c.mcc
    q = linspace(1e-6, 1., 1000)[:, newaxis, newaxis]
    eq_IntQ = '(q /((1+GAM *q)*(1+GAM *q)*(1+GAM *q))) * ((2. * q * log(q) + (1. + 2. * q) * (1. - q)) + 0.5 * (GAM * GAM * q * q * (1. - q)) / (1. + GAM * q))'
    f_KN = trapz(evaluate(eq_IntQ), q, axis=0)
    return const_ICloss * (gam[:, newaxis] - 1.) ** 2 * eps * f_KN

def n_dot_approx(eps0, eps1, gam, cosTheta):
    const_ICdirect = 3. * c.c * c.sig_T / (16.* c.pi* c.mcc)
    cT = cosTheta[:, newaxis, newaxis, newaxis]
    e1 = eps1[:, newaxis, newaxis] / c.mcc
    g = gam[:, newaxis]
    e0 = eps0 / c.mcc
    w = e1 / g
    bTheta = 2. * (1. + cT) * e0 * g
    f_direct = '1 + w*w/(2.*(1.-w)) - 2.*w/(bTheta*(1.-w)) + 2.*w*w/(bTheta*bTheta*(1.-w)*(1.-w))'
    kin_cond = (e0 >= e1 / (2. * (1. + cT) * g * g * (1. - (e1 / g)))) & (e1 < g) & (e0 < e1)
    return where(kin_cond, const_ICdirect * evaluate(f_direct) / (e0 * g * g), 0.)

def n_dot_iso_approx(eps0, eps1, gam):
    const_ICiso = 3. * c.c * c.sig_T / 4.
    e0 = eps0
    e1 = eps1[:,newaxis, newaxis]
    g = gam[:,newaxis]
    GAM = 4. * e0 * g / c.mcc
    q = e1/(GAM * (g * c.mcc - e1))
    kin_cond = (q < 1.) & (q > 1. / (4. * g ** 2)) & (e0 < e1)
    eq_Blum = "(2.*q*log(q)+(1.+2.*q)*(1.-q)+0.5*(GAM*GAM *q*q*(1.-q))/(1.+GAM*q))"
    return where(kin_cond, const_ICiso / (g ** 2 * e0) * evaluate(eq_Blum), 1e-250)

def n_dot_headon(eps0, eps1, gam):
    const_ICdirect = 3. * c.c * c.sig_T / (4. * c.mcc)
    cT = -1.
    e1 = eps1[:, newaxis, newaxis] / c.mcc
    g = gam[:, newaxis]
    e0 = eps0 / c.mcc
    w = e1 / g
    bTheta = 2. * (1. - cT) * e0 * g
    f_direct = '1 + w*w/(2.*(1.-w)) - 2.*w/(bTheta*(1.-w)) + 2.*w*w/(bTheta*bTheta*(1.-w)*(1.-w))'
    cond1 = (e1 <= g*bTheta/(1.+bTheta))
    cond2 = (e0 < e1)
    cond3 = (e1 < g)
    cond = cond1 & cond2 & cond3
    return where(cond, const_ICdirect * evaluate(f_direct) / (e0 * g * g), 0.)

def j(n_dot_approx_cT, eps0, gam, eps1, nph, ne): # n_dot_IC_approx_cT -- przyblizenie rozpraszania Comptona dla danego kata cT
    int1 = trapz(n_dot_approx_cT * ne[:,newaxis,:,newaxis], gam, axis=2)
    int2 = trapz(int1 * nph[:, newaxis], eps0, axis=2)
    return eps1 * int2