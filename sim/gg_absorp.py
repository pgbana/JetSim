from numpy import newaxis, trapz, sqrt, where
from numexpr import evaluate
import sim.const as c

def a_gg_monodirect(cosPhi, eps0, eps1, nph):
    C_gg_abs = (c.pi*c.r_e*c.r_e)/4.
    e0 = eps0[:,newaxis]
    e1 = eps1
    s = 0.5 * e0 * e1 * (1 - cosPhi)
    bet = where(c.mcc*c.mcc < s, sqrt(1. - c.mcc*c.mcc  / s), 1e-200)
    sig_s = "(1.-bet*bet)*((3.-bet**4)*log((1.+bet)/(1.-bet)) + 2.*bet*(bet*bet-2))"
    return trapz( C_gg_abs * nph[:,:, newaxis] * (1. - cosPhi) * evaluate(sig_s), eps0, axis=1)

def a_gg_monodirect_flat(cosPhi, eps0, eps1, nph):
    e0 = eps0
    e1 = eps1
    s = 2. * e0 * e1 * (1 - cosPhi)
    bet = where(4.*c.mcc*c.mcc < s, sqrt(1. - 4.*c.mcc*c.mcc  / s), 1e-200)
    sig_s = "(1.-bet*bet)*((3.-bet**4)*log((1.+bet)/(1.-bet)) + 2.*bet*(bet*bet-2))"
    return trapz( (c.pi*c.r_e*c.r_e)/4. * nph * (1. - cosPhi) * evaluate(sig_s), eps0, axis=0)

def cu_a_gg_monodirect(cosPhi, eps0, eps1, nph):

    e0 = eps0[:,newaxis]
    e1 = eps1
    s = 2. * e0 * e1 * (1.- cosPhi)
    bet = where(4.*c.mcc*c.mcc < s, sqrt(1. - 4.*c.mcc*c.mcc  / s), 1e-200)
    sig_s = "(1.-bet*bet)*((3.-bet**4)*log((1.+bet)/(1.-bet)) + 2.*bet*(bet*bet-2))"
    return trapz( (c.pi*c.r_e*c.r_e)/4. * nph[:, newaxis] * (1. - cosPhi) * evaluate(sig_s), eps0, axis=1)




def a_gg(cosPhi, eps0, eps1, nph):
    miu = cosPhi[:,newaxis]
    e0 = eps0[:,newaxis,newaxis]
    e1 = eps1
    e_th = 2. * c.mcc ** 2 / ((1. - miu) * e1)
    bet = where(e0 > e_th, sqrt(1. - e_th / e0), 0.)
    sig_s = "(1.-bet*bet)*((3.-bet**4)*log((1.+bet)/(1.-bet)) + 2.*bet*(bet*bet-2))"
    return trapz(trapz(3. / 32. * c.sig_T * nph[:, newaxis,newaxis] * (1. - miu) * evaluate(sig_s),
                       eps0,
                       axis=2),
                 cosPhi,
                 axis=1)