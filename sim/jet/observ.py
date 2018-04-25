from math import cos as mcos
from numpy import trapz, newaxis, ones, sqrt, exp, interp, loadtxt, ones_like, transpose, array, cos
from sim.helper import z_to_dL, nearest_id, nearest_array_id, progress
from sim.hdf5 import ow_data, or_data
import sim.inv_compt as ic
import sim.const as c
from scipy.interpolate import RegularGridInterpolator

def create(sim_name, obs_name, redshift, ThetaObs):
    obs = {}
    obs['redshift'] = redshift
    obs['ThetaObs'] = ThetaObs
    obs['cosThetaObs'] = mcos(ThetaObs)
    obs['dL'] = z_to_dL(redshift)
    ow_data(sim_name+'/'+obs_name, mode='w', **obs)

def EBL_abs(eps, redshf=0.1):
    z_axis = loadtxt('sim/tau_ax.csv', delimiter=',')
    Egam_ax = loadtxt('sim/Egam_ax.csv', delimiter=',') * c.mev_to_erg
    tau_grid = loadtxt('sim/opdep_fiducial.csv')
    tau_EBL_f = RegularGridInterpolator((Egam_ax, z_axis), tau_grid, bounds_error=False, fill_value=0)
    redshift = ones_like(eps) * redshf
    tau_EBL = tau_EBL_f(transpose(array([eps, redshift])))
    return exp(-tau_EBL)


def add_lyr(sim_name, lyr_name, lyr_name_ig, obs_name):

    def j_ex_rad(sim_name, lyr_name_ig):

        j_ex = ones((z_obs.size, epsH.size)) * 1e-200
        nex_zz, cosT_zz = or_data(sim_name + '/' + lyr_name_ig, 'nph_z_zex', 'cosT_z_zex')
        n_dot = ones((z_prim.size, epsH.size)) * 1e-200

        for id_z1 in range(z_obs.size):

            progress(id_z1, z_obs.size, lyr_name + ': ')

            int1 = trapz(trapz(n_dot_ic_prep * n_elec[id_z1, :, newaxis], gam, axis=2) * nex_zz[id_z1, :, newaxis, newaxis], epsL, axis=3)

            for id_z2 in range(z_prim.size):
                cosPsi_scat = sqrt(1 - cosT_zz[id_z1, id_z2] ** 2) * sqrt(1 - cosTheta_to ** 2) * cos(phi) +\
                                  cosT_zz[id_z1, id_z2] * cosTheta_to
                id_cosPsi = nearest_array_id(cosPsi, cosPsi_scat)
                n_dot_approx = int1[id_z2][id_cosPsi]
                n_dot[id_z2] = trapz(n_dot_approx, phi, axis=0)
            j_ex[id_z1] = trapz(epsH * n_dot, z_prim, axis=0)
        return j_ex

    eps, epsL, epsH, gam, z_obs, cosPsi, phi = or_data(sim_name+'/axis', 'eps', 'epsL', 'epsH', 'gam', 'z_obs', 'cosPsi', 'Phi')
    Gj, bj, B_z, R_z, S_z, z_prim = or_data(sim_name+'/'+lyr_name, 'Gj', 'bj', 'B_z', 'R_z', 'S_z', 'z_prim')
    j_syn_prep = or_data(sim_name+'/tmp/syn_prep_'+lyr_name, 'j_syn')
    n_dot_ic_prep = or_data(sim_name+'/tmp/ic_prep', 'n_dot_IC')
    n_elec, n_syn_ho, n_syn_to, a_SSA, EX_RAD = or_data(sim_name+'/'+lyr_name_ig, 'n_elec', 'n_syn_ho', 'n_syn_to', 'a_SSA', 'EX_RAD')

    obslyr = {}

    cosThetaObs, dL = or_data(sim_name+'/'+obs_name, 'cosThetaObs', 'dL')
    Dj = 1. / (Gj * (1. - bj * cosThetaObs)); obslyr['Dj'] = Dj
    cosTheta_ho = (cosThetaObs - bj) / (1. - cosThetaObs * bj); obslyr['cosTheta_ho'] = cosTheta_ho
    cosTheta_to = -(cosThetaObs - bj) / (1. - cosThetaObs * bj); obslyr['cosTheta_to'] = cosTheta_to
    id_cT_ho = nearest_id(cosPsi, cosTheta_ho)
    id_cT_to = nearest_id(cosPsi, cosTheta_to)

    obslyr['eps_obs'] = eps*Dj
    obslyr['epsL_obs'] = epsL * Dj
    obslyr['epsH_obs'] = epsH * Dj

    a_SSA_obs = a_SSA/Dj; obslyr['a_SSA_obs'] = a_SSA_obs
    j_syn_obs = Dj ** 2 * trapz(j_syn_prep * n_elec[:,:, newaxis], gam, axis=1); obslyr['j_syn_obs'] = j_syn_obs
    j_ic_ho_obs=Dj**2*ic.j(n_dot_ic_prep[id_cT_ho], epsL, gam, epsH, n_syn_ho, n_elec); obslyr['j_ic_ho_obs']=j_ic_ho_obs
    j_ic_to_obs=Dj**2*ic.j(n_dot_ic_prep[id_cT_to], epsL, gam, epsH, n_syn_to, n_elec); obslyr['j_ic_to_obs']=j_ic_to_obs
    j_ex_obs = ones((B_z.size, epsH.size)) * 1e-200

    if EX_RAD == True:
        j_ex_obs = Dj ** 2 * j_ex_rad(sim_name, lyr_name_ig)
    obslyr['j_ex_obs'] = j_ex_obs


    tau_SSA = -(a_SSA_obs * R_z[:,newaxis]); obslyr['tau_SSA'] = tau_SSA
    Fsyn_z_obs = j_syn_obs * exp(-tau_SSA) * S_z[:, newaxis] / dL ** 2; obslyr['Fsyn_z_obs'] = Fsyn_z_obs
    Fic_to_z_obs = j_ic_to_obs * S_z[:, newaxis] / dL ** 2; obslyr['Fic_to_z_obs'] = Fic_to_z_obs
    Fic_ho_z_obs = j_ic_ho_obs * S_z[:, newaxis] / dL ** 2; obslyr['Fic_ho_z_obs'] = Fic_ho_z_obs
    Fex_z_obs = j_ex_obs * S_z[:, newaxis] / dL ** 2; obslyr['Fex_z_obs'] = Fex_z_obs

    Ftot_z_obs = ones((S_z.size, eps.size)) * 1e-200
    Ftot_z_obs[:,:epsL.size] += Fsyn_z_obs[:]
    Ftot_z_obs[:,-epsH.size:] += (Fic_to_z_obs + Fic_ho_z_obs + Fex_z_obs); obslyr['Ftot_z_obs'] = Ftot_z_obs

    Fsyn = trapz(Fsyn_z_obs, z_obs, axis=0); obslyr['Fsyn'] = Fsyn
    Fic_to = trapz(Fic_to_z_obs, z_obs, axis=0); obslyr['Fic_to'] = Fic_to
    Fic_ho = trapz(Fic_ho_z_obs, z_obs, axis=0); obslyr['Fic_ho'] = Fic_ho
    Fex = trapz(Fex_z_obs, z_obs, axis=0); obslyr['Fex'] = Fex
    Ftot = trapz(Ftot_z_obs, z_obs, axis=0); obslyr['Ftot'] = Ftot

    obslyr['Lsyn'] = Fsyn * c.pi4 * dL ** 2 * epsL * Dj
    obslyr['Lic_to'] = Fic_to * c.pi4 * dL ** 2 * epsH * Dj
    obslyr['Lic_ho'] = Fic_ho * c.pi4 * dL ** 2 * epsH * Dj
    obslyr['Lex'] = Fex * c.pi4 * dL ** 2 * epsH * Dj
    obslyr['Ltot'] = Ftot * c.pi4 * dL ** 2 * eps * Dj

    ow_data(sim_name+'/'+obs_name, prefix=lyr_name+'/', **obslyr)

def finish(sim_name, obs_name, lyr_names):

    redshift = or_data(sim_name + '/' + obs_name, 'redshift')
    eps, epsL, epsH = or_data(sim_name + '/axis', 'eps', 'epsL', 'epsH')
    eFe = ones(eps.size) * 1e-200
    Ltot = ones(eps.size) * 1e-200
    Lsyn = ones(eps.size) * 1e-200
    Lic_ho = ones(eps.size) * 1e-200
    Lic_to = ones(eps.size) * 1e-200
    Lex = ones(eps.size) * 1e-200

    for lyr_name in lyr_names:
        eps_lyr, Ftot_lyr, Ltot_lyr= or_data(sim_name +'/'+obs_name, lyr_name+'/eps_obs', lyr_name+'/Ftot', lyr_name+'/Ltot')
        epsL_lyr, Lsyn_lyr =  or_data(sim_name +'/'+obs_name, lyr_name+'/epsL_obs', lyr_name+'/Lsyn')
        epsH_lyr, Lex_lyr, Lic_ho_lyr, Lic_to_lyr = or_data(sim_name + '/' + obs_name, lyr_name + '/epsH_obs',
                                                            lyr_name + '/Lex', lyr_name + '/Lic_ho', lyr_name + '/Lic_to')
        eFe += interp(eps, eps_lyr, Ftot_lyr * eps_lyr, left=1e-200, right=1e-200)
        Ltot += interp(eps, eps_lyr, Ltot_lyr, left=1e-200, right=1e-200)
        Lsyn += interp(eps, epsL_lyr, Lsyn_lyr, left=1e-200, right=1e-200)
        Lic_ho += interp(eps, epsH_lyr, Lic_ho_lyr, left=1e-200, right=1e-200)
        Lic_to += interp(eps, epsH_lyr, Lic_to_lyr, left=1e-200, right=1e-200)
        Lex += interp(eps, epsH_lyr, Lex_lyr, left=1e-200, right=1e-200)
    eFe_ebl = eFe * EBL_abs(eps, redshf=redshift)

    obs = {}
    obs['eps'] = eps
    obs['eFe'] = eFe
    obs['eFe_ebl'] = eFe_ebl
    obs['Ltot'] = Ltot
    obs['Lsyn'] = Lsyn
    obs['Lex'] = Lex
    obs['Lic_ho'] = Lic_ho
    obs['Lic_to'] = Lic_to

    ow_data(sim_name+'/'+obs_name, **obs)