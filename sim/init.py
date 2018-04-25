import sim.const as c
from numpy import logspace, diff, linspace
import sim.hdf5 as h5
import sim.inv_compt as ic

from os import makedirs
from os.path import exists, isfile


def load_param(file_name):
    param = {}
    with open(file_name) as ofile:
        for line in ofile:
            (key, value) = line.split('=')
            param[key] = float(value)
    return param


def folder(sim_name):
    if not exists(sim_name):
        makedirs(sim_name)

def axis_file(sim_name, accurancy=5, M_BH=1e9,
         log_gam_min=0, log_gam_max=9, gam_ax_size=9,
         log_eps_min = -20 , log_eps_max = 5, eps_ax_size = 28,
         eps_ic_min = 1e-12, eps_syn_max = 1e-5,
         Phi_size=10, cosPsi_size=7, theta_size=7,
         log_z_min = 0, log_z_max =7, z_size=8,
         u_size = 10):

    axis = {'M_BH': M_BH}
    R_sch = c.R_sch*M_BH
    axis['R_Sch']=R_sch

    gam_extended = logspace(log_gam_min, log_gam_max, gam_ax_size * accurancy * 2 + 1)
    gam05 = gam_extended[0::2]
    gam =  gam_extended[1:-1:2]
    axis['gam'] = gam
    axis['gamH'] = gam[gam >= 100.]
    axis['gam05'] = gam05
    axis['Dgam'] =gam05[1:] - gam05[:-1]

    eps = logspace(log_eps_min, log_eps_max, eps_ax_size * accurancy)
    axis['eps'] =eps
    epsL = eps[eps < eps_syn_max]
    epsH = eps[eps > eps_ic_min]
    epsVH = eps[eps > c.MeV100*0.01]
    axis['epsL'] =epsL
    axis['epsH'] =epsH
    axis['epsVH'] =epsVH

    z_extend = logspace(log_z_min, log_z_max, z_size * accurancy + 1)
    axis['z_Sch'] =z_extend[:-1]
    axis['Dz_Sch'] =diff(z_extend)
    axis['z_obs'] =z_extend[:-1] * R_sch
    axis['Dz_obs'] =diff(z_extend) * R_sch

    Phi = linspace(0, c.pi * 2., Phi_size * accurancy)
    axis['Phi'] =Phi
    cosPsi = linspace(-1.+1e-10, 1.-1e-10, cosPsi_size * accurancy)
    axis['cosPsi'] =cosPsi

    axis['u'] = linspace(0., 1., u_size*accurancy)

    h5.ow_data(file_name=sim_name + '/axis', mode='w', **axis)

def ic_prep(sim_name):
    if not isfile(sim_name+'/axis.hdf5'):
        print('Axis file error!')
    else:
        if not exists(sim_name+'/tmp'):
            makedirs(sim_name+'/tmp')
        epsL, gam, Dgam, gam05, epsH, cosPsi= h5.or_data(sim_name+'/axis', 'epsL', 'gam', 'Dgam', 'gam05', 'epsH', 'cosPsi')

        prep={'n_dot_IC': ic.n_dot_approx(epsL, epsH, gam, cosPsi),
              'IC_loss_rate': ic.loss_rate_approx(gam05, epsL),
              'n_dot_IC_iso': ic.n_dot_iso_approx(epsL, epsH, gam)}
        h5.ow_data(sim_name+'/tmp/ic_prep', mode='w', **prep)