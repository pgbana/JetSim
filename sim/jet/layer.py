from sim.hdf5 import or_data, ow_data
import sim.synchr as syn
from math import sqrt as msqrt
from numpy import power, newaxis, ones, trapz
import sim.const as c
from sim.jet.evolve import gen0, gen_next

def create(sim_name, lyr_name='lyr0',
         B0=1., ro=0.2,
         eta=1e-6, Q0=1e32, si=2.,
         Gj=10., gam_min=1.,
         q=1, b=1., R0=10.):

    z_obs, Dz_obs, gam05, epsL, gam, Dgam, cosPsi, R_Sch =\
        or_data(sim_name+'/axis', 'z_obs', 'Dz_obs', 'gam05', 'epsL', 'gam', 'Dgam', 'cosPsi', 'R_Sch')

    bj = msqrt(Gj ** 2 - 1.) / Gj

    z_prim, t_prim, Dt_prim = z_obs / Gj, z_obs / (bj * Gj * c.c), Dz_obs / (bj * Gj * c.c)

    lyr = {}

    lyr['lyr_name'] = lyr_name
    lyr['bj'] =bj
    lyr['B0'] =B0
    lyr['ro'] =ro
    lyr['eta'] =eta
    lyr['Q0'] =Q0
    lyr['si'] =si
    lyr['Gj'] =Gj
    lyr['gam_min'] =gam_min
    lyr['q'] =q
    lyr['b'] =b
    lyr['R0'] =R0

    lyr['z_prim'] = z_prim
    lyr['t_prim'] = t_prim
    lyr['Dt_prim'] = Dt_prim
    lyr['z_end_prim'] =z_obs[-1] * 0.1 / Gj

    R_z = (R0 * R_Sch) + ro * z_prim
    B_z = B0 * power((R0 * R_Sch) / R_z, b)
    S_z = c.pi * R_z ** 2
    t_acc = gam05 * c.mcc / (c.q * B_z[:, newaxis] * eta * c.c)
    cosPsi_prim = (cosPsi - bj) / (1. - cosPsi * bj)
    lyr['R_z'] =R_z
    lyr['B_z'] =B_z
    lyr['S_z'] =S_z
    lyr['t_acc'] =t_acc
    lyr['cosPsi_prim'] =cosPsi_prim

    adb_LR = ones((z_prim.size, gam05.size))
    adb_LR[1:] = 2. / 3. * (gam05 - 1.) * (ro * bj * c.c) / (R0 * R_Sch + ro * bj * c.c * t_prim[:-1][:, newaxis])
    adb_LR[0] = 1e-250
    syn_LR = syn.loss_rate(gam05, B_z)
    exIC_LR = ones((z_prim.size, gam05.size)) * 1e-200

    lyr['exIC_LR'] =exIC_LR
    lyr['adb_LR'] =adb_LR
    lyr['syn_LR'] =syn_LR

    ow_data(sim_name + '/' + lyr_name, mode='w', **lyr)

    prep = {'j_syn': syn.j_approx(epsL, gam, B_z),
            'a_SSA': syn.SSA_approx(epsL, gam, Dgam, B_z)}
    ow_data(sim_name + '/tmp/syn_prep_' + lyr_name, mode='w', **prep)

def new_gen(sim_name, lyr_name_ig, ex_rad=False, old_rad=False):
    epsL, g05, z_obs, = or_data(sim_name+'/axis', 'epsL', 'gam05', 'z_obs')
    rad = {}
    if old_rad==False:
        rad['IC_LR'] = ones((z_obs.size, g05.size))*1e-200
    else:
        IC_loss_rate = or_data(sim_name+'/tmp/ic_prep', 'IC_loss_rate')
        old_n_to, old_n_ho = or_data(sim_name+'/'+old_rad, 'n_syn_to', 'n_syn_ho')
        rad['IC_LR'] =  trapz(IC_loss_rate * old_n_to[:, newaxis], epsL, axis=2) + \
                        trapz(IC_loss_rate * old_n_ho[:, newaxis], epsL, axis=2)
    ow_data(sim_name+'/'+lyr_name_ig, mode='w', **rad)
    if ex_rad==False:
        ex = {'exIC_LR': ones((z_obs.size, g05.size)) * 1e-200}
        ex['EX_RAD'] = False
        ow_data(sim_name + '/' + lyr_name_ig, mode='r+', **ex)
    else:
        ex_rad['EX_RAD'] = True
        ow_data(sim_name + '/' + lyr_name_ig, mode='r+', **ex_rad)

def evolv(sim_name, lyr_name, iter_num=0, gen_num=3, ex_rad=False):
    lyr_name_i = lyr_name + '_' + str(iter_num)
    new_gen(sim_name, lyr_name_i+str(0), ex_rad=ex_rad)
    gen0(sim_name, lyr_name, iter_num)
    for id_gen in range(1, gen_num+1):
        new_gen(sim_name, lyr_name_i+str(id_gen), ex_rad=ex_rad, old_rad=lyr_name_i+str(id_gen-1))
        gen_next(sim_name, lyr_name, iter_num, gen_num=id_gen)