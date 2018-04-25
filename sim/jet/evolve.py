from sim.hdf5 import ow_data, or_data
from numpy import interp, log, power, ones, where, ones_like, diagflat, trapz, newaxis, nonzero, sqrt, exp
from numpy.linalg import solve

import sim.const as c

def save_gen(sim_name, lyr_name, lyr_name_ig, gam_max, Q_inj, n_elec, N_elec, n_ho, n_to, j_syn, N_dot, a_SSA, IClr):

    bj, S_z, z_prim, B_z= or_data(sim_name+'/'+lyr_name, 'bj', 'S_z', 'z_prim', 'B_z')
    gam = or_data(sim_name+'/axis', 'gam')

    lyr_ig = {}
    lyr_ig['gam_max'] =gam_max
    lyr_ig['Q_inj'] =Q_inj
    lyr_ig['n_elec'] =n_elec
    lyr_ig['N_elec'] =N_elec
    lyr_ig['n_syn_ho'] =n_ho
    lyr_ig['n_syn_to'] =n_to
    lyr_ig['j_syn'] =j_syn
    lyr_ig['N_dot'] =N_dot
    lyr_ig['a_SSA'] =a_SSA
    lyr_ig['IClr'] =IClr

    L_inj = trapz(c.mcc * trapz(Q_inj * gam, gam, axis=1), z_prim, axis=0)
    L_e = trapz(n_elec * gam, gam, axis=1) * c.mcc * bj * c.c * S_z
    L_B = S_z * (B_z)**2/ (8. * c.pi) * bj * c.c
    lyr_ig['L_inj'] = L_inj
    lyr_ig['eta_rad'] = 1. - L_e[-1]/L_inj
    lyr_ig['eta_eq'] = L_B/L_e
    ow_data(sim_name+'/'+lyr_name_ig, **lyr_ig)

def gam_max_fun(ax_gam05, tau_acc, total_loss_rate):
    tau_cool = ax_gam05/ (total_loss_rate + 1e-250)
    return interp(1., tau_acc / tau_cool, ax_gam05)

def Q_inj_fun(gam, z_prim, gam_max, r, gam_min, si, z_end_prim, Q0, q):
    if si == 2.:
        Q_t = Q0 / (log(gam_max) - log(gam_min)) #
    else:
        Q_t = Q0 * (2. - si) / (power(gam_max, 2. - si) - power(gam_min, 2. - si))
    return where((gam > gam_min) & (z_prim < z_end_prim),
                  Q_t * power(r, q) * power(gam/gam_min, -si) * exp(-gam / gam_max),
                  1e-250)

def solve_elec_eq(Ne, loss_rates, Q_inj, Dt, Dgam):
    S_i = Ne + Q_inj * Dt
    V2_matrix = diagflat(1. + Dt * loss_rates[:-1] / Dgam)
    V3_matrix = diagflat(-(Dt * loss_rates[1:-1]) / Dgam[:-1], 1)
    return solve(V2_matrix + V3_matrix, S_i)

def rad_transfer(epsL, z, j_syn, a_SSA, R):
    def tau_fun(alpha, z):
        return trapz(alpha, z, axis=0)
    def before(z, id_z, j_syn, a_SSA, R):
        id_z_before = nonzero(z[id_z] >= z)
        Dz = z[id_z] - z[id_z_before]
        Rn = sqrt(Dz**2 + R[id_z_before]**2)
        DOmega = 2. * c.pi * (1. - Dz/Rn)
        tau = tau_fun(a_SSA[id_z_before], Dz)
        return trapz(j_syn[id_z_before] * exp(-tau) * DOmega[:,newaxis], z[id_z_before], axis=0)

    def after(z, id_z, j_syn, a_SSA, R):
        id_z_after = nonzero(z[id_z] <= z)
        Dz = z[id_z_after] - z[id_z]
        Rn = sqrt(Dz**2 + R[id_z_after]**2)
        DOmega = 2. * c.pi * (1. - Dz/Rn)
        tau = tau_fun(a_SSA[id_z_after], Dz)
        return trapz(j_syn[id_z_after]  * exp(tau) * DOmega[:,newaxis], z[id_z_after], axis=0)

    n_headon = ones_like(j_syn) * 1e-250
    n_tailon = ones_like(j_syn) * 1e-250
    for id_z in range(len(z)):
        n_tailon[id_z] = before(z, id_z, j_syn, a_SSA, R) / (c.c * epsL)
        n_headon[id_z] = after(z, id_z, j_syn, a_SSA, R) / (c.c * epsL)

    return n_headon, n_tailon

def gen0(sim_name, lyr_name, iter_num):

    lyr_name_ig = lyr_name + '_' + str(iter_num) + str(0)

    z_prim, Dt_prim, t_acc, R_z, B_z, S_z, q, bj, Q0, si, R0, z_acc_end, syn_LR, adb_LR, gam_min =\
            or_data(sim_name + '/' + lyr_name, 'z_prim', 'Dt_prim', 't_acc', 'R_z', 'B_z', 'S_z', 'q',
                    'bj', 'Q0', 'si', 'R0', 'z_end_prim', 'syn_LR', 'adb_LR', 'gam_min')
    gam, gam05, Dgam, epsL = or_data(sim_name +'/axis', 'gam', 'gam05', 'Dgam', 'epsL')
    j_syn_apprx, a_SSA_apprx= or_data(sim_name+'/tmp/syn_prep_'+lyr_name, 'j_syn', 'a_SSA')
    exIC_LR, IC_LR= or_data(sim_name +'/' + lyr_name_ig, 'exIC_LR', 'IC_LR')
    
    
    z_size = z_prim.size; g_size = gam.size; eL_size = epsL.size; g05_size=gam05.size
    gam_max = ones(z_size)
    Q_inj = ones((z_size, g_size)) * 1e-200
    n_elec = ones((z_size, g_size)) * 1e-200
    N_elec = ones((z_size, g_size)) * 1e-200
    n_ho = ones((z_size, eL_size)) * 1e-200
    n_to = ones((z_size, eL_size)) * 1e-200
    j_syn = ones((z_size, eL_size)) * 1e-200
    N_dot = ones((z_size, eL_size)) * 1e-200
    a_SSA = ones((z_size, eL_size)) * 1e-200

    total_loss_rate = syn_LR + adb_LR + exIC_LR

    for id_z in range(z_size-1):
        gam_max[id_z] = gam_max_fun(gam05, t_acc[id_z], total_loss_rate[id_z])
        Q_inj[id_z] = Q_inj_fun(gam, z_prim[id_z], gam_max[id_z], R0/R_z[id_z], gam_min, si, z_acc_end, Q0, q)
        j_syn[id_z] = trapz(j_syn_apprx[id_z]*n_elec[id_z][:,newaxis], gam, axis=0)
        N_dot[id_z] = j_syn[id_z] * S_z[id_z] / epsL[:] * c.pi4
        N_elec[id_z+1] = solve_elec_eq(N_elec[id_z],
                                       total_loss_rate[id_z],
                                       Q_inj[id_z],
                                       Dt_prim[id_z],
                                       Dgam)
        n_elec[id_z + 1] = N_elec[id_z + 1] / S_z[id_z + 1]

    a_SSA[:-1] = trapz(a_SSA_apprx*n_elec[:-1, :, newaxis], gam, axis=1)
    n_ho, n_to = rad_transfer(epsL, z_prim, j_syn, a_SSA, R_z)
    save_gen(sim_name, lyr_name, lyr_name_ig, gam_max, Q_inj, n_elec, N_elec, n_ho, n_to, j_syn, N_dot, a_SSA, IC_LR)
    return n_ho, n_to

def gen_next(sim_name, lyr_name, iter_num, gen_num):
    lyr_name_ig = lyr_name + '_' + str(iter_num) + str(gen_num)

    z_prim, Dt_prim, t_acc, R_z, B_z, S_z, q, bj, Q0, si, R0,  z_acc_end, syn_LR, adb_LR, gam_min = or_data(
        sim_name + '/' + lyr_name, 'z_prim', 'Dt_prim', 't_acc', 'R_z', 'B_z', 'S_z', 'q',
        'bj', 'Q0', 'si', 'R0', 'z_end_prim', 'syn_LR', 'adb_LR', 'gam_min')
    gam, gam05, Dgam, epsL = or_data(sim_name + '/axis', 'gam', 'gam05', 'Dgam', 'epsL')
    j_syn_apprx, a_SSA_apprx = or_data(sim_name + '/tmp/syn_prep_' + lyr_name, 'j_syn', 'a_SSA')
    exIC_LR, IC_LR= or_data(sim_name + '/' + lyr_name_ig, 'exIC_LR', 'IC_LR')

    z_size = z_prim.size;
    g_size = gam.size;
    eL_size = epsL.size;
    g05_size = gam05.size
    gam_max = ones(z_size)
    Q_inj = ones((z_size, g_size)) * 1e-200
    n_elec = ones((z_size, g_size)) * 1e-200
    N_elec = ones((z_size, g_size)) * 1e-200
    n_ho = ones((z_size, eL_size)) * 1e-200
    n_to = ones((z_size, eL_size)) * 1e-200
    j_syn = ones((z_size, eL_size)) * 1e-200
    N_dot = ones((z_size, eL_size)) * 1e-200
    a_SSA = ones((z_size, eL_size)) * 1e-200

    total_loss_rate = syn_LR + adb_LR + exIC_LR + IC_LR

    for id_z in range(z_size - 1):
        gam_max[id_z] = gam_max_fun(gam05, t_acc[id_z], total_loss_rate[id_z])
        Q_inj[id_z] = Q_inj_fun(gam, z_prim[id_z], gam_max[id_z], R0 / R_z[id_z], gam_min, si, z_acc_end, Q0, q)
        j_syn[id_z] = trapz(j_syn_apprx[id_z] * n_elec[id_z][:, newaxis], gam, axis=0)
        N_dot[id_z] = j_syn[id_z] * S_z[id_z] / epsL[:] * c.pi4
        N_elec[id_z + 1] = solve_elec_eq(N_elec[id_z],
                                         total_loss_rate[id_z],
                                         Q_inj[id_z],
                                         Dt_prim[id_z],
                                         Dgam)
        n_elec[id_z+1] = N_elec[id_z+1]/S_z[id_z+1]

    a_SSA[:-1] = trapz(a_SSA_apprx * n_elec[:-1, :, newaxis], gam, axis=1)
    n_ho, n_to = rad_transfer(epsL, z_prim, j_syn, a_SSA, R_z)
    save_gen(sim_name, lyr_name, lyr_name_ig, gam_max, Q_inj, n_elec, N_elec, n_ho, n_to, j_syn, N_dot, a_SSA, IC_LR)
    return n_ho, n_to