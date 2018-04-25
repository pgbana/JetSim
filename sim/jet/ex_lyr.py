from sim.hdf5 import or_data, ow_data
from numpy import ones, sqrt, newaxis, trapz
import sim.const as c
from sim.helper import interp2D

def add_ex_rad(sim_name, ln1, ln2, ln2_ig):
    ex_rad = {}

    epsL = or_data(sim_name + '/axis', 'epsL')
    Gj, bj, R_z, z = or_data(sim_name+'/'+ln1, 'Gj', 'bj', 'R_z', 'z_prim')
    Gj_ex, bj_ex, R_ex, z_ex = or_data(sim_name+'/'+ln2, 'Gj', 'bj', 'R_z', 'z_prim')
    Ndot_ex = or_data(sim_name+'/'+ln2_ig, 'N_dot')

    G12 = Gj_ex * Gj * (1. - bj_ex*bj); ex_rad['G12'] = G12
    b12 = sqrt(G12 ** 2 - 1.) / G12; ex_rad['b12'] = b12
    Reff = 0.66666666667 * R_ex; ex_rad['R_eff'] = Reff

    nex_zz = ones((z.size, z_ex.size, epsL.size)) * 1e-200
    cosT_zz = ones((z.size, z_ex.size)) * 1e-200
    nex_z = ones((z.size, epsL.size)) * 1e-200
    D_zz = ones((z.size, z_ex.size)) * 1e-200

    for id_z in range(z.size):
        Dz = z_ex[id_z] - z_ex
        L = sqrt(Dz ** 2 + Reff ** 2)
        cosT_prim = Dz / L
        if Gj > Gj_ex:  # 1 -- spine, 2 -- sheath
            D_zz[id_z] = G12 * (1. - b12 * cosT_prim)
            cosT_zz[id_z] = (cosT_prim - b12) / (1. - b12 * cosT_prim)
            nph_prim = Ndot_ex / (8. * (c.pi ** 2) * (L ** 2) * c.c)[:, newaxis]
        else:           # 2 -- spine, 1 -- sheath
            D_zz[id_z] = G12 * (1. + b12 * cosT_prim)
            cosT_zz[id_z] = (cosT_prim + b12) / (1. + b12 * cosT_prim)
            nph_prim = Ndot_ex / (4. * (c.pi ) * (L ** 2) * c.c)[:, newaxis]

        n_ph = D_zz[id_z][:, newaxis] ** 2 * nph_prim
        eps = D_zz[id_z][:, newaxis] * epsL

        nex_zz[id_z] = interp2D(z, eps, n_ph, epsL)
        nex_z[id_z] = 2. * c.pi * trapz(nex_zz[id_z], z, axis=0)

    ex_rad['nph_z_zex'] = nex_zz
    ex_rad['nph_z'] = nex_z
    ex_rad['cosT_z_zex'] = cosT_zz
    ex_rad['D_z_zex'] = D_zz

    IC_LR_approx = or_data(sim_name+'/tmp/ic_prep', 'IC_loss_rate')
    IC_LR = trapz(IC_LR_approx * nex_z[:, newaxis], epsL, axis=2)
    ex_rad['exIC_LR'] =IC_LR

    ex_rad['EX_RAD'] = True

    return ex_rad


