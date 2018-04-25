import sim.gg_absorp as ggabs
import sim.inv_compt as ic
from sim.hdf5 import or_data, ow_data
from numpy import ones, trapz, exp, interp, newaxis, zeros, isnan, where
import sim.const as c

def calc(sim_name, lyr_name, lyr_name_ig, spine=False):

    def phot_absorption():
        a_gg_prim = ggabs.a_gg_monodirect(-1., epsL, epsVH_prim, n_syn_ho)
        a_gg = a_gg_prim / Dj

        tau_gg = ones((z_obs.size, epsVH.size)) * 1e-200
        p_phot_absorp = ones((z_obs.size, epsVH.size)) * 1e-200
        ITM_phot_absorp_z = ones((z_obs.size, epsVH.size, u.size)) * 1e-200
        ITM_phot_absorp_e = ones((z_obs.size, epsVH.size, u.size)) * 1e-200
        CDF_phot_absorp_e = zeros(epsL.size)
        CDF_phot_absorp_z = zeros(z_obs.size)


        for id_zem in range(z_obs.size):
            tau_gg[id_zem] = trapz(a_gg[id_zem:], z_obs[id_zem:], axis=0)
            p_phot_absorp[id_zem] = 1. - exp(-tau_gg[id_zem])

            for id_eVH in range(epsVH.size):
                for id_eL in range(2, epsL.size):
                    epsL_min = (c.mcc * c.mcc) / epsVH_prim[id_eVH]
                    if epsL[id_eL] < epsL_min:
                        CDF_phot_absorp_e[id_eL] = 0.
                    else:
                        CDF_phot_absorp_e[id_eL] = ggabs.a_gg_monodirect_flat(-1., epsL[:id_eL], epsVH_prim[id_eVH], n_syn_ho[id_zem,:id_eL]) / a_gg_prim[id_zem, id_eVH]
                CDF_phot_absorp_e[CDF_phot_absorp_e<0] = 0.
                ITM_phot_absorp_e[id_zem, id_eVH] = interp(u, CDF_phot_absorp_e, epsL)
                for id_zab in range(z_obs.size):
                    z_min = z_obs[id_zem]
                    if z_obs[id_zab] <= z_min:
                        CDF_phot_absorp_z[id_zab] = 0.
                    else:
                        CDF_phot_absorp_z[id_zab] = (1. - exp(-trapz(a_gg[id_zem:id_zab,id_eVH], z_obs[id_zem:id_zab], axis=0))) / (p_phot_absorp[id_zem, id_eVH])
                ITM_phot_absorp_z[id_zem, id_eVH] = interp(u, CDF_phot_absorp_z, z_obs, left=z_obs[id_zem], right=z_obs[-1])
        ITM_phot_absorp_z[isnan(ITM_phot_absorp_z)] = z_obs[-1]


        return tau_gg, p_phot_absorp, ITM_phot_absorp_z, ITM_phot_absorp_e, a_gg

    def phot_emission():

        Ndot_approx = ic.n_dot_headon(epsL, epsVH, gamH)
        Ndot = zeros((z_obs.size, epsVH.size, gamH.size))


        p_emiss = zeros((z_obs.size, gamH.size))
        ITM_emiss_evh = ones((z_obs.size, gamH.size, u.size)) * 0.
        CDF_emiss_evh = ones((z_obs.size, epsVH.size, gamH.size)) * 0.

        for id_z in range(z_obs.size):
            Ndot[id_z] = trapz(Ndot_approx * n_syn_ho[id_z], epsL, axis=2)
            p_emiss[id_z] = trapz(Ndot[id_z], epsVH, axis=0)
            for id_eVH in range(epsVH.size):
                if id_eVH > 0:
                    CDF_emiss_evh[id_z, id_eVH] = trapz(Ndot[id_z, :id_eVH], epsVH[:id_eVH], axis=0)/p_emiss[id_z]
            for id_g in range(gamH.size):
                ITM_emiss_evh[id_z, id_g] = interp(u, CDF_emiss_evh[id_z,:,id_g], epsVH, left=1e-200, right=1e-200)
                ITM_emiss_evh[id_z, id_g] = where(ITM_emiss_evh[id_z, id_g] < gamH[id_g]*c.mcc,ITM_emiss_evh[id_z, id_g], gamH[id_g]*c.mcc)
        return p_emiss, ITM_emiss_evh, CDF_emiss_evh

    def IC_loss_rate():

        dN_ho = trapz(IC_n_dot_prep * n_syn_ho[:,newaxis,newaxis], epsL, axis=3)
        dN_to = trapz(IC_n_dot_prep * n_syn_to[:,newaxis,newaxis], epsL, axis=3)

        id_epsLH = (epsH<c.MeV100/Dj).nonzero()[0]
        gam_dot_ho = trapz(dN_ho[:,id_epsLH,:] * epsH[id_epsLH, newaxis], epsH[id_epsLH], axis=1)/c.mcc
        gam_dot_to = trapz(dN_to * epsH[:, newaxis], epsH, axis=1)/c.mcc
        return gam_dot_ho + gam_dot_to

    def SYN_loss_rate():
        const_synLR = c.sig_T * c.c / (c.mcc * 6. * c.pi)
        return const_synLR * B_z[:, newaxis] ** 2 * (gamH ** 2 - 1.)

    def Delta_t():
        const_synLR = c.sig_T * c.c / (c.mcc * 6. * c.pi)
        return gamH/(const_synLR * B_z[:, newaxis] ** 2 * (gamH ** 2 - 1.)) * 0.1


    epsL, epsH, epsVH, z_obs, gamH, u= or_data(sim_name+'/axis', 'epsL', 'epsH', 'epsVH', 'z_obs', 'gamH', 'u')
    n_syn_ho, n_syn_to = or_data(sim_name+'/'+lyr_name_ig, 'n_syn_ho', 'n_syn_to')
    Gj, bj, B_z= or_data(sim_name+'/'+lyr_name, 'Gj', 'bj', 'B_z')

    Dj = 1./ (Gj*(1.-bj))
    epsVH_prim = epsVH/Dj

    IC_n_dot_prep = ic.n_dot_iso_approx(epsL, epsH, gamH)

    approx = {'z_obs': z_obs, 'Gj':Gj, 'bj':bj, 'Dj':Dj, 'B_z':B_z, 'u':u,
              'epsL':epsL, 'gamH':gamH, 'epsH':epsH, 'epsVH':epsVH, 'epsVH_prim':epsVH_prim}

    tau_gg, p_phot_absorp, ITM_phot_absorp_z, ITM_phot_absorp_e, a_gg = phot_absorption()
    ITM_phot_absorp_e[-1] = ITM_phot_absorp_e[-2]
    approx['tau_gg'] = tau_gg
    approx['p_phot_absorp'] = p_phot_absorp
    approx['ITM_phot_absorp_z'] = ITM_phot_absorp_z
    approx['ITM_phot_absorp_e'] = ITM_phot_absorp_e
    approx['a_gg'] = a_gg

    p_emiss, ITM_emiss_evh, CDF_emiss_evh = phot_emission()
    approx['p_emiss'] = p_emiss
    approx['ITM_emiss_evh']  = ITM_emiss_evh
    approx['CDF_emiss_evh']  = CDF_emiss_evh

    IC_loss_rate_LE = IC_loss_rate()
    approx['IC_loss_rate_LE'] = IC_loss_rate_LE

    syn_loss_rate = SYN_loss_rate()
    approx['syn_loss_rate'] = syn_loss_rate

    delta_t = Delta_t()
    approx['delta_t'] = delta_t

    ow_data(sim_name+'/tmp/'+'casc', mode='w', **approx)
