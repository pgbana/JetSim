from numpy import  append, amin, sqrt, cos, interp, array, ones
from numpy.random import rand
from sim.hdf5 import or_data, ow_data
from sim.helper import get_3col, get_2col, get_dict
from numpy import sum as npsum
import sim.const as c
from scipy.interpolate import RegularGridInterpolator
import sim.synchr as syn

class cLeptons:

    def __init__(self):

        self.bj = 0.
        self.B_z = 0.
        self.z_axis_obs = 0.
        self.Dj = 1.
        self.Gj = 1.

        self.gam_prim = array([])
        self.t_obs = []
        self.z_obs = []

        self.epsL_prim = None
        self.n_syn = None

        self.gam_dot_IC= None
        self.gam_dot_syn = None
        self.delta_t = None

        self.p_emiss = None
        self.ITM_phot_absorp_e = None
        self.ITM_emiss_evh = None

        self.IC_KN_losses = 0.
        self.syn_losses = 0.
        self.IC_Th_losses = 0.

    def load(self, sim_name):
        z_obs, epsVH_prim, u, gamH = or_data(sim_name+'/tmp/casc', 'z_obs', 'epsVH_prim', 'u', 'gamH')

        z_obs, Dj, B_z, epsL, bj, Gj= or_data(sim_name+'/tmp/casc', 'z_obs', 'Dj', 'B_z', 'epsL', 'bj', 'Gj')
        self.Gj = Gj
        self.bj = bj
        self.B_z = B_z
        self.z_axis_obs = z_obs
        self.Dj = Dj
        self.epsL_prim = epsL
        self.n_syn = ones(self.epsL_prim.size) * 1e-200

        IC_loss_rate, syn_loss_rate, delta_t= or_data(sim_name+'/tmp/casc', 'IC_loss_rate_LE','syn_loss_rate', 'delta_t')
        self.gam_dot_IC= RegularGridInterpolator((z_obs, gamH), IC_loss_rate, bounds_error=False, fill_value=0)
        self.gam_dot_syn = RegularGridInterpolator((z_obs, gamH), syn_loss_rate, bounds_error=False, fill_value=0)
        self.delta_t = RegularGridInterpolator((z_obs, gamH), delta_t, bounds_error=False, fill_value=0)

        p_emiss, ITM_phot_absorp_e, ITM_emiss_evh = or_data(sim_name+'/tmp/casc', 'p_emiss', 'ITM_phot_absorp_e', 'ITM_emiss_evh')
        self.p_emiss = RegularGridInterpolator((z_obs, gamH), p_emiss)
        self.ITM_phot_absorp_e = RegularGridInterpolator((z_obs, epsVH_prim, u), ITM_phot_absorp_e)
        self.ITM_emiss_evh = RegularGridInterpolator((z_obs, gamH, u), ITM_emiss_evh)

    def energy(self):
        return npsum(self.gam_prim)

    def inject(self, ph_abs):
        if ph_abs != False:
            self.gam_prim = append(self.gam_prim, [self.lep_spec(ph_abs['eps_obs'], ph_abs['z_obs'])])
            self.t_obs = append(self.t_obs, [ph_abs['t_obs'], ph_abs['t_obs']])
            self.z_obs = append(self.z_obs, [ph_abs['z_obs'], ph_abs['z_obs']])
            self.reject()

    def lep_spec(self, eps_obs, z_obs):
        eps1_prim = eps_obs/self.Dj
        u0 = rand(eps1_prim.size)
        def g1g2(e0_prim, e1_prim):
            e_cm = sqrt(e0_prim*e1_prim/(c.mcc*c.mcc))
            g_cm = e_cm
            b_cm = sqrt(1. - 1. / (g_cm * g_cm))
            g_c = (e0_prim / c.mcc + e1_prim / c.mcc) / (2. * e_cm)
            b_c = sqrt(1. - 1. / (g_c * g_c))
            u1 = cos(rand(e1_prim.size) * c.pi)
            g_plus = g_cm * g_c * (1. + b_c * b_cm * u1)
            g_minus = g_cm * g_c * (1. - b_c * b_cm * u1)
            return g_plus, g_minus
        e0_prim = self.ITM_phot_absorp_e(get_3col(z_obs, eps1_prim, u0))
        return g1g2(e0_prim, eps1_prim)

    def emit_IC(self, Delta_t):

        u1 = rand(self.gam_prim.size)
        P = self.p_emiss(get_2col(self.z_obs, self.gam_prim))*Delta_t
        id_emiss = (u1 <= P).nonzero()[0]
        e1_prim = self.ITM_emiss_evh(get_3col(self.z_obs[id_emiss], self.gam_prim[id_emiss], rand(id_emiss.size)))
        self.IC_KN_losses += npsum(e1_prim)
        self.gam_prim[id_emiss] -= e1_prim/c.mcc
        return get_dict('eps', e1_prim*self.Dj, 'zem', self.z_obs[id_emiss], 'tem', self.t_obs[id_emiss])

    def emit_syn(self, n=100, Delta_t=1.):
        n = int(n)
        gam_syn = self.gam_prim[0::n]
        z_syn = self.z_obs[0::n]
        B = interp(z_syn, self.z_axis_obs, self.B_z)
        for id_g in range(gam_syn.size):
            self.n_syn += syn.Psyn2(self.epsL_prim, gam_syn[id_g], B[id_g]) / (self.epsL_prim) * Delta_t * n

    def reject(self):
        id_lp = ((self.gam_prim > 200.) & (self.z_obs < self.z_axis_obs[-1])).nonzero()[0]
        self.gam_prim = self.gam_prim[id_lp]
        self.t_obs = self.t_obs[id_lp]
        self.z_obs = self.z_obs[id_lp]



    def make_step(self):
        Delta_t_prim = amin(self.delta_t(get_2col(self.z_obs, self.gam_prim))) * 0.01
        self.t_obs += Delta_t_prim * self.Dj
        self.z_obs += Delta_t_prim * self.Dj * c.c * self.bj

        syn_losses = self.gam_dot_syn(get_2col(self.z_obs, self.gam_prim)) * Delta_t_prim
        self.syn_losses += npsum(syn_losses) * c.mcc
        self.gam_prim -= syn_losses

        ic_losses = self.gam_dot_IC(get_2col(self.z_obs, self.gam_prim)) * Delta_t_prim
        self.IC_Th_losses += npsum(ic_losses) * c.mcc
        self.gam_prim -= ic_losses

        self.reject()
        return Delta_t_prim

    def finish(self, sim_name, casc_name):

        eps_syn = self.epsL_prim * self.Dj
        n_syn = self.n_syn / self.Dj
        leptons = {'lept_gam_prim': self.gam_prim, 'lept_z_obs':self.z_obs,
                   'lept_t_obs':self.t_obs, 'syn_losses':self.syn_losses,
                   'IC_KN_losses': self.IC_KN_losses, 'IC_Th_losses':self.IC_Th_losses,
                   'eps_syn':eps_syn,'n_syn':n_syn, 'Gj':self.Gj}
        ow_data(sim_name+'/'+casc_name, **leptons)











