from numpy import ones,  concatenate, logspace, power, sum, sqrt
from numpy import histogram,  diff, append, trapz
from numpy.random import rand
import sim.const as c
from sim.hdf5 import or_data, ow_data
from scipy.interpolate import RegularGridInterpolator
from sim.helper import get_dict, get_2col, get_3col

class cPhotons:

    def __init__(self):

        self.epsO = []
        self.temO = []
        self.zemO = []

        self.eps_abs = []
        self.z_abs = []
        self.z0 = 1.

        self.p_phot_absorp = None
        self.ITM_phot_absorp_z = None

    def load(self, sim_name):
        p_phot_absorp, ITM_phot_absorp_z, z_obs, epsVH, u, bj, Gj, z_obs_axis=\
            or_data(sim_name+'/tmp/casc', 'p_phot_absorp', 'ITM_phot_absorp_z', 'z_obs', 'epsVH', 'u', 'bj', 'Gj', 'z_obs')

        self.z0 = z_obs_axis[0]
        self.p_phot_absorp = RegularGridInterpolator((z_obs, epsVH), p_phot_absorp)
        self.ITM_phot_absorp_z = RegularGridInterpolator((z_obs, epsVH, u), ITM_phot_absorp_z)

    def PL_spec(self, e_min=1., e_max=100., si=2., n=1e3):
        e_rand = e_min * power(1. - rand(int(n)), -1. / (si - 1.))
        e_rand = e_rand[e_rand < e_max]
        while e_rand.size < n:
           e_rand = append(e_rand, e_min * power(1. - rand(int(n/2.)), -1. / (si - 1.)))
           e_rand = e_rand[e_rand < e_max]
        return e_rand[0:int(n)]

    def inject(self, Deps, Dtem, DN, si=0.):
        DN = int(DN)
        if si==0:
            eps = (Deps[1] - Deps[0]) * rand(DN) + Deps[0]
        else:
            eps = self.PL_spec(e_min=Deps[0], e_max=Deps[1], si=si, n=DN)
        tem = (Dtem[1] - Dtem[0]) * rand(DN) + Dtem[0]
        zem = ones(DN) * self.z0
        return get_dict('eps', eps, 'tem', tem, 'zem', zem)

    def absorb(self, ph0):
        u1 = rand(ph0['eps'].size)
        P = self.p_phot_absorp(get_2col(ph0['zem'], ph0['eps']))
        id_abs = (u1 <= P).nonzero()[0]
        if id_abs.size>0:
            eps_abs = ph0['eps'][id_abs]
            zem_abs = ph0['zem'][id_abs]
            tem_abs = ph0['tem'][id_abs]

            z_abs = self.ITM_phot_absorp_z(get_3col(zem_abs, eps_abs, rand(id_abs.size)))
            ph_abs = get_dict('eps_obs', eps_abs, 't_obs', tem_abs+(z_abs-zem_abs)/c.c, 'z_obs', z_abs)

            self.eps_abs.append(eps_abs)
            self.z_abs.append(z_abs)
        else:
            ph_abs = False

        id_esc = (u1 > P).nonzero()[0]

        if id_esc.size>0:
            self.epsO.append(ph0['eps'][id_esc])
            self.temO.append(ph0['tem'][id_esc])
            self.zemO.append(ph0['zem'][id_esc])

        return ph_abs

    def calc_power(self, eps):
        y_val, bins = histogram(eps['epsO'], logspace(-4, 4, 1601))
        bin_width = diff(bins)
        x_val = sqrt(bins[:-1] * bins[1:])
        print('sum: ', sum(eps['epsO'])*1e30)
        print('hist:', trapz(y_val / bin_width * x_val, x_val)*1e30)

    def finish(self, sim_name, casc_name):
        eps = concatenate(self.epsO).ravel()
        zem = concatenate(self.zemO).ravel()
        tem = concatenate(self.temO).ravel()
        eps_abs = concatenate(self.eps_abs).ravel()
        z_abs = concatenate(self.z_abs).ravel()
        photons = {'eps': eps, 'zem':zem, 'tem':tem,
                   'eps_abs':eps_abs, 'z_abs':z_abs}
        ow_data(sim_name+'/'+casc_name, **photons)