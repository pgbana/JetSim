from sim.hdf5 import ow_data
import sim.casc.photons as ph
import sim.casc.leptons as lp
import sys


def run(sim_name, casc_name='cascades',
        gammaN = 1000, stepN=1000, t_inj=1000., Emin=1., Emax=30., si=0, syn_fraction=100):

    param = {'gammaN':gammaN, 'stepN':stepN, 't_inj':t_inj, 'Emin':Emin, 'Emax':Emax, 'si':si,
             'syn_fraction': syn_fraction}

    phot = ph.cPhotons()
    lept = lp.cLeptons()
    phot.load(sim_name)
    lept.load(sim_name)

    ph_inj = phot.inject([1.6022 * Emin, 1.6023 * Emax], [0., t_inj], gammaN, si=si)
    param['ph_eps_inj'] = ph_inj['eps']
    param['ph_t_inj'] = ph_inj['tem']
    param['ph_z_inj'] = ph_inj['zem']
    param['E_inj'] = sum(ph_inj['eps'])
    ph_abs = phot.absorb(ph_inj)
    lept.inject(ph_abs)
    for id_i in range(int(stepN)):
        sys.stdout.write("                        Step: %d, lp_num: %d   \r" % (id_i, lept.gam_prim.size))
        sys.stdout.flush()
        Dt = lept.make_step()
        if lept.gam_prim.size == 0.: break
        ph0 = lept.emit_IC(Dt)
        if lept.gam_prim.size == 0.: break
        lept.emit_syn(n=syn_fraction, Delta_t=Dt)
        ph_abs = phot.absorb(ph0)
        lept.inject(ph_abs)
        if lept.gam_prim.size == 0.: break

    ow_data(sim_name+'/'+casc_name, mode='w', **param)
    phot.finish(sim_name, casc_name)
    lept.finish(sim_name, casc_name)
