import sim.init as init
import sim.jet.layer as lyr
import sim.jet.observ as obs
import sim.casc.approx as initcasc
import sim.casc.develop as dev
from sim.hdf5 import or_data

sim_name = 'cascades_in_jet'

jet = 'jet'
jet_obs = 'obs'
casc = 'casc'

sim_param = init.load_param(sim_name+'/sim.txt')
init.axis_file(sim_name, **sim_param)
init.ic_prep(sim_name)

lyr_param = init.load_param(sim_name+'/jet_param.txt')
lyr.create(sim_name, jet, **lyr_param)
lyr.evolv(sim_name, jet)

obs_param = init.load_param(sim_name+'/obs.txt')
obs.create(sim_name, jet_obs, **obs_param)
obs.add_lyr(sim_name, jet, jet+'_03', jet_obs)
obs.finish(sim_name, jet_obs, [jet])

initcasc.calc(sim_name, jet, jet+'_03')
casc_param = init.load_param(sim_name+'/'+casc+'.txt')
dev.run(sim_name, casc_name=casc, **casc_param)

