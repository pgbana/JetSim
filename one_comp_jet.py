import sim.init as init
import sim.jet.layer as lyr
import sim.jet.observ as obs


sim_name = 'one_comp_jet'

one_comp_jet = 'one_comp_jet'
one_comp_obs = 'one_comp_obs'

sim_param = init.load_param(sim_name+'/sim.txt')
init.axis_file(sim_name, **sim_param)
init.ic_prep(sim_name)

lyr_param = init.load_param(sim_name+'/jet_param.txt')
lyr.create(sim_name, one_comp_jet, **lyr_param)
lyr.evolv(sim_name, one_comp_jet)

obs_param = init.load_param(sim_name+'/obs.txt')
obs.create(sim_name, one_comp_obs, **obs_param)
obs.add_lyr(sim_name, one_comp_jet, one_comp_jet+'_03', one_comp_obs)
obs.finish(sim_name, one_comp_obs, [one_comp_jet])