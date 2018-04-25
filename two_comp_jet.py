import sim.init as init
import sim.jet.layer as lyr
import sim.jet.observ as obs
import sim.jet.ex_lyr as ex

sim_name = 'two_comp_jet'

def spine_sheath(spine_name='spine',
                 sheath_name='sheath',
                 obs_name='obs_name',
                 spine_file='spine',
                 sheath_file='sheath',
                 obs_file = 'obs'):

    init.folder(sim_name)
    sim_param = init.load_param(sim_name + '/sim.txt')
    init.axis_file(sim_name, **sim_param)
    init.ic_prep(sim_name)

    spine_param = init.load_param(sim_name + '/'+spine_file+'.txt')
    lyr.create(sim_name, spine_name, **spine_param)
    lyr.evolv(sim_name, spine_name, iter_num=0)

    sheath_param = init.load_param(sim_name + '/'+sheath_file+'.txt')
    lyr.create(sim_name, sheath_name, **sheath_param)
    lyr.evolv(sim_name, sheath_name, iter_num=0)

    for id_iter in range(3):
        ex_rad_from_sheath = ex.add_ex_rad(sim_name=sim_name,
                                           ln1=spine_name,
                                           ln2=sheath_name,
                                           ln2_ig=sheath_name + '_' + str(id_iter) + '3')
        lyr.evolv(sim_name, spine_name, iter_num=id_iter + 1, ex_rad=ex_rad_from_sheath)
        ex_rad_from_spine = ex.add_ex_rad(sim_name=sim_name,
                                          ln1=sheath_name,
                                          ln2=spine_name,
                                          ln2_ig=spine_name + '_' + str(id_iter) + '3')
        lyr.evolv(sim_name, sheath_name, iter_num=id_iter + 1, ex_rad=ex_rad_from_spine)

    obs_param = init.load_param(sim_name+'/'+obs_file+'.txt')
    obs.create(sim_name, obs_name, **obs_param)
    obs.add_lyr(sim_name, spine_name, spine_name + '_33', obs_name)
    obs.add_lyr(sim_name, sheath_name, sheath_name + '_33', obs_name)
    obs.finish(sim_name, obs_name, [spine_name, sheath_name])


spine_name = 'spine'; spine_file='spine'
sheath_name = 'sheath'; sheath_file = 'sheath'
obs_name = 'obs'; obs_file = 'obs'

spine_sheath(spine_name=spine_name, spine_file=spine_file,
             sheath_name=sheath_name, sheath_file=sheath_file,
             obs_name=obs_name, obs_file=obs_file)



