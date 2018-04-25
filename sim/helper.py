from numpy import fabs, newaxis, transpose, array, copy, interp, ones_like
from astropy.cosmology import FlatLambdaCDM
import sim.const as c
import sys

def z_to_dL(redshift):
    cosmo = FlatLambdaCDM(H0=73, Om0=0.27)
    dL = cosmo.luminosity_distance(redshift).value
    return dL * c.pc_to_cm * 1e6 # w cm

def nearest_value(array, value):
    idx = (fabs(array-value)).argmin()
    return array[idx]

def nearest_id(array, value):
    idx = (fabs(array-value)).argmin()
    return idx

def nearest_array_id(array, value):
    idx = (fabs(array-value[:,newaxis])).argmin(axis=1)
    return idx

def get_2col(first, second):
    return transpose(array([first, second]))

def get_3col(first, second, third):
    return transpose(array([first, second, third]))

def get_dict(*args):
    new_dict = {}
    for id_arg in range(0, len(args), 2):
        new_dict[args[id_arg]] = copy(args[id_arg+1])
    return new_dict

def interp2D(x0, y0, z0, y):
    z1 = ones_like(z0) * 1e-200
    for id_x in range(x0.size):
        z1[id_x] = interp(y, y0[id_x], z0[id_x], left=1e-200, right=1e-200)
    return z1

def progress(count, total, prefix):
    bar_len = 20
    filled_len = int(round(bar_len * count/float(total)))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('%s [%s]\r' % (prefix, bar))
    if count==(total-1):
        sys.stdout.write('\n' )
    sys.stdout.flush()