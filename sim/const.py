from numpy import pi as nppi
import astropy.constants as astropyc

pi = nppi
pi4 = pi*4.
c = astropyc.c.cgs.value
m_e = astropyc.m_e.cgs.value
mcc = m_e * c * c
h = astropyc.h.cgs.value
q = astropyc.e.esu.value
r_e = 2.81794032273e-13 # [cm]

pc_to_cm = astropyc.pc.cgs.value
erg_to_gev = 624.15064799632
hz_to_gev = 4.135665538536e-24
sig_T = 6.6524587158e-25
R_sch = 295407.146641595 # [M_sun]
MeV100 = 0.00016021773 # [erg]
mev_to_erg = 0.0000016021773