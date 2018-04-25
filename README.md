# JetSim - Relativistic Jet Simulation
JetSim is the numerical model of astrophysical collimated outflows of plasma, called *jets*. The JetSim is dedicated to jets from active galactic nuclei (AGNs). It include a number of astrophysical processes such as synchrotron emission, inverse Compton emission, adiabatic cooling. The JetSim is composed of three parts:
1. The stationary non-local inhomogeneous jet model;
2. The stationary stratified jet model;
3. The model of electromagnetic cascades in the extended jet.

The JetSim is written in Python. It use the following packages: `NumPy`, `NumExpr`, `AstroPy`, `Matplotlib`, `h5py`.

## Running the JetSim
The first model is run with `one_comp_jet.py` file; the second model with `two_comp_jet.py` and the third with `cascades_in_jet.py`. The parameter files of all models are in the corresponding directory.

## Full description

Full descrition of models can be find in my PhD thesis: "*Modelling of the non-thermal emission from
inhomogeneous jets in active galactic nuclei*".
