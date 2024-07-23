This repository contains the Python code written for my Part III (Master's)
Project at the University of Cambridge, entitled 'Simulating a few interacting 
particles in a flat band'. This involved exact numerical simulations of one and 
two bosons in a Kagome lattice, with a particular focus on investigating the 
effect of different interaction strengths on the system behaviour. The final project 
report is also included in both pdf and Latex form.

Included is the code used to run simulations, and generate, analyse 
and plot data. The main files are Kagome.py and Kagome2.py, which contain the core 
code, for 1 and 2 particles respectively: for building the systems and their 
Hamiltonians, computing time-evolution and physical observables, and producing 
plots of the system.

The majority of the other Python files each relate to particular types of simulation, 
or performing specific pieces of analysis. They generally contain functions for 
generating data, i.e. performing simulations with the desired initial states to calculate 
and save the required observables. They also contain functions for analyzing 
and plotting the data.

Exceptions to this are the KagomeAnimate.py file, which, given a file of 
the density at each site in the system against time, produces an animation
of it; and the files in the Complexity and Timing folder, which relate
to calculating the computation time of the various functions employed throughout 
the project.
