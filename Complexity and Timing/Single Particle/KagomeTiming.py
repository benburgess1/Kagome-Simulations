import Kagome as Kag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import cProfile
import timeit
import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

'''
time building lattice (dominated by eigh), state (dominated by matrix multiplication) for various lattice sizes
plot graphs vs N (=3*Lx*Ly), observe scalings

redo Kagome.py with sparse matrices and compare
'''



def time_Kagome_build(dim,repeat):
	Lx = dim[0]
	Ly = dim[1]
	N = 3*Lx*Ly
	psi_0 = np.zeros(N)
	
	time_vals = timeit.repeat(functools.partial(Kag.Kagome,psi_0,Lx,Ly), repeat=repeat, number=1)
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	
	return mean,stdev


def create_build_data():
	L = np.array([5,10,15,20,25,30,35,40,50,60])
	N = 3*L**2
	repeats = [10,5,5,5,5,5,5,3,3,3]
	
	means = []
	stdevs = []
	
	for i,L in enumerate(L):
		print(f'Evaluating {L}x{L}...')
		m,s = time_Kagome_build(dim=[L,L],repeat=repeats[i])
		means.append(m)
		stdevs.append(s)
	print('Done')
	
	np.savez('BuildTime.npz',L=L,N=N,means=means,stdevs=stdevs)
	

def time_Kagome_state(dim,repeat):
	Lx = dim[0]
	Ly = dim[1]
	N = 3*Lx*Ly
	psi_0 = np.zeros(N)
	psi_0[int(1.5*Lx*(Ly-1))]=1.
	
	system = Kag.Kagome(psi_0,Lx,Ly)
	
	time_vals = timeit.repeat(functools.partial(system.state,np.random.rand()), repeat=repeat, number=1)
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	
	return mean,stdev


def create_state_data():
	L = np.array([5,10,15,20,25,30,35,40,50,60])
	N = 3*L**2
	repeats = [10,5,5,5,5,5,3,3,3,3]
	
	means = []
	stdevs = []
	
	for i,L in enumerate(L):
		print(f'Evaluating {L}x{L}...')
		m,s = time_Kagome_state(dim=[L,L],repeat=repeats[i])
		means.append(m)
		stdevs.append(s)
	print('Done')
	
	np.savez('StateTime.npz',L=L,N=N,means=means,stdevs=stdevs)


create_state_data()
