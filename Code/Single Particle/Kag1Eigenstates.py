'''
This file contains functions for plotting the numerical eigenstates of 
single-particle kagome systems
'''


import Kagome as Kag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import cProfile



def plot_eigenstates(L=5,idxs=[0],print_eigvals=False):
	#Plot the density of eigenstates of specified indices for the given system size
	N = 3*L**2
	psi_0 = np.zeros(N)
	system = Kag.Kagome(psi_0=psi_0,Lx=L,Ly=L)
	
	if print_eigvals:
		print(system.eigvals)
	
	for idx in idxs:
		system.psi = system.eigvects[:,idx]
	
		fig,ax = plt.subplots()
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(f'$\omega={{{system.w[idx]}}}$')
	
		system.plot_current_state_tiled(fig,ax)
		plt.show()


def plot_several_eigenstates(L=10,num=10):
	#Plot the density of several of the eigenstates evenly spaced throughout the eigenspectrum
	N = 3*L**2
	psi_0 = np.zeros(N)
	system = Kag.Kagome(psi_0=psi_0,Lx=L,Ly=L)
	
	idxs = np.linspace(0,N-1,num)
	
	for idx in idxs:
		idx = int(idx)
		system.psi = system.eigvects[:,idx]
	
		fig,ax = plt.subplots()
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(f'$\omega={{{np.round(system.w[idx],3)}}}$')
	
		system.plot_current_state_tiled(fig,ax)
		plt.show()


def plot_wf_eigenstates(L=10,idxs=[0],print_eigvals=False,J=1.,bc='open'):
	#Plot the amplitude of eigenstates of specified indices
	N = 3*L**2
	psi_0 = np.zeros(N)
	system = Kag.Kagome(psi_0=psi_0,Lx=L,Ly=L,J=J,bc=bc)
	
	if print_eigvals:
		print(system.eigvals)
		
	for idx in idxs:
		idx = int(idx)
		system.psi = system.eigvects[:,idx]
	
		fig,ax = plt.subplots()
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(f'Re$(\psi), \omega={{{np.round(system.w[idx],5)}}}$')
		
		system.plot_re_wavefunction_tiled(fig,ax)
		plt.show()
		

def plot_several_wf_eigenstates(L=10,num=10):
	#Plot amplitudes of several eigenstates, evenly spaced throughout the eigenspectrum
	N = 3*L**2
	idxs = np.linspace(0,N-1,num)
	plot_wf_eigenstates(L=L,idxs=idxs)


def compare_J(J_vals,idxs=[0],L=10):
	#Compare eigenstates for different values of J (expecting no difference)
	for idx in idxs:
		fig,axs = plt.subplots()
		for i,J in enumerate(J_vals):
			plot_wf_eigenstates(L=L,idxs=[idx],J=J)

	
if __name__ == '__main__':
	plot_wf_eigenstates(L=10,idxs=[1,2,3,4,5,6],bc='periodic')
