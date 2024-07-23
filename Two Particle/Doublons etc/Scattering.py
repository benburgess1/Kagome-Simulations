'''
This file contains functions for generating and plotting data relating
to the scattering band in the strong-interaction limit
'''


import numpy as np
import matplotlib.pyplot as plt
import Kagome2 as Kag2
import Kagome as Kag
import scipy as sp
from matplotlib.lines import Line2D


def compare_noninteracting_scattering(L=5,U=10,ax=None,plot=False,system_file=None,lanczos=True,bc='periodic'):
	#Compare the eigensepctra for a non-interacting 2-particle system with a 
	#strongly-interacting scattering band (i.e. excluding the highest N_sites eigenvalues,
	#which are the doublon states)
	if system_file is not None:
		system2 = Kag2.system_from_file(system_file)
		U = system2.U
		L = system2.Lx
		if system2.Lx != system2.Ly:
			print('Warning: Lx not equal to Ly')
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=2,bc=bc)
		scatt_eigenfrequencies = system2.w[:N_states-N_sites]
		
	elif lanczos:	
		N_sites = 3*L**2
		#system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=2,bc=bc)
		N_states = int(0.5*N_sites*(N_sites+1))
		system1 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=0,bc=bc,skip_k=True,evolution_method='eigenvector')
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc,skip_k=True)
		#H = sp.sparse.csr_array(system2.H)
		scatt_eigenfrequencies = sp.sparse.linalg.eigsh(system2.H,k=N_states-N_sites,which='SA',return_eigenvectors=False)[::-1]
	
	else:
		N_sites = 3*L**2
		#system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=2,bc=bc)
		N_states = int(0.5*N_sites*(N_sites+1))
		system1 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=0,bc=bc,skip_k=True,evolution_method='eigenvector')
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,bc=bc,skip_k=True,evolution_method='eigenvector')
		scatt_eigenfrequencies = system2.w[:N_states-N_sites]
	
	
	#doublon_eigenfrequencies -= U + 8/U		#Shift downwards
	
	if ax is None:
		fig,ax = plt.subplots()
	ax.plot(np.arange(np.size(system1.w)),system1.w,color='r',marker=None,ls='-',label=f'U=0, Full Spectrum')
	ax.plot(np.arange(np.size(scatt_eigenfrequencies)),scatt_eigenfrequencies,color='b',marker=None,ls='-',label=f'U={U}, Scattering Band')
	
	ax.legend()
	ax.set_ylabel('Energy / J')
	ax.set_xlabel('Eigenstate Index')
	#ax.set_title(f'{L}x{L} System, bc='+bc)
	if plot:
		plt.show()
	
	
def plot_scattering_band(L=5,U=10,bc='periodic',ax=None,plot=False,lanczos=False,system_file=None,color='b',label=None):
	#Plot the scattering band for a strongly-interacting 2-particle system
	if system_file is not None:
		system = Kag2.system_from_file(system_file)
		U = system2.U
		L = system2.Lx
		if system2.Lx != system2.Ly:
			print('Warning: Lx not equal to Ly')
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		scatt_eigenfrequencies = system.w[:N_states-N_sites]
		
	elif lanczos:	
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc,skip_k=True)
		scatt_eigenfrequencies = sp.sparse.linalg.eigsh(system.H,k=N_states-N_sites,which='SA',return_eigenvectors=False)[::-1]
	
	else:
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,bc=bc,skip_k=True,evolution_method='eigenvector')
		scatt_eigenfrequencies = system.w[:N_states-N_sites]
		
	if ax is None:
		fig,ax = plt.subplots()
		
	ax.plot(np.arange(np.size(scatt_eigenfrequencies)),scatt_eigenfrequencies,color=color,ls='-',label=label)
	
	if plot:
		plt.show()
		
		
def compare_U_scattering_band(U_vals,L=5,bc='periodic',ax=None,plot=False,colors=None,labels='auto',plot_NI=True,color_NI='r'):
	#Plot the scattering band for various U values
	
	if ax is None:
		fig,ax = plt.subplots()
	
	if plot_NI:
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=0,bc=bc,skip_k=True,evolution_method='eigenvector')
		ax.plot(np.arange(N_states),system.eigvals,color=color_NI,label='U=0, Full Spectrum')
	
	if colors is None:
		colors = plt.cm.rainbow(np.linspace(0,1,len(U_vals)))
		
	if labels == 'auto':
		labels = [f'$U$ / $J={U}$' for U in U_vals]
		
	for i,U in enumerate(U_vals):
		plot_scattering_band(L=L,U=U,bc=bc,ax=ax,color=colors[i],label=labels[i] if labels is not None else None)
	
	ax.legend(fontsize=15)
	ax.set_ylabel('Energy / J',fontsize=15)
	ax.set_xlabel('Index',fontsize=15)
	#ax.set_title(f'{L}x{L} System, bc='+bc)
	if plot:
		plt.show()
	
	
	
if __name__ == '__main__':
	#compare_singleparticle_scattering(L=5,U=10,lanczos=True)
	
	compare_U_scattering_band(U_vals=[10,100,1000000],colors=['b','cyan','y'])
	
	
	
	
