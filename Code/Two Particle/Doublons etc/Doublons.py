'''
This file contains functions relating to obtaining and plotting the doublon
eigenspectrum.
'''

import numpy as np
import matplotlib.pyplot as plt
import Kagome2 as Kag2
import Kagome as Kag
import scipy as sp
from matplotlib.lines import Line2D


def compare_singleparticle_doublon(L=5,U=10,k=2.,ax=None,plot=False,system_file=None,lanczos=True,bc='periodic'):
	#Compare theoretical effective single particle spectrum with numerical doublon spectrum
	#k parameter controls the effective single particle hopping coefficient: J_eff = (k*J**2)/U. Used initially for troubleshooting;
	#correct 2nd order PT gives k=2
	if system_file is not None:
		system2 = Kag2.system_from_file(system_file)
		U = system2.U
		L = system2.Lx
		if system2.Lx != system2.Ly:
			print('Warning: Lx not equal to Ly')
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-k/U,bc=bc)
		doublon_eigenfrequencies = system2.w[N_states-N_sites:]
		
	elif lanczos:	
		N_sites = 3*L**2
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-k/U,bc=bc)
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc,skip_k=True)
		#H = sp.sparse.csr_array(system2.H)
		doublon_eigenfrequencies = sp.sparse.linalg.eigsh(system2.H,k=N_sites,which='LA',return_eigenvectors=False)
	
	else:
		N_sites = 3*L**2
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-k/U,bc=bc)
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,bc=bc,skip_k=True,evolution_method='eigenvector')
		doublon_eigenfrequencies = system2.w[N_states-N_sites:]
	
	
	doublon_eigenfrequencies -= U + 8/U		#Shift downwards
	
	if ax is None:
		fig,ax = plt.subplots()
	ax.plot(np.arange(np.size(system1.w)),system1.w,color='b',marker=None,ls='-',label=r'Single Particle, $J_{eff}$ / $J=$'+f'${-k/U}$')
	ax.plot(np.arange(np.size(doublon_eigenfrequencies)),doublon_eigenfrequencies,color='r',marker=None,ls='-',label=f'Doublon Band, $U$ / $J={U}$')
	
	ax.legend(fontsize=15)
	ax.set_ylabel('Energy / J',fontsize=15)
	ax.set_xlabel('Index',fontsize=15)
	#ax.set_title(f'{L}x{L} System, bc='+bc)
	if plot:
		plt.show()
	
	
def plot_doublon_eigenstates(idxs=[],L=5,U=100.):
	#Plot the amplitude of doublon eigenstates of specified indices
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	system = Kag2.Kagome2(psi_0,Lx=L,Ly=L,U=U)
	
	for i,idx in enumerate(idxs):
		system.psi = system.eigvects[:,idx]
		fig,ax = plt.subplots()
		system.plot_amplitude(fig,ax,uppernorm=1.,cbar_label_y=0.5)
		ax.set_title(f'$\omega - U - 8J^2/U$ = {np.round(system.w[idx] - U - 8/U,5)}')
		plt.show()
		

#The following functions here were used for troubleshooting the discrepancies between
#the doublon and effective single-particle spectra, when using open BCs.
#Periodic BCs were implemented later, and give much better agreement 'automatically' due 
#to the absence of edge effects		
def plot_lattice_neighbours(L=5):
	#Plot the kagome lattice with sites coloured according to their number of near neighbours. 
	N = 3*L**2
	psi_0 = np.zeros(N)
	system = Kag.Kagome(psi_0,Lx=L,Ly=L)
	fig,ax = plt.subplots()
	system.plot_lattice(ax,color='k',plot_sites=False)
	
	colors = {4:'k',3:'r',2:'y'}
	for site in system.sites:
		ax.plot(site.x,site.y,color=colors[len(site.nn)],marker='o')
		
	legend_elements = [Line2D([0],[0],color=colors[key],marker='o',ls='',label=key) for key in colors.keys()]
	ax.legend(handles=legend_elements,title='Near Neighbours:')
	
	
	ax.set_xticks([])
	ax.set_yticks([])
	
	ax.set_title(f'{L}x{L} Unit Cells')
	plt.show()
	
	
def hexagon_state_occupancies():
	#Plot the occupancies of numerical eigenstates (i.e. coefficients c) for the system in a hexagon eigenstate
	system = Kag2.double_occupied_hexagon_5x5_system(U=100.)
	
	fig,ax = plt.subplots()
	system.plot_state(fig,ax,t=0)
	plt.show()
	
	fig,ax = plt.subplots()
	ax.plot(np.arange(system.N),system.c_0,color='b',marker='x',ls='')
	ax.set_xlabel('State Index')
	ax.set_ylabel('c_0')
	plt.show()
	
	
def compare_singleparticle_doublon_states(idxs=[],L=5,U=10.):
	#Compare eigenstate amplitudes for numerical doublon states, and effective single particle
	N_sites = 3*L**2
	system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U)
	N_states = int(0.5*N_sites*(N_sites+1))
	system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U)
	
	for idx in idxs:
		fig,axs = plt.subplots(1,2)
		
		system1.psi = system1.eigvects[:,idx]*np.sqrt(2)
		system2.psi = system2.eigvects[:,idx+N_states-N_sites]			#Since doublon indexes start at N_states-N_sites for 2-particle system
		
		system1.plot_re_wavefunction_tiled(fig,axs[0],norm='1to1',plot_cbar=False)
		system2.plot_amplitude(fig,axs[1],uppernorm=1.,cbar_label_y=0.5)
		
		axs[0].set_title('Single Particle')
		axs[1].set_title('Doublon')
		plt.suptitle(f'State Index {idx}, $\omega$ = {np.round(system1.w[idx],5)}')
		
		fig.set_size_inches(12,5)
		
		plt.show()
		
	
def find_equiv_idx(system1,system2,idx,doublon_evects=None):
	#With the numerical doublon states and effective single particle states both enumerated, 
	#for a given doublon eigenstate, find the 'equivalent index' of the single-particle state
	#which 'most closely resembles' it - i.e. has the greatest overlap. Allows the eigenspectrum 
	#to be 're-ordered' so that the 'same' states appear on the same vertical line, allowing
	#easier comparison of their energies and interpretation of the spectrum
	psi = system1.eigvects[:,idx]
	
	amp = np.zeros((system2.N_sites,system2.N_sites))
	
	for i in range(system2.N_sites):
		if doublon_evects is not None:
			psi2 = doublon_evects[:,i]
		else:
			psi2 = system2.eigvects[:,int(system2.N-system2.N_sites+i)]
		amp[i,:] = system2.amp_from_psi(psi2)
	
	equiv_idx = np.argmax(np.abs(amp @ psi.T))
	if doublon_evects is None:
		equiv_idx += system2.N - system2.N_sites
	return equiv_idx
	
	
def compare_equiv_eigenvalues(L=5,U=100,k=2.,system_file=None,lanczos=True):
	#Plot eigenspectra of doublons and effective single-particles, with doublon values shifted 
	#to the 'equivalent index' as described above
	if system_file is not None:
		system2 = Kag2.system_from_file(system_file)
		U = system2.U
		L = system2.Lx
		if system2.Lx != system2.Ly:
			print('Warning: Lx not equal to Ly')
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-k/U)
		doublon_eigenfrequencies = system2.w[N_states-N_sites:]
		
	elif lanczos:	
		N_sites = 3*L**2
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-k/U)
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,skip_diag=True)
		H = sp.sparse.csr_array(system2.H)
		doublon_eigenfrequencies,doublon_eigenvectors = sp.sparse.linalg.eigsh(H,k=N_sites,which='LM',return_eigenvectors=True)
	
	else:
		N_sites = 3*L**2
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-k/U)
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U)
		doublon_eigenfrequencies = system2.w[N_states-N_sites:]
	
	
	doublon_eigenfrequencies -= U + 8/U		#Shift downwards
	equiv_idxs = np.zeros(N_sites)
	for i in range(N_sites):
		print(f'Evaluating index {i}...   ',end='\r')
		equiv_idxs[i] = find_equiv_idx(system1,system2,i,doublon_evects=doublon_eigenvectors)
	
	fig,ax = plt.subplots()
	ax.plot(equiv_idxs,doublon_eigenfrequencies,color='b',marker='x',ls='',label=f'Doublon')
	ax.plot(np.arange(np.size(system1.w)),system1.w,color='r',marker='x',ls='',label=f'Single Particle')
	ax.legend()
	ax.set_ylabel('Energy')
	ax.set_xlabel('Eigenstate Index')
	ax.set_title(f'{L}x{L} System, U = {U}, Doublon Eigenvalues Shifted to Equivalent Index')
	plt.show()
	
	
def compare_equiv_states(idxs=[],L=5,U=100):
	#Plot the amplitude of 'equivalent' doublon and single-particle states
	N_sites = 3*L**2
	system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U)
	N_states = int(0.5*N_sites*(N_sites+1))
	system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U)
	
	for idx in idxs:
		equiv_idx = find_equiv_idx(system1,system2,idx)
		fig,axs = plt.subplots(1,2)
		
		system1.psi = system1.eigvects[:,idx]*np.sqrt(2)
		system2.psi = system2.eigvects[:,equiv_idx]
		
		system1.plot_re_wavefunction_tiled(fig,axs[0],norm='1to1',plot_cbar=False)
		system2.plot_amplitude(fig,axs[1],uppernorm=1.,cbar_label_y=0.5)
		
		axs[0].set_title(f'Single Particle Index {idx}')
		axs[1].set_title(f'Doublon Index {equiv_idx - N_states + N_sites}')
		
		fig.set_size_inches(12,5)
		
		plt.show()
		
		
def plot_equiv_idxs(L=5,U=100.,system_file=None):
	#Plot doublon index (i.e. ascending order of eigenvalues) against the equivalent 
	#single particle index; deviations from the line y=x shows where discrepancies occur
	if system_file is not None:
		system2 = Kag2.system_from_file(system_file)
		U = system2.U
		L = system2.Lx
		if system2.Lx != system2.Ly:
			print('Warning: Lx not equal to Ly')
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U)
		
	else:
		N_sites = 3*L**2
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U)
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U)
	
	idxs = np.arange(N_sites,dtype=np.int16)
	equiv_idxs = np.zeros(N_sites)
	for idx in idxs:
		equiv_idxs[idx] = find_equiv_idx(system1,system2,idx) - N_states + N_sites		#Start counting when doublon indices start
		
	fig,ax = plt.subplots()
	ax.plot(idxs,equiv_idxs,color='b',ls='',marker='x')
	ax.plot(idxs,idxs,color='r',ls='--')
	ax.set_xlabel('Single Particle Index')
	ax.set_ylabel('Doublon Equivalent Index')
	ax.set_aspect('equal')
	plt.show()
	
	
def compare_8x8_5x5(filename='System_L8x8_J1_U100.npz',U=100.):
	#Compare 8x8 unit cell doublon spectrum to 5x5
	system8 = Kag2.system_from_file(filename)
	N_sites = 192
	N_states = int(0.5*N_sites*(N_sites+1))
	system1 = Kag.Kagome(np.zeros(N_sites),Lx=8,Ly=8,J=-2/U)
	
	system5 = Kag2.Kagome2(psi_0=np.zeros(2850),Lx=5,Ly=5,U=U)
	
	doublon_eigenfrequencies8 = system8.w[18336:]
	doublon_eigenfrequencies8 -= U + 8/U		#Shift downwards
	
	doublon_eigenfrequencies5 = system5.w[2775:]
	doublon_eigenfrequencies5 -= U + 8/U		#Shift downwards
	
	x_8 = np.arange(192)/191
	x_5 = np.arange(75)/74
	
	fig,ax = plt.subplots()
	ax.plot(x_8,system1.w,color='r',marker='x',ls='',label=f'8x8 Single Particle')
	ax.plot(x_8,doublon_eigenfrequencies8,color='b',marker='x',ls='',label=f'8x8 Doublon')
	ax.plot(x_5,doublon_eigenfrequencies5,color='cyan',marker='x',ls='',label=f'5x5 Doublon')
	ax.legend()
	ax.set_ylabel('Energy')
	ax.set_xlabel('Eigenstate Index / Max Eigenstate Index')
	plt.show()
	

def nn_weighted_shift(system):
	#Given a diagonalized kagome system, return the predicted theoretical single-particle
	#eigenvalues calculated within 2nd-order PT, but with the energy shifts modified according
	#to the density at each site in the numerical eigenstates. In the basic previous scheme, all 
	#sites were assumed to have 4 near neighbours, leading to a shift of 8J^2/U; now, the fact that
	#different numbers of near neighbours are present with open BCs is taken into account
	N_sites = system.N_sites
	N_states = system.N
	
	doublon_eigenfrequencies = system.w[N_states-N_sites:]
	
	for i in range(N_sites):
		system.psi = system.eigvects[:,N_states-N_sites+i]
		density = system.density_from_psi(system.psi)
		shift = 0
		for j in range(N_sites):
			shift += density[j]*len(system.sites[j].nn)/system.U			#NB no factor of 2 in numerator, since 'cancelled out' by normalizing density
			#shift += density[j]*4/system.U
		doublon_eigenfrequencies[i] -= shift + system.U
		
	return doublon_eigenfrequencies


def plot_nn_weighted_shift(L=5,U=100.,system_file=None):
	#Plot the eigenvalue predictions as weighted by the number of near neighbours and density at each
	#site, as described above
	if system_file is not None:
		system2 = Kag2.system_from_file(system_file)
		U = system2.U
		L = system2.Lx
		if system2.Lx != system2.Ly:
			print('Warning: Lx not equal to Ly')
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U)
		
	else:
		N_sites = 3*L**2
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U)
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U)
	
	doublon_eigenfrequencies = nn_weighted_shift(system2)
	
	fig,ax = plt.subplots()
	ax.plot(np.arange(np.size(doublon_eigenfrequencies)),doublon_eigenfrequencies,color='b',marker='x',ls='',label=f'Shifted Doublon Band')
	ax.plot(np.arange(np.size(system1.w)),system1.w,color='r',marker='x',ls='',label=f'Single Particle, J={-2/U}')
	ax.legend()
	ax.set_ylabel('Energy')
	ax.set_xlabel('Eigenstate Index')
	ax.set_title(f'{L}x{L} System, U={U}')
	plt.show()
	
	
def plot_weighted_shift_equiv_idx(L=5,U=100.,system_file=None):
	#Plot the weighted shift as above, but now also after shifting eigenvalues
	#to the equivalent index as described earlier
	if system_file is not None:
		system2 = Kag2.system_from_file(system_file)
		U = system2.U
		L = system2.Lx
		if system2.Lx != system2.Ly:
			print('Warning: Lx not equal to Ly')
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U)
		
	else:
		N_sites = 3*L**2
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U)
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U)
	
	doublon_eigenfrequencies = nn_weighted_shift(system2)
	
	equiv_idxs = np.array([find_equiv_idx(system1,system2,i) for i in range(N_sites)]) - N_states + N_sites
	
	fig,ax = plt.subplots()
	for i in range(N_sites):
		ax.plot(i,doublon_eigenfrequencies[equiv_idxs[i]],color='b',marker='x',ls='',label=f'Shifted Doublon Band' if i == 0 else None)
	ax.plot(np.arange(N_sites),system1.w,color='r',marker='x',ls='',label=f'Single Particle, J={-2/U}')
	ax.legend()
	ax.set_ylabel('Energy')
	ax.set_xlabel('Eigenstate Index')
	ax.set_title(f'{L}x{L} System, U={U}, Doublon Eigenvalues Shifted to Equivalent Index')
	plt.show()
	
	
def compare_L(L_vals,U=100,L_single=10):
	#For various values of L, plot the doublon eigenspectra and compare to the effective single-particle
	fig,ax = plt.subplots()
	
	colors = plt.cm.rainbow(np.linspace(0,1,len(L_vals)))
	for i,L in enumerate(L_vals):
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,skip_diag=True)
		H = sp.sparse.csr_array(system2.H)
		doublon_eigenfrequencies = sp.sparse.linalg.eigsh(H,k=N_sites,which='LM',return_eigenvectors=False)
		doublon_eigenfrequencies -= U + 8/U		#Shift downwards
		ax.plot(np.arange(N_sites)/(N_sites-1),doublon_eigenfrequencies,color=colors[i],marker='x',ls='',label=f'{L}x{L}')
	

	system1 = Kag.Kagome(psi_0=np.zeros(3*L_single**2),Lx=L_single,Ly=L_single,J=-2/U)
	ax.plot(np.arange(3*L_single**2)/(3*L_single**2-1),system1.w,color='k',ls=':',label=f'Single Particle')
	
	ax.legend()
	ax.set_ylabel('Energy')
	ax.set_xlabel('Eigenstate Index / Max Eigenstate Index')
	ax.set_title(f'U = {U}')
	plt.show()
	
	
def linear_fit(x,a,b):
	return a*x + b
	
	
def plot_energy_difference(L=5,U=100,lanczos=True,ax=None,plot=False,unit=1.,system_file=None,bc='periodic',plotfit=None,fit_idx1=0,fit_idx2='all',printparams=True):
	#Plot the difference between exact numerical doublon eigenvalues, and effective single-particle eigenvalues, against doublon energy
	if system_file is not None:
		system2 = Kag2.system_from_file(system_file)
		U = system2.U
		L = system2.Lx
		if system2.Lx != system2.Ly:
			print('Warning: Lx not equal to Ly')
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U,bc=bc)
		doublon_eigenfrequencies = system2.w[N_states-N_sites:]
		
	elif lanczos:	
		N_sites = 3*L**2
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U,bc=bc)
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc,skip_k=True)
		#H = sp.sparse.csr_array(system2.H)
		doublon_eigenfrequencies = sp.sparse.linalg.eigsh(system2.H,k=N_sites,which='LA',return_eigenvectors=False)
	
	else:
		N_sites = 3*L**2
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U,bc=bc)
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,bc=bc,skip_k=True,evolution_method='eigenvector')
		doublon_eigenfrequencies = system2.w[N_states-N_sites:]
	
	
	doublon_eigenfrequencies -= U + 8/U		#Shift downwards
	
	dE = system1.eigvals - doublon_eigenfrequencies
	
	if ax is None:
		fig,ax = plt.subplots()
	#ax.plot(system1.eigvals,dE,color='b',marker='x',ls='',label='Data')
	ax.plot(doublon_eigenfrequencies,dE/unit,color='b',marker='x',ls='',label='Data')
	
	if plotfit=='linear':
		if fit_idx2 == 'all':
			fit_idx2 = np.size(system1.eigvals)
		#popt,pcov=sp.optimize.curve_fit(linear_fit,system1.eigvals[fit_idx1:fit_idx2+1],dE[fit_idx1:fit_idx2+1])
		popt,pcov=sp.optimize.curve_fit(linear_fit,doublon_eigenfrequencies[fit_idx1:fit_idx2+1],dE[fit_idx1:fit_idx2+1]/unit)
		
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * E + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		#dE_fit = linear_fit(system1.eigvals,popt[0],popt[1])
		dE_fit = linear_fit(doublon_eigenfrequencies,popt[0],popt[1])
		#ax.plot(system1.eigvals,dE_fit,color='r',ls='--',label='Fit')
		ax.plot(doublon_eigenfrequencies,dE_fit,color='r',ls='--',label='Fit')
		ax.legend(fontsize=15)
		
	
	ax.set_xlabel(r'$E_{d}$ / $J$',fontsize=15)
	ax.set_ylabel(r'$\Delta E$ / $J$',fontsize=15)
	#ax.set_title(f'Energy Differences, {L}x{L} System, U={U}, bc='+bc)
	if plot:
		plt.show()
		
		
def generate_deltaE_fit_data(filename,U_vals,L=10,bc='periodic'):
	#Generate data for fitting linear relationships to the plots of Delta E vs E_d as above, for various values of U
	intercept_vals = np.zeros(np.size(U_vals))
	gradient_vals = np.zeros(np.size(U_vals))
	intercept_errs = np.zeros(np.size(U_vals))
	gradient_errs = np.zeros(np.size(U_vals))
	
	for i,U in enumerate(U_vals):
		print(f'Evaluating U={U_vals[i]}...')
		N_sites = 3*L**2
		system1 = Kag.Kagome(np.zeros(N_sites),Lx=L,Ly=L,J=-2/U,bc=bc)
		N_states = int(0.5*N_sites*(N_sites+1))
		system2 = Kag2.Kagome2(np.zeros(N_states),Lx=L,Ly=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc,skip_k=True)
		#H = sp.sparse.csr_array(system2.H)
		doublon_eigenfrequencies = sp.sparse.linalg.eigsh(system2.H,k=N_sites,which='LA',return_eigenvectors=False)
		doublon_eigenfrequencies -= U+8/U
		dE = system1.eigvals - doublon_eigenfrequencies
		
		popt,pcov=sp.optimize.curve_fit(linear_fit,doublon_eigenfrequencies,dE)
		perr = np.sqrt(np.diag(pcov))
		
		intercept_vals[i] = popt[1]
		gradient_vals[i] = popt[0]
		intercept_errs[i] = perr[1]
		gradient_errs[i] = perr[0]
		
	np.savez(filename,U_vals=U_vals,L=L,gradient_vals=gradient_vals,intercept_vals=intercept_vals,gradient_errs=gradient_errs,intercept_errs=intercept_errs)
		
		
	
	

if __name__ == '__main__':
	#plot_lattice_neighbours()
	
	#compare_singleparticle_doublon(L=10,U=100,lanczos=True,bc='periodic')
	
	#compare_8x8_5x5()
	
	#plot_doublon_eigenstates(idxs=[-1,-2,-3],U=100.)		
	
	#hexagon_state_occupancies()
	
	#compare_singleparticle_doublon_states(idxs=np.arange(start=53,stop=52,step=-1,dtype=np.int16),U=100.)
	
	#compare_equiv_states(idxs=[63])
	
	#plot_equiv_idxs(system_file='System_L8x8_J1_U100.npz')
	
	#plot_nn_weighted_shift(L=5,U=1e6)
	
	#plot_weighted_shift_equiv_idx(U=1e4)
	
	#compare_L(L_vals=[5,6,7,8,9,10],U=100,L_single=20)
	
	#compare_equiv_eigenvalues(L=7,U=100,lanczos=True)
	
	#plot_energy_difference(L=5,U=100,plotfit='linear',plot=True,unit=1e-3)
	
	generate_deltaE_fit_data('DoublonEnergyDelta_L10x10_U10_1000.npz',U_vals=np.logspace(1,3,13))
	
