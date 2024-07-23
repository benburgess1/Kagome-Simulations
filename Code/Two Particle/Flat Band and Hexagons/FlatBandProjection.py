'''
This file contains code relating to the OPDM and flat band projection
'''


import numpy as np
import matplotlib.pyplot as plt
import Kagome as Kag
import Kagome2 as Kag2
import scipy as sp
import matplotlib


def plot_projection_from_files(filenames,colors=None,labels='auto'):
	#Plot the total 'flat band occupancy' of the system over time, by projecting psi onto the flat band k states,
	#and finding the norm of the projected psi
	if colors == None:
		colors = plt.cm.rainbow(np.linspace(0,1,len(filenames)))
	
	#Create 'reference' system of correct size 
	#Don't need to calculate or diagonalize the Hamiltonian, or input an initial wavefunction
	data = np.load(filenames[0])
	L = data['Lx']
	system = Kag2.double_occupied_site_system(L=L,U=0,skip_H=True,skip_diag=True,evolution_method='eigenvector')
	
	fig,ax = plt.subplots()
	
	for i,name in enumerate(filenames):
		print(f'Evaluating file {i+1}')
		data = np.load(name)
		t = data['t']
		psi_data = data['psi']
		proj_vals = np.zeros(np.size(t))
		for j in range(np.size(t)):
			proj_vals[j] = np.sum(np.abs(system.flatband_projection(psi=psi_data[j,:]))**2)
			print(f'Progress: {np.round(100*(j+1)/np.size(t),2)}%',end='\r')
		if labels == 'auto':
			U = data['U']
			label = f'U={U}'
		elif labels is not None:
			label = labels[i]
		else:
			label = None
		ax.plot(t,proj_vals,color=colors[i],label=label)
		print('')
		
	ax.set_xlabel(r'$t$'+' / '+ r'$t_0$')
	ax.set_ylabel(r'$\sum_{k,k^\prime} |\langle k,k^\prime|\Psi \rangle|^2$')
	ax.set_title('Flat band projection density vs time')
	ax.legend()
	
	plt.show()
	
	
def plot_estate_projection(L=5,U=0.1,bc='periodic',which='top',num='auto',method='kspace'):
	#Plot the projection of eigenstates onto the (analytical) flat band states
	#i.e. summed over all values of k1,k2
	#For large L, only calculate requested number (num) of eigenvalues using sparse/Lanczos
	#For small L, just diagonalize Hamiltonian fully
	#Can choose to project onto either k-space or real-space representations of flat band eigenstates
	if num == 'auto':
		num = 3*L**2
		
	if L >= 6:
		system = Kag2.double_occupied_site_system(L=L,U=U,bc=bc)
		if which == 'top':
			evals,evects = sp.sparse.linalg.eigsh(system.H,k=num,which='LA',return_eigenvectors=True)
			evals = evals[::-1]
			evects = evects[:,::-1]
		elif which == 'bottom':
			evals,evects = sp.sparse.linalg.eigsh(system.H,k=num,which='SA',return_eigenvectors=True)
	else:
		system = Kag2.double_occupied_site_system(L=L,U=U,bc=bc,skip_diag=False,evolution_method='eigenvector')
		if which == 'top':
			evals = system.eigvals[-1:-(num+1):-1]
			evects = system.eigvects[:,-1:-(num+1):-1]
		elif which =='bottom':
			evals = system.eigvals[:num]
			evects = system.eigvects[:,:num]
	
		
	proj_vals = np.zeros(num)
	
	for i in range(num):
		if method == 'kspace':
			proj_vals[i] = np.linalg.norm(system.flatband_projection_kspace(psi=evects[:,i]))**2
		elif method == 'realspace':
			proj_vals[i] = np.linalg.norm(system.flatband_projection_realspace2(psi=evects[:,i]))**2
		print(f'Progress: {np.round(100*(i+1)/(num),2)}%',end='\r')
		
	fig,ax = plt.subplots()
	ax.plot(np.arange(num),proj_vals,color='b',marker='x',ls='')
	
	ax2 = ax.twinx()
	ax2.plot(np.arange(num),evals,color='r',marker='x',ls='')
	
	if method == 'kspace':
		ax.set_ylabel(r'$\sum_{k,k^\prime} |\langle k,k^\prime|\Psi \rangle|^2$')
	elif method == 'realspace':
		ax.set_ylabel(r'$\sum_{i,j} |\langle \phi_{i},\phi_{j}|\Psi \rangle|^2$')
		
		
	ax2.set_ylabel('E / J')
	ax.set_xlabel('Eigenstate Index')
	ax.set_title(f'Projection of eigenstates onto flat band, {L}x{L}, U={U}, bc={bc}')
	
	handles = [matplotlib.lines.Line2D([0],[0],color='b',marker='x',ls='',label='Projection'),
				matplotlib.lines.Line2D([0],[0],color='r',marker='x',ls='',label='Energy')]
	ax.legend(handles=handles)
	
	plt.show()
	
	
def plot_estate_projection_low(L=5,U=0.1,bc='periodic'):
	#Plot the projection of the lowest-energy eigenstates onto the flat band
	#For large L, only calculate flat band eigenvalues using sparse/Lanczos
	#For small L, just diagonalize Hamiltonian fully
	if L >= 6:
		system = Kag2.double_occupied_site_system(L=L,U=U,bc=bc)
		evals,evects = sp.sparse.linalg.eigsh(system.H,k=3*L**2,which='SA',return_eigenvectors=True)
	else:
		system = Kag2.double_occupied_site_system(L=L,U=U,bc=bc,skip_diag=False,evolution_method='eigenvector')
		evals = system.eigvals[:3*L**2]
		evects = system.eigvects[:,:3*L**2]
		
	proj_vals = np.zeros(3*L**2)
	
	for i in range(3*L**2):
		proj_vals[i] = np.linalg.norm(system.flatband_projection(psi=evects[:,i]))**2
		print(f'Progress: {np.round(100*(i+1)/(3*L**2),2)}%',end='\r')
		
	fig,ax = plt.subplots()
	ax.plot(np.arange(3*L**2),proj_vals,color='b',marker='x',ls='')
	
	ax2 = ax.twinx()
	ax2.plot(np.arange(3*L**2),evals,color='r',marker='x',ls='')
	
	ax.set_ylabel(r'$\sum_{k,k^\prime} |\langle k,k^\prime|\Psi \rangle|^2$')
	ax2.set_ylabel('E / J')
	ax.set_xlabel('Eigenstate Index')
	ax.set_title(f'Projection of mobile eigenstates onto flat band, {L}x{L}, U={U}, bc={bc}')
	
	handles = [matplotlib.lines.Line2D([0],[0],color='b',marker='x',ls='',label='Projection'),
				matplotlib.lines.Line2D([0],[0],color='r',marker='x',ls='',label='Energy')]
	ax.legend(handles=handles)
	
	plt.show()
	
	
def plot_OPDM_evals(system=None,L=5):
	#Plot the eigenvalues of the OPDM for the system with the hexagon initial state (i.e. a double-occupied
	#flat band state)
	if system is None:
		system = Kag2.two_hexagons_system(idx1=37,idx2=37,L=L,U=1,skip_diag=True,evolution_method='eigenvector',bc='periodic',skip_k=False)
	else:
		L = system.Lx
		if system.Ly != system.Lx:
			print('Warning: Lx and Ly not equal')
	
	OPDM = system.OPDM()
	evals,evects = np.linalg.eigh(OPDM)
	
	fig,ax = plt.subplots()
	ax.plot(np.arange(np.size(evals)),evals,marker='x',ls='',color='b')
	ax.set_title('OPDM Eigenvalues')
	ax.set_xlabel(r'$\alpha$')
	ax.set_ylabel(r'$n_{\alpha}$',rotation=0)
	plt.show()
	
	
def plot_OPDM_evects(system=None,L=5,idxs=[0]):
	#Plot the amplitude of eigenstates of the OPDM of the 2-particle system with the hexagon initial state
	if system is None:
		system = Kag2.two_hexagons_system(idx1=37,idx2=37,L=L,U=1,skip_diag=True,evolution_method='eigenvector',bc='periodic',skip_k=False)
	else:
		L = system.Lx
		if system.Ly != system.Lx:
			print('Warning: Lx and Ly not equal')
	
	OPDM = system.OPDM()
	evals,evects = np.linalg.eigh(OPDM)
	
	#Order descending
	evals = evals[::-1]
	evects = evects[:,::-1]
	
	system1 = Kag.Kagome(psi_0=np.zeros(3*L**2),Lx=L,Ly=L)
	
	for idx in idxs:
		fig,ax = plt.subplots()
		system1.psi = evects[:,idx]
		system1.plot_re_wavefunction_tiled(fig,ax)
		ax.set_title(r'$n_{\alpha}=$'+str(np.round(evals[idx],2)))
		plt.show()
		
		
def compare_opdm_proj_opdm(system=None,L=5):
	#Compare the OPDM with and without projection to the flat band for the given system
	#OPDM is plotted as a mpl.imshow() plot, i.e. representing the matrix as a heatmap
	if system is None:
		system = Kag2.two_hexagons_system(idx1=37,idx2=37,L=L,U=1,skip_diag=True,evolution_method='eigenvector',bc='periodic',skip_k=False)
	else:
		L = system.Lx
		if system.Ly != system.Lx:
			print('Warning: Lx and Ly not equal')
			
			
	fig,axs = plt.subplots(1,2)
	system.plot_opdm(fig=fig,ax=axs[0],plot=False,plot_cbar=False)
	
	
	opdm = system.calc_opdm()
	proj_opdm = system.opdm_fb_projection(opdm=opdm)
	print(np.trace(proj_opdm))
	
	system.plot_opdm(fig=fig,ax=axs[1],opdm=proj_opdm,plot=False)
	
	axs[0].set_title('OPDM')
	axs[1].set_title('Projected OPDM')
	plt.show()
	
	
def plot_estate_proj_opdm(L=5,U=0.1,bc='periodic',idxs=[0]):
	#Plot eigenstate amplitudes of OPDM after projection to flat band
	#For large L, only calculate flat band eigenvalues using sparse/Lanczos
	#For small L, just diagonalize Hamiltonian fully
	if L >= 6:
		system = Kag2.double_occupied_site_system(L=L,U=U,bc=bc,skip_diag=True,evolution_method='propagator',skip_k=False)
		evals,evects = sp.sparse.linalg.eigsh(system.H,k=3*L**2,which='SA',return_eigenvectors=True)
	else:
		system = Kag2.double_occupied_site_system(L=L,U=U,bc=bc,skip_diag=False,evolution_method='eigenvector',skip_k=False)
		evals = system.eigvals
		evects = system.eigvects
		
	#Order descending
	evals = evals[::-1]
	evects = evects[:,::-1]
	
	for idx in idxs:
		system.psi = evects[:,idx]
		opdm = system.calc_opdm()
		proj_opdm = system.opdm_fb_projection(opdm=opdm)
		
		fig,axs = plt.subplots(2,2)
		system.plot_density(fig=fig,ax=axs[0][0],plot_cbar=False,plot=False)
		system.plot_amplitude(fig=fig,ax=axs[0][1],plot_cbar=False,plot=False)
		system.plot_opdm(fig=fig,ax=axs[1][0],opdm=opdm,plot=False)
		system.plot_opdm(fig=fig,ax=axs[1][1],opdm=proj_opdm,plot=False)
		
		axs[0][0].set_title('Density')
		axs[0][1].set_title('Amplitude')
		axs[1][0].set_title('OPDM')
		axs[1][1].set_title('Projected OPDM')
		
		plt.suptitle(f'Index={idx}, E={np.round(evals[idx],4)}')
		
		plt.show()
		
		
def plot_tr_estate_proj_opdm(L=5,U=0.1,bc='periodic',descending=True,plot_energy=True,num=None):
	#For each of the eigenstates of a 2-particle kagome system, calculate the flat-band projected OPDM, take the trace and plot these traces
	#Shows the flat occupancy of each eigenstate
	#For weak interactions, expect some eigenstates have values of 2 (both particles flat band states), some have 1, some have 0
	#Strong interactions, bands get mixed, so expect to break down
	#For large L, only calculate flat band eigenvalues using sparse/Lanczos
	#For small L, just diagonalize Hamiltonian fully
	if L >= 6:
		system = Kag2.double_occupied_site_system(L=L,U=U,bc=bc,skip_diag=True,evolution_method='propagator',skip_k=False)
		evals,evects = sp.sparse.linalg.eigsh(system.H,k=num if num is not None else 3*L**2,which='SA',return_eigenvectors=True)
	else:
		system = Kag2.double_occupied_site_system(L=L,U=U,bc=bc,skip_diag=False,evolution_method='eigenvector',skip_k=False)
		evals = system.eigvals
		evects = system.eigvects
		
	if descending:
		evals = evals[::-1]
		evects = evects[:,::-1]
		
	if num is not None:
		evals = evals[:num]
		evects = evects[:,:num]
	
	tr_vals = np.zeros(np.size(evals))
	
	for i in range(np.size(evals)):
		system.psi = evects[:,i]
		opdm = system.calc_opdm()
		proj_opdm = system.opdm_fb_projection(opdm=opdm)
		tr_vals[i] += np.trace(proj_opdm)
		print(f'Progress: {np.round(100*(i+1)/np.size(evals),2)}%',end='\r')
	
	fig,ax = plt.subplots()
	ax.plot(np.arange(np.size(evals)),tr_vals,color='b',marker='x',ls='',label='Trace')
	ax.set_xlabel('Eigenstate Index')
	ax.set_ylabel('Tr(Projected OPDM)')
	ax.set_title('Eigenstate flat band character')
	
	if plot_energy:
		ax2 = ax.twinx()
		ax2.plot(np.arange(np.size(evals)),evals,color='r',marker='x',ls='',label='Energy')
		ax2.set_xlabel('Energy / J')
		
		handles = [matplotlib.lines.Line2D([0],[0],color='b',marker='x',ls='',label='Trace'),
				matplotlib.lines.Line2D([0],[0],color='r',marker='x',ls='',label='Energy')]
		ax.legend(handles=handles)
	
	plt.show()
	
	
def plot_fb_projection_fromfile(filename):
	#Plot flat band occupancy vs time from a file giving psi(t)
	data = np.load(filename)
	psi = data['psi']
	t = data['t']
	L = data['Lx']
	if L != data['Ly']:
		print('Warning: different Lx and Ly')
	system = Kag2.double_occupied_site_system(L=L,skip_diag=True,skip_k=False,evolution_method='eigenvector')
	
	proj_vals = np.zeros(np.size(t))
	for i in range(np.size(t)):
		proj_opdm = system.opdm_fb_projection(psi=psi[i,:])
		proj_vals[i] += np.real(np.trace(proj_opdm))
		print(f'Progress: {np.round(100*(i+1)/np.size(t))}%',end='\r')
	
	fig,ax = plt.subplots()
	ax.plot(t,proj_vals,color='b')
	
	ax.set_xlim(np.min(t),np.max(t))
	
	ax.set_xlabel(r'$t$ / $t_0$')
	ax.set_ylabel(r'$tr(\hat{P}\rho \hat{P}^{\dag})$')
	ax.set_title('OPDM Flat Band Projection vs t')
	
	plt.show()
	

def generate_trace_data(filename,U_vals,L=10,initial_state='hexagon',initial_site_idx=133,bc='periodic'):
	#Generate data of time-averaged flat band occupancy for different U
	means = np.zeros(np.size(U_vals))
	stdevs = np.zeros(np.size(U_vals))
	
	for i,U in enumerate(U_vals):
		print(f'Evaluating U = {U}...')
		if U <= 0.1:
			num_timesteps = 5
		else:
			num_timesteps = 10
		trace_vals = np.zeros(num_timesteps)
		if initial_state == 'hexagon':
			system = Kag2.two_hexagons_system(initial_site_idx,initial_site_idx,U=U,L=L,skip_diag=True,evolution_method='propagator',bc=bc,skip_k=False)
		elif initial_state == 'singlesite':
			system = Kag2.double_occupied_site_system(L=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc,initial_site_idx=initial_site_idx,skip_k=False)
		times = np.linspace(100,1000,num_timesteps)
		dt = times[1] - times[0]
		system.psi = system.state(times[0])
		trace_vals[0] = np.real(np.trace(system.opdm_fb_projection()))
		for j in range(1,num_timesteps):
			print(f'Progress: {np.round(100*j/num_timesteps,2)}%',end='\r')
			system.evolve_psi(dt)
			trace_vals[j] = np.real(np.trace(system.opdm_fb_projection()))
		print('Progress: 100.0%')
		means[i] = np.mean(trace_vals)
		stdevs[i] = np.std(trace_vals)
		print('')
		
	np.savez(filename,U_vals=U_vals,means=means,stdevs=stdevs,L=L,initial_state=initial_state,initial_site_idx=initial_site_idx,bc=bc)	
	

def plot_meantrace_data(filenames,ax=None,plot=False,logx=True,logy=False,plot_fit=None):
	#Plot data of time-averaged flat band occupancy for different U
	U_vals = np.array([])
	means = np.array([])
	stdevs = np.array([])
	for name in filenames:
		data = np.load(name)
		U_vals = np.concatenate((U_vals,data['U_vals']))
		means = np.concatenate((means,data['means']))
		stdevs = np.concatenate((stdevs,data['stdevs']))
		
	indices = np.argsort(U_vals)
	U_vals = U_vals[indices]
	means = means[indices]
	stdevs = stdevs[indices]
	
	if ax is None:
		fig,ax = plt.subplots()
	ax.errorbar(U_vals,means,yerr=stdevs,color='b',marker='x',ls='',label='Data')
	
	if logx:
		ax.set_xscale('log')
	if logy:
		ax.set_yscale('log')
		
	ax.set_xlabel(r'$U$ / $J$',fontsize=15)
	ax.set_ylabel(r'$\overline{N}_{FB}$',fontsize=15)
	#ax.set_title('Flat Band Character vs U')
	
	if plot:	
		plt.show()


def power_fit(x,a,n):
	return a*x**n
	
	
def linear_fit(x,a,b):
	return a + b*x
	
	
def plot_2minus_meantrace(filenames,ax=None,plot=False,logx=True,logy=True,plot_fit=None,fit_idx1=0,fit_idx2=1,printparams=True):
	#Plot 2 minus the flat-band occupancy, i.e. the 'non-flat-band occupancy', against U
	U_vals = np.array([])
	means = np.array([])
	stdevs = np.array([])
	for name in filenames:
		data = np.load(name)
		U_vals = np.concatenate((U_vals,data['U_vals']))
		means = np.concatenate((means,data['means']))
		stdevs = np.concatenate((stdevs,data['stdevs']))
		
	indices = np.argsort(U_vals)
	U_vals = U_vals[indices]
	means = means[indices]
	stdevs = stdevs[indices]
	
	if ax is None:
		fig,ax = plt.subplots()
	ax.errorbar(U_vals,2-means,yerr=stdevs,color='b',marker='x',ls='',label='Data')
	#ax.tick_params(labelsize=15)
	
	if logx:
		ax.set_xscale('log')
	if logy:
		ax.set_yscale('log')
		
	ax.set_xlabel(r'$U$ / $J$',fontsize=15)
	ax.set_ylabel(r'$2-\overline{N}_{FB}$',fontsize=15)
	#ax.set_title('Non-Flat Band Character vs U')
	
	#ax.set_xlim(right=0.2)
	
	if plot_fit == 'power':
		popt,pcov = sp.optimize.curve_fit(power_fit,U_vals[fit_idx1:fit_idx2+1],(2-means)[fit_idx1:fit_idx2+1])#,sigma=stdevs[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = power_fit(U_vals,popt[0],popt[1])
		ax.plot(U_vals,yfit,'r--',label='Fit')
		ax.legend(fontsize=15)
		if printparams:
			print('Fit y = A * U^n')
			print(f'A = {popt[0]} +/- {perr[0]}')
			print(f'n = {popt[1]} +/- {perr[1]}')
			
	
	elif plot_fit == 'linear_to_log':
		popt,pcov = sp.optimize.curve_fit(linear_fit,np.log(U_vals[fit_idx1:fit_idx2+1]),np.log((2-means)[fit_idx1:fit_idx2+1]))
		perr = np.sqrt(np.diag(pcov))
		A = np.exp(popt[0])
		A_err = 0.5*(np.exp(popt[0]+perr[0])+np.exp(popt[0]-perr[0]))
		yfit = power_fit(U_vals,A,popt[1])
		ax.plot(U_vals,yfit,'r--',label='Power Law Fit')
		ax.legend()
		if printparams:
			print('Fit y = A * U^n')
			print(f'A = {A} +/- {perr[0]}')
			print(f'n = {popt[1]} +/- {perr[1]}')
	
	if plot:	
		plt.show()

	
if __name__ == '__main__':
	#plot_projection_from_files(filenames=['N2_L5x5_SameHexagon_Periodic_T2000_dt20_J1_U0.1.npz'],colors=['b'])
	
	#plot_estate_projection(L=4,U=0.,which='top',num=60,method='kspace')
	
	#system = Kag2.two_hexagons_system(idx1=133,idx2=133,L=10,U=1,skip_diag=True,evolution_method='propagator',bc='periodic',skip_k=False)
	#system = Kag2.double_occupied_site_system(initial_site_idx=36,U=1.,L=5,skip_k=False,skip_diag=True,evolution_method='eigenvector',bc='periodic')
	
	
	#plot_OPDM_evals(system=system)
	
	#plot_OPDM_evects(system=system,idxs=[0,1])
	
	#compare_opdm_proj_opdm(system=system)
	
	#plot_estate_proj_opdm(L=5,idxs=[0])
	
	#plot_tr_estate_proj_opdm(L=5,plot_energy=True,num=75)
	
	#plot_fb_projection_fromfile(filename='N2_L5x5_SameHexagon_Periodic_T100_dt20_J1_U1000.npz')
	
	#generate_trace_data('MeanTrace_L10x10_SameHexagon_Periodic_U0.0017_0.056.npz',U_vals=[0.0017,0.0056,0.017,0.056],L=10,initial_site_idx=133)
	#generate_trace_data('MeanTrace_L10x10_SameHexagon_Periodic_U100_1000.npz',U_vals=np.logspace(2,3,6)[1:5],L=10,initial_site_idx=133)
	
	
	#plot_meantrace_data(['MeanTrace_L10x10_SameHexagon_Periodic_Combined_U0.001_1000.npz'],logy=True)
	
	plot_2minus_meantrace(['MeanTrace_L10x10_SameHexagon_Periodic_Combined_U0.001_1000.npz'],logy=True,plot_fit='linear_to_log',fit_idx2=12)
