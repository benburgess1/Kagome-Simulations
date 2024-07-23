'''
This file contains functions for generating and analyzing data relating
to the separation of particles
'''


import numpy as np
import matplotlib.pyplot as plt
import Kagome2 as Kag2
import scipy as sp


def plot_separation_from_psi_file(filenames,ax=None,colors=None,labels=None,plot=False):
	#Plot the expected separation between the two particles from a .npz data file containing
	#the full wavefunction psi at various times
	if colors == None:
		colors = plt.cm.rainbow(np.linspace(0,1,len(filenames)))
	
	#Create 'reference' system of correct size to enable use of exp_separation function
	#Don't need to calculate or diagonalize the Hamiltonian, or input an initial wavefunction
	data = np.load(filenames[0])
	L = data['Lx']
	system = Kag2.double_occupied_site_system(L=L,U=0,skip_H=True,skip_diag=True,evolution_method='eigenvector',skip_k=True)
	
	if ax is None:
		fig,ax = plt.subplots()
	
	for i,name in enumerate(filenames):
		data = np.load(name)
		t = data['t']
		psi_data = data['psi']
		d_vals = np.array([system.exp_separation(psi=psi_data[j,:]) for j in range(np.size(t))])
		if labels == 'auto':
			U = data['U']
			label = f'U={U}'
		elif labels is not None:
			label = labels[i]
		else:
			label = None
		ax.plot(t,d_vals,color=colors[i],label=label)
		
	ax.set_xlabel(r'$t$ / $t_0$')
	ax.set_ylabel(r'$\overline{\langle r \rangle}$ / $a$',rotation=0,labelpad=20,verticalalignment='center')
	ax.set_title('Expected particle separation vs time')
	ax.legend(fontsize=15,title=r'$U$ / $J$')
	
	ax.set_xlim(0,np.max(t))
	
	if plot:
		plt.show()
	
	
def generate_separation_data(filename,U_vals,L=10,initial_state='hexagon',initial_site_idx=133,bc='open',num_timesteps=100):
	#Generate data of time-averaged expected separation and standard deviations, for given U values 
	#Initial condition can be (doubly-occupied) flat band hexagon state, or doublon
	means = np.zeros(np.size(U_vals))
	stdevs = np.zeros(np.size(U_vals))
	
	for i,U in enumerate(U_vals):
		print(f'Evaluating U = {U}...')
		d_vals = np.zeros(num_timesteps)
		if initial_state == 'hexagon':
			system = Kag2.two_hexagons_system(initial_site_idx,initial_site_idx,U=U,L=L,skip_diag=True,evolution_method='propagator',bc=bc)
		elif initial_state == 'singlesite':
			system = Kag2.double_occupied_site_system(L=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc,initial_site_idx=initial_site_idx)
		if U < 1:
			b = 9.42 / U
			times = np.linspace(3*b,10*b,num_timesteps)
			dt = times[1] - times[0]
			system.psi = system.state(times[0])
			d_vals[0] = system.exp_separation()
			for j in range(1,num_timesteps):
				print(f'Progress: {np.round(100*j/num_timesteps,2)}%',end='\r')
				system.evolve_psi(dt)
				d_vals[j] = system.exp_separation()
			means[i] = np.mean(d_vals)
			stdevs[i] = np.std(d_vals)
			print('\n')
			
		else:
			times = np.linspace(100,500,num_timesteps)
			dt = times[1] - times[0]
			system.psi = system.state(times[0])
			d_vals[0] = system.exp_separation()
			for j in range(1,num_timesteps):
				print(f'Progress: {np.round(100*j/num_timesteps,2)}%',end='\r')
				system.evolve_psi(dt)
				d_vals[j] = system.exp_separation()
			means[i] = np.mean(d_vals)
			stdevs[i] = np.std(d_vals)
			print('\n')
			
	np.savez(filename,U_vals=U_vals,means=means,stdevs=stdevs,L=L,initial_state=initial_state,initial_site_idx=initial_site_idx,bc=bc)


def generate_separation_vs_t_data(filename,U_vals,times,L=10,initial_state='hexagon',initial_site_idx=133,bc='periodic'):
	#Generate data of expected separation against time, for given U values
	#Initial condition can be (doubly-occupied) flat band hexagon state, or doublon
	sep_vals = np.zeros((np.size(U_vals),np.size(times)))
	#dt = times[1] - times[0]
	
	for i,U in enumerate(U_vals):
		print(f'Evaluating U = {U}...')
		#d_vals = np.zeros(num_timesteps)
		if initial_state == 'hexagon':
			system = Kag2.two_hexagons_system(initial_site_idx,initial_site_idx,U=U,L=L,skip_diag=True,evolution_method='propagator',bc=bc)
		elif initial_state == 'singlesite':
			system = Kag2.double_occupied_site_system(L=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc,initial_site_idx=initial_site_idx)
		
		sep_vals[i,0] = system.exp_separation()
		print(f'Progress: {np.round(100/np.size(times),2)}%  ',end='\r')
		for j in range(1,np.size(times)):
			system.evolve_psi(times[j]-times[j-1])
			sep_vals[i,j] = system.exp_separation()
			print(f'Progress: {np.round(100*j/np.size(times),2)}%  ',end='\r')
		
		print('Progress: 100.0%     ')
			
	np.savez(filename,U_vals=U_vals,times=times,sep_vals=sep_vals,L=L,initial_state=initial_state,initial_site_idx=initial_site_idx,bc=bc)
				

def power_fit(x,a,n):
	return a*x**n


def plot_separation_data(filenames,ax=None,logx=True,logy=False,plot_fit=None,plot=False):
	#Plot time-averaged separation vs U from npz file
	U_vals = np.array([])
	means = np.array([])
	stdevs = np.array([])
	for name in filenames:
		data = np.load(name)
		U_vals = np.concatenate((U_vals,data['U_vals']))
		means = np.concatenate((means,data['means']))
		stdevs = np.concatenate((stdevs,data['stdevs']))
	
	if ax is None:
		fig,ax = plt.subplots()
	ax.errorbar(U_vals,means,yerr=stdevs,color='b',marker='x',ls='',label='Data')
	
	if logx:
		ax.set_xscale('log')
	if logy:
		ax.set_yscale('log')
	
	ax.set_ylim(bottom=0,top=2.5)
	ax.set_xlabel(r'$U$ / $J$',fontsize=15)
	ax.set_ylabel(r'$\overline{\langle r \rangle}$ / $a$',fontsize=15)
	#ax.set_title('Mean Separation vs U')
	
	if plot_fit == 'power':
		popt,pcov = sp.optimize.curve_fit(power_fit,U_vals[fit_idx1:fit_idx2+1],means[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = power_fit(U_vals,popt[0],popt[1])
		ax.plot(U_vals,yfit,'r--',label='Power Law Fit')
		ax.legend()
		if printparams:
			print('Fit y = A * U^n')
			print(f'A = {popt[0]} +/- {perr[0]}')
			print(f'n = {popt[1]} +/- {perr[1]}')
	if plot:		
		plt.show()
	
	
def plot_rdf_from_psi_file(filenames,time_idxs=[0],colors=None,labels='auto',bc='open'):
	#Plot the expected separation between the two particles from a .npz data file containing
	#the full wavefunction psi at various times
	if colors == None:
		colors = plt.cm.rainbow(np.linspace(0,1,len(filenames)))
	
	#Create 'reference' system of correct size to enable use of exp_separation function
	#Don't need to calculate or diagonalize the Hamiltonian, or input an initial wavefunction
	data = np.load(filenames[2])
	L = data['Lx']
	if 'bc' in data.keys():
		bc = data['bc']
	else:
		print('Warning: system bc not specified. Assumed open by default.')
	system = Kag2.double_occupied_site_system(L=L,U=0,skip_H=True,skip_diag=True,evolution_method='eigenvector',bc=bc)
	
	fig,axs = plt.subplots(1,len(time_idxs),sharey=True)
	
	for i,name in enumerate(filenames):
		data = np.load(name)
		t = data['t']
		psi_data = data['psi']
	
		if labels == 'auto':
			U = data['U']
			label = f'U={U}'
		elif labels is not None:
			label = labels[i]
		else:
			label = None
		for j,idx in enumerate(time_idxs):
			r,g = system.rdf(psi=psi_data[idx,:])
			ax = axs[j] if len(time_idxs)>1 else axs
			ax.plot(r,g,color=colors[i],label=label,marker='x',ls='')
			ax.set_title(f't = {np.round(t[idx],2)}')
			ax.set_xlabel(r'$r$')
			if j == 0:
				ax.set_ylabel(r'$g(r)$',rotation=0)
				ax.legend()
	
	plt.show()
	
	
def generate_rdf_data(filename,U_vals,L=10,initial_state='hexagon',initial_site_idx1=133,initial_site_idx2=133,bc='open',num_timesteps=100):
	#Generate data for time-averaged RDF and standard deviations for given U values
	#Initial condition can be (doubly-occupied) flat band hexagon state, or doublon
	system = Kag2.double_occupied_site_system(L=L,U=0,skip_H=True,skip_diag=True,evolution_method='eigenvector',bc=bc)
	r,g,counts = system.rdf(return_counts=True)
	num_r = len(r)
	
	means = np.zeros((np.size(U_vals),num_r))
	stdevs = np.zeros((np.size(U_vals),num_r))
	
	for i,U in enumerate(U_vals):
		print(f'Evaluating U = {U}...')
		g_vals = np.zeros((num_timesteps,num_r))
		
		if initial_state == 'hexagon':
			system = Kag2.two_hexagons_system(initial_site_idx1,initial_site_idx1,U=U,L=L,skip_diag=True,evolution_method='propagator',bc=bc)
		elif initial_state == 'singlesite':
			system = Kag2.double_occupied_site_system(L=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc,initial_site_idx=initial_site_idx1)
		elif initial_state == 'separatesites':
			system = Kag2.single_state_system(L=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc,initial_site_idx1=initial_site_idx1,initial_site_idx2=initial_site_idx2)
	
		times = np.linspace(200,500,num_timesteps)
		dt = times[1] - times[0]
		system.psi = system.state(times[0])
		g_vals[0,:] = system.rdf(return_r=False)
		for j in range(1,num_timesteps):
			print(f'Progress: {np.round(100*j/num_timesteps,2)}%',end='\r')
			system.evolve_psi(dt)
			g_vals[j,:] = system.rdf(return_r=False)
		means[i,:] = np.mean(g_vals,axis=0)
		stdevs[i,:] = np.std(g_vals,axis=0)
		print('Progress: 100.0%')
			
	np.savez(filename,U_vals=U_vals,r=r,counts=counts,means=means,stdevs=stdevs,L=L,initial_state=initial_state,initial_site_idx1=initial_site_idx1,initial_site_idx2=initial_site_idx2,bc=bc)


def plot_rdf_data(filename,ax=None,plot=False,colors=None,labels='auto',normalize_by_count=False,unit=1,markersize=5):
	#Plot time-averaged RDF data for each of the U values in the npz data file
	data = np.load(filename)
	U_vals = data['U_vals']#[::-1]
	r = data['r']
	means = data['means']
	stdevs = data['stdevs']
	
	if normalize_by_count:
		counts = data['counts']
		means = np.divide(means,counts)
		stdevs = np.divide(means,counts)
	
	if colors == None:
		colors = plt.cm.rainbow(np.linspace(0,1,len(U_vals)))
	
	if ax is None:	
		fig,ax = plt.subplots()
	for i,U in enumerate(U_vals):
		#i = np.size(U_vals) - j - 1
		if labels == 'auto':
			label = f'{U}'
		elif len(labels) == len(U_vals):
			label = labels[i]
		else:
			label = None
		ax.errorbar(r,means[i,:]/unit,yerr=stdevs[i,:]/unit,color=colors[i],marker='x',ls='',label=label,markersize=markersize)
		
	ax.set_xlabel(r'$r$ / $a$',fontsize=9)
	
	if normalize_by_count:
		#ax.set_title('Normalized Site Density Correlations, t=200-500')
		#ax.set_ylabel(r'$\frac{g(r)}{N(r)}$',rotation=0,fontsize=15,labelpad=20)
		ax.set_ylabel(r'$g(r)$',rotation=0,fontsize=9,labelpad=15)
		L = data['L']
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		ax.axhline(1/(N_states*unit),color='k',ls=':',label='No Correlations')
	else:
		ax.set_title('Site Density Correlations, t=200-500')
		ax.set_ylabel(r'$g(r)$',rotation=0)
		
	#ax.legend()
	if plot:
		plt.show()

	
if __name__ == '__main__':
	#plot_separation_from_psi_file(filenames=['N2_L10x10_SameHexagon_Periodic_T1000_dt20_J1_U0.1.npz','N2_L10x10_SameHexagon_Periodic_T1000_dt20_J1_U1.npz','N2_L10x10_SameHexagon_Periodic_T1000_dt20_J1_U10.npz','N2_L10x10_SameHexagon_Periodic_T1000_dt20_J1_U100.npz'],colors=['b','cyan','r','darkred'],labels='auto',plot=True)
	#plot_separation_from_psi_file(filenames=['N2_L10x10_SameHexagon_Periodic_T2000_dt10_J1_U0.1.npz'],colors=['b','cyan','r','darkred'],labels='auto',plot=True)
	#plot_separation_from_psi_file(filenames=['N2_L10x10_SameHexagon_Periodic_T1000_dt10_J1_U1.npz'],colors='b',plot=True)
	#generate_separation_data('Separation_L10x10_SameHexagon_Periodic_U10_1000.npz',U_vals=np.logspace(1,3,10),bc='periodic')
	#generate_separation_data('Separation_L10x10_SameHexagon_Periodic_U1_10.npz',U_vals=np.logspace(0,1,10)[1:9],bc='periodic')
	#generate_separation_data('Separation_L10x10_SameHexagon_Periodic_U0.001_1.npz',U_vals=np.logspace(-3,0,7),bc='periodic')
	
	generate_separation_vs_t_data('Separation_Vs_t_L10x10_SameHexagon_Periodic_U0.1_100.npz',U_vals=[0.1,1,5,10,100],times=np.arange(0,500.1,1))
	
	#generate_separation_data('Separation_Test_Open.npz',U_vals=[0.1],num_timesteps=3,bc='open')
	
	#plot_separation_data(filenames=['Separation_L10x10_SameHexagon_Periodic_U0.001_1.npz','Separation_L10x10_SameHexagon_Periodic_U1_10.npz','Separation_L10x10_SameHexagon_Periodic_U10_1000.npz'],logy=False)
	
	#plot_rdf_from_psi_file(filenames=['N2_L10x10_SameHexagon_Open_T1000_dt20_J1_U0.1.npz','N2_L10x10_SameHexagon_Open_T1000_dt20_J1_U1.npz','N2_L10x10_SameHexagon_Open_T1000_dt20_J1_U10.npz','N2_L10x10_SameHexagon_Open_T1000_dt20_J1_U100.npz'],time_idxs=[25])
	
	#generate_rdf_data('RDF_L10x10_SeparateSites_Periodic_U0.1_100.npz',U_vals=[0.1,1,10,100],bc='periodic',initial_state='separatesites',L=10,initial_site_idx1=103,initial_site_idx2=163)
	#plot_rdf_data('RDF_L10x10_SeparateSites_Periodic_U0.1_100.npz',colors=None,normalize_by_count=True)
