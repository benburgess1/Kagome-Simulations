'''
This file contains the code for timing different functions, and plotting the results.
'''


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import timeit
import functools
import Kagome2 as Kag2


#Compare diagonalization drivers.

def time_driver(H,driver,repeat):
	#Return mean and standard deviation of timing diagonalization with a given driver
	time_vals = timeit.repeat(functools.partial(sp.linalg.eigh,H,driver=driver),repeat=repeat,number=1)
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	return mean,stdev
	

def LxLy_system(Lx,Ly,U=1.,skip_diag=True,psi_0='auto',evolution_method='propagator'):
	#Returns a Lx x Ly, 2-particle system. Initial wavefunction is optionally a single occupied site, 
	#all states equally occupied, or a random occupancy of all states.
	N_sites = 3*Lx*Ly
	N_states = int(0.5*N_sites*(N_sites+1))
	if psi_0 == 'auto':
		psi_0 = np.zeros(N_states)
		psi_0[int(N_states/2)] = 1.
	elif psi_0 == 'ones':
		psi_0 = np.ones(N_states)/np.sqrt(N_states)
	elif psi_0 == 'random':
		psi_0 = np.random.rand(N_states)
		psi_0 = psi_0/np.linalg.norm(psi_0)
	system = Kag2.Kagome2(psi_0=psi_0,Lx=Lx,Ly=Ly,U=U,skip_diag=skip_diag,evolution_method=evolution_method)
	return system


def generate_driver_data(Lx_vals,Ly_vals,repeats,driver,filename):
	#Generate driver timing data for a series of different system sizes.
	N = np.size(Lx_vals)
	if np.size(Ly_vals) != N:
		print('Warning: Lx and Ly arrays of different size')
	N_sites_vals = 3*np.multiply(Lx_vals,Ly_vals)
	N_states_vals = 0.5*np.multiply(N_sites_vals,N_sites_vals+1)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	for i in range(N):
		Lx = Lx_vals[i]
		Ly = Ly_vals[i]
		print(f'Evaluating Lx = {Lx}, Ly = {Ly}...')
		system = LxLy_system(Lx,Ly,skip_diag=True)
		means[i],stdevs[i] = time_driver(system.H,driver,repeats[i])
		
	np.savez(filename,driver=driver,Lx_vals=Lx_vals,Ly_vals=Ly_vals,N_sites_vals=N_sites_vals,N_states_vals=N_states_vals,means=means,stdevs=stdevs)
		
		
def plot_driver_data(filename,ax,color='b',marker='x',ls='',label=None):
	#Plot driver timing data against number of basis states, from a npz file
	data = np.load(filename)
	means = data['means']
	stdevs = data['stdevs']
	N_states_vals = data['N_states_vals']
	if label == 'auto':
		label = data['driver']
	ax.errorbar(N_states_vals,means,yerr=stdevs,color=color,marker=marker,ls=ls,label=label)
	
	
def compare_drivers(filenames):
	#Compare various driver times on the same graph
	colors = plt.cm.rainbow(np.linspace(0,1,len(filenames)))
	fig,ax = plt.subplots()
	for i,name in enumerate(filenames):
		plot_driver_data(name,ax,color=colors[i],marker='x',ls='-')
		
	x = np.linspace(300,11000,100)
	A = 8e-11
	y = A*x**3
	
	ax.plot(x,y,color='k',ls=':',label=r'$\mathcal{O}(N^3)$')
	
	
	#Add L axis
	f = lambda x: (x/4.5)**0.25
	g = lambda x: 4.5*x**4
	ax2 = ax.secondary_xaxis('top',functions=(f,g))
	
	ax.set_xlabel(r'$N_{states}$')
	ax.set_ylabel('Time / s')
	ax2.set_xlabel('L')
	
	#ax2.set_xticklabels([3,4,5,6,7])
	ax.set_title('Diagonalization Time')
	ax.legend(title='Driver')
	
	ax2.set_xscale('log')
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax2.set_xticks(ticks=[3,4,5,6,7],labels=['3','4','5','6','7'])
	plt.show()
	
	
def cubic(x,a):
	#Cubic fit
	return a*x**3
	

def fit_cubic(filename,fit_idx1,printparams=True):
	#Fit a cubic relationship to given timing data
	data = np.load(filename)
	means = data['means']
	stdevs = data['stdevs']
	N_states_vals = data['N_states_vals']
	
	popt,pcov = sp.optimize.curve_fit(cubic,N_states_vals[fit_idx1:],means[fit_idx1:])
	perr = np.sqrt(np.diag(pcov))
	if printparams:
		print('Fit t = a * N^3')
		print(f'a = {popt[0]} +/- {perr[0]}')
	
	fig,ax = plt.subplots()
	ax.errorbar(N_states_vals,means,yerr=stdevs,color='b',marker='x',ls='',label='evd Data')
	
	x_fit = np.linspace(300,11e3,100)
	y_fit = cubic(x_fit,popt[0])
	
	ax.plot(x_fit,y_fit,color='r',ls=':',label='Cubic Fit')
	
	#Add L axis
	f = lambda x: (x/4.5)**0.25
	g = lambda x: 4.5*x**4
	ax2 = ax.secondary_xaxis('top',functions=(f,g))
	
	ax.set_xlabel(r'$N_{states}$')
	ax.set_ylabel('Time / s')
	ax2.set_xlabel('L')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.set_title('evd Diagonalization Time')
	
	ax.legend()
	plt.show()


def time_state(system,repeat,t=1.):
	#Returns mean and standard deviation of timing the state() function
	time_vals = timeit.repeat(functools.partial(system.state,t=t),repeat=repeat,number=1)
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	return mean,stdev
	

def generate_state_data(Lx_vals,Ly_vals,repeats,filename):
	#Generates timing data for state() function of various system sizes
	N = np.size(Lx_vals)
	if np.size(Ly_vals) != N:
		print('Warning: Lx and Ly arrays of different size')
	N_sites_vals = 3*np.multiply(Lx_vals,Ly_vals)
	N_states_vals = 0.5*np.multiply(N_sites_vals,N_sites_vals+1)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	for i in range(N):
		Lx = Lx_vals[i]
		Ly = Ly_vals[i]
		print(f'Evaluating Lx = {Lx}, Ly = {Ly}...')
		system = LxLy_system(Lx,Ly,skip_diag=False,psi_0='auto')
		means[i],stdevs[i] = time_state(system,repeats[i])
		
	np.savez(filename,Lx_vals=Lx_vals,Ly_vals=Ly_vals,N_sites_vals=N_sites_vals,N_states_vals=N_states_vals,means=means,stdevs=stdevs)
	
	
def propagate(psi,H,t=1.,hbar=1.):
	#Defines a test propagator function, without needing to create a full system.
	return sp.sparse.linalg.expm_multiply(-(1j*t/hbar)*H,psi)
	
	
def time_propagator(H,t,repeat):
	#Returns mean and standard deviation of propagator function
	N_states = H.shape[0]
	psi = np.zeros(N_states)
	psi[int(N_states/2)] = 1.
	time_vals = timeit.repeat(functools.partial(propagate,psi,H,t=t),repeat=repeat,number=1)
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	return mean,stdev
	
	
def generate_propagator_data(Lx_vals,Ly_vals,t,repeats,filename):
	#Generates propagator timing data for various system sizes
	N = np.size(Lx_vals)
	if np.size(Ly_vals) != N:
		print('Warning: Lx and Ly arrays of different size')
	N_sites_vals = 3*np.multiply(Lx_vals,Ly_vals)
	N_states_vals = 0.5*np.multiply(N_sites_vals,N_sites_vals+1)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	for i in range(N):
		Lx = Lx_vals[i]
		Ly = Ly_vals[i]
		print(f'Evaluating Lx = {Lx}, Ly = {Ly}...')
		system = LxLy_system(Lx,Ly,skip_diag=True,psi_0='random')
		H = sp.sparse.csc_array(system.H)
		means[i],stdevs[i] = time_propagator(H,t,repeats[i])
		
	np.savez(filename,t=t,Lx_vals=Lx_vals,Ly_vals=Ly_vals,N_sites_vals=N_sites_vals,N_states_vals=N_states_vals,means=means,stdevs=stdevs)
	
	
def propagate2(psi,H,t=1.,hbar=1.):
	#Alternative propagator routine, where expm() is calculated separately before multiplying
	#with state vector, instead of using expm_multiply()
	U = sp.sparse.linalg.expm(-(1j*t/hbar)*H)
	return U @ psi.T
	
	
def time_propagator2(H,repeat):
	#Return mean and standard deviation of timing alternative propagator routine
	N_states = H.shape[0]
	psi = np.zeros(N_states)
	psi[int(N_states/2)] = 1.
	time_vals = timeit.repeat(functools.partial(propagate2,psi,H),repeat=repeat,number=1)
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	return mean,stdev
	
	
def generate_propagator_data2(Lx_vals,Ly_vals,repeats,filename):
	#Generates timing data for alternative propagator routine for various system sizes
	N = np.size(Lx_vals)
	if np.size(Ly_vals) != N:
		print('Warning: Lx and Ly arrays of different size')
	N_sites_vals = 3*np.multiply(Lx_vals,Ly_vals)
	N_states_vals = 0.5*np.multiply(N_sites_vals,N_sites_vals+1)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	for i in range(N):
		Lx = Lx_vals[i]
		Ly = Ly_vals[i]
		print(f'Evaluating Lx = {Lx}, Ly = {Ly}...')
		system = LxLy_system(Lx,Ly,skip_diag=True)
		H = sp.sparse.csc_array(system.H)
		means[i],stdevs[i] = time_propagator2(H,repeats[i])
		
	np.savez(filename,Lx_vals=Lx_vals,Ly_vals=Ly_vals,N_sites_vals=N_sites_vals,N_states_vals=N_states_vals,means=means,stdevs=stdevs)


def compare_state_propagator(state_file,prop1_file,prop10_file,prop100_file):
	#Compares various evolution method times on the same graph
	state_data = np.load(state_file)
	state_means = state_data['means']
	state_stdevs = state_data['stdevs']
	state_N_states_vals = state_data['N_states_vals']
	
	prop1_data = np.load(prop1_file)
	prop1_means = prop1_data['means']
	prop1_stdevs = prop1_data['stdevs']
	prop1_N_states_vals = prop1_data['N_states_vals']
	
	prop10_data = np.load(prop10_file)
	prop10_means = prop10_data['means']
	prop10_stdevs = prop10_data['stdevs']
	prop10_N_states_vals = prop10_data['N_states_vals']
	
	prop100_data = np.load(prop100_file)
	prop100_means = prop100_data['means']
	prop100_stdevs = prop100_data['stdevs']
	prop100_N_states_vals = prop100_data['N_states_vals']
	
	fig,ax = plt.subplots()
	ax.errorbar(state_N_states_vals,state_means,yerr=state_stdevs,color='b',marker='x',ls='-',label='Eigenstate Decomposition')
	ax.errorbar(prop1_N_states_vals,prop1_means,yerr=prop1_stdevs,color='r',marker='x',ls='-',label='Propagator t=1')
	ax.errorbar(prop10_N_states_vals,prop10_means,yerr=prop10_stdevs,color='cyan',marker='x',ls='-',label='Propagator t=10')
	ax.errorbar(prop100_N_states_vals,prop100_means,yerr=prop100_stdevs,color='g',marker='x',ls='-',label='Propagator t=100')

	
	x = np.linspace(300,11000,100)
	A = 5e-9
	y2 = A*x**2
	
	ax.plot(x,y2,color='k',ls=':',label=r'$\mathcal{O}(N^2)$')
	

	B = 1e-6
	y = B*x
	
	ax.plot(x,y,color='k',ls='--',label=r'$\mathcal{O}(N)$')
	
	#Add L axis
	f = lambda x: (x/4.5)**0.25
	g = lambda x: 4.5*x**4
	ax2 = ax.secondary_xaxis('top',functions=(f,g))
	
	ax.set_xlabel(r'$N_{states}$')
	ax.set_ylabel('Time / s')
	ax2.set_xlabel('L')
	
	ax.set_title('Time-Evolution Comparison')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.legend()
	
	plt.show()


def generate_propagator_vs_t_data(t_vals,repeats,Lx,Ly,filename):
	#Generates propagator timing data for different evolution time increments
	N = np.size(t_vals)
	system = LxLy_system(Lx,Ly,skip_diag=True)
	H = sp.sparse.csr_array(system.H)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	for i,t in enumerate(t_vals):
		print(f'Evaluating t = {t}...')
		means[i],stdevs[i] = time_propagator(H,t,repeats[i])
		
	np.savez(filename,t_vals=t_vals,Lx=Lx,Ly=Ly,N_sites=system.N_sites,N_states=system.N,means=means,stdevs=stdevs)
	

def generate_state_vs_t_data(t_vals,repeats,Lx,Ly,filename):
	#Generates (eigenvector) state() timing data for different evolution time increments (expected to be independent)
	N = np.size(t_vals)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	system = LxLy_system(Lx,Ly,skip_diag=False,evolution_method='eigenvector')
	
	for i,t in enumerate(t_vals):
		print(f'Evaluating t = {t}...')
		means[i],stdevs[i] = time_state(system,repeats[i],t=t)
		
	np.savez(filename,t_vals=t_vals,Lx=Lx,Ly=Ly,N_sites=system.N_sites,N_states=system.N,means=means,stdevs=stdevs)


def compare_t_data(state_file,prop_file):
	#Compares timing data for propagator and eigenvector state() evolution methods against evolution time increment
	state_data = np.load(state_file)
	state_means = state_data['means']
	state_stdevs = state_data['stdevs']
	state_t_vals = state_data['t_vals']
	
	prop_data = np.load(prop_file)
	prop_means = prop_data['means']
	prop_stdevs = prop_data['stdevs']
	prop_t_vals = prop_data['t_vals']
	
	fig,ax = plt.subplots()
	ax.errorbar(state_t_vals,state_means,yerr=state_stdevs,color='b',marker='x',ls='-',label='Eigenstate Decomposition')
	ax.errorbar(prop_t_vals,prop_means,yerr=prop_stdevs,color='r',marker='x',ls='-',label='Propagator')
	
	x = np.linspace(1,1000,100)
	B = 4e-3
	y = B*x
	
	ax.plot(x,y,color='k',ls='--',label=r'$\mathcal{O}(N)$')
	
	ax.set_xlabel('t / s')
	ax.set_ylabel('Time / s')
	
	ax.set_title('Comparison of different evolution times t')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.legend()
	
	plt.show()
	

def quadratic(x,a):
	return a*x**2


def fit_quadratic(filename,fit_idx1,printparams=True):
	#Fits quadratic to eigenvector state() timing data
	data = np.load(filename)
	means = data['means']
	stdevs = data['stdevs']
	N_states_vals = data['N_states_vals']
	
	popt,pcov = sp.optimize.curve_fit(quadratic,N_states_vals[fit_idx1:],means[fit_idx1:])
	perr = np.sqrt(np.diag(pcov))
	if printparams:
		print('Fit t = a * N^2')
		print(f'a = {popt[0]} +/- {perr[0]}')
	
	fig,ax = plt.subplots()
	ax.errorbar(N_states_vals,means,yerr=stdevs,color='b',marker='x',ls='',label=r'$t_{eig}$ Data')
	
	x_fit = np.linspace(300,11e3,100)
	y_fit = quadratic(x_fit,popt[0])
	
	ax.plot(x_fit,y_fit,color='r',ls=':',label='Quadratic Fit')
	
	#Add L axis
	f = lambda x: (x/4.5)**0.25
	g = lambda x: 4.5*x**4
	ax2 = ax.secondary_xaxis('top',functions=(f,g))
	
	ax.set_xlabel(r'$N_{states}$')
	ax.set_ylabel('Time / s')
	ax2.set_xlabel('L')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.set_title('Eigenvalue Decomposition Evolution Time')
	
	ax.legend()
	plt.show()
	
	
def linear(x,a):
	return x*a
	
	
def fit_linear(filename,fit_idx1,printparams=True):
	#Performs linear fit to propagator timing data
	data = np.load(filename)
	means = data['means']
	stdevs = data['stdevs']
	N_states_vals = data['N_states_vals']
	
	popt,pcov = sp.optimize.curve_fit(linear,N_states_vals[fit_idx1:],means[fit_idx1:])
	perr = np.sqrt(np.diag(pcov))
	if printparams:
		print('Fit t = a * N')
		print(f'a = {popt[0]} +/- {perr[0]}')
	
	fig,ax = plt.subplots()
	ax.errorbar(N_states_vals,means,yerr=stdevs,color='b',marker='x',ls='',label=r'$t_{prop}$ Data (t=1)')
	
	x_fit = np.linspace(300,70e3,100)
	y_fit = linear(x_fit,popt[0])
	
	ax.plot(x_fit,y_fit,color='r',ls=':',label='Linear Fit')
	
	#Add L axis
	f = lambda x: (x/4.5)**0.25
	g = lambda x: 4.5*x**4
	ax2 = ax.secondary_xaxis('top',functions=(f,g))
	
	ax.set_xlabel(r'$N_{states}$')
	ax.set_ylabel('Time / s')
	ax2.set_xlabel('L')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.set_title('Propagator Evolution Time, Uniform State')
	
	ax.legend()
	plt.show()
	
	
def fit_linear_t(filename,fit_idx1,printparams=True):
	#Performs linear fit to propagator timing data vs evolution time increment
	data = np.load(filename)
	means = data['means']
	stdevs = data['stdevs']
	t_vals = data['t_vals']
	
	popt,pcov = sp.optimize.curve_fit(linear,t_vals[fit_idx1:],means[fit_idx1:])
	perr = np.sqrt(np.diag(pcov))
	if printparams:
		print('Fit time = a * t')
		print(f'a = {popt[0]} +/- {perr[0]}')
	
	fig,ax = plt.subplots()
	ax.errorbar(t_vals,means,yerr=stdevs,color='b',marker='x',ls='',label=r'$t_{prop}$ Data (6x6)')
	
	x_fit = np.linspace(1,1000,100)
	y_fit = linear(x_fit,popt[0])
	
	ax.plot(x_fit,y_fit,color='r',ls=':',label='Linear Fit')
	
	ax.set_xlabel(r't / s')
	ax.set_ylabel('Time / s')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.set_title('Propagator Evolution Time')
	
	ax.legend()
	plt.show()
	

def time_eigsh(H,k,repeat):
	#Time sparse diagonalization to find k eigenvectors
	time_vals = timeit.repeat(functools.partial(sp.sparse.linalg.eigsh,H,k=k),repeat=repeat,number=1)
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	return mean,stdev
	


def generate_eigsh_data(Lx_vals,Ly_vals,repeats,k,filename):
	#Generate sparse diagonalization timing data for different system sizes
	N = np.size(Lx_vals)
	if np.size(Ly_vals) != N:
		print('Warning: Lx and Ly arrays of different size')
	N_sites_vals = 3*np.multiply(Lx_vals,Ly_vals)
	N_states_vals = 0.5*np.multiply(N_sites_vals,N_sites_vals+1)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	for i in range(N):
		Lx = Lx_vals[i]
		Ly = Ly_vals[i]
		print(f'Evaluating Lx = {Lx}, Ly = {Ly}...')
		system = LxLy_system(Lx,Ly,skip_diag=True)
		H = sp.sparse.csr_array(system.H)
		means[i],stdevs[i] = time_eigsh(H,N_states_vals[i]-1 if k=='all' else k,repeats[i])
		
	np.savez(filename,k=k,Lx_vals=Lx_vals,Ly_vals=Ly_vals,N_sites_vals=N_sites_vals,N_states_vals=N_states_vals,means=means,stdevs=stdevs)


def plot_eigsh_data(filename,ax,color='b',marker='x',ls=''):
	#Plot sparse diagonalization timing data
	data = np.load(filename)
	means = data['means']
	stdevs = data['stdevs']
	N_states_vals = data['N_states_vals']
	k = data['k']
	ax.errorbar(N_states_vals,means,yerr=stdevs,color=color,marker=marker,ls=ls,label=f'k={k}')
	
	
def compare_k(filenames):
	#Compare sparse diagonalization time for different k
	colors = plt.cm.rainbow(np.linspace(0,1,len(filenames)))
	fig,ax = plt.subplots()
	for i,name in enumerate(filenames):
		plot_eigsh_data(name,ax,color=colors[i],marker='x',ls='-')
		
	x = np.linspace(300,5e4,100)
	A = 2e-10
	y2 = A*x**2
	
	ax.plot(x,y2,color='k',ls=':',label=r'$\mathcal{O}(N^2)$')
	
	x3 = np.linspace(300,3000,100)
	B = 1e-9
	y3 = B*x3**3
	
	ax.plot(x3,y3,color='k',ls='--',label=r'$\mathcal{O}(N^3)$')
	
	
	#Add L axis
	f = lambda x: (x/4.5)**0.25
	g = lambda x: 4.5*x**4
	ax2 = ax.secondary_xaxis('top',functions=(f,g))
	
	ax.set_xlabel(r'$N_{states}$')
	ax.set_ylabel('Time / s')
	ax2.set_xlabel('L')
	
	ax.set_title('Time to find k Eigenvectors')

	ax2.set_xscale('log')
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.legend()

	plt.show()
	
	
def generate_eigsh_vs_k_data(k_vals,repeats,filename,Lx=8,Ly=8):
	#Generate timing data for diagonalization time vs number of eigenvalues k
	N = np.size(k_vals)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	system = LxLy_system(Lx,Ly,skip_diag=True)
	H = sp.sparse.csr_array(system.H)
	for i,k in enumerate(k_vals):
		print(f'Evaluating k = {k}...')
		means[i],stdevs[i] = time_eigsh(H,k,repeats[i])
	np.savez(filename,k_vals=k_vals,Lx=Lx,Ly=Ly,N_sites=system.N_sites,N_states=system.N,means=means,stdevs=stdevs)
	
	
def eigsh_vs_k(filename):
	#Plot timing data for diagonalization time vs k
	data = np.load(filename)
	k_vals = data['k_vals']
	means = data['means']
	stdevs = data['stdevs']
	Lx = data['Lx']
	Ly = data['Ly']
	
	fig,ax = plt.subplots()
	
	ax.errorbar(k_vals,means,yerr=stdevs,color='b',marker='x',label='Data')
	
	x = np.linspace(10,300,100)
	A = 3e-4
	y2 = A*x**2
	
	ax.plot(x,y2,color='k',ls=':',label=r'$\mathcal{O}(N^2)$')
	
	ax.set_xlabel('k')
	ax.set_ylabel('Time / s')
	ax.set_title(f'Time to find k eigenvectors, {Lx}x{Ly} System')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.legend()
	
	plt.show()
	
	
def time_dens(system,repeat):
	#Time density() function
	time_vals = timeit.repeat(functools.partial(system.density),repeat=repeat,number=1)
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	return mean,stdev
	

def generate_dens_data(Lx_vals,Ly_vals,repeats,filename):
	#Generate timing data for density() function for various system sizes
	N = np.size(Lx_vals)
	if np.size(Ly_vals) != N:
		print('Warning: Lx and Ly arrays of different size')
	N_sites_vals = 3*np.multiply(Lx_vals,Ly_vals)
	N_states_vals = 0.5*np.multiply(N_sites_vals,N_sites_vals+1)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	for i in range(N):
		Lx = Lx_vals[i]
		Ly = Ly_vals[i]
		print(f'Evaluating Lx = {Lx}, Ly = {Ly}...')
		system = LxLy_system(Lx,Ly,psi_0='random',skip_diag=True,evolution_method='eigenvector')
		means[i],stdevs[i] = time_dens(system,repeats[i])
		
	np.savez(filename,Lx_vals=Lx_vals,Ly_vals=Ly_vals,N_sites_vals=N_sites_vals,N_states_vals=N_states_vals,means=means,stdevs=stdevs)


def plot_dens_data(filename,color='b',marker='x',ls='-'):
	#Plot density timing data
	data = np.load(filename)
	means = data['means']
	stdevs = data['stdevs']
	N_states_vals = data['N_states_vals']
	fig,ax = plt.subplots()
	ax.errorbar(N_states_vals,means,yerr=stdevs,color=color,marker=marker,ls=ls,label='Density Time')
	ax.set_ylabel('Time / s')
	ax.set_xlabel('N')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	A = 4e-6
	x = np.linspace(np.min(N_states_vals),np.max(N_states_vals),100)
	y = A*x
	ax.plot(x,y,color='k',ls=':',label=r'$\mathcal{O}(N)$')
	
	#Add L axis
	f = lambda x: (x/4.5)**0.25
	g = lambda x: 4.5*x**4
	ax2 = ax.secondary_xaxis('top',functions=(f,g))
	ax2.set_xlabel('L')
	
	ax.legend()
	
	plt.show()
	
	
def generate_propagator_vs_U_data(U_vals,repeats,Lx,Ly,filename):
	#Generates propagator timing data for systems with different interaction strengths
	N = np.size(U_vals)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	for i,U in enumerate(U_vals):
		print(f'Evaluating U = {U}...')
		system = LxLy_system(Lx,Ly,U=U,skip_diag=True)
		H = sp.sparse.csr_array(system.H)
		means[i],stdevs[i] = time_propagator(H,1,repeats[i])
		
	np.savez(filename,U_vals=U_vals,Lx=Lx,Ly=Ly,N_sites=system.N_sites,N_states=system.N,means=means,stdevs=stdevs)
	
	
def plot_propagator_vs_U_data(filename):
	#Plots data of propagator evolution time vs U
	data = np.load(filename)
	U_vals = data['U_vals']
	means = data['means']
	stdevs = data['stdevs']
	
	fig,ax = plt.subplots()
	ax.errorbar(U_vals,means,yerr=stdevs,color='b',ls='',marker='x',label='Propagator')
	
	ax.set_xlabel('U')
	ax.set_ylabel('Time / s')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	a = 1e-3
	y = a*U_vals
	
	ax.plot(U_vals,y,color='r',ls='--',label=r'$\mathcal{O}(U)$')
	
	ax.legend()
	
	plt.show()
	
	
def generate_diag_vs_U_data(U_vals,repeats,Lx,Ly,filename):
	#Generates diagonalization timing data for systems with different interaction strengths
	N = np.size(U_vals)
	means = np.zeros(N)
	stdevs = np.zeros(N)
	for i,U in enumerate(U_vals):
		print(f'Evaluating U = {U}...')
		system = LxLy_system(Lx,Ly,U=U,skip_diag=True,evolution_method='eigenvector')
		means[i],stdevs[i] = time_driver(system.H,'evd',repeats[i])
		
	np.savez(filename,U_vals=U_vals,Lx=Lx,Ly=Ly,N_sites=system.N_sites,N_states=system.N,means=means,stdevs=stdevs)
	
	

def plot_diag_vs_U_data(filename):
	#Plots data of diagonalization time vs U
	data = np.load(filename)
	U_vals = data['U_vals']
	means = data['means']
	stdevs = data['stdevs']
	
	fig,ax = plt.subplots()
	ax.errorbar(U_vals,means,yerr=stdevs,color='b',ls='',marker='x',label='Diagonalization')
	
	ax.set_xlabel('U')
	ax.set_ylabel('Time / s')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	plt.show()

if __name__ == '__main__':
	#generate_dens_data(Lx_vals=np.array([4,5,6,7,8,9,10]),Ly_vals=np.array([4,5,6,7,8,9,10]),repeats=[5,5,5,5,5,5,5],filename='dems_timing.npz')
	#fig,ax=plt.subplots()
	#plot_driver_data('evr_timing.npz',ax)
	#plt.show()
	
	#plot_dens_data('dens_timing.npz')
	
	#compare_drivers(['ev_timing.npz','evx_timing.npz','evr_timing.npz','evd_timing.npz'])
	
	#generate_propagator_data(Lx_vals=[3,4,5,6,7,8,9],Ly_vals=[3,4,5,6,7,8,9],repeats=[5,5,5,3,3,3,3,3,3],t=1.,filename='propagator_timing_t1_randpsi.npz')
	
	#generate_state_data(Lx_vals=[3,4,4,5,5,6,6,7],Ly_vals=[3,4,5,5,6,6,7,7],repeats=[5,5,5,3,3,3,3,3],filename='state_timing.npz')
	
	
	#compare_state_propagator('state_timing.npz','propagator_timing_t1.npz','propagator_timing_t10.npz','propagator_timing_t100.npz')
	
	#generate_propagator_vs_U_data(U_vals=np.logspace(0,3,10),repeats=5*np.ones(10,dtype=np.int16),Lx=6,Ly=6,filename='propagator_timing_U1_1000_L6x6.npz')
	
	#plot_propagator_vs_U_data('propagator_timing_U1_1000_L6x6.npz')
	
	#generate_diag_vs_U_data(U_vals=np.logspace(0,3,10),repeats=5*np.ones(10,dtype=np.int16),Lx=5,Ly=5,filename='diag_timing_U1_1000_L5x5.npz')
	
	plot_diag_vs_U_data('diag_timing_U1_1000_L5x5.npz')
	#generate_state_vs_t_data(t_vals=np.logspace(0,3,10),repeats=5*np.ones(10,dtype=np.int16),Lx=6,Ly=6,filename='state_timing_t1_1000_L6x6.npz')
	
	#compare_t_data('state_timing_t1_1000_L6x6.npz','propagator_timing_t1_1000_L6x6.npz')
	
	#fit_cubic('evd_timing.npz',fit_idx1=1)
	
	#fit_linear('propagator_timing_t1_fullpsi.npz',fit_idx1=4)
	
	#generate_eigsh_data(Lx_vals=[3,3,4,4],Ly_vals=[3,4,4,5],repeats=5*np.ones(8,dtype=np.int16),k='all',filename='eigsh_timing_kall.npz')
	
	#compare_k(['eigsh_timing_k10.npz','eigsh_timing_k50.npz','eigsh_timing_k100.npz','eigsh_timing_kall.npz'])
	
	#generate_eigsh_vs_k_data(k_vals=[10,20,30,50,100,200,300],repeats=[5,5,5,5,3,3,3],filename='eigsh_timing_k10_300_L8x8.npz')
	
	#eigsh_vs_k('eigsh_timing_k10_300_L8x8.npz')
	
