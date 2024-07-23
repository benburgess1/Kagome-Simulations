import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import Kagome as Kag
import cProfile
import timeit
import functools


'''
Copy/paste definition of H from Kagome.py
OR import Kagome and build a lattice, although would need to wait for diagonalization of H unnecessarily... not a big deal, probably quicker overall

With H obtained, time diagonalization using different routines

Plot/compare
'''

def time_func(H,func,repeat):
	if func == np.linalg.svd:
		time_vals = timeit.repeat(functools.partial(func,H,hermitian=True),repeat=repeat,number=1)
	else:
		time_vals = timeit.repeat(functools.partial(func,H),repeat=repeat,number=1)
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	return mean,stdev
	
def time_driver(H,driver,repeat):
	time_vals = timeit.repeat(functools.partial(sp.linalg.eigh,H,driver=driver),repeat=repeat,number=1)
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	return mean,stdev
	
def create_data(L,repeats):
	#L = np.array([5,10,15,20,25,30,35,40,50,60])
	N = 3*L**2
	#repeats = [10,5,5,5,5,5,3,3,3,3]
	
	npeigh_means = []
	npeigh_stdevs = []
	speigh_means = []
	speigh_stdevs = []
	npsvd_means = []
	npsvd_stdevs = []
	spsvd_means = []
	spsvd_stdevs = []
	
	for i,L_val in enumerate(L):
		print(f'Evaluating L = {L_val}...')
		system = Kag.Kagome(np.zeros(N[i]),L_val,L_val)
		print('Evaluating np.eigh...')
		m,s = time_func(system.H,np.linalg.eigh,repeats[i])
		npeigh_means.append(m)
		npeigh_stdevs.append(s)
		print('Evaluating sp.eigh...')
		m,s = time_func(system.H,sp.linalg.eigh,repeats[i])
		speigh_means.append(m)
		speigh_stdevs.append(s)
		print('Evaluating np.svd...')
		m,s = time_func(system.H,np.linalg.svd,repeats[i])
		npsvd_means.append(m)
		npsvd_stdevs.append(s)
		print('Evaluating sp.svd...')
		m,s = time_func(system.H,sp.linalg.svd,repeats[i])
		spsvd_means.append(m)
		spsvd_stdevs.append(s)
	
	
	np.savez('DiagCompare.npz', L=L, N=N, repeats=repeats,
								npeigh_means=npeigh_means, npeigh_stdevs=npeigh_stdevs,
								speigh_means=speigh_means, speigh_stdevs=speigh_stdevs,
								npsvd_means=npsvd_means, npsvd_stdevs=npsvd_stdevs,
								spsvd_means=spsvd_means, spsvd_stdevs=spsvd_stdevs)
	

def create_driver_data(L,repeats):
	#L = np.array([5,10,15,20,25,30,35,40,50,60])
	N = 3*L**2
	#repeats = [10,5,5,5,5,5,3,3,3,3]
	
	npeigh_means = []
	npeigh_stdevs = []
	spev_means = []
	spev_stdevs = []
	spevr_means = []
	spevr_stdevs = []
	spevd_means = []
	spevd_stdevs = []
	spevx_means = []
	spevx_stdevs = []
	
	for i,L_val in enumerate(L):
		print(f'Evaluating L = {L_val}...')
		system = Kag.Kagome(np.zeros(N[i]),L_val,L_val)
		print('Evaluating np.eigh...')
		m,s = time_func(system.H,np.linalg.eigh,repeats[i])
		npeigh_means.append(m)
		npeigh_stdevs.append(s)
		print('Evaluating sp.eigh, ev...')
		m,s = time_driver(system.H,'ev',repeats[i])
		spev_means.append(m)
		spev_stdevs.append(s)
		print('Evaluating sp.eigh, evr...')
		m,s = time_driver(system.H,'evr',repeats[i])
		spevr_means.append(m)
		spevr_stdevs.append(s)
		print('Evaluating sp.eigh, evd...')
		m,s = time_driver(system.H,'evd',repeats[i])
		spevd_means.append(m)
		spevd_stdevs.append(s)
		print('Evaluating sp.eigh, evx...')
		m,s = time_driver(system.H,'evx',repeats[i])
		spevx_means.append(m)
		spevx_stdevs.append(s)
	
	
	np.savez('DriverCompare.npz', L=L, N=N, repeats=repeats,
								npeigh_means=npeigh_means, npeigh_stdevs=npeigh_stdevs,
								spev_means=spev_means, spev_stdevs=spev_stdevs,
								spevr_means=spevr_means, spevr_stdevs=spevr_stdevs,
								spevd_means=spevd_means, spevd_stdevs=spevd_stdevs,
								spevx_means=spevx_means, spevx_stdevs=spevx_stdevs)


def reduced_driver_data(L,repeats):
	#L = np.array([5,10,15,20,25,30,35,40,50,60])
	N = 3*L**2
	#repeats = [10,5,5,5,5,5,3,3,3,3]
	
	npeigh_means = []
	npeigh_stdevs = []
	spevr_means = []
	spevr_stdevs = []
	spevd_means = []
	spevd_stdevs = []
	
	for i,L_val in enumerate(L):
		print(f'Evaluating L = {L_val}...')
		system = Kag.Kagome(np.zeros(N[i]),L_val,L_val)
		print('Evaluating np.eigh...')
		m,s = time_func(system.H,np.linalg.eigh,repeats[i])
		npeigh_means.append(m)
		npeigh_stdevs.append(s)
		print('Evaluating sp.eigh, evr...')
		m,s = time_driver(system.H,'evr',repeats[i])
		spevr_means.append(m)
		spevr_stdevs.append(s)
		print('Evaluating sp.eigh, evd...')
		m,s = time_driver(system.H,'evd',repeats[i])
		spevd_means.append(m)
		spevd_stdevs.append(s)
	
	
	np.savez('DriverCompareReduced.npz', L=L, N=N, repeats=repeats,
								npeigh_means=npeigh_means, npeigh_stdevs=npeigh_stdevs,
								spevr_means=spevr_means, spevr_stdevs=spevr_stdevs,
								spevd_means=spevd_means, spevd_stdevs=spevd_stdevs)
								
	
	
def plot_data(filename):
	data = np.load(filename)
	L = data['L']
	N = data['N']
	repeats = data['repeats']
	npeigh_means = data['npeigh_means']
	speigh_means = data['speigh_means']
	npsvd_means = data['npsvd_means']
	spsvd_means = data['spsvd_means']
	npeigh_stdevs = data['npeigh_stdevs']
	speigh_stdevs = data['speigh_stdevs']
	npsvd_stdevs = data['npsvd_stdevs']
	spsvd_stdevs = data['spsvd_stdevs']
	
	fig,ax = plt.subplots()
	
	x = np.linspace(N[0],N[-1],100)
	cub_fit = 1e-10*x**3
	
	ax.errorbar(N,npeigh_means,npeigh_stdevs,marker='x',color='b',label='np eigh')
	ax.errorbar(N,speigh_means,speigh_stdevs,marker='x',color='r',label='sp eigh')
	ax.errorbar(N,npsvd_means,npsvd_stdevs,marker='x',color='g',label='np svd')
	ax.errorbar(N,spsvd_means,spsvd_stdevs,marker='x',color='y',label='sp svd')
	ax.plot(x,cub_fit,'k:',label=r'$\mathcal{O}(N^3)$')
	
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlabel('N')
	ax.set_ylabel('Time/s')
	ax.legend()
	
	plt.show()
	
	
def plot_driver_data(filename):
	data = np.load(filename)
	L = data['L']
	N = data['N']
	repeats = data['repeats']
	npeigh_means = data['npeigh_means']
	spev_means = data['spev_means']
	spevr_means = data['spevr_means']
	spevd_means = data['spevd_means']
	spevx_means = data['spevx_means']
	npeigh_stdevs = data['npeigh_stdevs']
	spev_stdevs = data['spev_stdevs']
	spevr_stdevs = data['spevr_stdevs']
	spevd_stdevs = data['spevd_stdevs']
	spevx_stdevs = data['spevx_stdevs']
	
	fig,ax = plt.subplots()
	
	x = np.linspace(N[0],N[-1],100)
	cub_fit = 1e-10*x**3
	
	ax.errorbar(N,npeigh_means,npeigh_stdevs,marker='x',color='b',label='np eigh (evd)')
	ax.errorbar(N,spev_means,spev_stdevs,marker='x',color='r',label='sp ev')
	ax.errorbar(N,spevr_means,spevr_stdevs,marker='x',color='g',label='sp evr')
	ax.errorbar(N,spevd_means,spevd_stdevs,marker='x',color='y',label='sp evd')
	ax.errorbar(N,spevx_means,spevx_stdevs,marker='x',color='cyan',label='sp evx')
	ax.plot(x,cub_fit,'k:',label=r'$\mathcal{O}(N^3)$')
	
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlabel('N')
	ax.set_ylabel('Time/s')
	ax.legend()
	
	plt.show()
	
def plot_reduced_driver_data(filename):	
	data = np.load(filename)
	L = data['L']
	N = data['N']
	repeats = data['repeats']
	npeigh_means = data['npeigh_means']
	spevr_means = data['spevr_means']
	spevd_means = data['spevd_means']
	npeigh_stdevs = data['npeigh_stdevs']
	spevr_stdevs = data['spevr_stdevs']
	spevd_stdevs = data['spevd_stdevs']
	
	fig,ax = plt.subplots()
	
	x = np.linspace(N[0],N[-1],100)
	cub_fit = 1e-10*x**3
	
	ax.errorbar(N,npeigh_means,npeigh_stdevs,marker='x',color='b',label='np eigh (evd)')
	ax.errorbar(N,spevr_means,spevr_stdevs,marker='x',color='g',label='sp evr (default)')
	ax.errorbar(N,spevd_means,spevd_stdevs,marker='x',color='y',label='sp evd')
	ax.plot(x,cub_fit,'k:',label=r'$\mathcal{O}(N^3)$')
	
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlabel('N')
	ax.set_ylabel('Time/s')
	ax.legend()
	
	plt.show()
	
	
#create_data(L=np.array([5,10,15,20,25,30,35,40,50,60]),repeats=np.array([10,5,5,5,5,5,3,3,3,3]))

#create_driver_data(L=np.array([5,10,15,20,25,30]),repeats=np.array([10,5,5,5,5,5]))

#reduced_driver_data(L=np.array([5,10,15,20,25,30,35]),repeats=np.array([10,5,5,5,5,5,3]))
	
#plot_data('DiagCompare.npz')	

#plot_driver_data('DriverCompare.npz')
	
plot_reduced_driver_data('DriverCompareReduced.npz')
