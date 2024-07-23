'''
This file contains various functions for obtaining and plotting the eigenvalues
and eigenvectors of kagome systems.
'''

import numpy as np
import matplotlib.pyplot as plt
import Kagome2 as Kag2
import Kagome as Kag
import scipy as sp


def plot_eigenfrequencies(ax,system,color='b',ls='-',marker='x',label=None):
	#Given a kagome system, plot the eigenfrequencies
	ax.plot(np.arange(system.N),system.w,color=color,ls=ls,marker=marker,label=label)
	ax.set_xticks([])
	ax.set_ylabel(r'$\omega$ / rad s$^{-1}$')
	

def plot_several_eigenfrequencies(U_vals,colors,L=5,ls='-',marker=None):
	#For several values of U, build and diagonalize the 2-particle kagome system and plot the eigenfrequencies
	fig,ax = plt.subplots()
	for i,U in enumerate(U_vals):
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		psi_0 = np.zeros(N_states)
		system = Kag2.Kagome2(psi_0,Lx=L,Ly=L,U=U)
		plot_eigenfrequencies(ax,system,color=colors[i],label=f'U={U}',ls=ls,marker=marker)
	ax.legend()
	plt.show()
	
	
def plot_singleparticle_eigenfrequencies(color='b',L=5,J=1.,ls='-',marker='x'):
	#Plot the eigenfrequencies of a single-particle kagome system
	fig,ax = plt.subplots()
	N = 3*L**2
	psi_0 = np.zeros(N)
	system = Kag.Kagome(psi_0,Lx=L,Ly=L,J=J)
	plot_eigenfrequencies(ax,system,color=color,ls=ls,marker=marker)
	plt.show()
	

def first_order_shifts(U=1.,L=5):
	#Plot the interaction energy shifts calculated to first-order within perturbation theory
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	system = Kag2.Kagome2(psi_0,Lx=L,Ly=L,U=U)
	system_NI = Kag2.Kagome2(psi_0,Lx=L,Ly=L,U=0)
	
	H_int = system.H - system_NI.H
	
	dE = np.zeros(N_states)
	for i in range(N_states):
		v = system_NI.eigvects[:,i]
		dE[i] = v.T @ H_int @ v
		
	fig,ax = plt.subplots()
	ax.plot(np.arange(N_states),system_NI.eigvals,color='b',ls='-',label='U=0')
	ax.plot(np.arange(N_states),system_NI.eigvals+dE,color='r',ls='-',label=f'U={U} First Order')
	ax.plot(np.arange(N_states),system.eigvals,color='cyan',ls='-',label=f'U={U} Exact')
	
	ax.set_xticks([])
	ax.set_ylabel('Energy / J')
	ax.legend()
	
	plt.show()
	
	
def occupancy_vs_energy(U=1.,L=5,initial_state='single site',initial_state_idx=2070,hexagon_site_idx=43):
	#Plot the occupancy of the eigenstates against the eigenenergy, for different options of initial wavefunction state
	if initial_state == 'single site':
		N_sites = 3*L**2
		N_states = int(0.5*N_sites*(N_sites+1))
		psi_0 = np.zeros(N_states)
		psi_0[initial_state_idx] = 1
		system = Kag2.Kagome2(psi_0,Lx=L,Ly=L,U=U)
	elif initial_state == 'hexagon':
		system = Kag2.two_hexagons_system(idx1=hexagon_site_idx,idx2=hexagon_site_idx,L=L,U=U,bc='periodic')
		fig,ax=plt.subplots()
		system.plot_state(fig,ax)
		plt.show()
	
	fig,ax = plt.subplots()
	ax.plot(system.eigvals,system.c_0,'bx')
	ax.set_xlabel('Energy / J')
	ax.set_ylabel('Eigenstate Coefficient')
	ax.set_title(f'U = {U}')
	plt.show()
	
	
def extract_perturbed(U=1.,L=5,cutoff=2806,shift=True):
	#Return only the eigenvalues above a certain index; used to plot the interaction-shifted flat band energies
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	system = Kag2.Kagome2(psi_0,Lx=L,Ly=L,U=U)
	
	perturbed = system.eigvals[cutoff:]
	
	if shift:
		perturbed -= 4
	
	return perturbed
	

def power_const(x,a,b,c):
	return a*x**b + c


def plot_perturbed(ax,E,color='b',ls='-',label=None,plotfit=False,printparams=True,fitcolor='r',fitls=':',fitlabel='Fit'):
	#Plot the interaction-shifted flat band eigenenergies
	N = np.size(E)
	x = np.arange(N)
	ax.plot(x,E,color=color,ls=ls,label=label)
	
	if plotfit == 'power':
		popt,pcov = sp.optimize.curve_fit(power_const,x,E,p0=[1e-5,2.5,0.1])
		perr = np.sqrt(np.diag(pcov))
		E_fit = power_const(x,popt[0],popt[1],popt[2])
		ax.plot(x,E_fit,color=fitcolor,ls=fitls,label=fitlabel)
		if printparams:
			print('Fit y = a * x**b + c')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
			print(f'c = {popt[2]} +/- {perr[2]}')

	ax.set_ylabel(r'$\Delta E$',rotation=0)
	ax.set_xticks([])
	ax.legend()
	
	
def plot_several_perturbed(U_vals,L=5):
	#For several U values, plot the interaction-shifted flat band eigenenergies
	fig,ax = plt.subplots()
	colors = plt.cm.rainbow(np.linspace(0,1,np.size(U_vals)))
	for i,U in enumerate(U_vals):
		plot_perturbed(ax,extract_perturbed(U=U,L=L),color=colors[i],label=f'U={np.round(U,1)}')
		
	plt.show()
	
	
def generate_perturb_fit_data(filename,U_vals,L=5):
	#Generate data of performing power-law + constant fits to perturbed flat band eigenvalues, for various U
	a_vals = np.zeros(np.size(U_vals))
	b_vals = np.zeros(np.size(U_vals))
	c_vals = np.zeros(np.size(U_vals))
	a_err = np.zeros(np.size(U_vals))
	b_err = np.zeros(np.size(U_vals))
	c_err = np.zeros(np.size(U_vals))
	
	for i,U in enumerate(U_vals):
		print(f'Evaulating U={U}...')
		E = extract_perturbed(U=U,L=L)
		x = np.arange(np.size(E))
		popt,pcov = sp.optimize.curve_fit(power_const,x,E,p0=[1e-5,2.5,0.1])
		perr = np.sqrt(np.diag(pcov))
		
		a_vals[i] = popt[0]
		b_vals[i] = popt[1]
		c_vals[i] = popt[2]
		a_err[i] = perr[0]
		b_err[i] = perr[1]
		c_err[i] = perr[2]
		
	np.savez(filename,U_vals=U_vals,a_vals=a_vals,b_vals=b_vals,c_vals=c_vals,a_err=a_err,b_err=b_err,c_err=c_err)


def linear_fit(x,a,b):
	return a*x + b


def proportional_fit(x,a):
	return a*x
	
	
def plot_a(filename,plotfit=False,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot 'a' parameter (power law coefficient) of fits to perturbed eigenvalues vs U data
	data = np.load(filename)
	U = data['U_vals']
	a = data['a_vals']
	a_err = data['a_err']
	
	fig,ax = plt.subplots()
	ax.errorbar(U,a,yerr=a_err,color='b',marker='x',ls='',label='Data')
	
	if plotfit=='linear':
		popt,pcov=sp.optimize.curve_fit(linear_fit,U[fit_idx1:fit_idx2+1],a[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * U + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		a_fit = linear_fit(U,popt[0],popt[1])
		ax.plot(U,a_fit,color='r',ls='--',label='Fit')
		ax.legend()
		
	elif plotfit=='proportional':
		popt,pcov=sp.optimize.curve_fit(proportional_fit,U[fit_idx1:fit_idx2+1],a[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * U')
			print(f'a = {popt[0]} +/- {perr[0]}')
		a_fit = proportional_fit(U,popt[0])
		ax.plot(U,a_fit,color='r',ls='--',label='Fit')
		ax.legend()
	
	ax.set_xlabel('U')
	ax.set_ylabel('a',rotation=0)
	#ax.set_ylim(0,0.5)
	#ax.set_xlim(0,10.1)
	
	plt.show()		


def plot_b(filename,plotfit=False,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot 'b' parameter (power law exponent) of fits to perturbed eigenvalues vs U data
	data = np.load(filename)
	U = data['U_vals']
	b = data['b_vals']
	b_err = data['b_err']
	
	fig,ax = plt.subplots()
	ax.errorbar(U,b,yerr=b_err,color='b',marker='x',ls='',label='Data')
	
	if plotfit=='linear':
		popt,pcov=sp.optimize.curve_fit(linear_fit,U[fit_idx1:fit_idx2+1],b[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * U + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		b_fit = linear_fit(U,popt[0],popt[1])
		ax.plot(U,b_fit,color='r',ls='--',label='Fit')
		ax.legend()
		
	elif plotfit=='proportional':
		popt,pcov=sp.optimize.curve_fit(proportional_fit,U[fit_idx1:fit_idx2+1],b[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * U')
			print(f'a = {popt[0]} +/- {perr[0]}')
		b_fit = proportional_fit(U,popt[0])
		ax.plot(U,b_fit,color='r',ls='--',label='Fit')
		ax.legend()
	
	ax.set_xlabel('U')
	ax.set_ylabel('b',rotation=0)
	#ax.set_ylim(0,0.5)
	#ax.set_xlim(0,10.1)
	
	plt.show()
	
	
def plot_c(filename,plotfit=False,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot 'c' parameter (constant term) of fits to perturbed eigenvalues vs U data
	data = np.load(filename)
	U = data['U_vals']
	c = data['c_vals']
	c_err = data['c_err']
	
	fig,ax = plt.subplots()
	ax.errorbar(U,c,yerr=c_err,color='b',marker='x',ls='',label='Data')
	
	if plotfit=='linear':
		popt,pcov=sp.optimize.curve_fit(linear_fit,U[fit_idx1:fit_idx2+1],c[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * U + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		c_fit = linear_fit(U,popt[0],popt[1])
		ax.plot(U,c_fit,color='r',ls='--',label='Fit')
		ax.legend()
		
	elif plotfit=='proportional':
		popt,pcov=sp.optimize.curve_fit(proportional_fit,U[fit_idx1:fit_idx2+1],c[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * U')
			print(f'a = {popt[0]} +/- {perr[0]}')
		c_fit = proportional_fit(U,popt[0])
		ax.plot(U,c_fit,color='r',ls='--',label='Fit')
		ax.legend()
	
	ax.set_xlabel('U')
	ax.set_ylabel('c',rotation=0)
	#ax.set_ylim(0,0.5)
	#ax.set_xlim(0,10.1)
	
	plt.show()


def plot_max_eigenstates(num=1,U=1.,L=5,initial_state_idx=2070):
	#Plot density of the highest-energy eigenstates of the two-particle system
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	psi_0[initial_state_idx] = 1
	system = Kag2.Kagome2(psi_0,Lx=L,Ly=L,U=U)
	
	indices = np.argsort(np.abs(system.c_0))[-1:-(num+1):-1]
	
	for idx in indices:
		v_max = system.eigvects[:,idx]
		system.psi = v_max
		fig,ax = plt.subplots()
		system.plot_state(fig,ax,uppernorm=0.5,cbar_label_y=0.5)
		ax.set_title(f'$\omega$ = {np.round(system.w[idx],3)}')
		plt.show()
	
def plot_max_eigenstates_amp(num=1,U=1.,L=5,initial_state_idx=2070):
	#Plot amplitude of the highest-energy eigenstates of the two-particle system
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	psi_0[initial_state_idx] = 1
	system = Kag2.Kagome2(psi_0,Lx=L,Ly=L,U=U)
	
	indices = np.argsort(np.abs(system.c_0))[-1:-(num+1):-1]
	
	for idx in indices:
		v_max = system.eigvects[:,idx]
		system.psi = v_max
		fig,ax = plt.subplots()
		system.plot_amplitude(fig,ax,uppernorm=1.,cbar_label_y=0.5)
		ax.set_title(f'$\omega$ = {np.round(system.w[idx],3)}')
		plt.show()
		
		
def plot_max_eigenstates_components(num=1,U=1.,L=5,initial_state_idx=2070):
	#Plot occupancy (i.e. |c|^2) of the highest-occupancy eigenstates in the expansion of a single-site initial state
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	psi_0[initial_state_idx] = 1
	system = Kag2.Kagome2(psi_0,Lx=L,Ly=L,U=U)
	
	indices = np.argsort(np.abs(system.c_0))[-1:-(num+1):-1]
	
	fig,ax = plt.subplots()
	colors = plt.cm.rainbow(np.linspace(0,1,num))
	for i,idx in enumerate(indices):
		v_max = system.eigvects[:,idx]
		x = np.arange(np.size(v_max))
		ax.plot(x,v_max,color=colors[i],label=f'$\omega$ = {np.round(system.w[idx],3)}')
		#ax.set_title(f'$\omega$ = {np.round(system.w[idx],3)}')
	
	ax.legend()
	ax.set_xlabel('State Index')
	ax.set_ylabel('Occupancy')
	ax.set_title(f'U={U}')
	plt.show()
	
	

	
	


if __name__ == '__main__':
	#plot_several_eigenfrequencies(U_vals=[10],colors=['b'])
	
	#plot_singleparticle_eigenfrequencies(ls='-',marker=None,J=-1.)
	
	#first_order_shifts(U=0.1)
	
	#occupancy_vs_energy(U=0.,L=6,initial_state='hexagon',hexagon_site_idx=43)
	'''
	E = extract_perturbed(U=1.)
	fig,ax=plt.subplots()
	plot_perturbed(ax,E,plotfit='power',label='U=1')
	plt.show()
	'''
	
	#plot_several_perturbed(U_vals=np.arange(0.2,2.1,0.2))
	
	#generate_perturb_fit_data('EnergyShift_fit_params_U0.2_2.npz',U_vals=[0.2,0.5,0.75,1.,1.25,1.5])
	
	#plot_c('EnergyShift_fit_params_U0.2_2.npz')
	
	#plot_max_eigenstates_components(num=1,U=100.)
	
	#plot_max_eigenstates_amp(num=10,U=10.)
	

	
	

