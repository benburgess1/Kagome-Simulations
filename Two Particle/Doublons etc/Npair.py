'''
This file contains functions for generating and analysing data for the 
doublon fraction/probability (called 'Npair' throughout this file).
'''


import numpy as np
import matplotlib.pyplot as plt
import Kagome2 as Kag2
import scipy as sp


def npair(psi,L=5):
	#Calculate the doublon probability from the state psi 
	N_sites = 3*L**2
	idxs = [int(i*(N_sites-0.5*i+0.5)) for i in range(N_sites)]
	val = 0
	for idx in idxs:
		val += np.abs(psi[idx])**2
	return val
	
	
def plot_npair(filename,ax,color='b',ls='-',label=None):
	#Plot the doublon probability vs t from a npz file
	data = np.load(filename)
	psi = data['psi']
	L = data['Lx']
	t = data['t']
	
	npair_t = [npair(psi[i,:],L=L) for i in range(np.size(t))]
	
	ax.plot(t,npair_t,color=color,ls=ls,label=label)
	
	

def plot_various_npair(filenames,labels):
	#Plot doublon probability vs t for various npz files
	colors = plt.cm.rainbow(np.linspace(0,1,len(filenames)))
	fig,ax = plt.subplots()
	for i,name in enumerate(filenames):
		plot_npair(name,ax,color=colors[i],label=labels[i])
	ax.set_xlabel('t')
	ax.set_ylabel('$n_{pair}$')
	ax.legend()
	plt.show()
	
	
def generate_npair_data(filename,U_vals,L=10,initial_site_idx=133):
	#Generate data of time-averaged doublon probability and standard deviation for various U
	npair_means = []
	npair_stdevs = []
	
	#N_sites = 3*L**2
	#N_states = int(N_sites*(N_sites+1)/2)
	#psi_0 = np.zeros(N_states)
	#psi_0[initial_state_idx] = 1
	
	if L >= 6:
		times = np.arange(30,50.05,0.1)
	else:
		times = np.arange(20,50.05,0.1)
	
	for U in U_vals:
		print(f'Evaulating U={U}...')
		system = Kag2.double_occupied_site_system(L=L,U=U,initial_site_idx=initial_site_idx)
		system.psi = system.state(t=times[0])
		dt = 0.1
		npair_arr = np.zeros(np.size(times))
		for i in range(np.size(times)-1):
			print(f'Progress: {np.round(100*(i+1)/np.size(times),2)}%  ',end='\r')
			npair_arr[i] = npair(system.psi,L=L)
			system.evolve_psi(dt)
		npair_arr[np.size(times)-1] = npair(system.psi,L=L)
		npair_means.append(np.mean(npair_arr))
		npair_stdevs.append(np.std(npair_arr))
		print('Done')
		
	np.savez(filename,U_vals=U_vals,npair_means=npair_means,npair_stdevs=npair_stdevs,L=L,initial_site_idx=initial_site_idx)
	
	
def linear_fit(x,a,b):
	return a*x + b
	
def power_fit(x,a,b):
	return a*x**b
	

def plot_npair_data(filename,ax=None,plot=False,ylim=1,plot_lim=False,plot_fit=False,printparams=True,fit_idx1=0,fit_idx2=1,logx=True,logy=False):
	#Plot data of time-averaged doublon probability and standard deviation against U
	data = np.load(filename)
	U_vals = data['U_vals']
	npair_means = data['npair_means']
	npair_stdevs = data['npair_stdevs']
	
	if ax is None:
		fig,ax = plt.subplots()
	ax.errorbar(U_vals,npair_means,yerr=npair_stdevs,color='b',ls='',marker='x',label='Data')
	ax.set_xlabel(r'$U$ / $J$',fontsize=15)
	ax.set_ylabel(r'$\overline{P_d}$',fontsize=15)
	#ax.set_xlim(left=0)
	#ax.set_ylim(0.015,ylim)
	
	if plot_lim:
		ax.axhline(75/2850,color='k',ls=':',label='0.0263')
		ax.legend()
		
	if logx:
		ax.set_xscale('log')
	if logy:
		ax.set_yscale('log')
		
	if plot_fit == 'linear':
		popt,pcov=sp.optimize.curve_fit(linear_fit,U_vals[fit_idx1:fit_idx2+1],npair_means[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * U + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		y_fit = linear_fit(U_vals,popt[0],popt[1])
		ax.plot(U_vals,y_fit,color='r',ls='--',label='Linear Fit')
		ax.legend()
		
	
	elif plot_fit == 'power':
		popt,pcov=sp.optimize.curve_fit(power_fit,U_vals[fit_idx1:fit_idx2+1],npair_means[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * U^b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		y_fit = power_fit(U_vals,popt[0],popt[1])
		ax.plot(U_vals,y_fit,color='r',ls='--',label='Power Fit')
		ax.legend()
	if plot:
		plt.show()
	
	
def plot_1minus_npair(filename,ax=None,plot=False,ylim=1.,plot_fit=False,fit_idx1=0,fit_idx2=0,printparams=True,logx=True,logy=True):
	#Plot 1 minus time-averaged doublon probability and standard deviation vs U; gives 'non-doublon probability'
	data = np.load(filename)
	U_vals = data['U_vals']
	npair_means = data['npair_means']
	npair_stdevs = data['npair_stdevs']
	
	indices = np.argsort(U_vals)
	U_vals = U_vals[indices]
	npair_means = npair_means[indices]
	npair_stdevs = npair_stdevs[indices]
	
	if ax is None:
		fig,ax = plt.subplots()
	ax.errorbar(U_vals,1-npair_means,yerr=npair_stdevs,color='b',ls='',marker='x',label='Data')
	ax.set_xlabel(r'$U$ / $J$',fontsize=15)
	ax.set_ylabel(r'$1-\overline{P_d}$',fontsize=15)
	#ax.set_xlim(left=0)
	
	
	if plot_fit == 'power':
		if fit_idx2=='all':
			fit_idx2 = np.size(U_vals)-1
		popt,pcov=sp.optimize.curve_fit(power_fit,U_vals[fit_idx1:fit_idx2+1],1-npair_means[fit_idx1:fit_idx2+1],sigma=npair_stdevs[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * U^b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		y_fit = power_fit(U_vals,popt[0],popt[1])
		ax.plot(U_vals,y_fit,color='r',ls='--',label='Fit')
		ax.legend(fontsize=15)
		
	if logx:
		ax.set_xscale('log')
	if logy:
		ax.set_yscale('log')
		
	if plot:	
		plt.show()
		

def compare_offcentre(filename_centre,filename_offcentre,ylim=1.,plot_lim=False):
	#Compare time-averaged doublon probability vs U generated from simulations with the initial
	#state either in the centre or off-centre in the lattice
	data1 = np.load(filename_centre)
	U_vals1 = data1['U_vals']
	npair_means1 = data1['npair_means']
	npair_stdevs1 = data1['npair_stdevs']
	
	data2 = np.load(filename_offcentre)
	U_vals2 = data2['U_vals']
	npair_means2 = data2['npair_means']
	npair_stdevs2 = data2['npair_stdevs']
	
	fig,ax = plt.subplots()
	ax.errorbar(U_vals1,npair_means1,yerr=npair_stdevs1,color='b',ls='',marker='x',label='Centre')
	ax.errorbar(U_vals2,npair_means2,yerr=npair_stdevs2,color='r',ls='',marker='x',label='Off-Centre')
	ax.set_xlabel('U')
	ax.set_ylabel(r'$\langle n_{pair} \rangle$',rotation=0)
	ax.set_ylim(0,ylim)
	
	if plot_lim:
		ax.axhline(75/2850,color='k',ls=':',label='0.0263')
		
	ax.legend()
	
	plt.show()
	
	
def compare_L(filename4,filename5,filename6,filename7,ylim=1.,plot_lim=False,plot_fit=False,fit_idx1=0,fit_idx2=0,printparams=True):
	#Compare time-averaged doublon probability for different sizes of lattice
	data4 = np.load(filename4)
	U_vals4 = data4['U_vals']
	npair_means4 = data4['npair_means']
	npair_stdevs4 = data4['npair_stdevs']
	
	data5 = np.load(filename5)
	U_vals5 = data5['U_vals']
	npair_means5 = data5['npair_means']
	npair_stdevs5 = data5['npair_stdevs']
	
	data6 = np.load(filename6)
	U_vals6 = data6['U_vals']
	npair_means6 = data6['npair_means']
	npair_stdevs6 = data6['npair_stdevs']
	
	data7 = np.load(filename7)
	U_vals7 = data7['U_vals']
	npair_means7 = data7['npair_means']
	npair_stdevs7 = data7['npair_stdevs']
	
	fig,ax = plt.subplots()
	ax.errorbar(U_vals4,npair_means4,yerr=npair_stdevs4,color='cyan',ls='',marker='x',label='L=4')
	ax.errorbar(U_vals5,npair_means5,yerr=npair_stdevs5,color='b',ls='',marker='x',label='L=5')
	ax.errorbar(U_vals6,npair_means6,yerr=npair_stdevs6,color='r',ls='',marker='x',label='L=6')
	ax.errorbar(U_vals7,npair_means7,yerr=npair_stdevs7,color='g',ls='',marker='x',label='L=7')
	ax.set_xlabel('U')
	ax.set_ylabel(r'$\langle n_{pair} \rangle$',rotation=0)
	ax.set_ylim(0,ylim)
	
	if plot_fit == 'linear':
		popt,pcov=sp.optimize.curve_fit(linear_fit,U_vals4[fit_idx1:fit_idx2+1],npair_means4[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('4x4 Fit:')
			print('y = a * U + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		y_fit4 = linear_fit(U_vals4,popt[0],popt[1])
		ax.plot(U_vals4,y_fit4,color='cyan',ls='--',label=None)
		
		popt,pcov=sp.optimize.curve_fit(linear_fit,U_vals5[fit_idx1:fit_idx2+1],npair_means5[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('5x5 Fit:')
			print('y = a * U + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		y_fit5 = linear_fit(U_vals5,popt[0],popt[1])
		ax.plot(U_vals5,y_fit5,color='b',ls='--',label=None)
		
		popt,pcov=sp.optimize.curve_fit(linear_fit,U_vals6[fit_idx1:fit_idx2+1],npair_means6[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('6x6 Fit:')
			print('y = a * U + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		y_fit6 = linear_fit(U_vals6,popt[0],popt[1])
		ax.plot(U_vals6,y_fit6,color='r',ls='--',label=None)
		
		popt,pcov=sp.optimize.curve_fit(linear_fit,U_vals7[fit_idx1:fit_idx2+1],npair_means7[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('7x7 Fit:')
			print('y = a * U + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		y_fit7 = linear_fit(U_vals7,popt[0],popt[1])
		ax.plot(U_vals7,y_fit7,color='g',ls='--',label=None)
		
	if plot_lim:
		ax.axhline(48/1176,color='cyan',ls=':',label=None)
		ax.axhline(75/2850,color='b',ls=':',label=None)
		ax.axhline(108/5886,color='r',ls=':',label=None)
		ax.axhline(1/74,color='g',ls=':',label=None)
		
	ax.legend()
	
	plt.show()
	
	

if __name__ == '__main__':
	#plot_various_npair(filenames=['N2_L10x10_SameHexagon_T2000_dt10_J1_U0.1.npz'],labels=['U=0.1'])
	
	#system = Kag2.two_hexagons_system(L=6,idx1=43,idx2=43,U=0.1,skip_diag=False,evolution_method='eigenvector')
	#high_evals = system.eigvals[system.eigvals>4.001]
	#high_c0 = system.c_0[system.eigvals>4.001]
	#print(high_evals)
	#print(np.size(high_evals))
	#print(np.sum(np.abs(high_c0)**2))
	
	#npair_vals = [npair(system.eigvects[:,i],L=6) for i in range(5814,5886)]
	
	#print(npair_vals)
	
	arr = np.logspace(0,3,19)
	U_vals = []
	for i in range(19):
		if i % 2 == 1:
			U_vals.append(arr[i])
	U_vals = np.array(U_vals)
	#print(U_vals)
	
	generate_npair_data('npair_L10x10_U1_1000.npz',U_vals=U_vals,L=10,initial_site_idx=133)
	
	#plot_npair_data('npair_twosites_U0_1.npz',ylim=0.03,plot_lim=True,plot_fit='linear',fit_idx1=0,fit_idx2=7)
	
	#plot_1minus_npair('npair_U10_100.npz',ylim=0.15,plot_fit='power',fit_idx1=9,fit_idx2=18,plotlog=False)
	
	#compare_offcentre('npair_L5x5_U0_100.npz','npair_L5x5_offcentre_U0_100.npz',plot_lim=True,ylim=0.1)
	
	#compare_L('npair_L4x4_U0_100.npz','npair_L5x5_U0_100.npz','npair_L6x6_U0_100.npz','npair_L7x7_U0_100.npz',plot_lim=False,ylim=1.,plot_fit=False,fit_idx1=0,fit_idx2=7)

