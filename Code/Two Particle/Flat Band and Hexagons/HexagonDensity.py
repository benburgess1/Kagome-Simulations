'''
This file contains various functions for generating and analyzing
data relating to the density in the original hexagon, when expanding from 
the hexagon initial state.
'''


import numpy as np
import matplotlib.pyplot as plt
import Kagome2 as Kag2
import scipy as sp


def hexagon_density(density,idx,L=10):
	#Given the density at all sites in the lattice, returns the density within
	#the hexagon with lower-left site index idx
	idxs = [idx,idx+1,idx+2,idx+3,idx+3*L+1,idx+3*L-1]
	return(np.sum([density[i] for i in idxs]))
	
	
def plot_hexagon_density(ax,filename,ll_site_idx=None,color='b',ls='-',marker=None,label=None,plot_fit=None,fit_T='all',printparams=True,p0=None,plot_T=None):
	#Plot density in hexagon with specified lower-left site index over time, from file containing density at all sites in lattice
	#Various options for fitting models are specified
	data = np.load(filename)
	density = data['density']
	t = data['t']
	L = data['Lx']
	
	if plot_T is not None:
		density = density[t<=plot_T]
		t = t[t<=plot_T]
	
	if data['Lx'] != data['Ly']:
		print('Warning: rectangular lattice detected')
	
	if ll_site_idx is None:
		if 'll_site_idx' in data.keys():
			ll_site_idx = data['ll_site_idx']
		else:
			ll_site_idx = np.min(np.argwhere(density[0,:]))
			
	hex_density = np.array([hexagon_density(density[i,:],ll_site_idx,L=L) for i in range(np.size(t))])
	
	if label == 'auto':
		U = data['U']
		label = f'U={U}'
	
	ax.plot(t,hex_density,color=color,ls=ls,marker=marker,label=label)
	ax.set_xlim(0,np.max(t))
	
	if fit_T == 'all':
		fit_T = np.max(t)
	
	if plot_fit == 'gaussian':
		fit_density = np.extract(t<=fit_T,hex_density)
		fit_t = np.extract(t<=fit_T,t)
		popt,pcov = sp.optimize.curve_fit(gaussian_fit,fit_t,fit_density,p0=p0)
		perr = np.sqrt(np.diag(pcov))
		yfit = gaussian_fit(t,popt[0],popt[1])
		ax.plot(t,yfit,'r--',label='Gaussian Fit')
		if printparams:
			print('Fit y = A * exp(-0.5*(t/b)^2)')
			print(f'A = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
			
	elif plot_fit == 'gaussian2':			#Fixed density = 2 at t=0
		fit_density = np.extract(t<=fit_T,hex_density)
		fit_t = np.extract(t<=fit_T,t)
		popt,pcov = sp.optimize.curve_fit(gaussian_fit2,fit_t,fit_density,p0=p0)
		perr = np.sqrt(np.diag(pcov))
		yfit = gaussian_fit2(t,popt[0])
		ax.plot(t,yfit,'r--',label='Gaussian Fit')
		if printparams:
			print('Fit y = 2 * exp(-0.5*(t/b)^2)')
			print(f'b = {popt[0]} +/- {perr[0]}')
			
	elif plot_fit == 'gaussian2_const':			#Fixed density = 2 at t=0, and constant
		fit_density = np.extract(t<=fit_T,hex_density)
		fit_t = np.extract(t<=fit_T,t)
		popt,pcov = sp.optimize.curve_fit(gaussian2_const_fit,fit_t,fit_density,p0=p0)
		perr = np.sqrt(np.diag(pcov))
		yfit = gaussian2_const_fit(t,popt[0],popt[1])
		ax.plot(t,yfit,color='navy',ls=':',label='Fit')
		if printparams:
			print('Fit y = (2 - c) * exp(-0.5*(t/b)^2) + c')
			print(f'b = {popt[0]} +/- {perr[0]}')
			print(f'c = {popt[1]} +/- {perr[1]}')
			
	elif plot_fit == 'lorentzian':
		fit_density = np.extract(t<=fit_T,hex_density)
		fit_t = np.extract(t<=fit_T,t)
		popt,pcov = sp.optimize.curve_fit(lorentzian_fit,fit_t,fit_density,p0=p0)
		perr = np.sqrt(np.diag(pcov))
		yfit = lorentzian_fit(t,popt[0],popt[1])
		ax.plot(t,yfit,'r--',label='Lorentzian Fit')
		if printparams:
			print('Fit y = A / (x^2 + b^2)')
			print(f'A = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
			
	elif plot_fit == 'lorentzian2':			#Fixed density = 2 at t=0
		fit_density = np.extract(t<=fit_T,hex_density)
		fit_t = np.extract(t<=fit_T,t)
		popt,pcov = sp.optimize.curve_fit(lorentzian_fit2,fit_t,fit_density,p0=p0)
		perr = np.sqrt(np.diag(pcov))
		yfit = lorentzian_fit2(t,popt[0])
		ax.plot(t,yfit,'r--',label='Lorentzian Fit')
		if printparams:
			print('Fit y = 2 b^2 / (x^2 + b^2)')
			print(f'b = {popt[0]} +/- {perr[0]}')
			
	elif plot_fit == 'lorentzian_cos':
		fit_density = np.extract(t<=fit_T,hex_density)
		fit_t = np.extract(t<=fit_T,t)
		popt,pcov = sp.optimize.curve_fit(lorentzian_cos_fit,fit_t,fit_density,p0=p0)
		perr = np.sqrt(np.diag(pcov))
		yfit = lorentzian_cos_fit(t,popt[0],popt[1],popt[2],popt[3])
		ax.plot(t,yfit,'r--',label='Lorentzian + Cosine Fit')
		if printparams:
			print('Fit y = a / (x^2 + b^2) + c * cos(w*x)')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
			print(f'c = {popt[2]} +/- {perr[2]}')
			print(f'w = {popt[3]} +/- {perr[3]}')

	
	
def plot_various_hexagon_densities(filenames,colors=None,labels=None,ylim=2.,xlim=None,plot_fit=None,fit_T=[10],printparams=True,p0=None):
	#For various density vs t files, plot the density within original hexagon vs t
	fig,ax = plt.subplots()
	if colors is None:
		colors = plt.cm.rainbow(np.linspace(0,1,np.size(filenames)))
	for i,name in enumerate(filenames):
		plot_hexagon_density(ax,name,color=colors[i],label=labels[i] if labels is not None else 'auto',plot_fit=plot_fit,fit_T=fit_T[0] if len(fit_T)==1 else fit_T[i],printparams=printparams,p0=p0[i] if p0 is not None else None)
	
	ax.set_xlabel(r'$t$ / $t_0$')
	ax.set_ylabel(r'$\langle n \rangle$',rotation=0)
	ax.set_ylim(0,ylim)
	ax.legend()
	ax.set_title(f'Density in Original Hexagon')
	
	if xlim is not None:
		ax.set_xlim(0,xlim)
	
	plt.show()
	
	
def gaussian_fit(x,a,b):
	return a*np.exp(-0.5*(x/b)**2)
	

def gaussian_fit2(x,b):
	return 2*np.exp(-0.5*(x/b)**2)
	

def gaussian2_const_fit(x,b,c):
	return (2-c)*np.exp(-0.5*(x/b)**2) + c
	
	
def lorentzian_fit(x,a,b):
	return a/(x**2+b**2)
	

def lorentzian_fit2(x,b):
	a = 2*b**2
	return a/(x**2+b**2)
	
	
def lorentzian_cos_fit(x,a,b,c,w):
	return a/(x**2+b**2) + c*np.cos(w*x)
	
	
def generate_U_fit_data(filename,U_vals,L=10,ll_site_idx=133):
	#Generate Lorentzian fit data vs U for density in original hexagon
	a_vals = np.zeros(np.size(U_vals))
	b_vals = np.zeros(np.size(U_vals))
	a_err = np.zeros(np.size(U_vals))
	b_err = np.zeros(np.size(U_vals))

	
	for i,U in enumerate(U_vals):
		print(f'Evaulating U={U}...')
		system = Kag2.two_hexagons_system(idx1=ll_site_idx,idx2=ll_site_idx,U=U,L=10,skip_diag=True,evolution_method='propagator')
		T = 10/U
		dt = T/100
		times = np.arange(0,T+dt,dt)
		
		hex_density = np.array([hexagon_density(system.density(t),ll_site_idx,L=L) for t in times])
		
		popt,pcov = sp.optimize.curve_fit(lorentzian_fit,times,hex_density)
		perr = np.sqrt(np.diag(pcov))
		
		a_vals[i] = popt[0]
		b_vals[i] = popt[1]
		a_err[i] = perr[0]
		b_err[i] = perr[1]
		
	np.savez(filename,U_vals=U_vals,a_vals=a_vals,b_vals=b_vals,a_err=a_err,b_err=b_err,L=L,ll_site_idx=ll_site_idx)
	
	
def plot_b(filename,logscale=False,plot_fit=False,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot Lorentzian decay parameter 'b' vs U from npz file of data
	data = np.load(filename)
	U = data['U_vals']
	b = np.abs(data['b_vals'])			#Since may have converged to -b if unlucky
	b_err = data['b_err']
	
	fig,ax = plt.subplots()
	ax.errorbar(U,b,yerr=b_err,color='b',marker='x',ls='',label='10x10 Data')
	
	if plot_fit == 'power':
		popt,pcov = sp.optimize.curve_fit(power_fit,U[fit_idx1:fit_idx2+1],b[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = power_fit(U,popt[0],popt[1])
		ax.plot(U,yfit,'r--',label='Power Law Fit')
		if printparams:
			print('Fit y = A * U^n')
			print(f'A = {popt[0]} +/- {perr[0]}')
			print(f'n = {popt[1]} +/- {perr[1]}')
	
	ax.set_xlabel('U')
	ax.set_ylabel('b',rotation=0)
	if logscale:
		ax.set_yscale('log')
		ax.set_xscale('log')
	#ax.set_ylim(0,0.5)
	#ax.set_xlim(0,10.1)
	ax.legend()
	ax.set_title('HWHM vs U')
	plt.show()
	
	
def power_fit(x,a,n):
	return a*x**n
	
	
def generate_latetime_mean_data(filename,U_vals,L=10,ll_site_idx=133,dt=0.1):
	#Generate data of the late-time average value of the original hexagon density, vs U
	means = np.zeros(np.size(U_vals))
	stdevs = np.zeros(np.size(U_vals))
	
	for i,U in enumerate(U_vals):
		print(f'Evaulating U={U}...')
		system = Kag2.two_hexagons_system(idx1=ll_site_idx,idx2=ll_site_idx,U=U,L=10,skip_diag=True,evolution_method='propagator')
		
		if U <= 10:
			t1 = 70
		else:
			t1 = 40

		times = np.arange(t1,t1+20+dt,dt)
		
		dens = np.zeros((np.size(times),system.N_sites))
		
		system.psi = system.state(times[0])
	
		dens[0,:] = system.density_from_psi(system.psi)
		for j in range(1,np.size(times)):
			system.psi = system.evolve_psi(dt)
			dens[j,:] = system.density_from_psi(system.psi)

		hex_density = np.array([hexagon_density(dens[k,:],ll_site_idx,L=L) for k in range(np.size(times))])
		
		means[i] = np.mean(hex_density)
		stdevs[i] = np.std(hex_density)
		
	np.savez(filename,U_vals=U_vals,means=means,stdevs=stdevs,L=L,ll_site_idx=ll_site_idx)


def plot_mean_data(filename,logscale=False):
	#Plot data of the late-time average value of the original hexagon density vs U, from npz file
	data = np.load(filename)
	U_vals = data['U_vals']
	means = data['means']
	stdevs = data['stdevs']
	
	fig,ax = plt.subplots()
	ax.errorbar(U_vals,means,yerr=stdevs,color='b',marker='x',ls='',label='10x10 Data')
	
	ax.set_xlabel('U')
	ax.set_ylabel(r'$\langle n \rangle$',rotation=0)
	
	ax.set_title('Stationary Density vs U')
	
	if logscale:
		ax.set_xscale('log')
		ax.set_yscale('log')
		
	ax.legend()
	plt.show()
	
	
def generate_low_U_fit_data(filename,U_vals,L=10,ll_site_idx=133,bc='periodic'):
	#Generate Gaussian + constant fit parameters vs U. This function is used for low-U region (<~1)
	#where the gaussian + constant fit is suitable
	b_vals = np.zeros(np.size(U_vals))
	c_vals = np.zeros(np.size(U_vals))
	b_err = np.zeros(np.size(U_vals))
	c_err = np.zeros(np.size(U_vals))

	
	for i,U in enumerate(U_vals):
		print(f'Evaulating U={U}...')
		system = Kag2.two_hexagons_system(idx1=ll_site_idx,idx2=ll_site_idx,U=U,L=L,skip_diag=True,evolution_method='propagator',bc=bc,skip_k=True)
		T = 130/(3*U)
		times = np.linspace(0,T,100)
		dt = times[1] - times[0]
		
		dens = np.zeros((np.size(times),system.N_sites))
		
		#system.psi = system.state(times[0])
	
		dens[0,:] = system.density()
		for j in range(1,np.size(times)):
			system.psi = system.evolve_psi(dt)
			dens[j,:] = system.density()
			print(f'Progress: {np.round(100*(j+1)/np.size(times),2)}%',end='\r')

		hex_density = np.array([hexagon_density(dens[k,:],ll_site_idx,L=L) for k in range(np.size(times))])
		
		popt,pcov = sp.optimize.curve_fit(gaussian2_const_fit,times,hex_density,p0=[10/U,0.05])
		perr = np.sqrt(np.diag(pcov))
		
		b_vals[i] = popt[0]
		c_vals[i] = popt[1]
		b_err[i] = perr[0]
		c_err[i] = perr[1]
		
	np.savez(filename,U_vals=U_vals,b_vals=b_vals,c_vals=c_vals,b_err=b_err,c_err=c_err,L=L,ll_site_idx=ll_site_idx)


def plot_b_lowU(filename,ax=None,logscale=False,plot_fit=False,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot the gaussian decay constant vs U, from npz file
	data = np.load(filename)
	U = data['U_vals']
	b = np.abs(data['b_vals'])			#Since may have converged to -b if unlucky
	b_err = data['b_err']
	
	if ax is None:
		fig,ax = plt.subplots()
	ax.errorbar(U,b,yerr=b_err,color='b',marker='x',ls='',label='Data')
	
	if plot_fit == 'power':
		popt,pcov = sp.optimize.curve_fit(power_fit,U[fit_idx1:fit_idx2+1],b[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = power_fit(U,popt[0],popt[1])
		ax.plot(U,yfit,'r--',label='Fit')
		if printparams:
			print('Fit y = A * U^n')
			print(f'A = {popt[0]} +/- {perr[0]}')
			print(f'n = {popt[1]} +/- {perr[1]}')
	
	ax.set_xlabel(r'$U$ / $J$',fontsize=15)
	#ax.set_ylabel(r'$\tau$ / $t_0$',fontsize=15,rotation=0,labelpad=20)
	ax.set_ylabel(r'$\tau$ / $t_0$',fontsize=15)
	if logscale:
		ax.set_yscale('log')
		ax.set_xscale('log')

	ax.legend(fontsize=15)
	#ax.set_title('Decay Time vs U',fontsize=15)
	
	if ax is None:
		plt.show()
	
	
def plot_c_lowU(filename,logscale=False,plot_fit=False,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot the constant term 'c' in the Gaussian + constant fitting data vs U, from npz file
	data = np.load(filename)
	U = data['U_vals']
	c = data['c_vals']
	c_err = data['c_err']
	
	fig,ax = plt.subplots()
	ax.errorbar(U,c,yerr=c_err,color='b',marker='x',ls='',label='10x10 Data')
	
	if plot_fit == 'power':
		popt,pcov = sp.optimize.curve_fit(power_fit,U[fit_idx1:fit_idx2+1],c[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = power_fit(U,popt[0],popt[1])
		ax.plot(U,yfit,'r--',label='Power Law Fit')
		if printparams:
			print('Fit y = A * U^n')
			print(f'A = {popt[0]} +/- {perr[0]}')
			print(f'n = {popt[1]} +/- {perr[1]}')
	
	ax.set_xlabel('U')
	ax.set_ylabel('c',rotation=0)
	if logscale:
		ax.set_yscale('log')
		ax.set_xscale('log')

	ax.legend()
	ax.set_title('Stationary Density vs U')
	plt.show()
	
	
def generate_tpeak_data(filename,U_vals,L=10,ll_site_idx=133,bc='periodic'):
	#Generate data extracting the 'recombination time' of the second peak in the original hexagon density
	t_peak_vals = np.zeros(np.size(U_vals))
	t_peak_errs = np.zeros(np.size(U_vals))

	
	for i,U in enumerate(U_vals):
		print(f'Evaulating U={U}...')
		system = Kag2.two_hexagons_system(idx1=ll_site_idx,idx2=ll_site_idx,U=U,L=L,skip_diag=True,evolution_method='propagator',bc=bc,skip_k=True)
		if bc == 'periodic':
			t_p = 150/U
		else:
			t_p = 130/U
		times = np.linspace(0.5*t_p,1.2*t_p,200)
		dt = times[1] - times[0]
		t_peak_errs[i] = dt
		
		dens = np.zeros((np.size(times),system.N_sites))
		
		system.psi = system.state(times[0])
		dens[0,:] = system.density()
		for j in range(1,np.size(times)):
			system.psi = system.evolve_psi(dt)
			dens[j,:] = system.density()
			print(f'Progress: {np.round(100*(j+1)/np.size(times),2)}%',end='\r')

		hex_density = np.array([hexagon_density(dens[k,:],ll_site_idx,L=L) for k in range(np.size(times))])
		
		t_peak_vals[i] = times[np.argmax(hex_density)]
		
	np.savez(filename,U_vals=U_vals,t_peak_vals=t_peak_vals,t_peak_errs=t_peak_errs,L=L,ll_site_idx=ll_site_idx)
	
	
def plot_tpeak(filename,ax=None,logscale=True,plot_fit=None,fit_idx1=0,fit_idx2=5,printparams=True):
	#Plot the recombination time against U, from npz file
	data = np.load(filename)
	U_vals = data['U_vals']
	t_peak = data['t_peak_vals']
	t_peak_err = data['t_peak_errs']
	
	if ax is None:
		fig,ax = plt.subplots()
	ax.errorbar(U_vals,t_peak,yerr=t_peak_err,color='b',marker='x',ls='',label='Data')
	
	if plot_fit == 'power':
		popt,pcov = sp.optimize.curve_fit(power_fit,U_vals[fit_idx1:fit_idx2+1],t_peak[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = power_fit(U_vals,popt[0],popt[1])
		ax.plot(U_vals,yfit,'r--',label='Fit')
		if printparams:
			print('Fit y = A * U^n')
			print(f'A = {popt[0]} +/- {perr[0]}')
			print(f'n = {popt[1]} +/- {perr[1]}')
	
	if logscale:
		ax.set_xscale('log')
		ax.set_yscale('log')
		
	ax.set_xlabel(r'$U$ / $J$',fontsize=15)
	#ax.set_ylabel(r'$t_r$ / $t_0$',fontsize=15,rotation=0,labelpad=20,verticalalignment='center')
	ax.set_ylabel(r'$t_r$ / $t_0$',fontsize=15)
	
	ax.legend(fontsize=15)
	
	#ax.set_title('Recombination Time vs U',fontsize=15)
	
	if ax is None:
		plt.show()


if __name__ == '__main__':
	#plot_various_hexagon_densities(['N2_L10x10_SameHexagon_Periodic_T2000_dt10_J1_U100.npz'],colors=['b','r','cyan'],ylim=2.,xlim=None,plot_fit=False,fit_T=[50],p0=None)
	
	#generate_low_U_fit_data(filename='HexDensity_Periodic_Fit_Params_U0.01_1.npz',U_vals=np.logspace(-2,0,num=20))
	
	#plot_b_lowU('HexDensity_Fit_Params_U0.01_100Combined.npz',plot_fit=None,fit_idx1=0,fit_idx2=5,logscale=True)
	
	#plot_c_lowU('HexDensity_Fit_Params_U0.01_100Combined.npz',plot_fit=None,fit_idx1=3,fit_idx2=9,logscale=False)
	
	#generate_latetime_mean_data(filename='HexDensity_LateTimeMean_U1_100.npz',U_vals=[1.,1.2,1.4,1.6,1.8,2.,3.,4.,5.,6.,7.,8.,9.,10.,20.,30.,40.,50.,60.,70.,80.,90.,100.])
	
	#plot_mean_data('HexDensity_LateTimeMean_U1_100.npz')
	
	generate_tpeak_data('tpeak_Periodic_U0.01_2.npz',U_vals=np.logspace(-2,np.log10(2),20))
	
	#plot_tpeak('tpeak_U0.01_2.npz',plot_fit='power',fit_idx2=10)
	
