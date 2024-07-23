'''
This file contains functions for calculating and plotting the time-averaged
density at each site in the lattice, in order to visualize the extent and 
'shape' of the stationary density when expanding from a single site.
'''


import numpy as np
import matplotlib.pyplot as plt
import Kagome as Kag
import matplotlib
import scipy as sp


def calc_avg_density(filename,newfilename,t0=10):
	#From a file containing density vs time at each site in the lattice, 
	#save a file containing the time-averaged density from time t=t0 onwards
	
	data = np.load(filename)
	density = data['density']
	t = data['t']
	
	N_sites = density.shape[1]
	mean_density = np.array([np.mean(density[t>t0,i]) for i in range(N_sites)])
	stdev_density = np.array([np.std(density[t>t0,i]) for i in range(N_sites)])
	
	np.savez(newfilename,mean_density=mean_density,stdev_density=stdev_density,N_sites=N_sites)
	
	
def plot_avg_density(filename,L=20,log=True,plot_cbar=True):
	#Plot the time-averaged density at each site in the lattice
	
	data = np.load(filename)
	mean_density = data['mean_density']
	#if log:
	#	mean_density = np.log10(mean_density)
	mean_psi = np.sqrt(mean_density)		#Dummy 'wavefunction' giving the same density distribution, to allow plot_state_tiled method to be used
	
	
	system = Kag.Kagome(psi_0=mean_psi,Lx=L,Ly=L,skip_diag=True)
	
	fig,ax = plt.subplots()
	box = ax.get_position()
	box.x0 -= 0.1
	box.x1 -= 0.1
	ax.set_position(box)
	ax.set_xticks([])
	ax.set_yticks([])
	
	
	if log:
		norm = matplotlib.colors.LogNorm(vmin=np.min(mean_density), vmax=np.max(mean_density))
	else:
		norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(mean_density))

	cmap = plt.cm.Blues
	sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
	for i,site in enumerate(system.sites):
		tile = Kag.site_tile(site,a=1,color=cmap(norm(mean_density[i])))
		ax.add_patch(tile)	

	system.plot_lattice(ax,color='k',plot_sites=False,thickness=1)
	
	ax.set_aspect('equal')
	
	if plot_cbar:
		cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
		cbar = plt.colorbar(mappable=sm,cax=cax)
		cbar.set_label(r'$\langle n \rangle$',rotation=0,y=0.5,labelpad=10)
	
	plt.show()
	
	
def plot_avg_density_vs_idx(filename,ploterr=False,log=True):
	#Plots the time-averaged density against site index
	data = np.load(filename)
	mean_density = data['mean_density']
	stdev_density = data['stdev_density']
	N_sites = data['N_sites']
	
	fig,ax = plt.subplots()
	
	if ploterr:
		ax.errorbar(np.arange(N_sites),mean_density,yerr=stdev_density,color='b',ls='',marker='x',label='Density')
	else:
		ax.plot(np.arange(N_sites),mean_density,color='b',ls='',marker='x',label='Density')
	
	ax.axhline(y=(2/3)/(N_sites),color='r',ls=':',label='Dispersive Lattice Filling')
	
	ax.legend()
	ax.set_xlabel('Site Index')
	ax.set_ylabel(r'$\overline{\langle n \rangle}$')
	
	if log:
		ax.set_yscale('log')
	
	plt.show()
	
	
def power_fit(x,A,n):
	return A*x**n
	
	
def plot_avg_density_vs_r(filename,L=20,initial_site_idx=690,subtract_avg=True,ploterr=True,logy=False,logx=False,cutoff=None,plotfit=None,fit_idx1=0,fit_idx2=10,printparams=True):
	#Plot time-averaged density against radius r from initial site
	
	data = np.load(filename)
	mean_density = data['mean_density']
	stdev_density = data['stdev_density']
	N_sites = data['N_sites']
	
	system = Kag.Kagome(psi_0=np.zeros(N_sites),Lx=L,Ly=L,skip_diag=True)
	r = np.array([np.linalg.norm(site.r - system.sites[690].r) for site in system.sites])
	
	if subtract_avg:
		mean_density -= (2/3)/N_sites
		
	
	if cutoff is not None:
		idxs = np.where(mean_density>=cutoff)
		mean_density = mean_density[idxs]
		stdev_density = stdev_density[idxs]
		r = r[idxs]
		
	#Sort descending
	idxs = np.argsort(mean_density)[::-1]
	mean_density = mean_density[idxs]
	stdev_density = stdev_density[idxs]
	r = r[idxs]
	
	
	
	
	fig,ax = plt.subplots()
	if ploterr:
		ax.errorbar(r,mean_density,yerr=stdev_density,color='b',marker='x',ls='',label='Density')
	else:
		ax.plot(r,mean_density,color='b',marker='x',ls='',label='Density')
		
	if logx:
		#Exclude r=0
		mean_density = mean_density[1:]
		stdev_density = stdev_density[1:]
		r = r[1:]
		ax.set_xscale('log')
	if logy:
		if cutoff is None:
			ax.set_yscale('symlog')
		else:
			ax.set_yscale('log')
	
			
	if plotfit == 'power':
		popt,pcov = sp.optimize.curve_fit(power_fit,r[fit_idx1:fit_idx2+1],mean_density[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		xfit = np.linspace(r[fit_idx1],r[fit_idx2],100)
		yfit = power_fit(xfit,popt[0],popt[1])
		ax.plot(xfit,yfit,'r--',label='$y = A r^n$')
		if printparams:
			print(f'A = {popt[0]} +/- {perr[0]}')
			print(f'n = {popt[1]} +/- {perr[1]}')
		ax.legend()
		
		
		
	
	ax.set_xlabel('r / a')
	ax.set_ylabel(r'$\overline{\langle n \rangle}$')
	
	plt.show()
	
			
	
	
	
	
if __name__ == '__main__':
	#calc_avg_density('N1_L20x20_SingleSite_Open_T10000_dt1.npz','MeanDensity_AllSites_L20x20_T10000.npz',t0=10)
	
	#plot_avg_density('MeanDensity_AllSites_L20x20_T10000.npz',L=20)
	
	#plot_avg_density_vs_idx('MeanDensity_AllSites_L20x20_T10000.npz',ploterr=False)
	
	plot_avg_density_vs_r('MeanDensity_AllSites_L20x20_T10000.npz',ploterr=False,logy=True,logx=True,subtract_avg=True,cutoff=1e-3,plotfit='power',fit_idx1=1,fit_idx2=19)
