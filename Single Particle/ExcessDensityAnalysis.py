'''
This file contains the code for analyzing the 'excess' density in the original 
site (for single-site initial state one-particle simulations) above the predicted 
value of 1/9.
'''

import numpy as np
import matplotlib.pyplot as plt
import Kagome as Kag
import scipy as sp


def generate_means_data(L_vals,filename):
	#Generates mean late-time density data in original site (after dispersive density has propagated away),
	#for various system sizes
	means = np.zeros(np.size(L_vals))
	for i,L in enumerate(L_vals):
		print(f'Evaluating L = {L}...')
		
		N = 3*L**2
		psi_0 = np.zeros(N)
		
		if N%2 == 0:
			orig_site_idx = int(N/2 + 3*int(L/2))
		else:
			orig_site_idx = int(N/2)-1
			
		psi_0[orig_site_idx] = 1
		system = Kag.Kagome(psi_0=psi_0,Lx=L,Ly=L)
		
		times = np.arange(10,100.1,0.1)
		orig_site_density = np.zeros(np.size(times))
		for j,t in enumerate(times):
			orig_site_density[j] = system.density(t)[orig_site_idx]
			
		means[i] = np.mean(orig_site_density)
	
	np.savez(filename,means=means,L_vals=L_vals)
	print('Done')
	
	
def plot_means(filename,plotfit=False,plotasymptote=False):
	#Plots mean late-time density in original site, from file
	data = np.load(filename)
	means = data['means']
	L = data['L_vals']
	N = 3*L**2
	
	plt.plot(N,means,'b-')
	plt.xlabel('N')
	plt.ylabel('Mean Density')
	if plotfit:
		x = np.linspace(np.min(N),np.max(N),100)
		y = (1/9) + 2.2/x
		plt.plot(x,y,'r--')
	if plotasymptote:
		x = np.linspace(np.min(N),np.max(N),100)
		plt.plot(x,1/9*np.ones(100),'k--')
	plt.show()

	
def plot_log_excess(filename,plotfit=True,printparams=False,savefig=False,plotfig=True):
	#Plots log of mean excess density (above 1/9) in original site
	data = np.load(filename)
	means = data['means']
	excess = means - np.ones(np.size(means))/9
	L = data['L_vals']
	N = 3*L**2
	
	plt.plot(np.log(N),np.log(excess),'b-',label='Data')
	plt.xlabel('ln(N)')
	plt.ylabel(r'$\ln(\langle n \rangle - \frac{1}{9})$',rotation=0,labelpad=10)
	plt.yticks([-5,-6,-7,-8])
	plt.ylim(-8,-4.5)
	
	if plotfit:
		popt,pcov = sp.optimize.curve_fit(power_fit,N[2:],excess[2:])
		perr = np.sqrt(np.diag(pcov))
		xfit = np.linspace(N[2],N[-1],100)
		yfit = power_fit(xfit,popt[0],popt[1])
		plt.plot(np.log(xfit),np.log(yfit),'r--',label='$y = A N^n$')
		if printparams:
			print(f'A = {popt[0]} +/- {perr[0]}')
			print(f'n = {popt[1]} +/- {perr[1]}')
		plt.legend()
	
	plt.text(6.5,-5.4,f'A = {np.round(popt[0],3)} +/- {np.round(perr[0],3)}')
	plt.text(6.5,-5.6,f'n = {np.round(popt[1],3)} +/- {np.round(perr[1],3)}')
	
	if savefig:
		plt.savefig('Excess_Density_vs_N.png',bbox_inches='tight',dpi=600)
	
	if plotfit:
		plt.show()	
	
def power_fit(x,A,n):
	return A*x**n

if __name__ == '__main__':
	#generate_means_data(L_vals=np.array([5,7,9,11,13,15,17,19,21,23,25,27,29,31]),filename='test3.npz')
	plot_log_excess('test3.npz',savefig=True)
	#plot_means('test3.npz',plotfit=True,plotasymptote=True)
	
'''	
means = np.array([0.119524,0.118717,0.114143,0.113563])

excess = means - np.ones(4)/9

L = np.array([5,10,15,20])
N = 3*L**2


plt.plot(N,excess,'b-')
plt.xlabel('N')
plt.ylabel('Excess Density')
plt.ylim(0,0.01)
plt.show()
'''
