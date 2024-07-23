'''
This file contains the code for generating data and  performing frequency analysis of the 
density remaining in the original site, for a single-site initial state.
'''


import Kagome as Kag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import cProfile


def generate_L_data(L,filename,T=100):
	#Generate density vs time data for an L x L system
	N = 3*L**2
	psi_0 = np.zeros(N)
	if N%2 == 0:
		orig_site_idx = int(N/2 + 3*int(L/2))
	else:
		orig_site_idx = int(N/2)-1
	psi_0[orig_site_idx] = 1
	system = Kag.Kagome(psi_0=psi_0,Lx=L,Ly=L)
	times = np.arange(0,T+0.1,0.1)
	density = np.zeros((np.size(times),N))
	for i,t in enumerate(times):
		density[i,:] = system.density(t)
	#density = np.array([system.density(t) for t in times])
	np.savez(filename,t=times,density=density,orig_site_idx=orig_site_idx)
		

def generate_L_dataset(L_vals,filenames,T=100):
	#Generate density vs time data for various system sizes
	for i,L in enumerate(L_vals):
		print(f'Evaluating L={L}...')
		generate_data(L,filenames[i],T)
	print('Done')
	
	
def generate_J_data(J,filename,L=15,T=100):
	#Generate density vs time data for systems with different hopping coefficients J
	N = 3*L**2
	psi_0 = np.zeros(N)
	if N%2 == 0:
		orig_site_idx = int(N/2 + 3*int(L/2))
	else:
		orig_site_idx = int(N/2)-1
	psi_0[orig_site_idx] = 1
	system = Kag.Kagome(psi_0=psi_0,Lx=L,Ly=L,J=J)
	times = np.arange(0,T+0.1,0.1)
	density = np.zeros((np.size(times),N))
	for i,t in enumerate(times):
		density[i,:] = system.density(t)
	#density = np.array([system.density(t) for t in times])
	np.savez(filename,t=times,density=density,orig_site_idx=orig_site_idx)


def plot_orig_density(filename,plot_mean=False):
	#Plot density remaining in original site from file
	data = np.load(filename)
	t = data['t']
	density = data['density']
	orig_density = density[:,data['orig_site_idx']]
	plt.plot(t,orig_density,'b-')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$\langle n \rangle$',rotation=0)
	plt.title(filename)
	
	if plot_mean:
		mean = np.mean(orig_density[int(np.size(orig_density)/10):])
		plt.plot(t[int(np.size(orig_density)/10):],mean*np.ones(np.size(t[int(np.size(orig_density)/10):])),'r--',label=f'Mean = {np.round(mean,6)}')
		plt.legend()
	
	plt.show()
	
	
def plot_ft_orig_density(filename):
	#Plot Fourier Transform of density remaining in original site, from file
	data = np.load(filename)
	t = data['t']
	density = data['density']
	orig_density = density[:,data['orig_site_idx']]
	
	density_ft = np.fft.fft(orig_density)
	freq = np.fft.fftfreq(np.size(orig_density),d=0.1)
	
	density_ft = np.fft.fftshift(density_ft)
	freq = np.fft.fftshift(freq)
	
	density_ps = np.abs(density_ft)**2 			#Power spectrum
	
	plt.plot(freq,np.real(density_ft),'b-',label='Re')
	plt.plot(freq,np.imag(density_ft),'r-',label='Im')
	plt.xlabel('$Frequency / Hz$')
	plt.ylabel(r'$FT(\langle n \rangle)$',rotation=0)
	plt.legend()
	plt.title(filename)
	plt.show()
	
	plt.plot(freq,density_ps,'g-')
	plt.xlabel('$Frequency / Hz$')
	plt.ylabel(r'$|FT(\langle n \rangle)|^2$',rotation=0)
	plt.title(filename)
	plt.show()
	
	
def compare_ps(filenames):
	#Compare power spectra (i.e. |FT|^2) of different density vs time files
	for name in filenames:
		data = np.load(name)
		t = data['t']
		density = data['density']
		orig_density = density[:,data['orig_site_idx']]
	
		density_ft = np.fft.fft(orig_density)
		freq = np.fft.fftfreq(np.size(orig_density),d=0.1)
	
		density_ft = np.fft.fftshift(density_ft)
		freq = np.fft.fftshift(freq)
	
		density_ps = np.abs(density_ft)**2
		
		plt.plot(freq,density_ps,label=name)
	plt.xlabel('$Frequency / Hz$')
	plt.ylabel(r'$|FT(\langle n \rangle)|^2$',rotation=0)
	plt.legend()
	plt.show()
	
	
	
		
if __name__ == '__main__':
	#L_vals = [5,10,15,20]
	#filenames = ['Density_5x5_T100.npz','Density_10x10_T100.npz','Density_15x15_T100.npz','Density_20x20_T100.npz']
	#generate_dataset(L_vals,filenames,T=100)
	
	plot_orig_density('Density_5x5_T100_J1.npz',plot_mean=True)
	#plot_ft_orig_density('Density_5x5_T100.npz')
	
	#compare_ps(filenames)
	
	#generate_J_data(2,'Density_15x15_T100_J2.npz')
	
	#plot_ft_orig_density('Density_20x20_T100_J1.npz')
	
	'''
	plot_orig_density('Density_5x5_T100.npz')
	plot_orig_density('Density_10x10_T100.npz')
	plot_orig_density('Density_15x15_T100.npz')
	plot_orig_density('Density_20x20_T100.npz')
	'''
	
