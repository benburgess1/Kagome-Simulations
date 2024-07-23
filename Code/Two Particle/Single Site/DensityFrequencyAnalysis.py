'''
This file contains the code for performing frequency analysis of density in
the original site for 2-particle simulations with a single-site initial state.
'''

import numpy as np
import matplotlib.pyplot as plt
import DensityAnalysis as DA
import Kagome2 as Kag2

def plot_FT(ax,filename,color='b',ls='-',label=None,freq_unit='rad',scale_frequency=False,freq_scale_factor=None):
	#Plot Fourier Transform of density in original site
	data = np.load(filename)
	t = data['t']
	density = data['density']
	
	dt = t[1] - t[0]
	if np.abs(dt - (t[2] - t[1])) > 1e-6:
		print('Warning: variable time steps detected')
	
	if 'orig_site_idx' in data.keys():
		orig_site_idx = data['orig_site_idx']
	else:
		orig_site_idx = np.argwhere(density[0,:]>0.1)[0][0]
		if np.size(np.argwhere(density[0,:]>0.1)) > 1:
			print('Warning: more than one site with initial density > 0.1')
		
	orig_density = density[:,orig_site_idx]
	
	density_ft = np.fft.fft(orig_density)
	freq = np.fft.fftfreq(np.size(orig_density),d=dt)
	
	density_ft = np.fft.fftshift(density_ft)
	freq = np.fft.fftshift(freq)
	if freq_unit == 'rad':
		freq = freq*2*np.pi
		
	if scale_frequency:			#Express frequencies in units of J_eff
		if freq_scale_factor is not None:
			freq = freq/freq_scale_factor
		else:
			freq = freq/(2/data['U'])
	
	density_ps = np.abs(density_ft)**2
	
	ax.plot(freq,density_ps,color=color,ls=ls,label=label)
	
	if freq_unit == 'rad':
		ax.set_xlabel(r'$\omega$ / rad s$^{-1}$')
	elif freq_unit =='Hz':
		ax.set_xlabel('Frequency / Hz')
	else:
		print('Uknown frequency unit')
		
	if scale_frequency:
		ax.set_xlabel(r'$\omega$ / $J_{eff}$')
		
	ax.set_ylabel(r'$|FT(\langle n \rangle)|^2$',rotation=0)
	
	
def plot_various_orig_density_FT(filenames,colors=None,labels=None,scale_frequency=False):
	#Plot FT of various original site density files
	fig,ax = plt.subplots()
	if colors is None:
		colors = plt.cm.rainbow(np.linspace(0,1,np.size(filenames)))
	for i,name in enumerate(filenames):
		plot_FT(ax,name,color=colors[i],label=labels[i] if labels is not None else name,scale_frequency=scale_frequency)
	
	ax.legend()
	ax.set_title(f'Original Site Density Frequency Spectrum')
	
	plt.show()
	
	
def extract_peak(spectrum,freq):	
	#Extract peak of maximum intensity in Fourier spectrum
		
	freq_pos = np.extract(freq>1e-6,freq)		#Exclude f=0 peak and take only positive half of spectrum
	spectrum_pos = np.extract(freq>1e-6,spectrum)
	
	spectrum_max = np.max(spectrum_pos)
	freq_max = freq_pos[np.argwhere(spectrum_pos==spectrum_max)[0][0]]
	
	return freq_max
	
	
def find_peak(filename,freq_unit='rad'):
	#Given density vs t file, find maximum intensity frequency peak
	data = np.load(filename)
	t = data['t']
	density = data['density']
	
	dt = t[1] - t[0]
	if np.abs(dt - (t[2] - t[1])) > 1e-6:
		print('Warning: variable time steps detected')
	
	if 'orig_site_idx' in data.keys():
		orig_site_idx = data['orig_site_idx']
	else:
		orig_site_idx = np.argwhere(density[0,:]>0.1)[0][0]
		if np.size(np.argwhere(density[0,:]>0.1)) > 1:
			print('Warning: more than one site with initial density > 0.1')
		
	orig_density = density[:,orig_site_idx]
	
	density_ft = np.fft.fft(orig_density)
	freq = np.fft.fftfreq(np.size(orig_density),d=dt)
	
	density_ft = np.fft.fftshift(density_ft)
	freq = np.fft.fftshift(freq)
	if freq_unit == 'rad':
		freq = freq*2*np.pi
	
	density_ps = np.abs(density_ft)**2
	
	freq_max = extract_peak(density_ps,freq)
	
	print(freq_max)
	
	
def plot_peaks(filenames,U_vals,freq_unit='rad',color='b'):
	#Plot maximum intensity frequency peaks vs U
	fig,ax = plt.subplots()
	for i,name in enumerate(filenames):
		data = np.load(name)
		t = data['t']
		density = data['density']
	
		dt = t[1] - t[0]
		if np.abs(dt - (t[2] - t[1])) > 1e-6:
			print('Warning: variable time steps detected')
	
		if 'orig_site_idx' in data.keys():
			orig_site_idx = data['orig_site_idx']
		else:
			orig_site_idx = np.argwhere(density[0,:]>0.1)[0][0]
			if np.size(np.argwhere(density[0,:]>0.1)) > 1:
				print('Warning: more than one site with initial density > 0.1')
		
		orig_density = density[:,orig_site_idx]
	
		density_ft = np.fft.fft(orig_density)
		freq = np.fft.fftfreq(np.size(orig_density),d=dt)
	
		density_ft = np.fft.fftshift(density_ft)
		freq = np.fft.fftshift(freq)
		if freq_unit == 'rad':
			freq = freq*2*np.pi
		
		density_ps = np.abs(density_ft)**2
		
		freq_max = extract_peak(density_ps,freq)
		
		ax.plot(U_vals[i],freq_max,color=color,ls='',marker='x')
		
	ax.set_xlabel('U')
	if freq_unit == 'rad':
		ax.set_ylabel(r'$\omega_{peak}$ / rad s$^{-1}$')
	elif freq_unit == 'Hz':
		ax.set_ylabel(r'$f_{peak}$ / rad s$^{-1}$')
	
	plt.show()
	
	
def generate_peak_data(filename,U_vals,T=100,dt=0.1,L=5,initial_state_idx=2070,initial_site_idx=36,freq_unit='rad'):
	#Generate data for maximum intensity frequency components for different U
	freq_peaks = np.zeros(np.size(U_vals))
	
	for i,U in enumerate(U_vals):
		print(f'Evaulating U={U}...')
		Lx=L
		Ly=L
		N_sites = 3*Lx*Ly
		N = int(N_sites*(N_sites+1)/2)
		psi_0 = np.zeros(N)
		psi_0[initial_state_idx] = 1
		system = Kag2.Kagome2(psi_0=psi_0,Lx=Lx,Ly=Ly,U=U)
		
		times = np.arange(0,T+dt,dt)
		orig_density = np.array([system.density(t)[initial_site_idx] for t in times])
		
		density_ft = np.fft.fft(orig_density)
		freq = np.fft.fftfreq(np.size(orig_density),d=dt)
	
		density_ft = np.fft.fftshift(density_ft)
		freq = np.fft.fftshift(freq)
		if freq_unit == 'rad':
			freq = freq*2*np.pi
		
		density_ps = np.abs(density_ft)**2
		
		freq_peaks[i] = extract_peak(density_ps,freq)
		
	np.savez(filename,U_vals=U_vals,freq_peaks=freq_peaks,L=L,T=T,dt=dt,orig_state_idx=initial_state_idx,orig_site_idx=initial_site_idx)


def linear_fit(x,a,b):
	return a*x + b	

	
def plot_peak_data(filename,freq_unit='rad',plotfit=False,fit_idx1=0,fit_idx2=0):
	#Plot maximum intensity frequency component vs U data
	data = np.load(filename)
	U = data['U_vals']
	freq_peaks = data['freq_peaks']
	
	fig,ax = plt.subplots()
	ax.plot(U,freq_peaks,color='b',marker='x',ls='',label='Data')	
	
	if plotfit=='linear':
		popt,pcov=sp.optimize.curve_fit(linear_fit,U[fit_idx1:fit_idx2+1],freq_peaks[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * U + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		freq_fit = linear_fit(U,popt[0],popt[1])
		ax.plot(U,freq_fit,color='r',ls='--',label='Fit')
		ax.legend()
		
	ax.set_xlabel('U')
	if freq_unit == 'rad':
		ax.set_ylabel(r'$\omega_{peak}$ / rad s$^{-1}$')
	elif freq_unit == 'Hz':
		ax.set_ylabel(r'$f_{peak}$ / rad s$^{-1}$')
		
	plt.show()
	
	
def generate_U_data(filename,U_vals,L=5,initial_state_idx=2070,initial_site_idx=36,T_factor=50,dt_factor=5):
	#Generate data for density vs time and its FT, for various U values.
	#Scale the total simulation time and time increment appropriately to obtain comparable spectra
	
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	psi_0[initial_state_idx] = 1
	
	N_times = int(T_factor*dt_factor + 1)
	
	density_t = np.zeros((np.size(U_vals),N_times))		
	density_PS = np.zeros((np.size(U_vals),N_times))
	times = np.zeros((np.size(U_vals),N_times))
	w = np.zeros((np.size(U_vals),N_times))
	
	for i,U in enumerate(U_vals):
		print(f'Evaluating U = {U}...')
		system = Kag2.Kagome2(psi_0=psi_0,Lx=L,Ly=L,U=U,skip_diag=False,evolution_method='eigenvector')
		T = T_factor*U
		dt = U / dt_factor
		time_vals = np.arange(0,T+dt,dt)
		times[i,:] = time_vals
		orig_density = np.array([system.density(t)[initial_site_idx] for t in time_vals])
		density_t[i,:] = orig_density
		orig_density_PS = np.abs(np.fft.fftshift(np.fft.fft(orig_density)))**2
		density_PS[i,:] = orig_density_PS
		w_vals = np.fft.fftshift(np.fft.fftfreq(N_times,d=dt))*2*np.pi
		w[i,:] = w_vals
		
	np.savez(filename,U_vals=U_vals,density_t=density_t,density_PS=density_PS,times=times,w=w,L=L,initial_state_idx=initial_state_idx,initial_site_idx=initial_site_idx,T_factor=T_factor,dt_factor=dt_factor)
		
		
def plot_spectra_fromdata(filename,scale_frequencies=True,sp_filename=None):
	#Plot frequency power spectra vs U
	data = np.load(filename)
	U_vals = data['U_vals']
	density_PS = data['density_PS']
	w = data['w']
	
	fig,ax = plt.subplots()
	colors = plt.cm.rainbow(np.linspace(0,1,np.size(U_vals)))
	for i,U in enumerate(U_vals):
		if scale_frequencies:
			w[i,:] = w[i,:] / (2/U)
		ax.plot(w[i,:],density_PS[i,:],color=colors[i],label=f'U={U}')
		
	if sp_filename is not None:
		plot_FT(ax,sp_filename,color='k',ls=':',label='Single Particle',scale_frequency=False)
		
	ax.legend()
	ax.set_title('Original Site Density Frequency Spectrum vs U')
	ax.set_ylabel(r'$|FT(\langle n \rangle)|^2$',rotation=0)
	if scale_frequencies:
		ax.set_xlabel(r'$\omega$ / $J_{eff}$')
	else:
		ax.set_xlabel(r'$\omega$')
		
	ax.set_xlim(-8,8)
	ax.set_ylim(-100,2e3)
	plt.show()
		
		
		
	
if __name__ == '__main__':
	'''
	fig,ax = plt.subplots()
	#plot_FT(ax,'N2_L5x5_T100_dt0.1_J1_U0.npz',color='r',ls=':',label='U=0')
	plot_FT(ax,'N2_L5x5_T1000_dt0.1_J1_U0.5.npz',color='b',ls='-',label='U=0.5')
	plot_FT(ax,'N2_L5x5_T1000_dt0.1_J1_U0.75.npz',color='r',ls='-',label='U=0.75')
	plot_FT(ax,'N2_L5x5_T1000_dt0.1_J1_U1.npz',color='cyan',ls='-',label='U=1')
	plot_FT(ax,'N2_L5x5_T1000_dt0.1_J1_U1.5.npz',color='g',ls='-',label='U=1.5')
	#plot_FT(ax,'N2_L5x5_T100_dt0.1_J1_U1.npz',label='U=1')
	ax.legend()
	plt.show()
	'''
	
	#filenames = ['N2_L5x5_T100_dt0.1_J1_U0.1.npz','N2_L5x5_T100_dt0.1_J1_U0.5.npz','N2_L5x5_T100_dt0.1_J1_U1.npz','N2_L5x5_T100_dt0.1_J1_U2.npz','N2_L5x5_T100_dt0.1_J1_U3.npz','N2_L5x5_T100_dt0.1_J1_U4.npz','N2_L5x5_T100_dt0.1_J1_U5.npz']
	#plot_peaks(filenames=filenames,U_vals=[0.1,0.5,1,2,3,4,5],color='b')
	#plot_various_orig_density_FT(filenames)
	
	#find_peak('N2_L5x5_T100_dt0.1_J1_U5.npz')
	
	#generate_peak_data('FreqPeaks_L5x5_U0.1_10.npz',U_vals=[0.1,0.3,0.5,0.6,0.7,0.8,0.9,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,8.,9.,10.])
	
	#plot_peak_data('FreqPeaks_L5x5_U0.1_10.npz')
	
	#filenames = ['N2_L5x5_T5000_dt5_J1_U100.npz','N2_L5x5_T500000_dt500_J1_U10000.npz','N1_L5x5_T5000_dt5_J-0.02.npz']
	#filenames = ['N2_L5x5_T5000_dt5_J1_U100.npz','N2_L5x5_T500000_dt500_J1_U10000.npz','N2_L5x5_T50000_dt40_J1_U1000.npz']
	#filenames = ['N2_L5x5_T5000_dt5_J1_U100.npz','N2_L5x5_T100_dt0.01_J1_U100.npz','N1_L5x5_T5000_dt5_J-0.02.npz']
	#labels = ['N=2, U=100','N=2, U=10000','N=2, U=1000']
	#colors = ['b','r','cyan']
	#plot_various_orig_density_FT(filenames,labels=labels,colors=colors,scale_frequency=True)
	
	
	#generate_U_data('OrigDensity_L6x6_U100_100000.npz',U_vals=[100,300,1000,3000,10000,30000,100000],T_factor=250,L=6,initial_state_idx=3933,initial_site_idx=46)
	
	plot_spectra_fromdata('OrigDensity_L6x6_U100_100000.npz',sp_filename='N2_L6x6_T5000_dt0.4_J1_U0.npz')
	
	
	
	
	
	
	
	
