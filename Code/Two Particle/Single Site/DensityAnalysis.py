'''
This file contains the code for generating and analyzing data for an initial state
of two particles occupying a single site.
'''


import numpy as np
import matplotlib.pyplot as plt
import Kagome2 as Kag2
import sys
sys.path.append('/Users/benburgess/Library/CloudStorage/OneDrive-Personal/University/Part III/Project/Kagome/Single Particle')
import Kag1FT as Kag1FT
import scipy as sp


def plot_orig_density(ax,filename,color='b',ls='-',label=None,plot_mean=False,frac=10,rolling_avg=False,n_avg=5,gaussian_kernel=False,d=2,plot_fraction=False,scale_time=False):
	#Plot the density in the original site against time
	data = np.load(filename)
	t = data['t']
	density = data['density']
	
	if 'orig_site_idx' in data.keys():
		orig_site_idx = data['orig_site_idx']
	else:
		orig_site_idx = np.argwhere(density[0,:]>0.1)[0][0]
		if np.size(np.argwhere(density[0,:]>0.1)) > 1:
			print('Warning: more than one site with initial density > 0.1')
		
	orig_density = density[:,orig_site_idx]
	
	if plot_fraction:
		orig_density = orig_density / data['N_particles']
	
	if rolling_avg:
		orig_density = rolling_average(orig_density,n_avg)
		
	elif gaussian_kernel:
		orig_density = gaussian_kernel_smooth(t,orig_density,d)
		
	if scale_time:
		t = t / np.max(t)
	
	ax.plot(t,orig_density,color=color,label=label,ls=ls)
	
	if scale_time:
		ax.set_xlabel(r't / t$_{max}$')
	else:
		ax.set_xlabel(r'$t$ / $t_0$',fontsize=15)
	if plot_fraction:
		ax.set_ylabel(r'$\frac{\langle n \rangle}{N}$',rotation=0,labelpad=10,fontsize=15)
	else:
		ax.set_ylabel(r'$\langle n \rangle$',rotation=0,labelpad=10,fontsize=15,verticalalignment='center')
	#ax.set_title(filename)
	ax.set_yticks([0,1,2])
	ax.set_xlim(t[0],t[-1])
	
	if plot_mean:
		mean = np.mean(orig_density[int(np.size(orig_density)/frac):])
		ax.plot(t[int(np.size(orig_density)/frac):],mean*np.ones(np.size(t[int(np.size(orig_density)/frac):])),'r--',label=f'Mean = {np.round(mean,4)}')


def rolling_average(data,n_avg):
	#Given data, smooth with a window kernel, i.e. replacing each data point by the average of the n_avg points either side
	N = np.size(data)
	data_avg = np.zeros(N)
	for i in range(N):
		if i < n_avg:
			data_avg[i] = np.mean(data[:i+n_avg+1])
		elif N-i < n_avg+1:
			data_avg[i] = np.mean(data[i-n_avg:])
		else:
			data_avg[i] = np.mean(data[i-n_avg:i+n_avg+1])
	return data_avg


def gaussian_kernel_smooth(xdata,ydata,d):
	#Smooth data with a gaussian kernel
	N = np.size(ydata)
	data_smoothed = np.zeros(N)
	for i in range(N):
		data_smoothed[i] = np.sum([K(xdata[i],xdata[j],d)*ydata[j] for j in range(N)])/np.sum([K(xdata[i],xdata[j],d) for j in range(N)])
	return data_smoothed


def K(x1,x2,d):		
	#Gaussian Kernel
	return np.exp(-0.5*((x1-x2)/d)**2)

def plot_density_NI_ratio(ax,density_filename,non_interacting_filename,apply_offset=False,color='b',ls='-',label=None):
	#Plot ratio of density in original site with and without interactions
	data = np.load(density_filename)
	t = data['t']
	density = data['density']
	
	if 'orig_site_idx' in data.keys():
		orig_site_idx = data['orig_site_idx']
	else:
		orig_site_idx = np.argwhere(density[0,:]>0.1)[0][0]
		if np.size(np.argwhere(density[0,:]>0.1)) > 1:
			print('Warning: more than one site with initial density > 0.1')
		
	orig_density = density[:,orig_site_idx]
	
	
	data_NI = np.load(non_interacting_filename)
	density_NI = data_NI['density']
	orig_density_NI = density_NI[:,orig_site_idx]			#Assume system sizes, initial site, times etc. all same between two files
	
	if apply_offset:
		orig_density = orig_density - np.mean(orig_density[800:])
		orig_density_NI = orig_density_NI - np.mean(orig_density_NI[800:])
		
	ratio = orig_density / orig_density_NI
	
	ax.plot(t,ratio,color=color,label=label,ls=ls)
	ax.set_xlabel('t / s')
	ax.set_ylabel(r'$\frac{\langle n_U \rangle}{\langle n_0 \rangle}$',rotation=0,labelpad=10)
	#ax.set_title(filename)
	#ax.set_yticks([0,1])
	ax.set_xlim(t[0],t[-1])
	
	
def plot_various_orig_density_avg(n_avg=50):
	#Plot various files of density in original site, with rolling average kernel
	fig,ax = plt.subplots()
	#plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U0.npz',color='y',ls=':',rolling_average=True,n_avg=n_avg,label='U=0')
	#plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U0.1.npz',color='k',rolling_average=True,n_avg=n_avg,label='U=0.1')
	#plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U0.5.npz',color='b',rolling_average=True,n_avg=n_avg,label='U=0.5')
	plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U1.npz',color='b',rolling_average=True,n_avg=n_avg,label='U=1.0')
	plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U2.npz',color='r',rolling_average=True,n_avg=n_avg,label='U=2.0')
	plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U5.npz',color='cyan',rolling_average=True,n_avg=n_avg,label='U=5.0')
	
	t_crit = n_avg*0.1
	ax.axvline(t_crit,color='k',ls='--')
	ax.text(t_crit*1.2,0.8,f't={t_crit} s')
	
	ax.set_ylim(0,1)
	
	ax.legend()
	ax.set_title(f'Original Site Density, n_avg={n_avg}')
	
	plt.show()


def plot_various_orig_density_GK(d=2):
	#Plot various files of density in original site, with Gaussian kernel
	fig,ax = plt.subplots()
	#plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U0.npz',color='y',ls=':',gaussian_kernel=True,d=d,label='U=0')
	#plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U0.1.npz',color='k',gaussian_kernel=True,d=d,label='U=0.1')
	#plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U0.5.npz',color='b',gaussian_kernel=True,d=d,label='U=0.5')
	plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U1.npz',color='b',gaussian_kernel=True,d=d,label='U=1.0')
	plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U2.npz',color='r',gaussian_kernel=True,d=d,label='U=2.0')
	plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J1_U5.npz',color='cyan',gaussian_kernel=True,d=d,label='U=5.0')
	
	ax.set_ylim(0,1)
	
	ax.legend()
	ax.set_title(f'Original Site Density, Gaussian Kernel, d={d}')
	
	plt.show()
	
	
def plot_various_orig_density(filenames,colors=None,labels=None,gaussian_kernel=False,d=2,ylim=2.,plot_fraction=False,plot_mean=False,scale_time=False):
	#Plot various files of density in original site
	fig,ax = plt.subplots()
	if colors is None:
		colors = colors = plt.cm.rainbow(np.linspace(0,1,np.size(filenames)))
	for i,name in enumerate(filenames):
		plot_orig_density(ax,name,color=colors[i],gaussian_kernel=gaussian_kernel,d=d,label=labels[i] if labels is not None else name,plot_fraction=plot_fraction,plot_mean=plot_mean,scale_time=scale_time)
	
	ax.set_ylim(0,ylim)
	ax.legend()
	ax.set_title(f'Original Site Density')
	
	plt.show()
	
	
def exp_constant_fit(filename,rolling_avg=False,n_avg=50,gaussian_kernel=False,d=2,plotlog=True,plotfit=True,printparams=True):
	#Fit an exponential + constant to original site density data
	data = np.load(filename)
	t = data['t']
	density = data['density']
	
	if 'orig_site_idx' in data.keys():
		orig_site_idx = data['orig_site_idx']
	else:
		orig_site_idx = np.argwhere(density[0,:]>0.1)[0][0]
		if np.size(np.argwhere(density[0,:]>0.1)) > 1:
			print('Warning: more than one site with initial density > 0.1')
		
	orig_density = density[:,orig_site_idx]
	
	if rolling_avg:
		orig_density = rolling_average(orig_density,n_avg)
		
	elif gaussian_kernel:
		orig_density = gaussian_kernel_smooth(t,orig_density,d)
	
	if plotfit:
		popt,pcov = sp.optimize.curve_fit(exp_const,t,orig_density,p0=[0.4,0.1,0.1])
		perr = np.sqrt(np.diag(pcov))
		dens_fit = exp_const(t,popt[0],popt[1],popt[2])
		#plt.plot(np.log(xfit),np.log(yfit),'r--',label='$y = A N^n$')
		if printparams:
			print('Fit y = a * exp(-b*t) + c')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
			print(f'c = {popt[2]} +/- {perr[2]}')
	
	fig,ax = plt.subplots()
	
	if plotlog:
		#Subtract constant, then plot <n> and fit. If no fit, subtract mean of final data points
		if plotfit:
			ax.plot(t,np.log(orig_density-popt[2]),color='b',label='Data')
			ax.plot(t,np.log(dens_fit-popt[2]),color='r',ls=':',label='Fit')
		else:
			mean = np.mean(orig_density[800:])
			ax.plot(t,np.log(orig_density-mean),color='b',label='Data')
	else:
		ax.plot(t,orig_density,color='b',label='Data')
		if plotfit:
			ax.plot(t,dens_fit,color='r',ls=':',label='Fit')

	ax.set_xlabel('t / s')
	if plotlog:
		ax.set_ylabel(r'$\ln(\langle n \rangle)$',rotation=0,labelpad=10)
	else:
		ax.set_ylabel(r'$\langle n \rangle$',rotation=0,labelpad=10)
	ax.set_title(filename)
	#ax.set_yticks([0,1,2])
	ax.set_xlim(t[0],t[-1])
	ax.legend()
	
	plt.show()
	

def exp_const(x,a,b,c):
	#Exponential + constant function for fitting
	return a*np.exp(-b*x) + c
	
def linear_fit(x,a,b):
	#Linear fitting function
	return a*x + b
	
def proportional_fit(x,a):
	#Proportional fitting function
	return a*x


def plot_b(filename,plotfit=False,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot exponential decay parameter vs U
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
	ax.set_ylim(0,0.5)
	ax.set_xlim(0,10.1)
	
	plt.show()
	
	
def generate_U_fit_data(filename,U_vals,L=5,initial_state_idx=2070,initial_site_idx=36,T=50,dt=0.1):
	#Generate exponential + constant fit parameters for various U
	a_vals = np.zeros(np.size(U_vals))
	b_vals = np.zeros(np.size(U_vals))
	c_vals = np.zeros(np.size(U_vals))
	a_err = np.zeros(np.size(U_vals))
	b_err = np.zeros(np.size(U_vals))
	c_err = np.zeros(np.size(U_vals))
	
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
		orig_density = gaussian_kernel_smooth(times,orig_density,d=2)
		
		popt,pcov = sp.optimize.curve_fit(exp_const,times,orig_density,p0=[0.4,0.1,0.1])
		perr = np.sqrt(np.diag(pcov))
		
		a_vals[i] = popt[0]
		b_vals[i] = popt[1]
		c_vals[i] = popt[2]
		a_err[i] = perr[0]
		b_err[i] = perr[1]
		c_err[i] = perr[2]
		
	np.savez(filename,U_vals=U_vals,a_vals=a_vals,b_vals=b_vals,c_vals=c_vals,a_err=a_err,b_err=b_err,c_err=c_err,L=L,T=T,dt=dt,orig_state_idx=initial_state_idx,orig_site_idx=initial_site_idx)
	

def power_fit(x,a,b):
	#Power law fitting function
	return a*(x**b)
		

def plot_c(filename,plotfit=False,fit_idx1=0,fit_idx2=0,printparams=True,logy=False,logx=False):
	#Plot constant from exponential + constant fits vs U
	data = np.load(filename)
	U = data['U_vals']
	c = data['c_vals']
	c_err = data['c_err']
	
	fig,ax = plt.subplots()
	ax.errorbar(U,c,yerr=c_err,color='b',marker='x',ls='',label='Data')
	
	if plotfit=='power':
		popt,pcov=sp.optimize.curve_fit(power_fit,U[fit_idx1:fit_idx2+1],c[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * x**b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		c_fit = power_fit(U,popt[0],popt[1])
		ax.plot(U,c_fit,color='r',ls='--',label='Fit')
		ax.legend()
	
	ax.set_xlabel('U')
	ax.set_ylabel('c',rotation=0)
	#ax.set_ylim(0,0.5)
	#ax.set_xlim(0,10.1)
	
	if logx:
		ax.set_xscale('log')
	if logy:
		ax.set_yscale('log')
	
	plt.show()	
	

def generate_J_fit_data(filename,J_vals,L=5,initial_state_idx=2070,initial_site_idx=36,T=50,dt=0.1):
	#Generate exponential + constant fit parameters vs J 
	a_vals = np.zeros(np.size(J_vals))
	b_vals = np.zeros(np.size(J_vals))
	c_vals = np.zeros(np.size(J_vals))
	a_err = np.zeros(np.size(J_vals))
	b_err = np.zeros(np.size(J_vals))
	c_err = np.zeros(np.size(J_vals))
	
	for i,J in enumerate(J_vals):
		print(f'Evaulating J={J}...')
		Lx=L
		Ly=L
		N_sites = 3*Lx*Ly
		N = int(N_sites*(N_sites+1)/2)
		psi_0 = np.zeros(N)
		psi_0[initial_state_idx] = 1
		system = Kag2.Kagome2(psi_0=psi_0,Lx=Lx,Ly=Ly,J=J)
		
		times = np.arange(0,T+dt,dt)
		orig_density = np.array([system.density(t)[initial_site_idx] for t in times])
		orig_density = gaussian_kernel_smooth(times,orig_density,d=2)
		
		popt,pcov = sp.optimize.curve_fit(exp_const,times,orig_density,p0=[0.5,0.12,0.1])
		perr = np.sqrt(np.diag(pcov))
		
		a_vals[i] = popt[0]
		b_vals[i] = popt[1]
		c_vals[i] = popt[2]
		a_err[i] = perr[0]
		b_err[i] = perr[1]
		c_err[i] = perr[2]
		
	np.savez(filename,J_vals=J_vals,a_vals=a_vals,b_vals=b_vals,c_vals=c_vals,a_err=a_err,b_err=b_err,c_err=c_err,L=L,T=T,dt=dt,orig_state_idx=initial_state_idx,orig_site_idx=initial_site_idx)
		

def plot_b_vs_J(filename,plotfit=False,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot exponential decay parameter vs J
	data = np.load(filename)
	J = data['J_vals']
	b = data['b_vals']
	b_err = data['b_err']
	
	fig,ax = plt.subplots()
	ax.errorbar(J,b,yerr=b_err,color='b',marker='x',ls='',label='Data')
	
	if plotfit=='linear':
		popt,pcov=sp.optimize.curve_fit(linear_fit,J[fit_idx1:fit_idx2+1],b[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * J + b')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		b_fit = linear_fit(J,popt[0],popt[1])
		ax.plot(U,b_fit,color='r',ls='--',label='Fit')
		ax.legend()
		
	elif plotfit=='proportional':
		popt,pcov=sp.optimize.curve_fit(proportional_fit,J[fit_idx1:fit_idx2+1],b[fit_idx1:fit_idx2+1])
		perr=np.sqrt(np.diag(pcov))
		if printparams:
			print('Fit y = a * J')
			print(f'a = {popt[0]} +/- {perr[0]}')
		b_fit = proportional_fit(J,popt[0])
		ax.plot(U,b_fit,color='r',ls='--',label='Fit')
		ax.legend()
	
	
	ax.set_xlabel('J')
	ax.set_ylabel('b',rotation=0)
	
	f = lambda x: 1/x
	ax2 = ax.secondary_xaxis('top',functions=(f,f))
	ax2.set_xticks([0.3,0.4,0.5,0.6,0.8,1.0,1.5,2.0,3.0])
	ax2.set_xlabel('U / J')
	
	#ax.set_ylim(0,0.5)
	ax.set_xlim(0.1,3.5)
	
	plt.show()
	
	
def plot_localized_density(filename,r=1.55):
	#Plot total density within a radius r of the original site
	data = np.load(filename)
	t = data['t']
	density = data['density']
	orig_site_idx = 165		#(Here, input manually)

	L = data['Lx']
	N_states = data['N_states']
	
	system = Kag2.Kagome2(psi_0=np.zeros(N_states),Lx=L,Ly=L,skip_H=True,skip_diag=True,evolution_method='eigenvector',skip_k=True)
	
	site_idxs = []
	for i in range(len(system.sites)):
		if np.linalg.norm(system.sites[i].r-system.sites[orig_site_idx].r) < r:
			site_idxs.append(i)

	stat_density = np.zeros(np.size(t))
	for i in range(np.size(t)):
		for idx in site_idxs:
			stat_density[i] += density[i,idx]
			
	mean = np.mean(stat_density/2)
			
	fig,ax = plt.subplots()
	
	ax.plot(t,stat_density/2,color='r',label=r'$U$ / $J = 1$')
	ax.axhline(mean,color='k',ls=':',label=f'Mean = {np.round(mean,3)}')
	
	ax.set_ylim(0,0.2)
	ax.set_xlim(np.min(t),np.max(t))
	
	ax.set_ylabel(r'$\frac{\langle n \rangle}{N}$',fontsize=25,rotation=0,labelpad=15,verticalalignment='center')
	ax.set_xlabel(r'$t$ / $t_0$',fontsize=15)
	ax.set_yticks([0,0.1,0.2])
	ax.set_yticklabels([0,0.1,0.2],fontsize=15)
	ax.set_xticks([4000,4500,5000])
	ax.set_xticklabels([4000,4500,5000],fontsize=15)
	ax.legend(fontsize=15)
	
	
	ax.set_title('Density in Original Site Vicinity',fontsize=15)
	
	plt.show()
	
			
			
	

if __name__ == '__main__':
	#generate_J_fit_data('Density_Fit_Params_Centre_J0.1_10.npz',[0.2,0.3,0.5,0.6,0.8,1.,1.25,1.5,1.75,2.,2.5,3.],initial_state_idx=2070,initial_site_idx=36)
	
	#plot_c('Density_Fit_Params_OffCentre_U0.1_10.npz',logx=False,logy=False,plotfit='power',fit_idx1=5,fit_idx2=8)
	
	#plot_b_vs_J('Density_Fit_Params_Centre_J0.2_3.npz',plotfit=False,fit_idx1=2,fit_idx2=10)
	
	#exp_constant_fit('N2_L5x5_T100_dt0.1_J3_U1.npz',gaussian_kernel=True,plotlog=False)
	
	'''
	fig,ax = plt.subplots()
	plot_orig_density(ax,'N2_L5x5_T100_dt0.1_J0.1_U1.npz',color='k',label='5x5, Centre, J=0.1',gaussian_kernel=True)
	#plot_orig_density(ax,'N2_L5x5_OffCentre_T100_dt0.1_J1_U1.npz',color='b',label='5x5, Off Centre, U=1',gaussian_kernel=False)
	#plot_orig_density(ax,'N2_L5x5_OffCentre_T100_dt0.1_J1_U0.3.npz',color='cyan',label='5x5, Off Centre, U=0.3',gaussian_kernel=False)
	#plot_orig_density(ax,'N2_L7x7_OffCentre_T100_dt0.1_J1_U1.npz',color='cyan',label='7x7, Off Centre',gaussian_kernel=True)
	#plot_orig_density(ax,'N2_L7x7_VeryOffCentre_T100_dt0.1_J1_U1.npz',color='g',label='7x7, Very Off Centre',gaussian_kernel=True)
	
	ax.set_ylim(0,1)
	ax.legend()
	plt.show()
	'''
	
	#filenames = ['N2_L5x5_T100_dt0.1_J1_U1.npz','N2_L5x5_T100_dt0.1_J1_U2.npz','N2_L5x5_T100_dt0.1_J1_U3.npz','N2_L5x5_T100_dt0.1_J1_U4.npz','N2_L5x5_T100_dt0.1_J1_U5.npz']
	#filenames = ['N2_L10x10_SingleSite_Periodic_T5000_dt5_J1_U100.npz','N1_L10x10_Periodic_T5000_dt5_J-0.02.npz']
	#filenames = ['N1_L5x5_T5000_dt5_J-0.02.npz','N1_L5x5_T100_dt0.1_J1.npz']
	filenames = ['N2_L5x5_SeparateSites_Periodic_T100_dt0.1_J1_U0.npz','N2_L5x5_SeparateSites_Periodic_T100_dt0.1_J1_U10.npz','N2_L5x5_SeparateSites_Periodic_T100_dt0.1_J1_U100.npz']
	labels = ['U=0','U=10','U=100']
	colors = ['r','b','cyan']
	plot_various_orig_density(filenames,colors=colors,labels=labels,plot_fraction=False,ylim=1.,gaussian_kernel=False,d=5,plot_mean=False,scale_time=False)
	
	#plot_localized_density(filename='N2_L10x10_SingleSite_Periodic_T4000_5000_dt20_J1_U1.npz')





