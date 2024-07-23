import numpy as np
import matplotlib.pyplot as plt

def SingleParticle():
	data = np.load('Density_5x5_T100_J1.npz')
	t = data['t']
	density = data['density']
	orig_site_density = density[:,data['orig_site_idx']]
	
	mean = np.mean(orig_site_density[int(np.size(orig_site_density)/10):])

	fig,ax = plt.subplots()
	
	ax.plot(t,orig_site_density,'b-')
	ax.plot(t[int(np.size(orig_site_density)/10):],mean*np.ones(np.size(t[int(np.size(orig_site_density)/10):])),'r--',label=f'Mean = {np.round(mean,3)}')
	
	ax.set_xlabel('t / s')
	ax.set_ylabel(r'$\langle n \rangle$',rotation=0,labelpad=10)
	ax.set_title('Original Site Density: N=1')
	ax.set_yticks([0.0,0.5,1.0])
	ax.set_xlim(0,100)
	ax.legend()
	
	plt.show()

def Noninteracting_2Particle():
	data = np.load('N2_SingleSite_L5x5_J1_U0.npz')
	t = data['t']
	psi = data['psi']
	density = data['density']
	
	orig_state_idx = 2070
	orig_site_idx = 36
	
	orig_site_density = density[:,orig_site_idx]
	
	mean = np.mean(orig_site_density[int(np.size(orig_site_density)/10):])

	fig,ax = plt.subplots()
	
	ax.plot(t,orig_site_density,'b-')
	ax.plot(t[int(np.size(orig_site_density)/10):],mean*np.ones(np.size(t[int(np.size(orig_site_density)/10):])),'r--',label=f'Mean = {np.round(mean,3)}')
	
	ax.set_xlabel('t / s')
	ax.set_ylabel(r'$\langle n \rangle$',rotation=0,labelpad=10)
	ax.set_title('Original Site Density: N=2, U=0')
	ax.set_yticks([0.0,0.5,1.0,1.5,2.0])
	ax.set_xlim(0,100)
	ax.legend()
	
	plt.show()
	
	
def Comparison():
	data1 = np.load('Density_5x5_T100_J1.npz')
	t = data1['t']
	density1 = data1['density']
	orig_site_density1 = density1[:,data1['orig_site_idx']]
	
	data2 = np.load('N2_SingleSite_L5x5_J1_U0.npz')
	density2 = data2['density']
	
	orig_site_idx2 = 36
	
	orig_site_density2 = density2[:,orig_site_idx2]
	
	#mean = np.mean(orig_site_density[int(np.size(orig_site_density)/10):])

	fig,ax = plt.subplots()
	
	ax.plot(t,orig_site_density1,'k-',label='N=1')
	ax.plot(t,orig_site_density2/2,'r:',label='N=2')
	#ax.plot(t[int(np.size(orig_site_density)/10):],mean*np.ones(np.size(t[int(np.size(orig_site_density)/10):])),'r--',label=f'Mean = {np.round(mean,3)}')
	
	ax.set_xlabel('t / s')
	ax.set_ylabel(r'$\frac{\langle n \rangle}{N}$',rotation=0,labelpad=10,fontsize=15)
	ax.set_title('Fraction of Particles in Original Site')
	ax.set_yticks([0.0,0.5,1.0])
	ax.set_xlim(0,100)
	ax.legend()
	
	plt.show()
	

if __name__ == '__main__':
	Comparison()
