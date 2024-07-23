import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import cProfile
import Kagome as Kag


def long_time():
	data = np.load('L20x20_dt1.npz')
	time = data['t']
	psi = data['psi']
	abs_psi2 = np.real(psi)**2 + np.imag(psi)**2
	
	
	i = 50
	t = time[i]
	Lx=20
	Ly=20
	psi_0 = psi[i,:]
	system = Kag.Kagome(psi_0=psi_0,Lx=Lx,Ly=Ly)
	
	
	fig,ax=plt.subplots()
	
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title(f't={t}')
	#ax.set_aspect('equal')
	
	#cbar=plt.colorbar(mappable=sm,ax=ax)
	#cbar.set_label(r'$|\Psi^2|$',rotation=0)
	
	system.plot_state_tiled(0,fig,ax,norm = matplotlib.colors.Normalize(vmin=0, vmax=0.15),cbar_label_y=0.5)
	plt.show()
	
def long_time_wf():
	data = np.load('L20x20_dt1.npz')
	time = data['t']
	psi = data['psi']
	abs_psi2 = np.real(psi)**2 + np.imag(psi)**2
	
	
	i = 50
	t = time[i]
	Lx=20
	Ly=20
	psi_0 = psi[i,:]
	system = Kag.Kagome(psi_0=psi_0,Lx=Lx,Ly=Ly)
	
	
	fig,ax=plt.subplots()
	
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title(f't={t}')
	#ax.set_aspect('equal')
	
	#cbar=plt.colorbar(mappable=sm,ax=ax)
	#cbar.set_label(r'$|\Psi^2|$',rotation=0)
	
	system.plot_wavefunction_tiled(0,fig,ax,norm = matplotlib.colors.Normalize(vmin=-0.3, vmax=0.3),cbar_label_y=0.5)
	plt.show()
	
	
def estate_wf():
	Lx=5
	Ly=5
	psi_0 = np.zeros(3*Lx*Ly)
	psi_0[24] = 1/np.sqrt(6)
	psi_0[25] = -1/np.sqrt(6)
	psi_0[38] = 1/np.sqrt(6)
	psi_0[36] = -1/np.sqrt(6)
	psi_0[22] = 1/np.sqrt(6)
	psi_0[23] = -1/np.sqrt(6)
	system = Kag.Kagome(psi_0=psi_0,Lx=Lx,Ly=Ly)

	
	times = np.arange(0,40,10)
	system.plot_wf_evolution_tiled(times,norm=matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5),cbar_label_y=1.05)
	plt.show()
	
	
def estate2_wf():
	Lx=5
	Ly=5
	psi_0 = np.zeros(3*Lx*Ly)
	x = 1/np.sqrt(12)
	psi_0[24] = x
	psi_0[25] = -x
	psi_0[39] = x
	psi_0[40] = -x
	psi_0[53] = x
	psi_0[51] = -x
	psi_0[50] = x
	psi_0[48] = -x
	psi_0[34] = x
	psi_0[35] = -x
	psi_0[22] = x
	psi_0[23] = -x
	
	system = Kag.Kagome(psi_0=psi_0,Lx=Lx,Ly=Ly)

	
	times = np.arange(0,40,10)
	system.plot_wf_evolution_tiled(times,norm=matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5),cbar_label_y=1.05)
	plt.show()

'''
Lx=5
Ly=5
psi_0 = np.zeros(3*Lx*Ly)
system = Kag.Kagome(psi_0=psi_0,Lx=Lx,Ly=Ly)
fig,ax=plt.subplots()
system.plot_lattice(ax,label_sites=True,fontsize=5)
plt.show()
'''

estate2_wf()

def estate_density():
	Lx=5
	Ly=5
	psi_0 = np.zeros(3*Lx*Ly)
	psi_0[24] = 1/np.sqrt(6)
	psi_0[25] = -1/np.sqrt(6)
	psi_0[38] = 1/np.sqrt(6)
	psi_0[36] = -1/np.sqrt(6)
	psi_0[22] = 1/np.sqrt(6)
	psi_0[23] = -1/np.sqrt(6)
	system = Kag.Kagome(psi_0=psi_0,Lx=Lx,Ly=Ly)

	
	times = np.arange(0,40,10)
	system.plot_evolution_tiled(times,norm=matplotlib.colors.Normalize(vmin=0., vmax=0.2),cbar_label_y=1.1)
	plt.show()

#estate_density()
