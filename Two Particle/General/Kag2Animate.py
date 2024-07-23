'''
This file contains the code for animating the density evolution in the kagome lattice.
'''


import Kagome2 as Kag2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import cProfile

matplotlib.rcParams['animation.ffmpeg_path'] = r'/Users/benburgess/Library/CloudStorage/OneDrive-Personal/University/Part III/Project/Kagome/ffmpeg'


def animate(i,ax,lattice_sites,density,t,cmap,norm):
	#Animation function
	for j in range(len(lattice_sites)):
		lattice_sites[j].set_color(cmap(norm(density[i,j])))
	ax.set_title(f'$t$ / $t_0$={np.round(t[i],2)}',x=0.42,horizontalalignment='left',fontsize=15)
	return lattice_sites


def animate_kagome(system,density,t,cmap=plt.cm.rainbow,norm='auto',save_vid=False,filename=None):
	#Given density array (shape (len(t) x N_sites)), create animation of density at each timestep in array of times t.
	if norm == 'auto':
		norm = matplotlib.colors.Normalize(vmin=0, vmax=Kag2.uppernorm(np.max(density)))
	elif norm == '0to1':
		norm = matplotlib.colors.Normalize(vmin=0, vmax=1.)
	sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
	fig,ax = plt.subplots()

	ax.set_xticks([])
	ax.set_yticks([])
	#ax.set_aspect('equal')
	cbar=plt.colorbar(mappable=sm,ax=ax)
	cbar.set_label(r'$\langle n \rangle$',rotation=0,y=0.5,labelpad=5,fontsize=18,verticalalignment='center')
	#cbar.set_ticks([0,0.05,0.1,0.15])
	#cbar.set_ticklabels(ticklabels=[0,0.05,0.1,0.15],fontsize=15)
	
	lattice_sites = tuple(ax.add_patch(Kag2.site_tile(site,a=system.a,color=cmap(norm(density[0,i])))) for i,site in enumerate(system.sites))
	system.plot_lattice(ax,color='k',plot_sites=False)
	anim = animation.FuncAnimation(fig, animate, frames=np.size(t), fargs=(ax,lattice_sites,density,t,cmap,norm), interval=50, blit=False)
	
	plt.show()
	
	if save_vid:
		anim.save(filename,writer=animation.FFMpegWriter(fps=10))
		
	

if __name__ == '__main__':
	data = np.load('N2_L10x10_SameHexagon_Periodic_T1000_dt20_J1_U0.1.npz')
	t = data['t']
	#psi = data['psi']
	density = data['density']
	
	#Lx=5
	#Ly=5
	#psi_0 = psi[0,:]		#Doesn't actually matter what psi_0 is here, since all system is just initialised to provide lattice points; all computation of states has been done previously and loaded into this script.
	#system = Kag2.Kagome2(psi_0=psi_0,Lx=Lx,Ly=Ly,skip_diag=True,evolution_method='eigenvector')
	system = Kag2.double_occupied_site_system(L=10,skip_diag=True,evolution_method='eigenvector',skip_k=True)
	
	animate_kagome(system,density,t,norm = matplotlib.colors.Normalize(vmin=0, vmax=0.01),cmap=plt.cm.Blues,save_vid=False,filename='N2_L5x5_SeparateSites_Periodic_T100_dt1_J1_U0.mp4')


