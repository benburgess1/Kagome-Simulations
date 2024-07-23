import Kagome as Kag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import cProfile



		

def animate(i,ax,lattice_points,abs_psi2,t,cmap,norm):
	#system.psi = psi[i,:]
	for j in range(len(lattice_points)):
		lattice_points[j].set_color(cmap(norm(abs_psi2[i,j])))
	ax.set_title(f't={np.round(t[i],2)}')
	return lattice_points


def animate_kagome(system,psi,t,cmap=plt.cm.rainbow,norm=matplotlib.colors.Normalize(vmin=0, vmax=0.1),markersize=5):
	abs_psi2 = np.real(psi)**2 + np.imag(psi)**2
	if norm == 'auto':
		norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(self.abs_psi2()))
	elif norm == '0to1':
		norm = matplotlib.colors.Normalize(vmin=0, vmax=1.)
	sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
	fig,ax = plt.subplots()

	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_aspect('equal')
	cbar=plt.colorbar(mappable=sm,ax=ax)
	cbar.set_label(r'$|\Psi^2|$',rotation=0)
	
	#site_tile(site,a=self.a,color=cmap(norm(self.abs_psi2()[i])))
	
	#lattice_points = tuple(ax.plot(site.x,site.y,marker='o',color=cmap(norm(abs_psi2[0,i])),markersize=markersize)[0] for i,site in enumerate(system.sites))
	lattice_points = tuple(ax.add_patch(Kag.site_tile(site,a=system.a,color=cmap(norm(abs_psi2[0,i])))) for i,site in enumerate(system.sites))
	system.plot_lattice(ax,color='k',plot_sites=False)
	ani = animation.FuncAnimation(fig, animate, frames=np.size(t), fargs=(ax,lattice_points,abs_psi2,t,cmap,norm), interval=50, blit=False)
	
	plt.show()


if __name__ == '__main__':
	data = np.load('test2.npz')
	t = data['t']
	psi = data['psi']
	Lx=10
	Ly=10
	psi_0 = np.zeros(3*Lx*Ly)		#Doesn't actually matter what psi_0 is here, since all system is just initialised to provide lattice points; all computation of states has been done previously and loaded into this script.
	system = Kag.Kagome(psi_0=psi_0,Lx=Lx,Ly=Ly)


	animate_kagome(system,psi,t,markersize=3)

