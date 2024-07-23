'''
This file plots a kagome lattice diagram, with lattice vectors indicated.
'''

import Kagome as Kag
import numpy as np
import matplotlib.pyplot as plt


def plot_lattice_diagram(ax,Lx=3,Ly=3,plotfig=True,thickness=1,markersize=5):
	#Plot the lattice diagram

	system = Kag.Kagome(psi_0=np.zeros(3*Lx*Ly),Lx=Lx,Ly=Ly)
	
	system.plot_lattice(ax,color='k',thickness=thickness,markersize=markersize,plot_lattice_vectors=False)
	
	ax.arrow(0,0,system.a1[0],system.a1[1],length_includes_head=True,facecolor='r',edgecolor='r',width=0.02,head_length=system.a/10)
	ax.arrow(0,0,system.a2[0],system.a2[1],length_includes_head=True,color='r',width=0.02,head_length=system.a/10)
	
	ax.text(system.a1[0]-0.25,system.a1[1]+0.09,r'$\bf{a}$'+r'$_1$',fontsize=15,color='r')
	ax.text(system.a2[0]-0.35,system.a2[1]-0.05,r'$\bf{a}$'+r'$_2$',fontsize=15,color='r')
	
	ax.set_xticks([])
	ax.set_yticks([])
	
	if plotfig:
		plt.show()
	
if __name__ == '__main__':
	fig,ax = plt.subplots()
	plot_lattice_diagram(ax,L=3)
	
	
