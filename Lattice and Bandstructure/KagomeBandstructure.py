'''
This file plots the Kagome bandstructure
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def E1(k,a=1.,J=1.):
	#Return the eigenvalue of the lowest band at wavevector k
	
	#Lattice vectors:
	a1 = a*np.array([1,0])
	a2 = a*np.array([0.5,np.sqrt(3)/2])
	a3 = a2 - a1
	
	c1 = np.cos(np.dot(k,a1)/2)
	c2 = np.cos(np.dot(k,a2)/2)
	c3 = np.cos(np.dot(k,a3)/2)
	
	return J*(-1-np.sqrt(4*(c1**2+c2**2+c3**2)-3))
	
def E2(k,a=1.,J=1.):
	#Return the eigemvalue of the middle band at wavevector k
	
	#Lattice vectors:
	a1 = a*np.array([1,0])
	a2 = a*np.array([0.5,np.sqrt(3)/2])
	a3 = a2 - a1
	
	c1 = np.cos(np.dot(k,a1)/2)
	c2 = np.cos(np.dot(k,a2)/2)
	c3 = np.cos(np.dot(k,a3)/2)
	
	return J*(-1+np.sqrt(4*(c1**2+c2**2+c3**2)-3))
	
def E3(k,a=1.,J=1.):
	#Return the eigemvalue of the flat band at wavevector k
	return 2*J
	
	
def hexagon_patch(l,a,yscale=0.03,facecolor=None,edgecolor='k'):
	#Returns a hexagonal patch for adding to plots
	
	ll = l + a*np.array([0.5,-yscale*np.sqrt(3)/2])
	lr = ll + a*np.array([1,0])
	r = lr + a*np.array([0.5,yscale*np.sqrt(3)/2])
	ur = r + a*np.array([-0.5,yscale*np.sqrt(3)/2])
	ul = ur - a*np.array([1,0])
	#l = ul - a*np.array([0.5,(3/125)*np.sqrt(3)/2])
	return matplotlib.patches.Polygon(np.array([l,ll,lr,r,ur,ul]),facecolor=facecolor,edgecolor=edgecolor)
	

def plot_bandstructure(ax,a=1.,J=1.,savefig=False,plotfig=True,yscale=0.03):
	#Plots the kagome bandstructure for the path in the BZ Gamma -> M -> K -> Gamma
	
	k_G = np.zeros(2)
	#k_M = np.pi/a*np.array([1,1/np.sqrt(3)])
	#k_K = 2*np.pi/(a*np.sqrt(3))*np.array([1,0])
	k_M = 2*np.pi/(a*np.sqrt(3))*np.array([0,1])
	k_K = 2*np.pi/(a*np.sqrt(3))*np.array([1/np.sqrt(3),1])
	
	#fig,ax = plt.subplots()
	
	#Gamma to M
	N_GM = 100
	x_GM = np.arange(0,N_GM+1,1)
	E1_GM = np.zeros(N_GM+1)
	E2_GM = np.zeros(N_GM+1)
	E3_GM = np.zeros(N_GM+1)
	dk_GM = (k_M-k_G)/N_GM
	for i in range(N_GM+1):
		k = k_G + i*dk_GM
		E1_GM[i] = E1(k,a=a,J=J)
		E2_GM[i] = E2(k,a=a,J=J)
		E3_GM[i] = E3(k,a=a,J=J)
		
	#M to K
	N_MK = 50
	x_MK = np.arange(N_GM,N_GM+N_MK+1,1)
	E1_MK = np.zeros(N_MK+1)
	E2_MK = np.zeros(N_MK+1)
	E3_MK = np.zeros(N_MK+1)
	dk_MK = (k_K-k_M)/N_MK
	for i in range(N_MK+1):
		k = k_M + i*dk_MK
		E1_MK[i] = E1(k,a=a,J=J)
		E2_MK[i] = E2(k,a=a,J=J)
		E3_MK[i] = E3(k,a=a,J=J)
		
	#K to Gamma
	N_KG = 100
	x_KG = np.arange(N_GM+N_MK,N_GM+N_MK+N_KG+1,1)
	E1_KG = np.zeros(N_KG+1)
	E2_KG = np.zeros(N_KG+1)
	E3_KG = np.zeros(N_KG+1)
	dk_KG = (k_G-k_K)/N_KG
	for i in range(N_KG+1):
		k = k_K + i*dk_KG
		E1_KG[i] = E1(k,a=a,J=J)
		E2_KG[i] = E2(k,a=a,J=J)
		E3_KG[i] = E3(k,a=a,J=J)
	
	ax.plot(x_GM,E1_GM,'r-')
	ax.plot(x_GM,E2_GM,'b-')
	ax.plot(x_GM,E3_GM,'g-')
	ax.plot(x_MK,E1_MK,'r-')
	ax.plot(x_MK,E2_MK,'b-')
	ax.plot(x_MK,E3_MK,'g-')
	ax.plot(x_KG,E1_KG,'r-')
	ax.plot(x_KG,E2_KG,'b-')
	ax.plot(x_KG,E3_KG,'g-')
	
	#Add BZ diagram
	BZ = hexagon_patch(np.array([20,-1]),30,yscale=yscale,facecolor='w',edgecolor='k')
	ax.add_patch(BZ)
	
	ax.plot([50],[-1],'ko',markersize=5)
	ax.plot([50],[-1+yscale*30*np.sqrt(3)/2],'ko',markersize=5)
	ax.plot([65],[-1+yscale*30*np.sqrt(3)/2],'ko',markersize=5)
	
	ax.text(40,-1,'$\Gamma$',verticalalignment='center',fontsize=15)
	ax.text(50,-1+yscale*30*np.sqrt(3)/2+0.2,'M',horizontalalignment='center',fontsize=15)
	ax.text(65,-1+yscale*30*np.sqrt(3)/2+0.2,'K',horizontalalignment='center',fontsize=15)
	
	
	ax.set_ylabel('Energy / J',fontsize=15)
	ax.set_yticks(np.arange(-4,2.1,1))
	ax.set_yticklabels(np.arange(-4,2.1,1),fontsize=15)
	ax.set_xticks([0,N_GM,N_GM+N_MK,N_GM+N_MK+N_KG])
	ax.set_xticklabels(['$\Gamma$','M','K','$\Gamma$'],fontsize=20)
	ax.set_xlim(0,N_GM+N_MK+N_KG)
	
	if savefig:
		plt.savefig('KagomeBandstructure.png',dpi=300)
	
	if plotfig:
		plt.show()
	
	
	
	
if __name__ == '__main__':
	fig,ax = plt.subplots()
	plot_bandstructure(ax,savefig=True)
	
	
	
	
	
