'''
This file contains various functions for calculating quantities in 
the reciprocal k-space
'''


import numpy as np
import matplotlib.pyplot as plt
import Kagome as Kag
import Kagome2 as Kag2


def H_k(k,J=1.,a=1.):
	#Return the (single-particle) Hamiltonian in k-space at a given k value
	H = np.zeros((3,3))
	
	a1 = a*np.array([0.5,0])
	a2 = a*np.array([0.25,np.sqrt(3)/4])
	a3 = a2 - a1
	
	c1 = np.cos(np.dot(k,a1))
	c2 = np.cos(np.dot(k,a2))
	c3 = np.cos(np.dot(k,a3))
	
	H[0,1] = c1
	H[1,0] = c1
	H[0,2] = c2
	H[2,0] = c2
	H[1,2] = c3
	H[2,1] = c3
	
	return -2*J*H
	
	
def comp_exp(system,idx,k):
	#Return the complex exponential exp(i k dot r), where r is the position of the site with index idx
	site = system.sites[idx]
	return np.cos(np.dot(k,site.r)) - 1j*np.sin(np.dot(k,site.r))
	
	
def plot_eigvects(k,L=10,dens=False,wf=True,bands=[0,1,2]):		#basis vectors for k should be the non-orthogonal lattice vectors
	#Plot the eigenvectors of the matrix H(k)
	a1 = np.array([1,0])
	a2 = np.array([0.5,np.sqrt(3)/2])
	k_cart = a1*k[0] + a2*k[1]		#NB a1 and a2 already normalised here
	H = H_k(k_cart)
	evals,evects = np.linalg.eigh(H)

	N = 3*L**2
	system = Kag.Kagome(psi_0=np.zeros(N),Lx=L,Ly=L)
	for j in bands:
		v = evects[:,j]
		psi = np.zeros(N,dtype=np.complex128)
		for i in range(L**2):
			for l in range(3):
				r_lv = system.sites[3*i+l].r_lv + np.array([0.5,0.5])
				psi[3*i+l] = v[l] * np.sqrt(2) * np.sin(r_lv[0]*k[0]) * np.sin(r_lv[1]*k[1]) / L
				#psi[3*i+l] = v[l] * comp_exp(system,3*i+l,k) / L
		system.psi = psi
		if wf:
			fig,ax = plt.subplots()
			system.plot_re_wavefunction_tiled(fig,ax)
			ax.set_aspect('equal')
			ax.set_title(f'$\omega$ = {np.round(evals[j],3)}')
			plt.show()
		if dens:
			fig,ax = plt.subplots()
			system.plot_state_tiled(fig,ax,cmap=plt.cm.Blues)
			ax.set_aspect('equal')
			plt.show()
			
			
def plot_eigvects_old(k,L=10,dens=False,wf=True,bands=[0,1,2]):
	#Old method for plotting eigenvectors of H(k)
	R = np.array([[-0.5,-np.sqrt(3)/2],[np.sqrt(3)/2,-0.5]])
	
	k1 = k
	k2 = np.transpose(R@k1.T)
	k3 = np.transpose(R@k2.T)
	
	H1 = H_k(k1)
	H2 = H_k(k2)
	H3 = H_k(k3)
	evals1,evects1 = np.linalg.eigh(H1)
	evals2,evects2 = np.linalg.eigh(H2)
	evals3,evects3 = np.linalg.eigh(H3)
	
	#print(evals1[0],evals2[0],evals3[0])

	N = 3*L**2
	system = Kag.Kagome(psi_0=np.zeros(N),Lx=L,Ly=L)
	
	for j in bands:
		v1 = evects1[:,j]
		v2 = evects2[:,j]
		v3 = np.abs(evects3[:,j])
		#print(v1,v2,v3)
		#print(v)
		psi = np.zeros(N,dtype=np.complex128)
		for l in range(3):
			for i in range(L**2):
				r = system.sites[3*i+l].r - 0.5*(L-1)*(system.a1 + system.a2)
				#psi[3*i+l] = comp_exp(system,3*i+l,k)
				psi[3*i+l] +=  np.cos(np.dot(r,k1)) * v1[l] / L
				psi[3*i+l] +=  np.cos(np.dot(r,k2)) * v2[l] / L
				psi[3*i+l] +=  np.cos(np.dot(r,k3)) * v3[l] / L
		psi = psi / np.linalg.norm(psi)
		system.psi = psi
		if wf:
			fig,ax = plt.subplots()
			system.plot_re_wavefunction_tiled(fig,ax)
			ax.set_aspect('equal')
			plt.show()
		if dens:
			fig,ax = plt.subplots()
			system.plot_state_tiled(fig,ax,cmap=plt.cm.Blues)
			ax.set_aspect('equal')
			plt.show()
			

def check_orthogonality(L=4):
	#Checks that the 2-particle k-states |k1,k2> are orthnormal, by calculating the inner product between an arbitrarily
	#chosen |k1,k2> state, and all the other states
	system = Kag2.double_occupied_site_system(L=L,U=0,skip_diag=True,evolution_method='eigenvector',bc='periodic')
	
	k1 = system.k_BZ[int(L**2/4),:]
	k2 = system.k_BZ[int(2*L**2/3),:]
	
	psi_k1k2 = system.psi_k(k1,k2)
	
	N = int(0.5*L**2*(L**2+1))
	overlap = np.zeros(N,dtype=np.complex(128))
	n = 0
	
	for i in range(L**2):
		for j in range(i,L**2):
			psi_kikj = system.psi_k(system.k_BZ[i,:],system.k_BZ[j,:])
			overlap[n] = np.vdot(psi_k1k2,psi_kikj)
			n += 1
			
	fig,ax = plt.subplots()
	ax.plot(np.arange(N),np.abs(overlap),'b-')
	ax.set_xlabel(r'Index of $|k,k^\prime \rangle$')
	ax.set_ylabel(r'$|\langle k_{1},k_{2}|k,k^\prime \rangle|$')
	ax.set_title(f'$k_1$ = {np.round(k1,2)}, $k_2$ = {np.round(k2,2)}')
	plt.show()
	
	

	


		
	
	
if __name__ == '__main__':
	#plot_eigvects_old(4*np.pi/(10*np.sqrt(3)) * np.array([1,0]),dens=False,bands=[0])
	
	check_orthogonality(L=5)
	
	
		
