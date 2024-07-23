'''
This file contains code relating to generating and plotting data relating
to the eigenvalues and eigenvectors of the (interaction-shifted) flat band states.
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.patches as mplPatch
import matplotlib.lines as mplLines
import matplotlib
import kspace as kspace
import Kagome as Kag
import Kagome2 as Kag2
import scipy as sp
from Npair import npair


def v(k,t=1e-6):
	#Return the flat band eigenvector v(k) of the momentum-space (single-particle) Hamiltonian
	c1 = np.cos(0.5*k[0])
	c2 = np.cos(0.25*k[0] + 0.25*np.sqrt(3)*k[1])
	c3 = np.cos(-0.25*k[0] + 0.25*np.sqrt(3)*k[1])
	s1 = np.sin(0.5*k[0])
	s2 = np.sin(0.25*k[0] + 0.25*np.sqrt(3)*k[1])
	s3 = np.sin(-0.25*k[0] + 0.25*np.sqrt(3)*k[1])
	if np.abs(s1) < t and np.abs(s2) < t and np.abs(s3) < t:
		return np.array([1,-0.5+0.5j*np.sqrt(3),-0.5-0.5j*np.sqrt(3)])/np.sqrt(3)
	elif np.abs(s1) < t:
		return np.array([1,-1,0])/np.sqrt(2)
	elif np.abs(s2) < t:
		return np.array([1,0,-1])/np.sqrt(2)
	elif np.abs(s3) < t:
		return np.array([0,1,-1])/np.sqrt(2)
	else:
		v = np.array([s3,-s2,s1])
		return v / np.linalg.norm(v)
		
	#if c1 != 1:
	#	v = np.array([s3,-s2,s1])
	#	return v / np.linalg.norm(v)
		
	#if s1 != 0 and s2 != 0 and s3 != 0:
	#	v = np.array([s3,-s2,s1])
	#	return v / np.linalg.norm(v)
	#elif c2 != 1:
	#	return np.array([1,-1,0])/np.sqrt(2)
	#else:
	#	return np.array([1,-0.5+0.5j*np.sqrt(3),-0.5-0.5j*np.sqrt(3)])/np.sqrt(3)
	
	
def plot_v(ax,a=1.,J=1.,savefig=False,plotfig=True):
	#Plot the three components of v(k) on a path Gamma -> M -> K -> Gamma
	#in the BZ
	
	k_G = np.zeros(2)
	#k_M = np.pi/a*np.array([1,1/np.sqrt(3)])
	#k_K = 2*np.pi/(a*np.sqrt(3))*np.array([1,0])
	k_M = 2*np.pi/(a*np.sqrt(3))*np.array([0,1])
	k_K = 2*np.pi/(a*np.sqrt(3))*np.array([1/np.sqrt(3),1])
	
	
	#fig,ax = plt.subplots()
	
	#Gamma to M
	N_GM = 100
	x_GM = np.arange(0,N_GM+1,1)
	v_GM = np.zeros((N_GM+1,3))
	dk_GM = (k_M-k_G)/N_GM
	for i in range(N_GM+1):
		k = k_G + i*dk_GM
		v_GM[i,:] = v(k)
		
	#M to K
	N_MK = 50
	x_MK = np.arange(N_GM,N_GM+N_MK+1,1)
	v_MK = np.zeros((N_MK+1,3))
	dk_MK = (k_K-k_M)/N_MK
	for i in range(N_MK+1):
		k = k_M + i*dk_MK
		v_MK[i,:] = v(k)
		
	#K to Gamma
	N_KG = 100
	x_KG = np.arange(N_GM+N_MK,N_GM+N_MK+N_KG+1,1)
	v_KG = np.zeros((N_KG+1,3))
	dk_KG = (k_G-k_K)/N_KG
	for i in range(N_KG+1):
		k = k_K + i*dk_KG
		v_KG[i,:] = v(k)
	
	ax.plot(x_GM,v_GM[:,0],'r-',label='v1')
	ax.plot(x_GM,v_GM[:,1],'b-',label='v2')
	ax.plot(x_GM,v_GM[:,2],'g-',label='v3')
	ax.plot(x_MK,v_MK[:,0],'r-')
	ax.plot(x_MK,v_MK[:,1],'b-')
	ax.plot(x_MK,v_MK[:,2],'g-')
	ax.plot(x_KG,v_KG[:,0],'r-')
	ax.plot(x_KG,v_KG[:,1],'b-')
	ax.plot(x_KG,v_KG[:,2],'g-')
	

	
	ax.set_ylabel('v',rotation=0)
	ax.set_xticks([0,N_GM,N_GM+N_MK,N_GM+N_MK+N_KG])
	ax.set_xticklabels(['$\Gamma$','M','K','$\Gamma$'],fontsize=15)
	ax.set_xlim(0,N_GM+N_MK+N_KG)
	
	ax.legend()
	
	if savefig:
		plt.savefig('KagomeBandstructure.png',dpi=300)
	
	if plotfig:
		plt.show()
		
		
def compare_ed(a=1.):
	#Compare the numerically obtained v(k) with the analytical plot above
	k_G = np.zeros(2)
	k_M = 2*np.pi/(a*np.sqrt(3))*np.array([0,1])
	k_K = 2*np.pi/(a*np.sqrt(3))*np.array([1/np.sqrt(3),1])
	
	k_GM = np.array([k_G + i * (k_M-k_G)/100 for i in range(100)])
	k_MK = np.array([k_M + i * (k_K-k_M)/50 for i in range(50)])
	k_KG = np.array([k_K + i * (k_G-k_K)/100 for i in range(101)])		#Includes final point
	
	k_vals = np.concatenate((k_GM,k_MK,k_KG),axis=0)
	
	#print(k_vals)
	
	v_vals = np.array([v(k_vals[i,:]) for i in range(np.shape(k_vals)[0])])
	x_vals = np.arange(np.shape(k_vals)[0])
	
	v_ed = np.zeros((np.shape(k_vals)[0],3))
	
	for i in range(np.shape(k_vals)[0]):
		H = kspace.H_k(k_vals[i,:])
		evals,evects = np.linalg.eigh(H)
		v_ed[i,:] = evects[:,2] * np.sign(evects[0,2])		#Ensures plots not thrown off by eigenvectors being found with extra factor of -1
	
	fig,ax = plt.subplots()
	
	ax.plot(x_vals,v_vals[:,0],'r-',label='v1')
	ax.plot(x_vals,v_vals[:,1],'b-',label='v2')
	ax.plot(x_vals,v_vals[:,2],'g-',label='v3')
	
	ax.plot(x_vals,v_ed[:,0],'y',ls='--',label='v1 Numerical')
	ax.plot(x_vals,v_ed[:,1],'cyan',ls='--',label='v2 Numerical')
	ax.plot(x_vals,v_ed[:,2],'lime',ls='--',label='v3 Numerical')

	ax.set_ylabel('v',rotation=0)
	ax.set_xticks([0,100,150,250])
	ax.set_xticklabels(['$\Gamma$','M','K','$\Gamma$'],fontsize=15)
	ax.set_xlim(0,250)
	
	ax.legend()
	
	plt.show()
	
	
def s(k):
	#Returns un-normalized v(k)
	s1 = np.sin(0.5*k[0])
	s2 = np.sin(0.25*k[0] + 0.25*np.sqrt(3)*k[1])
	s3 = np.sin(-0.25*k[0] + 0.25*np.sqrt(3)*k[1])
	return np.array([s3,-s2,s1])
	
	
def c(k,N_uc=100,r_H=np.array([0,0])):
	#Returns coefficients c(k) of expansion of flat-band hexagon state 
	#in terms of k-space eigenstates
	s1 = np.sin(0.5*k[0])
	s2 = np.sin(0.25*k[0] + 0.25*np.sqrt(3)*k[1])
	s3 = np.sin(-0.25*k[0] + 0.25*np.sqrt(3)*k[1])
	
	return np.sqrt(s1**2 + s2**2 + s3**2) * (np.cos(np.dot(k,r_H))+1j*np.sin(np.dot(k,r_H))) * 1j * np.sqrt(2/(3*N_uc))
	
	
def compare_ck(a=1.):
	#Compares numerical and analytical c(k) (hexagon coefficients as above)
	k_G = np.zeros(2)
	k_M = 2*np.pi/(a*np.sqrt(3))*np.array([0,1])
	k_K = 2*np.pi/(a*np.sqrt(3))*np.array([1/np.sqrt(3),1])
	
	k_GM = np.array([k_G + i * (k_M-k_G)/100 for i in range(100)])
	k_MK = np.array([k_M + i * (k_K-k_M)/50 for i in range(50)])
	k_KG = np.array([k_K + i * (k_G-k_K)/100 for i in range(101)])		#Includes final point
	
	k_vals = np.concatenate((k_GM,k_MK,k_KG),axis=0)
	
	#print(k_vals)
	
	#c_vals = np.array([np.dot(v(k_vals[i,:]),s(k_vals[i,:])) for i in range(np.shape(k_vals)[0])])
	c_vals = np.array([c(k_vals[i,:]) for i in range(np.shape(k_vals)[0])])
	x_vals = np.arange(np.shape(k_vals)[0])
	
	fig,ax = plt.subplots()
	
	ax.plot(x_vals,c_vals,'b-')

	ax.set_ylabel(r'$\sqrt{\frac{3N}{2}}|c_k|$',rotation=0,labelpad=20)
	ax.set_xticks([0,100,150,250])
	ax.set_xticklabels(['$\Gamma$','M','K','$\Gamma$'],fontsize=15)
	ax.set_xlim(0,250)

	
	plt.show()
	
	
def BrillouinZone(a=1.):
	#Returns the coordinates of the six vertices of the 1st BZ in k-space
	c1 = np.array([1/3,1/np.sqrt(3)])*2*np.pi/a #+ dk/100
	R = np.array([[0.5,-np.sqrt(3)/2],[np.sqrt(3)/2,0.5]])
	vertices = np.zeros((6,2))
	vertices[0,:] = c1
	for n in range(5):
		c1 = np.transpose(R @ c1.T)
		vertices[n+1,:] = c1
	return vertices


def BZ_k_vals(L=10,a=1.,shift='vertical',plot_BZ=False):
	#Returns all values of k within the first BZ (i.e. the discrete values, 
	#determined by the lattice size). Optionally, shift the BZ in different ways
	#to include different sets of the 'edge' k-values.
	b1 = np.array([1,-1/np.sqrt(3)])*2*np.pi/a
	b2 = np.array([0,1])*4*np.pi/(a*np.sqrt(3))
	b1_hat = b1/np.linalg.norm(b1)
	b2_hat = b2/np.linalg.norm(b2)
	dk = np.linalg.norm(b2)/L
	
	c1 = np.array([1/3,1/np.sqrt(3)])*2*np.pi/a #+ dk/100
	R = np.array([[0.5,-np.sqrt(3)/2],[np.sqrt(3)/2,0.5]])
	
	vertices = BrillouinZone(a=a)
		
	if shift == 'vertical':
		vertices[:,1] += dk/100
		
	elif shift == 'rotate':
		t = 0.001
		R_small = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
		for i in range(6):
			vertices[i,:] = np.transpose(R_small @ vertices[i,:].T)
		
	elif shift == 'enlarge':
		t = 1.0001
		vertices *= t
		
	elif shift == 'diagonal':
		x = np.array([0.5,1])
		for i in range(6):
			vertices[i,:] += x*dk/100
		
	BZ = mplPath.Path(vertices)
	k_vals = np.array([n*b1_hat*dk + m*b2_hat*dk for n in np.arange(-L,L+1,1) for m in np.arange(-L,L+1,1)])
	k_BZ = k_vals[BZ.contains_points(k_vals),:]
	#print(np.shape(k_BZ))
	
	if plot_BZ:
		fig,ax = plt.subplots()
	
		ax.plot(k_vals[:,0],k_vals[:,1],color='b',marker='x',ls='',label='Outisde')
		ax.plot(k_BZ[:,0],k_BZ[:,1],color='r',marker='x',ls='',label='Inside')
	
		BZPatch = mplPatch.Polygon(vertices)
		ax.add_patch(BZPatch)
		
		ax.arrow(0,0,b1[0],b1[1],length_includes_head=True,width=0.1,color='y')
		ax.arrow(0,0,b2[0],b2[1],length_includes_head=True,width=0.1,color='y')
	
		ax.legend()
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		plt.show()
		
	return k_BZ
	
	
def BZ_k_vals_alt(L=10,a=1.,plot_BZ=False,foldback=True):
	#Alternative method for calculating BZ k-values, via backfolding
	b1 = np.array([1,-1/np.sqrt(3)])*2*np.pi/a
	b2 = np.array([0,1])*4*np.pi/(a*np.sqrt(3))
	b1_hat = b1/np.linalg.norm(b1)
	b2_hat = b2/np.linalg.norm(b2)
	dk = np.linalg.norm(b2)/L

	k_vals = np.array([n*b1_hat*dk + m*b2_hat*dk for n in np.arange(0,L,1) for m in np.arange(0,L,1)])

	#print(np.shape(k_vals))
	
	vertices = BrillouinZone(a=a)*1.0001
	BZ = mplPath.Path(vertices)
	
	for i in range(np.shape(k_vals)[0]):
		k = k_vals[i,:]
		#print(k,BZ.contains_points([k]))
		if not BZ.contains_points([k]):
			if BZ.contains_points([k-b1]):
				k -= b1
			elif BZ.contains_points([k-b2]):
				k -= b2
			elif BZ.contains_points([k-b1-b2]):
				k -= b1+b2
	
	if plot_BZ:
		fig,ax = plt.subplots()
	
		ax.plot(k_vals[:,0],k_vals[:,1],color='r',marker='x',ls='',label='k Values')
	
		BZPatch = mplPatch.Polygon(vertices)
		ax.add_patch(BZPatch)
		
		ax.arrow(0,0,b1[0],b1[1],length_includes_head=True,width=0.1,color='y')
		ax.arrow(0,0,b2[0],b2[1],length_includes_head=True,width=0.1,color='y')
	
		ax.legend()
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		plt.show()
		
	return k_vals
	
	
def plot_hexagon(L=10,a=1.,r_H='centre',wf=True,dens=False):
	#Plot a hexagon eigenstate (either density or amplitude)
	N_sites = 3*L**2
	system = Kag.Kagome(psi_0=np.zeros(N_sites),Lx=L,Ly=L,a=a)
	
	if r_H == 'centre':
		r_H = 0.5*(L-1)*(system.a1+system.a2)
		
	elif r_H == 'off-centre':
		r_H = 0.5*(L+1)*(system.a1+system.a2)
	
	elif r_H == 'bottom left':
		r_H = 0.5*(system.a1+system.a2)
	
	elif r_H == 'top right':
		r_H = (L-1.5)*(system.a1+system.a2)
	
	k_BZ = BZ_k_vals(L=L,a=a)
	
	psi = np.zeros(N_sites,dtype=np.complex128)
	
	for k in k_BZ:
		c_k = c(k,N_uc=L**2,r_H=r_H)
		v_k = v(k)
		for i in range(L**2):
			for j in range(3):
				psi[3*i+j] += c_k * v_k[j] * kspace.comp_exp(system,3*i+j,k) / L
				
				
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
	
	
def component_shift(k,k1,q,U=1.,N=100,a=1.):
	#Plot the 'component shift' of the |k,k1> -> |k+q,k1-q> contribution to the hexagon eigenstate, as defined in OneNote (Flat Band Theory, 16/02/24)
	shift = 0
	#a1 contribution
	shift += np.sin(0.5*a*(k+q)[0])*np.sin(0.5*a*(k1-q)[0])*np.sin(0.5*a*k1[0])*np.sin(0.5*a*k[0])
	shift += np.sin(0.25*a*(k+q)[0]+0.25*np.sqrt(3)*a*(k+q)[1])*np.sin(0.25*a*(k1-q)[0]+0.25*np.sqrt(3)*a*(k1-q)[1])*np.sin(0.25*a*k1[0]+0.25*np.sqrt(3)*a*k1[1])*np.sin(0.25*a*k[0]+0.25*np.sqrt(3)*a*k[1])
	shift += np.sin(-0.25*a*(k+q)[0]+0.25*np.sqrt(3)*a*(k+q)[1])*np.sin(-0.25*a*(k1-q)[0]+0.25*np.sqrt(3)*a*(k1-q)[1])*np.sin(-0.25*a*k1[0]+0.25*np.sqrt(3)*a*k1[1])*np.sin(-0.25*a*k[0]+0.25*np.sqrt(3)*a*k[1])
	return shift*4*U/(9*N**3)
	
	
def k_shift(k,k_BZ,U=1.,N=100.,a=1.):
	#Plot the total shift at wavevector k (summing over k1,q) to the hexagon eigenstate, as defined in OneNote (Flat Band Theory, 16/02/24)
	shift = 0
	for k1 in k_BZ:
		for q in k_BZ:
			shift += component_shift(k,k1,q,U=U,N=N,a=a)
	return shift
		
		
def linear_fit(x,a,b):
	return a + b*x	


def plot_k_shift_vs_k(L=10,U=1.,a=1.,plot_fit=None,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot the total k-shift against k
	k_BZ = BrillouinZone(L=L,a=a)
	k_BZ_abs = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(k_BZ.shape[0])])
	
	k_BZ_abs_unique,idxs = np.unique(k_BZ_abs.round(decimals=4),return_index=True)
	
	k_shift_vals_unique = np.zeros(len(idxs))
	c_vals_unique = np.zeros(len(idxs))
	
	for i in range(len(idxs)):
		print(f'Progress: {np.round(100*(i+1)/len(idxs),2)}%    ',end='\r')
		idx = idxs[i]
		k_shift_vals_unique[i] = k_shift(k_BZ[idx,:],k_BZ,U=U,N=L**2,a=a)
		c_vals_unique[i] = np.abs(c(k_BZ[idx,:],N_uc=L**2))
	print('')

	
	fig,ax = plt.subplots()
	ax2 = ax.twinx()
	
	ax.plot(k_BZ_abs_unique,k_shift_vals_unique,color='b',marker='x',ls='')
	ax2.plot(k_BZ_abs_unique,c_vals_unique,color='r',marker='x',ls='')
	
	ax.set_xlabel(r'$|k|$')
	ax2.set_ylabel(r'$|c_{k}|$',rotation=0)
	ax.set_ylabel(r'$\Delta E$',rotation=0)
	ax.set_title(f'Energy shift vs k, U={U}')
	
	ax.set_ylim(0,1.25e-4)
	ax2.set_ylim(0,0.125)
	
	if plot_fit == 'linear':
		popt,pcov = sp.optimize.curve_fit(linear_fit,k_BZ_abs_unique[fit_idx1:fit_idx2+1],k_shift_vals_unique[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = linear_fit(k_BZ_abs_unique,popt[0],popt[1])
		ax.plot(k_BZ_abs_unique,yfit,'k--')
		if printparams:
			print('Fit y = a + b*x')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
	
	if plot_fit is None:
		handles = [mplLines.Line2D([0],[0],color='b',marker='x',ls='',label=r'$\Delta E$'),mplLines.Line2D([0],[0],color='r',marker='x',ls='',label=r'$|c_{k}|$')]
	elif plot_fit == 'linear':
		handles = [mplLines.Line2D([0],[0],color='b',marker='x',ls='',label=r'$\Delta E$'),mplLines.Line2D([0],[0],color='r',marker='x',ls='',label=r'$|c_{k}|$'),mplLines.Line2D([0],[0],color='k',ls='--',label='Linear Fit')]

		
	ax.legend(handles=handles,loc='upper left')

	plt.show()
	

def plot_weighted_shift_vs_k(L=10,U=1.,a=1.,plot_fit=None,fit_idx1=0,fit_idx2=0,printparams=True,plot_ck=False):
	#Plot the total k-shift, weighted by the occupancy |c(k)|, against k
	k_BZ = BrillouinZone(L=L,a=a)
	k_BZ_abs = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(k_BZ.shape[0])])
	
	k_BZ_abs_unique,idxs = np.unique(k_BZ_abs.round(decimals=4),return_index=True)
	
	k_shift_vals_unique = np.zeros(len(idxs))
	c_vals_unique = np.zeros(len(idxs))
	
	for i in range(len(idxs)):
		print(f'Progress: {np.round(100*(i+1)/len(idxs),2)}%    ',end='\r')
		idx = idxs[i]
		k_shift_vals_unique[i] = k_shift(k_BZ[idx,:],k_BZ,U=U,N=L**2,a=a)
		c_vals_unique[i] = np.abs(c(k_BZ[idx,:],N_uc=L**2))
	print('')

	
	fig,ax = plt.subplots()
	
	k_shift_scaled = np.divide(k_shift_vals_unique,c_vals_unique)
	
	ax.plot(k_BZ_abs_unique,k_shift_scaled,color='b',marker='x',ls='')
	
	handles = [mplLines.Line2D([0],[0],color='b',marker='x',ls='',label=r'$\frac{\Delta E}{|c_{k}|^2}$')]
	
	if plot_ck:
		ax2 = ax.twinx()
		ax2.plot(k_BZ_abs_unique,c_vals_unique,color='r',marker='x',ls='')
		ax2.set_ylabel(r'$|c_{k}|$',rotation=0)
		#ax2.set_ylim(0,0.125)
		handles.append(mplLines.Line2D([0],[0],color='r',marker='x',ls='',label=r'$|c_{k}|$'))
	
	ax.set_xlabel(r'$|k|$')
	ax.set_ylabel(r'$\frac{\Delta E}{|c_{k}|^2}$',rotation=0,fontsize=15)
	ax.set_title(f'Energy shift vs k, U={U}')
	
	ax.set_xlim(0,4)
	ax.set_ylim(bottom=0)
	ax.set_yticks([0,0.0005,0.001,0.0015,0.002])
	
	if plot_fit == 'linear':
		popt,pcov = sp.optimize.curve_fit(linear_fit,k_BZ_abs_unique[fit_idx1:fit_idx2+1],k_shift_scaled[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = linear_fit(k_BZ_abs_unique,popt[0],popt[1])
		ax.plot(k_BZ_abs_unique,yfit,'k--')
		handles.append(mplLines.Line2D([0],[0],color='k',ls='--',label='Linear Fit'))
		if printparams:
			print('Fit y = a + b*x')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
	
	ax.legend(handles=handles,loc='upper left')

	plt.show()
	
	
def component_shift2(k,k1,U=0.1,N_uc=100):
	#Alternative definition of component shift in terms of v(k)
	v_k = v(k)
	v_k1 = v(k1)
	
	shift = np.linalg.norm(np.multiply(v_k,v_k1))**2
	
	return shift*2*U/N_uc
	
	
def plot_component_shift(ax,k1,U=0.1,L=10,a=1.,color='b',marker='x',ls='',label=None):
	#Plot alternative component shift vs k
	k_BZ = BrillouinZone(L=L,a=a)
	k_BZ_abs = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(k_BZ.shape[0])])
	
	k_BZ_abs_unique,idxs = np.unique(k_BZ_abs.round(decimals=4),return_index=True)
	
	shift_vals = np.array([component_shift2(k_BZ[idx,:],k1,U=U,N_uc=L**2) for idx in idxs])
	
	ax.plot(k_BZ_abs_unique,shift_vals,color=color,marker=marker,ls=ls,label=label if label is not None else None)
	
	
def plot_component_weighted_shift(ax,U=0.1,L=10,a=1.,color='b',marker='x',ls='',label=None):
	#Plot alternative component shift, weighted by |c(k)|
	k_BZ = BrillouinZone(L=L,a=a)
	k_BZ_abs = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(k_BZ.shape[0])])
	
	k_BZ_abs_unique,idxs = np.unique(k_BZ_abs.round(decimals=4),return_index=True)
	
	shift_vals = np.zeros(np.size(idxs))
	for i in range(np.size(idxs)):
		for k1 in k_BZ:
			c_k1 = c(k1,N_uc=L**2)
			shift = component_shift2(k_BZ[idxs[i],:],k1,U=U,N_uc=L**2)
			shift_vals[i] += shift#*np.abs(c_k1)**2
	
	ax.plot(k_BZ_abs_unique,shift_vals,color=color,marker=marker,ls=ls,label=label if label is not None else None)
	
	
def plot_shifts_vs_k2(L=10,U=0.1,a=1.,plot_fit=None,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot various alternative component shifts for different k2
	k_M = np.array([0,1])*2*np.pi/(a*np.sqrt(3))
	k_K = np.array([1/np.sqrt(3),1])*2*np.pi/(a*np.sqrt(3))
	
	k1_vals = [k_M/2,k_M,k_K/2,k_K]
	labels = [r'$k_{M}/2$',r'$k_{M}$',r'$k_{K}/2$',r'$k_{K}$']
	colors = ['b','r','cyan','y']
	
	fig,ax = plt.subplots()
	for i in range(4):
		plot_component_shift(ax,k1_vals[i],U=U,L=L,a=a,color=colors[i],label=labels[i])

	ax.set_xlabel(r'$|k|$')
	ax.set_ylabel(r'$\Delta \omega_{k,k^\prime}$',rotation=0)
	ax.set_title(r'Energy Shift of $|k,k^\prime \rangle$ States')
	
	if plot_fit == 'linear':
		popt,pcov = sp.optimize.curve_fit(linear_fit,k_BZ_abs_unique[fit_idx1:fit_idx2+1],k_shift_vals_unique[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = linear_fit(k_BZ_abs_unique,popt[0],popt[1])
		ax.plot(k_BZ_abs_unique,yfit,'k--')
		if printparams:
			print('Fit y = a + b*x')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		
	ax.legend(title=r'$k^\prime$')

	plt.show()
	
	
def plot_weighted_shifts_vs_k2(L=10,U=0.1,a=1.,plot_fit=None,fit_idx1=0,fit_idx2=0,printparams=True):
	#Plot weighted average energy shift (summed over k1)
	fig,ax = plt.subplots()

	plot_component_weighted_shift(ax,U=U,L=L,a=a,color='b',marker='x',ls='')

	ax.set_xlabel(r'$|k|$')
	ax.set_ylabel(r'$\sum_{k^\prime}\Delta \omega_{k,k^\prime}|c_{k^\prime}|^2$')
	ax.set_title(r'Weighted Average Energy Shift of $|k \rangle$')
	
	if plot_fit == 'linear':
		popt,pcov = sp.optimize.curve_fit(linear_fit,k_BZ_abs_unique[fit_idx1:fit_idx2+1],k_shift_vals_unique[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = linear_fit(k_BZ_abs_unique,popt[0],popt[1])
		ax.plot(k_BZ_abs_unique,yfit,'k--')
		if printparams:
			print('Fit y = a + b*x')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		
	#ax.legend(title=r'$k_{1}$')

	plt.show()
	

def plot_top_evals(L=10,U=0.1,k=10,bc='open',reverse=True,xticks=True):
	#Plot the highest-energy eigenvalues of a 2-particle kagome system
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	system = Kag2.Kagome2(psi_0=psi_0,Lx=L,Ly=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc)
	evals = sp.sparse.linalg.eigsh(system.H,k=k,return_eigenvectors=False,which='LA')
	if reverse:
		evals = evals[::-1]


	fig,ax = plt.subplots()
	ax.plot(np.arange(k),evals,color='b',marker='x',ls='')
	ax.axhline(y=4,color='r',ls='--',label='E=4J')
	ax.set_ylabel('E / J')
	ax.set_xlabel('Index')
	if not xticks:
		ax.set_xticks([])
		ax.set_xlabel('')
	ax.legend()

	ax.set_title(f'{L}x{L}, U={U}, bc='+bc)
	plt.show()
		
	
def plot_top_estates(L=10,U=0.1,k=10,bc='open'):
	#Plot the amplitude of the highest-energy eigenstates of a 2-particle Kagome system
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	system = Kag2.Kagome2(psi_0=psi_0,Lx=L,Ly=L,U=U,skip_diag=True,evolution_method='propagator',bc=bc)
	evals,evects = sp.sparse.linalg.eigsh(system.H,k=k,return_eigenvectors=True,which='LA')
	
	for i in range(k):
		system.psi = evects[:,i]
		fig,ax = plt.subplots()
		system.plot_amplitude(fig,ax)
	
		ax.set_xticks([])
		ax.set_yticks([])

		ax.set_title(f'E = {np.round(evals[i],3)}, n_pair={np.round(npair(system.psi,L=L),3)}')
		plt.show()
	
	
def psi_k(k1,k2,system):
	#Returns the state |k1,k2> in the real-space basis for the given system
	L = system.Lx
	if system.Ly != L:
		print('Warning: Lx and Ly must be equal')
	N_uc = L**2
	v_k1 = v(k1)
	v_k2 = v(k2)
	
	psi = np.zeros(system.N,dtype=np.complex128)
	
	for i in range(N_uc):
		for a in range(3):
			for j in range(N_uc):
				for b in range(3):
					idx = system.state_idx(3*i+a,3*j+b)
					p1 = np.dot(k1,system.sites[3*i+a].r)
					p2 = np.dot(k2,system.sites[3*j+b].r)
					psi[idx] += v_k1[a]*v_k2[b]*(np.cos(p1+p2)-1j*np.sin(p1+p2))/N_uc
					if i==j and a==b:
						psi[idx] *= np.sqrt(2)		#Since basis state defined with 1/sqrt(2) normalization factor
	if np.array_equal(k1,k2):
		psi /= np.sqrt(2)
					
	return psi
	
	
def c_k1k2(k1,k2,psi,system):
	#Returns the occupancy (i.e. inner product) of the state |k1,k2> for a given state |psi>
	psi_k1k2 = psi_k(k1,k2,system)
	return np.vdot(psi_k1k2,psi)
	
	
def k_weight(k,system,psi,k_BZ=None):
	#Returns the sum over k1 of |c_k,k1|^2, giving the 'weight' of the (single-particle) wavevector k present 
	#in the wavefunction psi
	if k_BZ is None:
		L = system.Lx
		if system.Ly != L:
			print('Warning: Lx and Ly must be equal')
	
		k_BZ = BZ_k_vals(L=L,a=system.a)
	
	val = 0
	
	for j in range(np.shape(k_BZ)[0]):
		k1 = k_BZ[j,:]
		c = c_k1k2(k,k1,psi,system)
		val += np.abs(c)**2 #*2
		#if np.array_equal(k,k1):
		#	val -= np.abs(c)**2
	
	return val
	
	
def calculate_k_weights(psi,system,k_BZ=None):
	#Calculates 'weights' of all values of k within the BZ
	if k_BZ is None:
		L = system.Lx
		if system.Ly != L:
			print('Warning: Lx and Ly must be equal')
		k_BZ = BZ_k_vals(L=L,a=system.a)
	weight_vals = np.zeros(np.shape(k_BZ)[0])
	for i in range(np.shape(k_BZ)[0]):
		print(f'Evaluating k value {i+1} of {np.shape(k_BZ)[0]}...   ',end='\r')
		weight_vals[i] = k_weight(k_BZ[i,:],system,psi,k_BZ=k_BZ)
		
	return weight_vals
	

def hexagon_patch(r,a,facecolor='w',edgecolor='k'):
	#Returns a hexagonal mpl patch
	l = r - a*np.array([1,0])
	ll = l + a*np.array([0.5,-np.sqrt(3)/2])
	lr = ll + a*np.array([1,0])
	r = lr + a*np.array([0.5,np.sqrt(3)/2])
	ur = r + a*np.array([-0.5,np.sqrt(3)/2])
	ul = ur - a*np.array([1,0])
	return matplotlib.patches.Polygon(np.array([l,ll,lr,r,ur,ul]),fc=facecolor,ec=edgecolor)

	
def plot_k_weights(psi,system,k_BZ=None,plot_BZ=True,cmap=plt.cm.Blues,uppernorm='auto',plot_cbar=True,cbar_label=False,cbar_label_y=0.5,title=None,plot=True,save=False,filename=None,plot_wf=True):
	#Plots the 'weights' of all values of k within the BZ. Plot is a heatmap of the BZ.
	if k_BZ is None:
		L = system.Lx
		if system.Ly != L:
			print('Warning: Lx and Ly must be equal')
		k_BZ = BZ_k_vals(L=L,a=system.a)
	
	weights = calculate_k_weights(psi,system,k_BZ=k_BZ)
	
	if uppernorm == 'auto':
		norm = matplotlib.colors.Normalize(vmin=0, vmax=Kag2.uppernorm_func(np.max(weights)))
	sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
	
	dk = 4*np.pi/(system.a*np.sqrt(3)*system.Lx)
	
	if plot_wf:
		fig,axs = plt.subplots(1,2)
		ax = axs[1]
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])
		
		
		for i in range(np.shape(k_BZ)[0]):
			k = k_BZ[i,:]
			ax.add_patch(hexagon_patch(k,dk/np.sqrt(3),facecolor=cmap(norm(weights[i])),edgecolor=None))
			
		if plot_BZ:
			BZ = BrillouinZone(a=system.a)
			ax.add_patch(mplPatch.Polygon(BZ,fc=(0,0,0,0),ec='k'))
		
		scale = 1.5
		xmax = np.max(np.abs(k_BZ[:,0]))
		ymax = np.max(np.abs(k_BZ[:,1]))
		ax.set_xlim(-scale*xmax,scale*xmax)
		ax.set_ylim(-scale*ymax,scale*ymax)
			
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax)
			if cbar_label:
				cbar.set_label(r'$\sum_{k^\prime} |c_{k,k^\prime}|^2$',rotation=0,y=cbar_label_y)
			cbar.ax.locator_params(nbins=5)
			
		ax.set_title(r'$\sum_{k^\prime} |\langle k,k^\prime|\Psi \rangle|^2$')
		
		axs[0].set_position(pos=[axs[0].get_position().x0-0.05,ax.get_position().y0,axs[0].get_position().width,ax.get_position().height])
		system.psi = psi
		system.plot_amplitude(fig,axs[0],plot_cbar=True,label_cbar=False)
		axs[0].set_title(r'$\Psi$')
		
		plt.suptitle(title,y=0.85)
		
		if save:
			if filename is None:
				filename = 'k_spectrum_'+title+'.png'
			plt.savefig(filename,dpi=300)
	
		if plot:
			plt.show()
		
	else:
		fig,ax = plt.subplots()
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])
		
		for i in range(np.shape(k_BZ)[0]):
			k = k_BZ[i,:]
			ax.add_patch(hexagon_patch(k,dk/np.sqrt(3),facecolor=cmap(norm(weights[i])),edgecolor=None))
			
		if plot_BZ:
			BZ = BrillouinZone(a=system.a)
			ax.add_patch(mplPatch.Polygon(BZ,fc=(0,0,0,0),ec='k'))
		
		scale = 1.5
		xmax = np.max(np.abs(k_BZ[:,0]))
		ymax = np.max(np.abs(k_BZ[:,1]))
		ax.set_xlim(-scale*xmax,scale*xmax)
		ax.set_ylim(-scale*ymax,scale*ymax)
			
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax)
			if cbar_label:
				cbar.set_label(r'$\sum_{k^\prime} |c_{k,k^\prime}|^2$',rotation=0,y=cbar_label_y)
			cbar.ax.locator_params(nbins=5)
			
		ax.set_title(r'$\sum_{k^\prime} |\langle k,k^\prime|\Psi \rangle|^2$')
		plt.suptitle(title,y=0.85)
		
		if save:
			if filename is None:
				filename = 'k_spectrum_'+title+'.png'
			plt.savefig(filename,dpi=300)
	
		if plot:
			plt.show()
		
	
def plot_estate_k_weights(idxs=[0],L=10,U=1.,a=1.,bc='periodic',plot_wf=False,plot=True,save=False):		#idxs of descending eigenvalues
	#For eigenstates with specified indices (relative to the highest-energy), plot the 'weights' of k in the BZ
	system = Kag2.double_occupied_site_system(L=L,U=U,a=a,skip_diag=True,skip_H=False,evolution_method='propagator',bc=bc)
	num = np.max(idxs)+1
	evals,evects = sp.sparse.linalg.eigsh(system.H,k=num,return_eigenvectors=True,which='LA')
	
	k_BZ = BZ_k_vals(L=L,a=a)
	
	for i,idx in enumerate(idxs):
		psi = evects[:,num-idx-1]
		plot_k_weights(psi,system,k_BZ=k_BZ,title=f'E[{idx}]={np.round(evals[num-idx-1],4)}',plot_BZ=True,plot_wf=plot_wf,plot=plot,save=save)
		

def plot_c_kk1(k1,psi,system,k_BZ=None,plot_BZ=True,cmap=plt.cm.Blues,uppernorm='auto',plot_cbar=True,cbar_label=False,cbar_label_y=0.5,title=None,plot=True,save=False,filename=None,plot_wf=True):
	#Plot the values of c_k,k1 within the BZ for fixed k1, showing the occupancy of given states
	if k_BZ is None:
		L = system.Lx
		if system.Ly != L:
			print('Warning: Lx and Ly must be equal')
		k_BZ = BZ_k_vals(L=L,a=system.a)
	
	c_kk1_vals = np.array([np.abs(c_k1k2(k_BZ[i,:],k1,psi,system))**2 for i in range(np.shape(k_BZ)[0])])
	
	if uppernorm == 'auto':
		norm = matplotlib.colors.Normalize(vmin=0, vmax=Kag2.uppernorm_func(np.max(c_kk1_vals)))
	sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
	
	dk = 4*np.pi/(system.a*np.sqrt(3)*system.Lx)
	
	if plot_wf:
		fig,axs = plt.subplots(1,2)
		ax = axs[1]
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])
		
		for i in range(np.shape(k_BZ)[0]):
			k = k_BZ[i,:]
			ax.add_patch(hexagon_patch(k,dk/np.sqrt(3),facecolor=cmap(norm(c_kk1_vals[i])),edgecolor=None))
			
		if plot_BZ:
			BZ = BrillouinZone(a=system.a)
			ax.add_patch(mplPatch.Polygon(BZ,fc=(0,0,0,0),ec='k'))
			
		#Draw fixed	k1 vector as an arrow
		ax.arrow(0,0,k1[0],k1[1],length_includes_head=True,width=0.1,color='r',label=r'$k^\prime$')
		ax.annotate(r'$k^\prime$',(k1[0]+dk/4,k1[1]-dk/4))
		
		scale = 1.5
		xmax = np.max(np.abs(k_BZ[:,0]))
		ymax = np.max(np.abs(k_BZ[:,1]))
		ax.set_xlim(-scale*xmax,scale*xmax)
		ax.set_ylim(-scale*ymax,scale*ymax)
			
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax)
			if cbar_label:
				cbar.set_label(r'$\sum_{k^\prime} |c_{k,k^\prime}|^2$',rotation=0,y=cbar_label_y)
			cbar.ax.locator_params(nbins=5)
			
		ax.set_title(r'$|\langle k,k^\prime|\Psi \rangle|^2$')#+f'$k^\prime$={k1}')
		
		axs[0].set_position(pos=[axs[0].get_position().x0-0.05,ax.get_position().y0,axs[0].get_position().width,ax.get_position().height])
		system.psi = psi
		system.plot_amplitude(fig,axs[0],plot_cbar=True,label_cbar=False)
		axs[0].set_title(r'$\Psi$')
		
		plt.suptitle(title,y=0.85)
		
		if save:
			if filename is None:
				filename = 'k_spectrum_'+title+'.png'
			plt.savefig(filename,dpi=300)
	
		if plot:
			plt.show()
			
			
def plot_estate_c_kk1(idxs=[0],k1_vals=[np.zeros(2)],L=10,U=1.,a=1.,bc='periodic',plot_wf=True,plot=True,save=False):		#idxs of descending eigenvalues
	#For eigenstates with specifid indices, plot the c_k,k1 for fixed k1 within the BZ
	system = Kag2.double_occupied_site_system(L=L,U=U,a=a,skip_diag=True,skip_H=False,evolution_method='propagator',bc=bc)
	num = np.max(idxs)+1
	evals,evects = sp.sparse.linalg.eigsh(system.H,k=3*L**2,return_eigenvectors=True,which='LA')
	
	k_BZ = BZ_k_vals(L=L,a=a)
	
	b1 = np.array([1,-1/np.sqrt(3)])*2*np.pi/a
	b2 = np.array([0,1])*4*np.pi/(a*np.sqrt(3))
	b1_hat = b1/np.linalg.norm(b1)
	b2_hat = b2/np.linalg.norm(b2)
	dk = np.linalg.norm(b2)/L
	
	for i,idx in enumerate(idxs):
		psi = evects[:,3*L**2-idx-1]
		if len(k1_vals)>1:
			k1 = (k1_vals[i][0]*b1_hat+k1_vals[i][1]*b2_hat)*dk			#k1 supplied in basis of elementary steps in BZ
		else:
			k1 = (k1_vals[0][0]*b1_hat+k1_vals[0][1]*b2_hat)*dk
		plot_c_kk1(k1,psi,system,k_BZ=k_BZ,title=f'E[{idx}]={np.round(evals[3*L**2-idx-1],4)}',plot_BZ=True,plot_wf=plot_wf,plot=plot,save=save)
		

def A(q,psi,system,k_BZ=None,L=10,a=1.):
	#Return A(q) coefficient defined in OneNote: Flat Band Theory (23/02/24)
	if k_BZ is None:
		k_BZ = BZ_k_vals(L=L,a=a)
	
	val = 0
	for i in range(np.shape(k_BZ)[0]):
		k = k_BZ[i,:]
		val += c_k1k2(k,q-k,psi,system)*np.vdot(v(k),v(q-k))/L**2
	
	return val


def plot_A2(psi,system,L=10,a=1.,plot_fit=None,fit_idx1=0,fit_idx2=1,printparams=True):
	#Plot A(q)^2 against |q|
	k_BZ = BZ_k_vals(L=L,a=a)
	q = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(L**2)])
	
	q_unique,idxs = np.unique(q.round(decimals=4),return_index=True)
	
	A_q2 = np.array([np.abs(A(k_BZ[idx,:],psi,system,L=L,a=a,k_BZ=k_BZ))**2 for idx in idxs])
	
	fig,ax=plt.subplots()
	ax.plot(q_unique,A_q2,color='b',marker='x',ls='',label='Data')
	ax.set_xlabel(r'$|q|$')
	ax.set_ylabel(r'$|a_q|^2$',rotation=0)
	
	if plot_fit == 'linear':
		popt,pcov = sp.optimize.curve_fit(linear_fit,q_unique[fit_idx1:fit_idx2+1],A_q2[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = linear_fit(q_unique,popt[0],popt[1])
		ax.plot(q_unique,yfit,'r--',label='Linear Fit')
		ax.legend()
		if printparams:
			print('Fit y = a + b*x')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		
	plt.show()
		

def shift1(q,k_BZ=None,L=5,a=1.):
	#Return the value of the interaction energy shift using the (approximate) expression from Torma et al. 2018
	#(See OneNote Flat Band Theory 28/02/24)
	if k_BZ is None:
		k_BZ = BZ_k_vals(L=L,a=a)
	
	val = 0
	for i in range(np.shape(k_BZ)[0]):
		k = k_BZ[i,:]
		val += np.vdot(v(k),v(q-k))
	
	return np.abs(val)**2
	

def shift2(q,k_BZ=None,L=5,a=1.,include_mobileband=False):
	#Return the alternative (less approximate) value for the interaction energy shift from Torma et al. 2018
	if k_BZ is None:
		k_BZ = BZ_k_vals(L=L,a=a)
	
	val = 0
	for i in range(np.shape(k_BZ)[0]):
		k = k_BZ[i,:]
		val += np.abs(np.vdot(v(k),v(q-k)))**2
		
	#Optionally, add contribution from k=0 state in 2nd band, also of energy E=4J:
	if include_mobileband:
		v_0 = np.array([1,-0.5-0.5j*np.sqrt(3),-0.5+0.5j*np.sqrt(3)])/np.sqrt(3)
		val += np.abs(np.vdot(v_0,v(q)))**2
	
	return val
	
	
def shift3(q,k_BZ=None,L=5,a=1.,include_mobileband=False):
	#Return the 'energy shift' calculated by taking dot product (as in shift1()), but 'rolling' one vector by one place
	#Hoped that this might account for the three eigenvalues at each q, but doesn't match numerical spectrum
	if k_BZ is None:
		k_BZ = BZ_k_vals_alt(L=L,a=a)
	
	val = 0
	for i in range(np.shape(k_BZ)[0]):
		k = k_BZ[i,:]
		v1 = v(k)
		v2 = np.roll(v(q-k),1)
		val += np.abs(np.vdot(v1,v2))**2
		
	#Optionally, add contribution from k=0 state in 2nd band, also of energy E=4J:
	if include_mobileband:
		v_0 = np.array([1,-0.5-0.5j*np.sqrt(3),-0.5+0.5j*np.sqrt(3)])/np.sqrt(3)
		val += np.abs(np.vdot(v_0,v(q)))**2
	
	return val
	

def shift4(q,k_BZ=None,L=5,a=1.,include_mobileband=False):
	#Return the 'energy shift' calculated by taking dot product (as in shift1()), but 'rolling' one vector by two places
	#Hoped that this might account for the three eigenvalues at each q, but doesn't match numerical spectrum
	if k_BZ is None:
		k_BZ = BZ_k_vals_alt(L=L,a=a)
	
	val = 0
	for i in range(np.shape(k_BZ)[0]):
		k = k_BZ[i,:]
		v1 = v(k)
		v2 = np.roll(v(q-k),2)
		val += np.abs(np.vdot(v1,v2))**2
		
	#Optionally, add contribution from k=0 state in 2nd band, also of energy E=4J:
	if include_mobileband:
		v_0 = np.array([1,-0.5-0.5j*np.sqrt(3),-0.5+0.5j*np.sqrt(3)])/np.sqrt(3)
		val += np.abs(np.vdot(v_0,v(q)))**2
	
	return val
	
	
def shift5(q,k_BZ=None,L=5,a=1.,include_mobileband=False):
	#Returns interaction energy shift as in shift1(), but splitting q between the k values of the two vectors
	if k_BZ is None:
		k_BZ = BZ_k_vals_alt(L=L,a=a)
	
	val = 0
	for i in range(np.shape(k_BZ)[0]):
		k = k_BZ[i,:]
		v1 = v(k+q/2)
		v2 = v(k-q/2)
		val += np.abs(np.vdot(v1,v2))**2
		
	#Optionally, add contribution from k=0 state in 2nd band, also of energy E=4J:
	if include_mobileband:
		v_0 = np.array([1,-0.5-0.5j*np.sqrt(3),-0.5+0.5j*np.sqrt(3)])/np.sqrt(3)
		val += np.abs(np.vdot(v_0,v(q)))**2
	
	return val
	
	
def plot_all_shifts(L=5,a=1,U=0.1,ax=None,plot=False,plot_numerical=False,plot_fit=None,fit_idx1=0,fit_idx2='all',printparams=True):
	#Plots various of the different shifts defined above
	if ax is None:
		fig,ax = plt.subplots()
	
	k_BZ = BZ_k_vals_alt(L=L,a=a)
	q = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(L**2)])
	
	q_unique,idxs = np.unique(q.round(decimals=4),return_index=True)
	
	shift2_vals = np.array([shift2(k_BZ[idx,:],L=L,a=a,k_BZ=k_BZ) for idx in idxs])
	shift3_vals = np.array([shift3(k_BZ[idx,:],L=L,a=a,k_BZ=k_BZ) for idx in idxs])
	shift4_vals = np.array([shift4(k_BZ[idx,:],L=L,a=a,k_BZ=k_BZ) for idx in idxs])
	
	E_shift_2 = shift2_vals*U/(3*L**2)
	E_shift_3 = shift3_vals*U/(3*L**2)
	E_shift_4 = shift4_vals*U/(3*L**2)
	
	if plot_numerical:
		plot_evals_vs_q(L=L,U=U,ax=ax,label='Numerical',color='r',shift=True)
	
	ax.plot(q_unique,E_shift_2,color='b',marker='x',ls='',label=r'$\Delta E_{q}^{(1)}$')
	ax.plot(q_unique,E_shift_3,color='g',marker='x',ls='',label=r'$\Delta E_{q}^{(2)}$')
	ax.plot(q_unique,E_shift_4,color='y',marker='x',ls='',label=r'$\Delta E_{q}^{(3)}$')
	
	
	ax.set_xlabel(r'$|q|$ / $a^{-1}$')
	ax.set_ylabel(r'$\Delta E_q$ / $J$',rotation=0,labelpad=15)
	ax.set_title(f'{L}x{L} System, U={U}, bc=periodic')
	ax.legend()
		
	if plot_fit == 'linear':
		if fit_idx2 == 'all':
			fit_idx2 = L-1
		popt,pcov = sp.optimize.curve_fit(linear_fit,q_unique[fit_idx1:fit_idx2+1],E_shift[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = linear_fit(q_unique,popt[0],popt[1])
		ax.plot(q_unique,yfit,'b--',label='Linear Fit')
		ax.legend(fontsize=15)
		if printparams:
			print('Fit y = a + b*x')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
	
	if plot:
		plt.show()	
	
	
def plot_evals_vs_q(L=5,U=0.1,bc='periodic',ax=None,color='b',marker='x',ls='',label=None,plot=False,shift=False,plotfit=None,fit_idx1=0,fit_idx2=1,printparams=True):
	#Plots numerical eigenvalues vs |q|
	system = Kag2.double_occupied_site_system(L=L,U=U,skip_diag=True,evolution_method='propagator',skip_k=False,bc=bc)
	evals,evects = sp.sparse.linalg.eigsh(system.H,which='LA',k=3*L**2,return_eigenvectors=True)
	if shift:
		evals -= 4
	evals_unique,idxs = np.unique(evals.round(decimals=6),return_index=True)
	evects_unique = evects[:,idxs]
	
	N = system.N_uc + 1
	N_comb = int(0.5*N*(N+1))
	psi_k1k2_vals = np.zeros((system.N,N_comb),dtype=np.complex128)
	count = 0
	for i in range(N):
		psi_k1 = system.fb_psi_k_sp[:,i]
		for j in range(i,N):
			psi_k2 = system.fb_psi_k_sp[:,j]
			psi_k1k2_vals[:,count] = Kag2.direct_product(psi_k1,psi_k2)
			count += 1
			print(f'Progress: {np.round(100*count/N_comb,2)}%',end='\r')
	
	q_vals = np.zeros(np.size(idxs))
	
	for i,idx in enumerate(idxs):
		c_q = calc_c_q(evects[:,idx],system,psi_k1k2_vals=psi_k1k2_vals)
		q = np.linalg.norm(system.k_BZ[np.argmax(c_q),:])
		q_vals[i] = q
	
	if ax is None:
		fig,ax = plt.subplots()
		
	ax.plot(q_vals,evals_unique,color=color,marker=marker,ls=ls,label='Numerical Values')
	
	if plot_fit == 'linear':
		if fit_idx2 == 'all':
			fit_idx2 = L-1
		popt,pcov = sp.optimize.curve_fit(linear_fit,q_unique[fit_idx1:fit_idx2+1],evals_unique[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = linear_fit(q_unique,popt[0],popt[1])
		ax.plot(q_unique,yfit,'b--',label='Fit')
		ax.legend(fontsize=13)
		if printparams:
			print('Fit y = a + b*x')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
	
	ax.set_xlabel(r'$|q|$ / $a^{-1}$',fontsize=13)
	ax.set_ylabel(r'$\Delta E_q$ / $J$',fontsize=13)
	#if label is not None:
	#	ax.legend(fontsize=13)
	if plot:
		plt.show()
		

def plot_sum_evals_vs_q(L=5,U=0.1,a=1.,bc='periodic',ax=None,color='b',marker='x',ls='',label=None,plot=False):
	#Plots sum of eigenvalues at each q, vs q
	system = Kag2.double_occupied_site_system(L=L,U=U,a=a,skip_diag=True,evolution_method='propagator',skip_k=False,bc=bc)
	evals,evects = sp.sparse.linalg.eigsh(system.H,which='LA',k=3*L**2,return_eigenvectors=True)
	evals -= 4
	evals_unique,idxs = np.unique(evals.round(decimals=10),return_index=True)
	evects_unique = evects[:,idxs]
	
	N = system.N_uc + 1
	N_comb = int(0.5*N*(N+1))
	psi_k1k2_vals = np.zeros((system.N,N_comb),dtype=np.complex128)
	count = 0
	for i in range(N):
		psi_k1 = system.fb_psi_k_sp[:,i]
		for j in range(i,N):
			psi_k2 = system.fb_psi_k_sp[:,j]
			psi_k1k2_vals[:,count] = Kag2.direct_product(psi_k1,psi_k2)
			count += 1
			print(f'Progress: {np.round(100*count/N_comb,2)}%',end='\r')
	
	q_vals = np.zeros((np.size(idxs),2))
	q_mag_vals = np.zeros(np.size(idxs))
	
	for i,idx in enumerate(idxs):
		c_q = calc_c_q(evects[:,idx],system,psi_k1k2_vals=psi_k1k2_vals)
		q_vals[i,:] = system.k_BZ[np.argmax(c_q),:]
		q_mag_vals[i] = np.linalg.norm(system.k_BZ[np.argmax(c_q),:])
	
	q_mag_unique,idxs_q = np.unique(np.round(q_mag_vals,4),return_index=True)
	evals_summed = np.array([np.sum(evals_unique[np.round(q_mag_vals,4)==q]) for q in q_mag_unique])
	
	shift2_vals = np.array([shift2(q_vals[i,:],L=L,a=a,k_BZ=system.k_BZ) for i in idxs_q])*U/system.N_sites
	
	if ax is None:
		fig,ax = plt.subplots()
		
	ax.plot(q_mag_unique,evals_summed,color=color,marker=marker,ls=ls,label=label)
	ax.plot(q_mag_unique,shift2_vals,color='b',marker='x',ls='',label='Analytical')
	ax.set_xlabel(r'$|q|$')
	ax.set_ylabel(r'$\sum_{q}\Delta E_{q}$ / $J$')
	if label is not None:
		ax.legend()
	if plot:
		plt.show()
		

def calc_c_q(psi,system,psi_k1k2_vals=None):
	#Calculates total occupancy of all |k1,k2> states with k1+k2=q in the state |psi>
	b1_hat = system.b1/np.linalg.norm(system.b1)
	b2_hat = system.b2/np.linalg.norm(system.b2)
	dkx = np.linalg.norm(system.b2)/system.Lx
	dky = np.linalg.norm(system.b2)/system.Ly

	BZPath = mplPath.Path(system.BZ*(1+min(dkx,dky)/100))
	
	N = system.N_uc + 1
	N_comb = int(0.5*N*(N+1))
	c_q = np.zeros(system.N_uc)
	count = 0
	for i in range(N):
		if i != N-1:
			k1 = system.k_BZ[i,:]
		else:		#Case of 2nd Band Gamma point
			k1 = np.zeros(2)
		if psi_k1k2_vals is None:
			psi_k1 = system.fb_psi_k_sp[:,i]
		for j in range(i,N):
			if j != N-1:
				k2 = system.k_BZ[j,:]
			else:
				k2 = np.zeros(2)
				
			q = backfold(k1+k2,BZPath,system.b1,system.b2)
			
			if psi_k1k2_vals is None:
				psi_k2 = system.fb_psi_k_sp[:,j]
				psi_k1k2 = Kag2.direct_product(psi_k1,psi_k2)
			else:
				psi_k1k2 = psi_k1k2_vals[:,count]

			dec = int(1-np.floor(np.log10(dkx)))
			idx = np.where((np.round(system.k_BZ[:,0],dec)==np.round(q[0],dec))&(np.round(system.k_BZ[:,1],dec)==np.round(q[1],dec)))[0]
			c_q[idx] += np.abs(np.vdot(psi_k1k2,psi))**2
			count += 1
			print(f'Progress: {np.round(100*count/N_comb,2)}%',end='\r')
	return c_q		
	
	
def compare_12(q,L=5,a=1):
	#Compares shift1() and shift2() (defined above)
	k_BZ = BZ_k_vals(L=L,a=a)
	
	b1 = np.array([1,-1/np.sqrt(3)])*2*np.pi/a
	b2 = np.array([0,1])*4*np.pi/(a*np.sqrt(3))
	b1_hat = b1/np.linalg.norm(b1)
	b2_hat = b2/np.linalg.norm(b2)
	dk = np.linalg.norm(b2)/L
	
	q = np.array(q[0]*b1_hat + q[1]*b2_hat)*dk
	
	print(shift1(q,k_BZ=k_BZ,L=L,a=a))
	print(shift2(q,k_BZ=k_BZ,L=L,a=a))
	
	
def plot_shift2(L=5,a=1,U=0.1,ax=None,plot=False,plot_numerical=False,plot_fit=None,fit_idx1=0,fit_idx2='all',printparams=True):
	#Plots shift2() vs q
	if ax is None:
		fig,ax = plt.subplots()
	
	k_BZ = BZ_k_vals_alt(L=L,a=a)
	q = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(L**2)])
	
	q_unique,idxs = np.unique(q.round(decimals=4),return_index=True)
	
	shift = np.array([shift2(k_BZ[idx,:],L=L,a=a,k_BZ=k_BZ) for idx in idxs])
	
	E_shift = shift*U/(3*L**2)
	
	if plot_numerical:
		system = Kag2.double_occupied_site_system(L=L,U=U,a=a,skip_diag=True,evolution_method='propagator',skip_k=True)
		evals = sp.sparse.linalg.eigsh(system.H,which='LA',k=3*L**2,return_eigenvectors=False)
		top_evals_unique = np.unique(np.round(evals,6))[-1:-(len(q_unique)+1):-1] - 4
		ax.plot(q_unique,top_evals_unique,color='r',marker='x',ls='',label='Numerical')
	
	ax.plot(q_unique,E_shift,color='b',marker='x',ls='',label='Theoretical')
	ax.set_xlabel(r'$|q|$ / $a^{-1}$',fontsize=15)
	ax.set_ylabel(r'$\Delta E_q$ / $J$',rotation=0,fontsize=15,labelpad=20)
	ax.set_title('Energy Eigenvalue Shifts',fontsize=15)
		
	if plot_fit == 'linear':
		if fit_idx2 == 'all':
			fit_idx2 = L-1
		popt,pcov = sp.optimize.curve_fit(linear_fit,q_unique[fit_idx1:fit_idx2+1],E_shift[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		yfit = linear_fit(q_unique,popt[0],popt[1])
		ax.plot(q_unique,yfit,'b--',label='Linear Fit')
		ax.legend(fontsize=15)
		if printparams:
			print('Fit y = a + b*x')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
	
	if plot:
		plt.show()

	
	
def plot_k_states(L=10,idxs=[0]):
	#Plot (analytical) single-particle k states
	system = Kag.Kagome(psi_0=np.zeros(3*L**2),Lx=L,Ly=L,bc='periodic')
	
	fb_evects = system.eigvects[:,-1:-(L**2+2):-1].astype(dtype=np.complex128)
	k_BZ = BZ_k_vals_alt(L=L,plot_BZ=False)
	psi_k_vals = np.zeros((3*L**2,L**2+1),dtype=np.complex128)
	for i in range(L**2):
		k = k_BZ[i,:]
		for j in range(L**2):
			for a in range(3):
				idx = 3*j + a
				p = np.dot(k,system.sites[idx].r)
				psi_k_vals[idx,i] = v(k,t=1e-6)[a]*(np.cos(p)-1j*np.sin(p))/L
				
	
	v_0 = np.array([1,-0.5-0.5j*np.sqrt(3),-0.5+0.5j*np.sqrt(3)])/np.sqrt(3)
	for j in range(L**2):
		for a in range(3):
			idx = 3*j + a
			psi_k_vals[idx,L**2] = v_0[a]/L
			
			
	for idx in idxs:
		fig,ax=plt.subplots()
		system.psi = psi_k_vals[:,idx]
		system.plot_re_wavefunction_tiled(fig,ax)
		ax.set_title(f'k = {k_BZ[idx,:]}')
		plt.show()
		
		
def plot_wannier_states(L=10,idxs=[0]):
	#Plot FT of k-states, i.e. Wannier states
	system = Kag.Kagome(psi_0=np.zeros(3*L**2),Lx=L,Ly=L,bc='periodic')
	
	fb_evects = system.eigvects[:,-1:-(L**2+2):-1].astype(dtype=np.complex128)
	k_BZ = BZ_k_vals_alt(L=L,plot_BZ=False)
	psi_k_vals = np.zeros((3*L**2,L**2+1),dtype=np.complex128)
	for i in range(L**2):
		k = k_BZ[i,:]
		for j in range(L**2):
			for a in range(3):
				idx = 3*j + a
				p = np.dot(k,system.sites[idx].r)
				psi_k_vals[idx,i] = v(k,t=1e-6)[a]*(np.cos(p)-1j*np.sin(p))/L
				
	
	v_0 = np.array([1,-0.5-0.5j*np.sqrt(3),-0.5+0.5j*np.sqrt(3)])/np.sqrt(3)
	for j in range(L**2):
		for a in range(3):
			idx = 3*j + a
			psi_k_vals[idx,L**2] = v_0[a]/L
			
			
	psi_wannier_vals = np.zeros((3*L**2,L**2),dtype=np.complex128)
	for i in range(L**2):
		#r_i = (system.sites[3*i].r + system.sites[3*i+1].r + system.sites[3*i+2].r)/3
		r_i = system.sites[3*i].r + 0.5*(system.a1+system.a2)
		for j in range(L**2):
			k = k_BZ[j,:]
			p = np.dot(k,r_i)
			psi_k = psi_k_vals[:,j]
			psi_wannier_vals[:,i] += (np.cos(p) + 1j*np.sin(p))*psi_k/L
			
	for idx in idxs:
		fig,axs=plt.subplots(1,2)
		system.psi = psi_wannier_vals[:,idx]
		print(np.max(np.real(psi_wannier_vals[:,idx])))
		print(np.max(np.imag(psi_wannier_vals[:,idx])))
		system.plot_re_wavefunction_tiled(fig,axs[0])
		system.plot_im_wavefunction_tiled(fig,axs[1])
		axs[0].set_title(r'$Re(W_i)$')
		axs[1].set_title(r'$Im(W_i)$')
		#plt.suptitle(f'i = {idx}, $r_i$ = {np.round((system.sites[3*idx].r + system.sites[3*idx+1].r + system.sites[3*idx+2].r)/3,2)}')
		plt.suptitle(f'i = {idx}, $r_i$ = {np.round(system.sites[3*idx].r + 0.5*(system.a1+system.a2),2)}')
		plt.show()	
	
	
	#A = np.conjugate(psi_wannier_vals).T @ psi_wannier_vals
	#plt.imshow(np.abs(A))
	#plt.show()
	
	
def backfold(k,BZPath,b1,b2):
	#Backfold a given k vector into the 1st BZ, to make sure that an equivalent set of k points
	#within the BZ is always chosen
	b3 = b1 + b2
	if not BZPath.contains_points([k]):
		if BZPath.contains_points([k+b1]):
			k += b1
		elif BZPath.contains_points([k+b2]):
			k += b2
		elif BZPath.contains_points([k+b3]):
			k += b3
		elif BZPath.contains_points([k-b1]):
			k -= b1
		elif BZPath.contains_points([k-b2]):
			k -= b2
		elif BZPath.contains_points([k-b3]):
			k -= b3
		else:
			print('Problem!')
			
	#Handle cases where k is exactly on lower/left edge of BZ - need to fold to opposite side to be consistent
	#Checking b3 (=b1+b2) first ensures correct handling of cases where adding b1 or b2 alone wouldn't return
	#a point from the original set of k_BZ points.
	elif BZPath.contains_points([k+b3]):
		k += b3
	elif BZPath.contains_points([k+b1]):
		k += b1
	elif BZPath.contains_points([k+b2]):
		k += b2
	
	
	return k
	
	
def plot_q_weight(psi,system,psi_k1k2_vals=None,plot_BZ=True,cmap=plt.cm.Blues,uppernorm='auto',plot_cbar=True,cbar_label=False,cbar_label_y=0.5,title=None,plot=True,save=False,filename=None,plot_wf=True):
	#Plot (optionally) wavefunction amplitude, and corresponding weight of states |k,q-k> summed over k values in the BZ (vs q in the BZ)
	b1_hat = system.b1/np.linalg.norm(system.b1)
	b2_hat = system.b2/np.linalg.norm(system.b2)
	dkx = np.linalg.norm(system.b2)/system.Lx
	dky = np.linalg.norm(system.b2)/system.Ly
			
	BZPath = mplPath.Path(system.BZ*(1+min(dkx,dky)/100))
	
	N = system.N_uc + 1
	N_comb = int(0.5*N*(N+1))
	c_q = np.zeros(system.N_uc)
	
	count = 0
	for i in range(N):
		if i != N-1:
			k1 = system.k_BZ[i,:]
		else:		#Case of 2nd Band Gamma point
			k1 = np.zeros(2)
		if psi_k1k2_vals is None:
			psi_k1 = system.fb_psi_k_sp[:,i]
		for j in range(i,N):
			if j != N-1:
				k2 = system.k_BZ[j,:]
			else:
				k2 = np.zeros(2)
				
			q = backfold(k1+k2,BZPath,system.b1,system.b2)
			
			if psi_k1k2_vals is None:
				psi_k2 = system.fb_psi_k_sp[:,j]
				psi_k1k2 = Kag2.direct_product(psi_k1,psi_k2)
			else:
				psi_k1k2 = psi_k1k2_vals[:,count]

			dec = int(1-np.floor(np.log10(dkx)))
			idx = np.where((np.round(system.k_BZ[:,0],dec)==np.round(q[0],dec))&(np.round(system.k_BZ[:,1],dec)==np.round(q[1],dec)))[0]
			c_q[idx] += np.abs(np.vdot(psi_k1k2,psi))**2
			count += 1
			print(f'Progress: {np.round(100*count/N_comb,2)}%',end='\r')
	
	if uppernorm == 'auto':
		norm = matplotlib.colors.Normalize(vmin=0, vmax=Kag2.uppernorm_func(np.max(c_q)))
	sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
	
	#dk = 4*np.pi/(system.a*np.sqrt(3)*system.Lx)
	dk = dkx
	
	if plot_wf:
		fig,axs = plt.subplots(1,2)
		ax = axs[1]
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])
		
		for i in range(system.N_uc):
			k = system.k_BZ[i,:]
			ax.add_patch(hexagon_patch(k,dk/np.sqrt(3),facecolor=cmap(norm(c_q[i])),edgecolor=None))
			
		if plot_BZ:
			#BZ = BrillouinZone(a=system.a)
			ax.add_patch(mplPatch.Polygon(system.BZ,fc=(0,0,0,0),ec='k'))
			
		scale = 1.5
		xmax = np.max(np.abs(system.k_BZ[:,0]))
		ymax = np.max(np.abs(system.k_BZ[:,1]))
		ax.set_xlim(-scale*xmax,scale*xmax)
		ax.set_ylim(-scale*ymax,scale*ymax)
			
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax)
			if cbar_label:
				cbar.set_label(r'$\sum_{k} |c_{k,q-k}|^2$',rotation=0,y=cbar_label_y)
			cbar.ax.locator_params(nbins=5)
			
		ax.set_title(r'$\sum_{k} |c_{k,q-k}|^2$')#+f'$k^\prime$={k1}')
		
		axs[0].set_position(pos=[axs[0].get_position().x0-0.05,ax.get_position().y0,axs[0].get_position().width,ax.get_position().height])
		system.psi = psi
		system.plot_amplitude(fig,axs[0],plot_cbar=True,label_cbar=False)
		axs[0].set_title(r'$\Psi$')
		
		plt.suptitle(title,y=0.85)
		
		if save:
			if filename is None:
				filename = 'k_spectrum_'+title+'.png'
			plt.savefig(filename,dpi=300)
	
		if plot:
			plt.show()
	
	
def plot_estate_q(idxs=[0],L=10,U=1.,a=1.,bc='periodic',plot_wf=True,plot=True,save=False):		#idxs of descending eigenvalues
	#Plot (optionally) eigenstate wavefunction amplitude for given indices, and corresponding weight of states |k,q-k> summed over k values in the BZ
	#(vs q values in the BZ)
	system = Kag2.double_occupied_site_system(L=L,U=U,a=a,skip_diag=True,skip_H=False,evolution_method='propagator',bc=bc,skip_k=False)
	num = np.max(idxs)+1
	evals,evects = sp.sparse.linalg.eigsh(system.H,k=3*L**2,return_eigenvectors=True,which='LA')
	
	N = system.N_uc + 1
	N_comb = int(0.5*N*(N+1))
	
	psi_k1k2_vals = np.zeros((system.N,N_comb),dtype=np.complex128)
	
	count = 0
	for i in range(N):
		#if i != N-1:
		#	k1 = system.k_BZ[i,:]
		#else:		#Case of 2nd Band Gamma point
		#	k1 = np.zeros(2)
		psi_k1 = system.fb_psi_k_sp[:,i]
		for j in range(i,N):
		#	if j != N-1:
		#		k2 = system.k_BZ[j,:]
		#	else:
		#		k2 = np.zeros(2)
			psi_k2 = system.fb_psi_k_sp[:,j]
			psi_k1k2_vals[:,count] = Kag2.direct_product(psi_k1,psi_k2)
			count += 1
			print(f'Progress: {np.round(100*count/N_comb,2)}%',end='\r')
	
	'''
	k_BZ = BZ_k_vals(L=L,a=a)
	
	b1 = np.array([1,-1/np.sqrt(3)])*2*np.pi/a
	b2 = np.array([0,1])*4*np.pi/(a*np.sqrt(3))
	b1_hat = b1/np.linalg.norm(b1)
	b2_hat = b2/np.linalg.norm(b2)
	dk = np.linalg.norm(b2)/L
	'''
	for i,idx in enumerate(idxs):
		psi = evects[:,3*L**2-idx-1]
		plot_q_weight(psi,system,psi_k1k2_vals=psi_k1k2_vals,title=f'E[{idx}]={np.round(evals[3*L**2-idx-1],4)}',plot_BZ=True,plot_wf=plot_wf,plot=plot,save=save)
		
	
def plot_K_weight(psi,system,psi_k1k2_vals=None,plot_BZ=True,cmap=plt.cm.Blues,uppernorm='auto',plot_cbar=True,cbar_label=False,cbar_label_y=0.5,title=None,plot=True,save=False,filename=None,plot_wf=True):
	#Plot weights of K states |k,k-K> (i.e. K = k1-k2) summed over k in the BZ, and optionally plot the amplitude of |psi>
	b1_hat = system.b1/np.linalg.norm(system.b1)
	b2_hat = system.b2/np.linalg.norm(system.b2)
	dkx = np.linalg.norm(system.b2)/system.Lx
	dky = np.linalg.norm(system.b2)/system.Ly
			
	BZPath = mplPath.Path(system.BZ*(1+min(dkx,dky)/100))
	
	N = system.N_uc + 1
	N_comb = int(0.5*N*(N+1))
	c_K = np.zeros(system.N_uc)
	
	count = 0
	for i in range(N):
		if i != N-1:
			k1 = system.k_BZ[i,:]
		else:		#Case of 2nd Band Gamma point
			k1 = np.zeros(2)
		if psi_k1k2_vals is None:
			psi_k1 = system.fb_psi_k_sp[:,i]
		for j in range(i,N):
			if j != N-1:
				k2 = system.k_BZ[j,:]
			else:
				k2 = np.zeros(2)
				
			K = backfold(k1-k2,BZPath,system.b1,system.b2)
			
			if psi_k1k2_vals is None:
				psi_k2 = system.fb_psi_k_sp[:,j]
				psi_k1k2 = Kag2.direct_product(psi_k1,psi_k2)
			else:
				psi_k1k2 = psi_k1k2_vals[:,count]

			dec = int(1-np.floor(np.log10(dkx)))
			idx = np.where((np.round(system.k_BZ[:,0],dec)==np.round(K[0],dec))&(np.round(system.k_BZ[:,1],dec)==np.round(K[1],dec)))[0]
			c_K[idx] += np.abs(np.vdot(psi_k1k2,psi))**2
			count += 1
			print(f'Progress: {np.round(100*count/N_comb,2)}%',end='\r')
	
	if uppernorm == 'auto':
		norm = matplotlib.colors.Normalize(vmin=0, vmax=Kag2.uppernorm_func(np.max(c_K)))
	sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
	
	#dk = 4*np.pi/(system.a*np.sqrt(3)*system.Lx)
	dk = dkx
	
	if plot_wf:
		fig,axs = plt.subplots(1,2)
		ax = axs[1]
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])
		
		for i in range(system.N_uc):
			k = system.k_BZ[i,:]
			ax.add_patch(hexagon_patch(k,dk/np.sqrt(3),facecolor=cmap(norm(c_K[i])),edgecolor=None))
			
		if plot_BZ:
			#BZ = BrillouinZone(a=system.a)
			ax.add_patch(mplPatch.Polygon(system.BZ,fc=(0,0,0,0),ec='k'))
			
		scale = 1.5
		xmax = np.max(np.abs(system.k_BZ[:,0]))
		ymax = np.max(np.abs(system.k_BZ[:,1]))
		ax.set_xlim(-scale*xmax,scale*xmax)
		ax.set_ylim(-scale*ymax,scale*ymax)
			
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax)
			if cbar_label:
				cbar.set_label(r'$\sum_{k} |c_{k+K,k}|^2$',rotation=0,y=cbar_label_y)
			cbar.ax.locator_params(nbins=5)
			
		ax.set_title(r'$\sum_{k} |c_{k+K,k}|^2$')#+f'$k^\prime$={k1}')
		
		axs[0].set_position(pos=[axs[0].get_position().x0-0.05,ax.get_position().y0,axs[0].get_position().width,ax.get_position().height])
		system.psi = psi
		system.plot_amplitude(fig,axs[0],plot_cbar=True,label_cbar=False)
		axs[0].set_title(r'$\Psi$')
		
		plt.suptitle(title,y=0.85)
		
		if save:
			if filename is None:
				filename = 'k_spectrum_'+title+'.png'
			plt.savefig(filename,dpi=300)
	
		if plot:
			plt.show()
			
	
def plot_estate_K(idxs=[0],L=10,U=1.,a=1.,bc='periodic',plot_wf=True,plot=True,save=False):		#idxs of descending eigenvalues
	#Plot weights of K states |k,k-K> (i.e. K = k1-k2) summed over k in the BZ for eigenstates, and optionally plot the eigenstate amplitude
	system = Kag2.double_occupied_site_system(L=L,U=U,a=a,skip_diag=True,skip_H=False,evolution_method='propagator',bc=bc,skip_k=False)
	num = np.max(idxs)+1
	evals,evects = sp.sparse.linalg.eigsh(system.H,k=3*L**2,return_eigenvectors=True,which='LA')
	
	N = system.N_uc + 1
	N_comb = int(0.5*N*(N+1))
	
	psi_k1k2_vals = np.zeros((system.N,N_comb),dtype=np.complex128)
	
	count = 0
	for i in range(N):
		psi_k1 = system.fb_psi_k_sp[:,i]
		for j in range(i,N):
			psi_k2 = system.fb_psi_k_sp[:,j]
			psi_k1k2_vals[:,count] = Kag2.direct_product(psi_k1,psi_k2)
			count += 1
			print(f'Progress: {np.round(100*count/N_comb,2)}%',end='\r')


def plot_q_K_weight(psi,system,psi_k1k2_vals=None,plot_BZ=True,cmap=plt.cm.Blues,uppernorm='auto',plot_cbar=True,cbar_label=False,cbar_label_y=0.5,title=None,plot=True,save=False,filename=None,plot_wf=True):
	#Plot wavefunction amplitude, total momentum q
	b1_hat = system.b1/np.linalg.norm(system.b1)
	b2_hat = system.b2/np.linalg.norm(system.b2)
	dkx = np.linalg.norm(system.b2)/system.Lx
	dky = np.linalg.norm(system.b2)/system.Ly
	
	#print(system.b1)
	#print(system.b2)
	#done=False
			
	BZPath = mplPath.Path(system.BZ*(1+min(dkx,dky)/100))
	
	N = system.N_uc + 1
	N_comb = int(0.5*N*(N+1))
	c_q = np.zeros(system.N_uc)
	c_K = np.zeros(system.N_uc)
	
	count = 0
	for i in range(N):
		if i != N-1:
			k1 = system.k_BZ[i,:]
		else:		#Case of 2nd Band Gamma point
			k1 = np.zeros(2)
		if psi_k1k2_vals is None:
			psi_k1 = system.fb_psi_k_sp[:,i]
		for j in range(i,N):
			if j != N-1:
				k2 = system.k_BZ[j,:]
			else:
				k2 = np.zeros(2)
			
			q = backfold(k1+k2,BZPath,system.b1,system.b2)
			K = backfold(k1-k2,BZPath,system.b1,system.b2)
			
			if psi_k1k2_vals is None:
				psi_k2 = system.fb_psi_k_sp[:,j]
				psi_k1k2 = Kag2.direct_product(psi_k1,psi_k2)
			else:
				psi_k1k2 = psi_k1k2_vals[:,count]

			dec = int(1-np.floor(np.log10(dkx)))
			q_idx = np.where((np.round(system.k_BZ[:,0],dec)==np.round(q[0],dec))&(np.round(system.k_BZ[:,1],dec)==np.round(q[1],dec)))[0]
			K_idx = np.where((np.round(system.k_BZ[:,0],dec)==np.round(K[0],dec))&(np.round(system.k_BZ[:,1],dec)==np.round(K[1],dec)))[0]
			val = np.abs(np.vdot(psi_k1k2,psi))**2
			c_q[q_idx] += val
			c_K[K_idx] += val
			
			#if np.round(np.linalg.norm(K),3)==np.round(dkx,3) and not done and val > 0.01:
			#	print(k1,k2,q,K)
			#	done=True
			
			count += 1
			print(f'Progress: {np.round(100*count/N_comb,2)}%',end='\r')
	
	if uppernorm == 'auto':
		norm2 = matplotlib.colors.Normalize(vmin=0, vmax=Kag2.uppernorm_func(np.max(c_K)))
		norm1 = matplotlib.colors.Normalize(vmin=0, vmax=Kag2.uppernorm_func(np.max(c_q)))
	sm1 = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm1)
	sm2 = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm2)
	
	#dk = 4*np.pi/(system.a*np.sqrt(3)*system.Lx)
	dk = dkx
	
	if plot_wf:
		fig,axs = plt.subplots(1,3)
		fig.set_size_inches(10,3)
		ax = axs[1]
		ax.set_position(pos=[ax.get_position().x0-0.05,ax.get_position().y0-0.08,ax.get_position().width,ax.get_position().height])
		axs[0].set_position(pos=[axs[0].get_position().x0-0.1,ax.get_position().y0,ax.get_position().width,ax.get_position().height])
		axs[2].set_position(pos=[axs[2].get_position().x0,ax.get_position().y0,axs[2].get_position().width,axs[2].get_position().height])
		axs[0].set_aspect(1,adjustable='datalim')
		
		#axs[0].set_xlim(-0.5,5.5)
		#axs[0].set_ylim(-0.5,5.5)
		
		
		ax.set_aspect(1,adjustable='datalim')
		ax.set_xticks([])
		ax.set_yticks([])
		
		for i in range(system.N_uc):
			k = system.k_BZ[i,:]
			ax.add_patch(hexagon_patch(k,dk/np.sqrt(3),facecolor=cmap(norm1(c_q[i])),edgecolor=None))
			
		if plot_BZ:
			#BZ = BrillouinZone(a=system.a)
			ax.add_patch(mplPatch.Polygon(system.BZ,fc=(0,0,0,0),ec='k'))
			
		scale = 1.5
		xmax = np.max(np.abs(system.k_BZ[:,0]))
		ymax = np.max(np.abs(system.k_BZ[:,1]))
		ax.set_xlim(-scale*xmax,scale*xmax)
		ax.set_ylim(-scale*ymax,scale*ymax)
			
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm1,cax=cax)
			if cbar_label:
				cbar.set_label(r'$\sum_{k} |c_{k,k-q}|^2$',rotation=0,y=cbar_label_y)
			cbar.ax.locator_params(nbins=5)
			
		ax.set_title(r'$\sum_{k} |c_{k,q-k}|^2$')
		
		axs[2].set_aspect(1,adjustable='datalim')
		axs[2].set_xticks([])
		axs[2].set_yticks([])
		
		for i in range(system.N_uc):
			k = system.k_BZ[i,:]
			axs[2].add_patch(hexagon_patch(k,dk/np.sqrt(3),facecolor=cmap(norm2(c_K[i])),edgecolor=None))
			
		if plot_BZ:
			axs[2].add_patch(mplPatch.Polygon(system.BZ,fc=(0,0,0,0),ec='k'))

		axs[2].set_xlim(-scale*xmax,scale*xmax)
		axs[2].set_ylim(-scale*ymax,scale*ymax)
			
		if plot_cbar:
			cax2 = fig.add_axes([axs[2].get_position().x1+0.01,axs[2].get_position().y0,0.02,axs[2].get_position().height])
			cbar2 = plt.colorbar(mappable=sm2,cax=cax2)
			if cbar_label:
				cbar2.set_label(r'$\sum_{k} |c_{k+K,k}|^2$',rotation=0,y=cbar_label_y)
			cbar2.ax.locator_params(nbins=5)
			
		axs[2].set_title(r'$\sum_{k} |c_{k+K,k}|^2$')
		
		system.psi = psi
		system.plot_amplitude(fig,axs[0],plot_cbar=True,label_cbar=False)
		axs[0].set_title(r'$\Psi$')
	
		plt.suptitle(title,y=0.975)
		
		if save:
			if filename is None:
				filename = 'k_spectrum_'+title+'.png'
			plt.savefig(filename,dpi=300)
	
		if plot:
			plt.show()
					

def plot_estate_q_K(idxs=[0],L=10,U=1.,a=1.,bc='periodic',plot_wf=True,plot=True,save=False):		#idxs of descending eigenvalues
	#Plot the eigenstate amplitude, q weights and K weights in the BZ on the same figure
	system = Kag2.double_occupied_site_system(L=L,U=U,a=a,skip_diag=True,skip_H=False,evolution_method='propagator',bc=bc,skip_k=False)
	num = np.max(idxs)+1
	evals,evects = sp.sparse.linalg.eigsh(system.H,k=3*L**2,return_eigenvectors=True,which='LA')
	
	N = system.N_uc + 1
	N_comb = int(0.5*N*(N+1))
	
	psi_k1k2_vals = np.zeros((system.N,N_comb),dtype=np.complex128)
	
	count = 0
	for i in range(N):
		psi_k1 = system.fb_psi_k_sp[:,i]
		for j in range(i,N):
			psi_k2 = system.fb_psi_k_sp[:,j]
			psi_k1k2_vals[:,count] = Kag2.direct_product(psi_k1,psi_k2)
			count += 1
			print(f'Progress: {np.round(100*count/N_comb,2)}%',end='\r')
	
	'''
	k_BZ = BZ_k_vals(L=L,a=a)
	
	b1 = np.array([1,-1/np.sqrt(3)])*2*np.pi/a
	b2 = np.array([0,1])*4*np.pi/(a*np.sqrt(3))
	b1_hat = b1/np.linalg.norm(b1)
	b2_hat = b2/np.linalg.norm(b2)
	dk = np.linalg.norm(b2)/L
	'''
	for i,idx in enumerate(idxs):
		psi = evects[:,3*L**2-idx-1]
		plot_q_K_weight(psi,system,psi_k1k2_vals=psi_k1k2_vals,title=f'E[{idx}]={np.round(evals[3*L**2-idx-1],4)}',plot_BZ=True,plot_wf=plot_wf,plot=plot,save=save)
		
		
def V_kk(q,L=5,U=0.1,k_BZ=None):
	#Interaction matrix V_(k,k')(q) defined in Torma et al. 2018
	N_uc = L**2
	N_sites = 3*N_uc
	
	if k_BZ is None:
		k_BZ = BZ_k_vals_alt(L=L,plot_BZ=False)
		
	V = np.zeros((N_uc,N_uc),dtype=np.complex128)
	for i in range(N_uc):
		k1 = k_BZ[i,:]
		v_k1 = v(k1)
		v_qk1 = v(q-k1)
		for j in range(i,N_uc):
			k2 = k_BZ[j,:]
			v_k2 = v(k2)
			v_qk2 = v(q-k2)
			val = 0
			for l in range(3):
				val += np.conjugate(v_k1[l])*np.conjugate(v_qk1[l])*v_k2[l]*v_qk2[l]*U/N_uc
			V[i,j] = val
			V[j,i] = np.conjugate(val)
	
	return V
			
def plot_V_evals(q,L=5,U=0.1,a=1.):
	#Plot eigenvalues of interaction matrix V_(k,k')(q) defined above
	b1 = np.array([1,-1/np.sqrt(3)])*2*np.pi/a
	b2 = np.array([0,1])*4*np.pi/(a*np.sqrt(3))
	b1_hat = b1/np.linalg.norm(b1)
	b2_hat = b2/np.linalg.norm(b2)
	dk = np.linalg.norm(b2)/L
	
	q_true = (q[0]*b1_hat + q[1]*b2_hat)*dk
	
	
	k_BZ = BZ_k_vals_alt(L=L,plot_BZ=False)
	
	V = V_kk(q=q_true,L=L,U=U,k_BZ=k_BZ)
	
	evals,evects = np.linalg.eigh(V)
	
	fig,ax = plt.subplots()
	ax.plot(np.arange(np.size(evals)),evals,color='b',marker='x',ls='')
	ax.set_ylabel('E/J')
	ax.set_xlabel('Eigenstate Index')
	plt.show()
	
	
def compare_V_numerical(L=5,U=0.1):
	#Compare eigenvalues of V_(k,k')(q) matrix with exact numerical eigenvalues of the full Hamiltonian
	N_uc = L**2
	N_sites = 3*L**2
	k_BZ = BZ_k_vals_alt(L=L,plot_BZ=False)
	k_BZ_mag = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(N_uc)])
	q_unique,idxs = np.unique(k_BZ_mag,return_index=True)
	
	
	fig,ax = plt.subplots()
	
	n=0
	for idx in idxs:
		print(f'Evaluating q value {n+1} out of {np.size(idxs)}...',end='\r')
		q = k_BZ[idx,:]
		V = V_kk(q=q,L=L,U=U,k_BZ=k_BZ)
		evals,evects = np.linalg.eigh(V)
		evals_nonzero = evals[evals>U/100]
		for i in range(np.size(evals_nonzero)):
			ax.plot(k_BZ_mag[idx],evals_nonzero[i],color='b',marker='x',ls='')
		n += 1
			
	print('')
	print('Evaluating numerical spectrum...')
	plot_evals_vs_q(L=L,U=U,ax=ax,label=None,color='r',shift=True)
	
	ax.set_ylabel(r'$\Delta E_q$ / $J$',rotation=0)
	ax.set_xlabel(r'$q$ / $a^{-1}$')
	ax.set_ylim(bottom=0)
	
	handles = [mplLines.Line2D([0],[0],color='b',marker='x',ls='',label='Analytical'),
				mplLines.Line2D([0],[0],color='r',marker='x',ls='',label='Numerical')]
	
	ax.legend(handles=handles)
	ax.set_title(f'Energy of Bound Particle Pairs, {L}x{L} system, U={U}, bc=periodic')
	
	plt.show()


def generate_V_numerical_data(filename,L=10,U=0.1,bc='periodic'):
	#Generate data calculating interaction matrices V_(k,k')(q) for all q, and corresponding eigenvalues, and the exact numerical eigenspectrum
	N_uc = L**2
	N_sites = 3*L**2
	k_BZ = BZ_k_vals_alt(L=L,plot_BZ=False)
	k_BZ_mag = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(N_uc)])
	q_unique,idxs = np.unique(np.round(k_BZ_mag,4),return_index=True)
	
	V_evals = np.zeros((np.size(idxs),3))
	for i,idx in enumerate(idxs):
		print(f'Evaluating q value {i+1} out of {np.size(idxs)}...',end='\r')
		q = k_BZ[idx,:]
		V = V_kk(q=q,L=L,U=U,k_BZ=k_BZ)
		evals,evects = np.linalg.eigh(V)
		V_evals[i,:] = evals[evals>U/100]
		
	#V_evals = np.reshape(V_evals,(3*np.size(idxs)))
			
	print('')
	print('Evaluating numerical spectrum...')
	
	system = Kag2.double_occupied_site_system(L=L,U=U,skip_diag=True,evolution_method='propagator',skip_k=False,bc=bc)
	evals,evects = sp.sparse.linalg.eigsh(system.H,which='LA',k=3*L**2,return_eigenvectors=True)
	evals -= 4
	evals_unique,idxs = np.unique(evals.round(decimals=6),return_index=True)
	evects_unique = evects[:,idxs]
	
	print('Evaluating values of q...')
	
	N = system.N_uc + 1
	N_comb = int(0.5*N*(N+1))
	psi_k1k2_vals = np.zeros((system.N,N_comb),dtype=np.complex128)
	count = 0
	for i in range(N):
		psi_k1 = system.fb_psi_k_sp[:,i]
		for j in range(i,N):
			psi_k2 = system.fb_psi_k_sp[:,j]
			psi_k1k2_vals[:,count] = Kag2.direct_product(psi_k1,psi_k2)
			count += 1
			print(f'Progress: {np.round(100*count/N_comb,2)}%',end='\r')
	
	q_vals = np.zeros(np.size(idxs))
	
	for i,idx in enumerate(idxs):
		c_q = calc_c_q(evects[:,idx],system,psi_k1k2_vals=psi_k1k2_vals)
		q = np.linalg.norm(system.k_BZ[np.argmax(c_q),:])
		q_vals[i] = q
	
	np.savez(filename,L=L,U=U,evals_numerical=evals_unique,qvals_numerical=q_vals,evals_analytical=V_evals,qvals_analytical=q_unique)


def compare_V_numerical_fromdata(filename,ax=None,plot=False,plot_V=True,plot_fit=False,fit_idx1=0,fit_idx2=1,printparams=True):
	#Given npz files containing data generated above, plot the data
	data = np.load(filename)
	evals_numerical = data['evals_numerical']
	qvals_numerical = data['qvals_numerical']
	evals_analytical = data['evals_analytical']
	qvals_analytical = data['qvals_analytical']
	
	idxs = np.argsort(evals_numerical)[::-1]
	evals_numerical = evals_numerical[idxs]
	qvals_numerical = qvals_numerical[idxs]
	
	if ax is None:
		fig,ax = plt.subplots()
	
	ax.plot(qvals_numerical,evals_numerical,color='b',marker='x',ls='')
	if plot_V:
		for i in range(3):
			ax.plot(qvals_analytical,evals_analytical[:,i],color='r',marker='x',ls='')
		
	ax.set_ylabel(r'$\Delta E_q$ / $J$',fontsize=13)
	ax.set_xlabel(r'$q$ / $a^{-1}$',fontsize=13)
	ax.set_ylim(bottom=0)
	
	
	handles = [mplLines.Line2D([0],[0],color='b',marker='x',ls='',label='Numerical Values')]
	
	if plot_V:
		handles.append(mplLines.Line2D([0],[0],color='r',marker='x',ls='',label='Analytical Values'))
				
	if plot_fit == 'linear':
		if plot_V:
			fit_col = 'b'
		else:
			fit_col = 'r'
		popt,pcov = sp.optimize.curve_fit(linear_fit,qvals_numerical[fit_idx1:fit_idx2+1],evals_numerical[fit_idx1:fit_idx2+1])
		perr = np.sqrt(np.diag(pcov))
		qfit = np.linspace(0,np.max(qvals_numerical),20)
		yfit = linear_fit(qfit,popt[0],popt[1])
		ax.plot(qfit,yfit,color=fit_col,ls='--')
		if printparams:
			print('Fit y = a + b*x')
			print(f'a = {popt[0]} +/- {perr[0]}')
			print(f'b = {popt[1]} +/- {perr[1]}')
		handles.insert(1,mplLines.Line2D([0],[0],color=fit_col,marker=None,ls='--',label='Linear Fit'))
	
	ax.legend(handles=handles,fontsize=13)
		
	if plot:
		plt.show()
		
		
def generate_shift2_numerical_data(filename,L=10,U=0.1,bc='periodic'):
	#Generate data of the (approximate) shift2() formula from Torma et al. 2018
	N_uc = L**2
	N_sites = 3*L**2
	k_BZ = BZ_k_vals_alt(L=L,plot_BZ=False)
	k_BZ_mag = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(N_uc)])
	q_unique,idxs = np.unique(np.round(k_BZ_mag,4),return_index=True)
	
	shift_vals = np.zeros(np.size(idxs))
	for i,idx in enumerate(idxs):
		print(f'Evaluating q value {i+1} out of {np.size(idxs)}...',end='\r')
		q = k_BZ[idx,:]
		shift_vals[i] = shift2(q,k_BZ=k_BZ,L=L)*U/N_sites

	
	np.savez(filename,L=L,U=U,shift_vals=shift_vals,q_vals=q_unique)
	
	
def V_kk_approx(q,L=5,U=0.1,k_BZ=None):
	#Return approximate interaction matrix defined in Torma et al. 2018
	N_uc = L**2
	N_sites = 3*N_uc
	
	if k_BZ is None:
		k_BZ = BZ_k_vals_alt(L=L,plot_BZ=False)
		
	V = np.zeros((N_uc,N_uc),dtype=np.complex128)
	for i in range(N_uc):
		k1 = k_BZ[i,:]
		v_k1 = v(k1)
		v_qk1 = v(q-k1)
		for j in range(i,N_uc):
			k2 = k_BZ[j,:]
			v_k2 = v(k2)
			v_qk2 = v(q-k2)
			val = 0
			#for l in range(3):
			val += np.vdot(v_k1,v_qk1)*np.vdot(v_k2,v_qk2)*U/N_sites
			V[i,j] = val
			V[j,i] = np.conjugate(val)
	
	return V
	
	
def compare_V_V_approx(L=5,U=0.1):
	#Compare eigenvalues of exact and approximate interaction matrices defined above
	#NB the 'exact' interaction matrix contains an implicit isolated flat band approximation,
	#hence doesn't agree exactly with numerical spectrum
	N_uc = L**2
	N_sites = 3*L**2
	k_BZ = BZ_k_vals_alt(L=L,plot_BZ=False)
	k_BZ_mag = np.array([np.linalg.norm(k_BZ[i,:]) for i in range(N_uc)])
	q_unique,idxs = np.unique(np.round(k_BZ_mag,5),return_index=True)
	
	
	fig,ax = plt.subplots()
	
	n=0
	done=False
	for idx in idxs:
		print(f'Evaluating q value {n+1} out of {np.size(idxs)}...',end='\r')
		q = k_BZ[idx,:]
		
			
		V = V_kk(q=q,L=L,U=U,k_BZ=k_BZ)
		V_approx = V_kk_approx(q=q,L=L,U=U,k_BZ=k_BZ)
		evals,evects = np.linalg.eigh(V)
		evals_nonzero = evals[evals>U/100]
		
		if np.linalg.norm(q) > 2 and not done:
			max_eval = np.max(evals_nonzero)
			max_evect = evects[:,evals==max_eval]
			
			#print(f'A_k={max_evect}')
			done = True
			fig1,ax1 = plt.subplots()
			ax1.plot(np.abs(max_evect),'kx',label='Abs')
			ax1.plot(np.real(max_evect),'bx',label='Re')
			ax1.plot(np.imag(max_evect),'rx',label='Im')
			ax1.set_title(f'q={np.round(q,3)}')
			ax1.legend()
			#plt.show()
			
		evals_approx,evects_approx = np.linalg.eigh(V_approx)
		evals_approx_nonzero = evals_approx[evals_approx>U/1000]
		for i in range(np.size(evals_nonzero)):
			ax.plot(k_BZ_mag[idx],evals_nonzero[i],color='b',marker='x',ls='')
		for i in range(np.size(evals_approx_nonzero)):
			ax.plot(k_BZ_mag[idx],evals_approx_nonzero[i],color='r',marker='x',ls='')
		n += 1
	
	ax.set_ylabel(r'$\Delta E_q$ / $J$')
	ax.set_xlabel(r'$q$ / $a^{-1}$')
	ax.set_ylim(bottom=0)
	
	handles = [mplLines.Line2D([0],[0],color='b',marker='x',ls='',label='Exact'),
				mplLines.Line2D([0],[0],color='r',marker='x',ls='',label='Approximate')]
	
	ax.legend(handles=handles)

	ax.set_title(f'Comparison of $V(kk^\prime,q)$ Matrices, {L}x{L} system, U={U}, bc=periodic')
	
	plt.show()
	
		



if __name__ == '__main__':
	compare_V_V_approx(L=10)
	#plot_V_evals(q=[0,1],L=6,U=0.1)
	#generate_V_numerical_data('FB_Energy_Shifts_L10x10_U0.1.npz',L=10,U=0.1)
	#generate_shift2_numerical_data('FB_Energy_Shifts_Approx_L10x10_U0.1.npz',L=10,U=0.1)
	#compare_V_numerical_fromdata('test.npz')
	#compare_V_numerical(L=10,U=0.1)
	
	#plot_sum_evals_vs_q(L=6,plot=True,color='r',label='Summed Pair Energies')
	#plot_evals_vs_q(L=6,plot=True)
	#plot_all_shifts(plot=True,plot_numerical=True)
	#plot_estate_q(idxs=[0],L=4,U=0.1)
	
	#plot_estate_K(idxs=[0,1,7,13],L=6,U=0.1)
	#[0,1,7,13,16,19,25,31,37,38,39,45]
	
	#plot_estate_q_K(idxs=[0,1,7,13],L=10,U=0.1)
	
	#fig,ax=plt.subplots()
	#plot_v(ax)
	
	#compare_ck()
	
	#compare_ed()
	
	#BrillouinZone(L=4,plot_BZ=True)
	
	#plot_hexagon(L=10,r_H='centre')
	
	#plot_k_shift_vs_k(U=0.1,plot_fit=None,fit_idx1=1,fit_idx2=9)
	
	#plot_weighted_shift_vs_k(U=0.1,plot_fit='linear',fit_idx1=1,fit_idx2=9,plot_ck=False)
	
	#plot_top_evals(L=6,U=0.1,k=108,bc='periodic')
	
	#plot_shifts_vs_k2(L=10)
	
	#plot_weighted_shifts_vs_k2()
	
	#plot_top_estates(L=5,k=4,bc='periodic')
	
	#BZ_k_vals_alt(L=6,plot_BZ=True)

	#plot_estate_k_weights(idxs=[0],L=5,U=0.1,plot_wf=True,save=False,plot=True)
	
	#plot_estate_c_kk1(idxs=[13,14,15],k1_vals=[np.array([0,0])],L=4,U=0.1)
	
	#system = Kag2.two_hexagons_system(130,130,U=0.1,skip_diag=True,evolution_method='eigenvector',L=10,bc='periodic')
	#psi = system.psi_0
	
	#plot_A2(psi,system,L=10,plot_fit=None,fit_idx1=1,fit_idx2=13)
	
	#compare_12(q=np.array([-1,1]))
	
	#plot_shift2(plot_numerical=True,plot_fit='linear',L=5,plot=True)
	
	#plot_all_shifts(exact_E=[0.0353,0.028,0.0202,0.0181,0.0134,0.0071,0.0076,0.0082,0.0108,0.0118,0.0071,0.0066,0.0079,0.0066,0.0081],plot_fit=None,fit_idx2=4)
	
	#flatband_projection(L=10,plot_mean=False)
	
	#plot_k_states(L=10,idxs=[0,1,2])
	
	#plot_wannier_states(L=10,idxs=[25])
	
