'''
This file contains the core code for running two-particle simulations.

The kagome system is defined as a class, which upon initialization,
builds the lattice and calculates the Hamiltonian matrix. Methods are 
defined for time-evolution and calculating observables such as the density, 
and producing various plots.

System parameters are:

psi_0: initial state
Lx: number of unit cells in x direction
Ly: number of unit cells in y direction
a: lattice constant
J: hopping coefficient
U: interaction strength
hbar: reduced Planck constant
evolution_method: whether to calculate time evolution using the eigenvector or propagator method
bc: boundary conditions, either open or periodic

'''


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import matplotlib.path as mplPath
import matplotlib.patches as mplPatch
import matplotlib.lines as mplLines
import cProfile
import Kagome as Kag



class Lattice_Site:
	#Class for lattice sites. Attributes are position, a list of 
	#near-neighbour lattice sites, and the sublattice A, B or C
	def __init__(self,r,index=None,sublattice=None):
		self.r = r
		self.x = self.r[0]
		self.y = self.r[1]
		self.nn = []
		self.sublattice = sublattice
		self.index = index
	
	
	def add_nn(self,neighbour):
		if neighbour not in self.nn:
			self.nn.append(neighbour)
		if self not in neighbour.nn:
			neighbour.nn.append(self)
	
		

class Kagome2:
	def __init__(self,psi_0,Lx,Ly,a=1.,J=1.,U=1.,hbar=1.,skip_H=False,skip_diag=False,skip_k=False,evolution_method='propagator',bc='periodic'):
		#Wavefunction
		self.psi_0 = psi_0
		self.psi = np.copy(psi_0)
		
		#Hamiltonian parameters
		self.J = J
		self.U = U
		self.hbar = hbar
		
		#Lattice parameters
		self.a = a
		self.Lx = Lx
		self.Ly = Ly
		self.N_uc = self.Lx*self.Ly
		self.N_sites = 3*self.Lx*self.Ly
		
		#Lattice Vectors
		self.a1 = self.a*np.array([1.,0.])
		self.a2 = self.a*np.array([0.5,np.sqrt(3)/2])
		
		#N = number of basis states
		self.N = int(self.N_sites*(self.N_sites+1)/2)
		if np.size(self.psi_0) != self.N:
			print('Warning: psi_0 not of correct dimensions for lattice')
		
		#Set evolution method and boundary conditions
		self.evolution_method = evolution_method
		self.bc = bc
		
		#Build lattice
		self.sites = []
		self.build_lattice()
		
		#Build index arrays
		self.site_index_array = self.build_site_array()
		self.state_index_array = self.build_state_array()

		#Optionally build and diagonalize Hamiltonian matrix
		if not skip_H:
			self.H = np.zeros((self.N,self.N))
			self.build_H()
		
		if not skip_diag:
			self.eigvals,self.eigvects = np.linalg.eigh(self.H)
			self.w = self.eigvals/self.hbar
			self.c_0 = self.psi_0 @ self.eigvects
			
		if self.evolution_method == 'propagator':
			self.H = sp.sparse.csr_array(self.H)
		
		if evolution_method == 'eigenvector' and skip_diag:
			print('Warning: Ensure that eigenvectors/values are input explicitly if required')
	
		#Optionally calculate reciprocal lattice vectors, Brillouin Zone and k-space eigenstates
		if not skip_k:
			self.b1 = np.array([1,-1/np.sqrt(3)])*2*np.pi/self.a
			self.b2 = np.array([0,1])*4*np.pi/(self.a*np.sqrt(3))
			self.BZ = self.BrillouinZone()
			self.k_BZ = self.BZ_k_vals()
			#print('yep')
			self.fb_psi_k_sp = self.calc_fb_psi_k_sp()
			#self.fb_psi_k_2p = self.calc_fb_psi_k_2p()
			self.fb_projector = self.calc_fb_projector()
			
		self.opdm = np.zeros((self.N_sites,self.N_sites))
		

	#Functions for constructing the lattice and Hamiltonian:
	def add_site(self,site):
		#Add a site to the lattice
		self.sites.append(site)
		
	def build_lattice(self):
		#Build lattice of Lx x Ly unit cells, including storing all near neighbours of each site
		for i in range(self.Lx):
			for j in range(self.Ly):
				r_uc = i*self.a1 + j*self.a2		#Unit cell location
				
				A = Lattice_Site(r_uc,sublattice='A',index=3*(self.Ly*i + j))		#Index = 3(Ly*i + j) for A site; +1 for B site, +2 for C site
				B = Lattice_Site(r_uc + 0.5*self.a1,sublattice='B',index=3*(self.Ly*i+j)+1)		
				C = Lattice_Site(r_uc + 0.5*self.a2,sublattice='C',index=3*(self.Ly*i+j)+2)
				
				A.add_nn(B)
				A.add_nn(C)
				B.add_nn(C)
				
				if j > 0:			
					A.add_nn(self.sites[3*(self.Ly*i+j) - 1])						#Connection between current A site and previous C site (down one row)
					
				if i > 0:
					A.add_nn(self.sites[3*(self.Ly*(i-1)+j) + 1])			#Connection between current A site and B site on previous 'column'
					if j < self.Ly - 1:
						C.add_nn(self.sites[3*(self.Ly*(i-1)+j) + 4])			#Between current C site and B site on previous column and up one row
					
				self.add_site(A)
				self.add_site(B)
				self.add_site(C)
				
		if self.bc == 'periodic':
			#Add near neighbours between sites on opposite sides of lattice if using periodic boundary conditions
			for j in range(self.Ly):
				self.sites[3*j].add_nn(self.sites[3*(self.Ly*(self.Lx-1)+j)+1])		#A and B sites on left and right sides
				if j < self.Ly-1:
					self.sites[3*j+2].add_nn(self.sites[3*(self.Ly*(self.Lx-1)+j+1)+1])		#C and B on left and right sides
				else:
					self.sites[3*j+2].add_nn(self.sites[3*(self.Ly*(self.Lx-1))+1])			#Top left and bottom right corners
					
			for i in range(self.Lx):
				self.sites[3*self.Ly*i].add_nn(self.sites[3*self.Ly*(i+1)-1])		#A and C sites on bottom and top rows
				if i < self.Lx-1:
					self.sites[3*self.Ly*i+1].add_nn(self.sites[3*self.Ly*(i+2)-1])	#B and C sites on bottom and top rows
				#Corner hopping already added earlier
					
	
	
	def build_H(self):
		#Build the Hamiltonian matrix for the 2-particle system.
		#H[i,j] annhilates particle in state j and creates in i.
		for i in range(self.N_sites):
			for j in range(i,self.N_sites):
				#idx = self.state_idx(i,j)
				idx = self.state_index_array[i,j]
				
				if i == j:
					#On site repulsion
					self.H[idx,idx] = self.U
					
					#Hopping - matrix elements are -J*sqrt(2)
					for neighbour in self.sites[j].nn:
						nn_idx = neighbour.index
						#nn_state_idx = self.state_idx(i,nn_idx)
						nn_state_idx = self.state_index_array[i,nn_idx]
						self.H[nn_state_idx,idx] = -self.J*np.sqrt(2)

				else:
					#Need to deal with case that hops to double occupied site correctly
					#Hopping from j site
					for neighbour in self.sites[j].nn:
						nn_idx = neighbour.index
						#nn_state_idx = self.state_idx(i,nn_idx)
						nn_state_idx = self.state_index_array[i,nn_idx]
						if nn_idx == i:
							self.H[nn_state_idx,idx] = -self.J*np.sqrt(2)
						else:
							self.H[nn_state_idx,idx] = -self.J
					
					#Hopping from i site
					for neighbour in self.sites[i].nn:
						nn_idx = neighbour.index
						#nn_state_idx = self.state_idx(nn_idx,j)
						nn_state_idx = self.state_index_array[nn_idx,j]
						if nn_idx == j:
							self.H[nn_state_idx,idx] = -self.J*np.sqrt(2)
						else:
							self.H[nn_state_idx,idx] = -self.J
					
					
	def state_idx(self,i,j):		
		#Return index of state b_i^+ b_j^+ |0>
		if i<= j:
			return int(i*(self.N_sites - 0.5*i - 0.5) + j)
		else:
			return int(j*(self.N_sites - 0.5*j - 0.5) + i)		#Swap j and i so that i is always the lower site index
		
	
	def build_site_array(self):			
		#Builds an array whose x-th row has elements [i,j] giving the two sites occupied by the state with index x 
		arr = np.zeros((self.N,2),dtype=int)
		x = 0
		for i in range(self.N_sites):
			for j in range(i,self.N_sites):
				arr[x,0] = i
				arr[x,1] = j
				x += 1
		return arr
		
		
	def build_state_array(self):
		#Builds an array whose element [i,j] is the index of the state corresponding to b_i^+ b_j^+ |0>
		arr = np.zeros((self.N_sites,self.N_sites),dtype=np.intc)
		for i in range(self.N_sites):
			for j in range(i,self.N_sites):
				idx = self.state_idx(i,j)
				arr[i,j] = idx
				arr[j,i] = idx
		return arr
				
	
	#Fucntions for performing operations on the wavefunction psi
	def evolve_psi(self,t):
		#Evolve psi forward from the current state by a time t
		if self.evolution_method == 'propagator':
			self.psi = sp.sparse.linalg.expm_multiply((-1j*t/self.hbar)*self.H,self.psi)
		elif self.evolution_method == 'eigenvector':
			exp = np.cos(self.w*t) - 1j*np.sin(self.w*t)
			c = self.psi @ self.eigvects
			self.psi = np.transpose(self.eigvects @ np.transpose(np.multiply(c,exp)))
		return self.psi
	
	
	def state(self,t):
		#Calculate and return psi at time t
		if t == 0:
			self.psi = self.psi_0
		else:
			if self.evolution_method == 'propagator':
				self.psi = sp.sparse.linalg.expm_multiply((-1j*t/self.hbar)*self.H,self.psi_0)
			elif self.evolution_method == 'eigenvector':
				exp = np.cos(self.w*t) - 1j*np.sin(self.w*t)
				self.psi = np.transpose(self.eigvects @ np.transpose(np.multiply(self.c_0,exp)))
		return self.psi
	
	
	def density(self,psi=None,t=None):		
		#Return array (length N_sites) of expected number of particles, i.e. density, on each site in lattice
		if psi is not None:
			self.psi = psi
		elif t is not None:
			self.psi = self.state(t)
		dens = np.zeros(self.N_sites)		
		for idx,val in enumerate(self.psi):
			[i,j] = self.site_index_array[idx,:]
			dens[i] += np.abs(val)**2
			dens[j] += np.abs(val)**2
		return dens

		
	def amplitude(self,psi=None,t=None):		
		#Return array (length N_sites), whose i-th element is the sum over j of coefficients of basis states c_i^+ c_j^+ |0>,
		#i.e. all basis states which contribute to site i
		#Limited physical interpretation, but can help visualise states with a large deal of coherence between basis states,
		#by showing sites where amplitude is positive or negative
		#If only one basis state contributes to each site, then |amplitude|^2 gives density
		if psi is not None:
			self.psi = psi
		elif t is not None:
			self.psi = self.state(t)
		amp = np.zeros(self.N_sites)		
		for idx,val in enumerate(self.psi):
			[i,j] = self.site_index_array[idx,:]
			amp[i] += val
			amp[j] += val
		return amp
		
		
	def npair(self):
		#Calculate and return probability that system is in a doublon state (two particles on the same site)
		#i.e. total density in doublon sites, divided by N_particles = 2
		idxs = np.diag(self.state_index_array)
		val = 0
		for idx in idxs:
			val += np.abs(psi[idx])**2
		return val
		
		
	def exp_separation(self,psi=None,t=None):
		#Calculate and return expected value of separation between particles
		if psi is not None:
			self.psi = psi
		elif t is not None:
			self.psi = self.state(t)
		d = 0
		for idx,val in enumerate(self.psi):
			[i,j] = self.site_index_array[idx,:]
			if self.bc == 'periodic':
				#Calculate distance as minimum among 4 possible 'backfolds' of r_j (j>i always)
				r1 = np.linalg.norm(self.sites[j].r - self.sites[i].r)
				r2 = np.linalg.norm(self.sites[j].r - self.sites[i].r - self.Lx*self.a1)
				r3 = np.linalg.norm(self.sites[j].r - self.sites[i].r - self.Ly*self.a2)
				r4 = np.linalg.norm(self.sites[j].r - self.sites[i].r - self.Lx*self.a1 - self.Ly*self.a2)
				r = np.min([r1,r2,r3,r4])
			else:
				r = np.linalg.norm(self.sites[j].r - self.sites[i].r)
			d += r*np.abs(val)**2
		return d
		
		
	def rdf(self,psi=None,t=None,return_r=True,normalize_by_count=False,return_counts=False):
		#Calculate and return 'radial distribution function' - i.e. density-density correlations vs site separation
		#Returns array of g(r) for each unique separation r between pairs of sites, and optionally the array of unique r
		if psi is not None:
			self.psi = psi
		elif t is not None:
			self.psi = self.state(t)
		r_vals = np.zeros(self.N)
		g_vals = np.zeros(self.N)
		for idx,val in enumerate(self.psi):
			[i,j] = self.site_index_array[idx,:]
			if self.bc == 'periodic':
				#Calculate distance as minimum among 4 possible 'backfolds' of r_j (j>i always)
				r1 = np.linalg.norm(self.sites[j].r - self.sites[i].r)
				r2 = np.linalg.norm(self.sites[j].r - self.sites[i].r - self.Lx*self.a1)
				r3 = np.linalg.norm(self.sites[j].r - self.sites[i].r - self.Ly*self.a2)
				r4 = np.linalg.norm(self.sites[j].r - self.sites[i].r - self.Lx*self.a1 - self.Ly*self.a2)
				r5 = np.linalg.norm(self.sites[j].r - self.sites[i].r + self.Ly*self.a2)
				r6 = np.linalg.norm(self.sites[j].r - self.sites[i].r - self.Lx*self.a1 + self.Ly*self.a2)
				#Need to consider adding Ly as well
				#Don't need to consider adding Lx, since j>i
				r_vals[idx] = np.min([r1,r2,r3,r4,r5,r6])
				#if r_vals[idx] > 4:
				#	print([i,j])
				#	print([r1,r2,r3,r4])
			else:
				r_vals[idx] = np.linalg.norm(self.sites[j].r - self.sites[i].r)
			g_vals[idx] += np.abs(val)**2
		
		r_vals = np.round(r_vals,4)
		r_unique,counts = np.unique(r_vals,return_counts=True)
		#fig,ax = plt.subplots()
		#ax.plot(r_unique,counts,color='b',marker='x',ls='')
		#ax.set_xlabel('r')
		#ax.set_ylabel('Count')
		#ax.set_title('Frequency of different inter-site separations')
		#plt.show()
		g_unique = np.array([np.sum(g_vals[r_vals==r]) for r in r_unique])
		
		if normalize_by_count:
			g_unique = np.divide(g_unique,counts)
		
		if return_r and return_counts:
			return r_unique,g_unique,counts
		elif return_r:
			return r_unique,g_unique
		elif return_counts:
			return g_unique,counts
		else:
			return g_unique
			
			
	def calc_opdm(self,psi=None,t=None):
		#Return (N_sites x N_sites) one-particle density matrix
		if psi is not None:
			self.psi = psi
		elif t is not None:
			self.psi = self.state(t)
			
		rho = np.zeros((self.N_sites,self.N_sites),dtype=np.complex128)
		for k in range(self.N_sites):
			for i in range(self.N_sites):
				idx1 = self.state_index_array[i,k]
				for j in range(self.N_sites):
				#for k in range(self.N_sites):
					idx2 = self.state_index_array[j,k]
					
					val = np.conjugate(self.psi[idx1])*self.psi[idx2]
					if i == k:
						val *= np.sqrt(2)
					if j == k:
						val *= np.sqrt(2)
						
					rho[i,j] += val

		self.opdm = rho	
		return rho
			
			
	#Functions for dealing with k-space
	def BrillouinZone(self):
		#Returns the vertices of the 1st Brillouin zone in k-space
		c1 = np.array([1/3,1/np.sqrt(3)])*2*np.pi/self.a #+ dk/100
		R = np.array([[0.5,-np.sqrt(3)/2],[np.sqrt(3)/2,0.5]])
		vertices = np.zeros((6,2))
		vertices[0,:] = c1
		for n in range(5):
			c1 = np.transpose(R @ c1.T)
			vertices[n+1,:] = c1
		return vertices
	
		
	def BZ_k_vals(self,shift=True):
		#Returns the values of k within the 1st BZ
		#k values are calculated on the interval [0,2pi), then backfolded as necessary into the BZ
		#For consistency with backfolding, the BZ path is infinitesimally enlarged before testing whether k values lie inside
		#or not, to remove ambiguity with those values that lie exactly on the edge
		b1_hat = self.b1/np.linalg.norm(self.b1)
		b2_hat = self.b2/np.linalg.norm(self.b2)
		dkx = np.linalg.norm(self.b2)/self.Lx
		dky = np.linalg.norm(self.b2)/self.Ly
			
		BZPath = mplPath.Path(self.BZ*(1+min(dkx,dky)/100))
		
		k_BZ = np.array([n*b1_hat*dkx + m*b2_hat*dky for n in np.arange(0,self.Lx,1) for m in np.arange(0,self.Ly,1)])
		
		for i in range(np.shape(k_BZ)[0]):
			k = k_BZ[i,:]
			if not BZPath.contains_points([k]):
				if BZPath.contains_points([k-self.b1]):
					k -= self.b1
				elif BZPath.contains_points([k-self.b2]):
					k -= self.b2
				elif BZPath.contains_points([k-self.b1-self.b2]):
					k -= self.b1+self.b2
		
		return k_BZ
	
	
	def v(self,k,t=1e-6):
		#Returns the eigenvector of the k-space Hamiltonian corresponding to the flat band
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
	
	
	def psi_k_sp(self,k):
		#Returns the single-particle flat band eigenstate with wavevector k for a system of the same size as self
		v_k = self.v(k)
		psi = np.zeros(self.N_sites,dtype=np.complex128)
		
		for i in range(self.N_uc):
			for a in range(3):
				idx = 3*i+a
				p = np.dot(k,self.sites[idx].r)	
				psi[idx] += v_k[a]*(np.cos(p)-1j*np.sin(p))/np.sqrt(self.N_uc)
						
		return psi
		
	
	def calc_fb_psi_k_sp(self):
		#Returns (N_sites x N_uc+1) array whose columns are all the single particle k-eigenstates with energy E=2J
		#Includes the k=0 state from the 2nd band
		psi = np.zeros((self.N_sites,self.N_uc+1),dtype=np.complex128)
		for i in range(self.N_uc):
			psi[:,i] = self.psi_k_sp(self.k_BZ[i,:])
			
		v_0 = np.array([1,-0.5-0.5j*np.sqrt(3),-0.5+0.5j*np.sqrt(3)])/np.sqrt(3)
		for i in range(self.N_uc):
			for a in range(3):
				idx = 3*i + a
				psi[idx,self.N_uc] = v_0[a]/np.sqrt(self.N_uc)
		return psi
		
		
	def calc_fb_projector(self):
		#Calculates and returns the projection operator to the flat band k states, P = sum_k |k><k|
		P = np.zeros((self.N_sites,self.N_sites),dtype=np.complex128)
		for i in range(self.N_uc+1):
			k = self.fb_psi_k_sp[:,i]
			P += np.outer(k,np.conjugate(k))
		return P
			
	
	def calc_fb_psi_k_2p(self):
		#Returns (N x 0.5*(N_uc+1)*(N_uc+2)) array whose columns are the two particle |k,k'> eigenstates
		#formed from the direct products of all the single particle flat band |k> states
		#fb_psi_k_sp = self.calc_fb_psi_k_sp()
		
		#A = np.conjugate(fb_psi_k_sp.T) @ fb_psi_k_sp
		#fig,ax=plt.subplots()
		#ax.imshow(np.abs(A))
		#plt.show()
		
		psi = np.zeros((self.N,int(0.5*(self.N_uc+1)*(self.N_uc+2))),dtype=np.complex128)
		n = 0
		for i in range(self.N_uc+1):
			for j in range(i,self.N_uc+1):
				psi[:,n] = direct_product(self.fb_psi_k_sp[:,i],self.fb_psi_k_sp[:,j])
				if i == j:
					psi[:,n] /= np.sqrt(2)
					
		A = np.conjugate(psi.T) @ psi
		fig,ax=plt.subplots()
		ax.imshow(np.abs(fb_psi_k_sp))
		plt.show()
					
		return psi
	
	
	def psi_kold(self,k1,k2):
		#Returns the state |k1,k2> in the real-space basis for the given system
		if self.Ly != self.Lx:
			print('Warning: Lx and Ly must be equal')
		v_k1 = self.v(k1)
		v_k2 = self.v(k2)
		
		psi = np.zeros(self.N,dtype=np.complex128)
		
		for i in range(self.N_uc):
			for a in range(3):
				for j in range(self.N_uc):
					for b in range(3):
						idx = self.state_idx(3*i+a,3*j+b)
						p1 = np.dot(k1,self.sites[3*i+a].r)
						p2 = np.dot(k2,self.sites[3*j+b].r)
						psi[idx] += v_k1[a]*v_k2[b]*(np.cos(p1+p2)-1j*np.sin(p1+p2))/self.N_uc
						if i==j and a==b:
							psi[idx] *= np.sqrt(2)		#Since basis state defined with 1/sqrt(2) normalization factor
		if np.array_equal(k1,k2):
			psi /= np.sqrt(2)
						
		return psi
		
		
	def opdm_fb_projection(self,opdm=None,psi=None,t=None):
		#Returns the density matrix produced by projecting the OPDM onto the flat band
		#The trace of this density matrix gives the 'flat band character' of the state
		if opdm is not None:
			self.opdm = opdm
		else:
			self.opdm = self.calc_opdm(psi=psi,t=t)
			
		#evals,evects = np.linalg.eigh(opdm)
			
		#proj_evects = self.fb_projector @ evects
		
		#proj_opdm = np.zeros((self.N_sites,self.N_sites),dtype=np.complex128)
		
		#for i in range(self.N_sites):
		#	proj_opdm += evals[i]*np.outer(proj_evects[:,i],np.conjugate(proj_evects[:,i]))
		
		proj_opdm = self.fb_projector @ self.opdm @ np.conjugate(self.fb_projector.T)
		
		return proj_opdm
		
		
	#Functions for real-space flat band eigenstates (hexagons)
	def hex(self,idx):
		#Return single-particle hexagon state on a lattice with Ly unit cells in each 'column'
		#idx is index of lower-left site of the hexagon; should be B site
		
		#Find (i,j) coordinate of unit cell in lattice vector basis
		idx0 = idx - 1
		i = idx0 // (3*self.Ly)
		j = (idx0 % (3*self.Ly)) / 3
		
		if j == self.Ly - 1 and i == self.Lx - 1:
			#Top corner case
			idxs = [idx,idx+1,3*self.Ly*i,3*self.Ly*i+1,3*(self.Ly-1)+2,3*(self.Ly-1)]
		elif j == self.Ly - 1:
			#Hexagon wraps over top of lattice
			idxs = [idx,idx+1,3*self.Ly*i,3*self.Ly*i+1,idx+3*self.Ly+1,idx+3*self.Ly-1]
		elif i == self.Lx - 1:
			#Hexagon wraps around side of lattice
			idxs = [idx,idx+1,idx+2,idx+3,3*j+2,3*j]
		else:
			idxs = [idx,idx+1,idx+2,idx+3,idx+3*self.Ly+1,idx+3*self.Ly-1]
		
		psi = np.zeros(self.N_sites)
		A = 1/np.sqrt(6)
		n=0
		for i in idxs:
			psi[int(i)] = A*(-1)**n
			n += 1
		
		return psi
		
		
	def psi_h(self,idx1,idx2):
		#Return two-particle real space flat band state formed from the direct product
		#of two single-particle hexagon states, at sites with the given indices
		#The direct product function takes account of normalization
		
		hex1 = self.hex(idx1)
		hex2 = self.hex(idx2)
		
		return direct_product(hex1,hex2)
		
		
	def flatband_projection_realspace(self,psi=None,t=None):
		#Returns the projection of the wavefunction onto the flat band, 
		#by expressing the wavefunction in a basis of only the flat band real space hexagon states
		#NB: this function is kept here for the sake of a complete account of the record 
		#of work; however, in fact, since the hexagons are not orthonormal, they do not 
		#form a suitable basis for the flat band. This function therefore should not be used in future
		
		if psi is not None:
			self.psi = psi
		elif t is not None:
			self.psi = self.state(t)
		
		psi_projected = np.zeros(self.N,dtype=np.complex128)
		
		for i in range(self.N_uc):
			for j in range(i,self.N_uc):
				psi_h = self.psi_h(idx1=i,idx2=j)
				c_h = np.vdot(psi_h,self.psi)
				psi_projected += c_h * psi_h
		
		return psi_projected
		
		
	def flatband_projection_realspace2(self,psi=None,t=None):
		#Returns the projection of the wavefunction onto the flat band, 
		#by expressing the wavefunction in a basis of only the flat band real space eigenstates,
		#obtained from diagonalizing a single particle system of the same size
		#These do form an orthonormal set, unlike the hexagons above
		system1 = Kag.Kagome(psi_0=np.zeros(self.N_sites),Lx=self.Lx,Ly=self.Ly,J=self.J,a=self.a,hbar=self.hbar,bc=self.bc)
		
		evects = system1.eigvects[:,-1:-(self.N_uc+1):-1]
		
		if psi is not None:
			self.psi = psi
		elif t is not None:
			self.psi = self.state(t)
		
		psi_projected = np.zeros(self.N,dtype=np.complex128)
		
		for i in range(self.N_uc):
			for j in range(i,self.N_uc):
				psi = direct_product(evects[:,i],evects[:,j])
				c = np.vdot(psi,self.psi)
				psi_projected += c * psi
		
		return psi_projected

	
	#Functions for plotting the system
	def plot_lattice(self,ax,color='blue',plot_sites=True,label_sites=False,fontsize=10,thickness=1,plot_sublattices=False,sublattice_colors={'A':'red','B':'green','C':'yellow'}):
		#Plot the lattice with lines between sites
		#Optionally, colour the sites according to sublattice (A, B or C), and label sites with their site index
		marker = 'o' if plot_sites else None
		dx = self.a*self.Lx/100
		dy = self.a*self.Ly/100
		for i,site in enumerate(self.sites):
			for nn in site.nn:
				if np.linalg.norm(site.r-nn.r) < self.a:		#Don't connect between 'near neighbours' on opposite sides of lattice in periodic case
					ax.plot(np.array([site.x,nn.x]),np.array([site.y,nn.y]),color=color,marker=marker,linewidth=thickness)
			if label_sites:
				if site.sublattice == 'A':
					ax.text(site.x-dx,site.y+dy,str(i),horizontalalignment='right',verticalalignment='bottom',fontsize=fontsize)
				elif site.sublattice == 'B':
					ax.text(site.x+dx,site.y+dy,str(i),horizontalalignment='left',verticalalignment='bottom',fontsize=fontsize)
				elif site.sublattice == 'C':
					ax.text(site.x-dx*np.sqrt(2),site.y,str(i),horizontalalignment='right',verticalalignment='center',fontsize=fontsize)
		if plot_sublattices:
			for site in self.sites:
				ax.plot(site.x,site.y,marker='o',color=sublattice_colors[site.sublattice])
			pointA = matplotlib.lines.Line2D([0],[0],label='A Site',color=sublattice_colors['A'],marker='o',ls='')
			pointB = matplotlib.lines.Line2D([0],[0],label='B Site',color=sublattice_colors['B'],marker='o',ls='')
			pointC = matplotlib.lines.Line2D([0],[0],label='C Site',color=sublattice_colors['C'],marker='o',ls='')
			ax.legend(handles=[pointA,pointB,pointC])
	
	
	def plot_BZ(self,plot_recip_vectors=True):
		#Plot the Brillouin zone and the discrete k values inside it for the system
		fig,ax = plt.subplots()
		
		ax.plot(self.k_BZ[:,0],self.k_BZ[:,1],color='b',marker='x',ls='',label='Inside')
	
		BZPatch = mplPatch.Polygon(self.BZ,ec='k',fc=(0,0,0,0))
		ax.add_patch(BZPatch)
		
		if plot_recip_vectors:
			ax.arrow(0,0,self.b1[0],self.b1[1],length_includes_head=True,width=0.1,color='r')
			ax.annotate(r'$b_1$',(self.b1[0]*0.95,self.b1[1]*0.85),fontsize=13)
			ax.arrow(0,0,self.b2[0],self.b2[1],length_includes_head=True,width=0.1,color='r')
			ax.annotate(r'$b_2$',(self.b2[0]+self.b2[1]*0.05,self.b2[1]*0.95),fontsize=13)
	
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		plt.show()
			

	def plot_density(self,fig=None,ax=None,plot=False,density=None,psi=None,t=None,cmap=plt.cm.Blues,uppernorm='auto',plot_lattice=True,lattice_thickness=1,plot_cbar=True,cbar_label_y=0.5):
		#Plot the density. If psi specified, density will be evaluated from psi; otherwise, if t specified, density will be evaluated at time t;
		#otherwise, density is evaluated from the current self.psi
		if density is None:
			if psi is not None:
				self.psi = psi
			elif t is not None:
				self.psi = self.state(t)
			density = self.density()
		
		if ax is None:
			fig,ax = plt.subplots()

		if uppernorm == 'auto':
			norm = matplotlib.colors.Normalize(vmin=0, vmax=uppernorm_func(np.max(density)))
		else: 
			norm = matplotlib.colors.Normalize(vmin=0, vmax=uppernorm)
		sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
		for i,site in enumerate(self.sites):
			tile = site_tile(site,a=self.a,color=cmap(norm(density[i])))
			ax.add_patch(tile)
		
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax)
			cbar.set_label(r'$\langle n \rangle$',rotation=0,y=cbar_label_y)
			cbar.ax.locator_params(nbins=5)
			
		if plot_lattice:
			self.plot_lattice(ax,color='k',plot_sites=False,thickness=lattice_thickness)
			
		ax.set_xticks([])
		ax.set_yticks([])
		
		if plot:
			plt.show()
			
			
	def plot_amplitude(self,fig=None,ax=None,plot=False,psi=None,t=None,cmap=plt.cm.bwr,uppernorm='auto',plot_lattice=True,lattice_thickness=1,plot_cbar=True,label_cbar=True,cbar_label_y=0.5):
		#Plot the amplitude. If psi specified, amplitude will be evaluated from psi; otherwise, if t specified, amplitude will be evaluated at time t;
		#otherwise, amplitude is evaluated from the current self.psi
		if psi is not None:
			self.psi = psi
		elif t is not None:
			self.psi = self.state(t)
		amp = self.amplitude()
		
		if ax is None:
			fig,ax = plt.subplots()
		
		if uppernorm == 'auto':
			val = uppernorm_func(np.max(np.abs(amp)))
			norm = matplotlib.colors.Normalize(vmin=-val, vmax=val)
		else:
			norm = matplotlib.colors.Normalize(vmin=-uppernorm, vmax=uppernorm)
		sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
		for i,site in enumerate(self.sites):
			tile = site_tile(site,a=self.a,color=cmap(norm(amp[i])))
			ax.add_patch(tile)
		
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax)
			if label_cbar:
				cbar.set_label(r'$Re(\Psi)$',rotation=0,y=cbar_label_y)
			cbar.ax.locator_params(nbins=5)
			
		if plot_lattice:
			self.plot_lattice(ax,color='k',plot_sites=False,thickness=lattice_thickness)
			
		ax.set_xticks([])
		ax.set_yticks([])
		
		if plot:
			plt.show()
	
	
	def plot_density_evolution(self,times,plot_lattice=True,lattice_thickness=1,uppernorm='auto',cbar_label_y=0.5):
		#Plot the density at 4 different times on the same figure
		if len(times)==4:
			fig,axs = plt.subplots(2,2)
			for i,t in enumerate(times):
				ax = axs[0 if i<2 else 1,i%2]
				self.plot_density(fig,ax,t=t,plot_lattice=plot_lattice,lattice_thickness=lattice_thickness,uppernorm=uppernorm,cbar_label_y=cbar_label_y)
				ax.set_title(f't={np.round(t,2)}')
				ax.set_xticks([])
				ax.set_yticks([])
			plt.show()
		else:
			print('Must be 4 specified times')
	
	
	def plot_amplitude_evolution(self,times,plot_lattice=True,lattice_thickness=1,uppernorm='auto',cbar_label_y=0.5):
		#Plot the amplitude at 4 different times on the same figure
		if len(times)==4:
			fig,axs = plt.subplots(2,2)
			for i,t in enumerate(times):
				ax = axs[0 if i<2 else 1,i%2]
				self.plot_wavefunction(t,fig,ax,plot_lattice=plot_lattice,lattice_thickness=lattice_thickness,uppernorm=uppernorm,cbar_label_y=cbar_label_y)
				ax.set_title(f't={np.round(t,2)}')
				ax.set_xticks([])
				ax.set_yticks([])
			plt.show()	
		else:
			print('Must be 4 specified times')
			
	
	def plot_opdm(self,fig=None,ax=None,plot=False,opdm=None,psi=None,t=None,cmap=plt.cm.bwr,uppernorm='auto',plot_cbar=True):
		#Plot the real part of the one-particle density matrix as an imshow() plot
		if opdm is not None:
			self.opdm = opdm
		else:
			self.opdm = self.calc_opdm(psi=psi,t=t)
		
		if uppernorm == 'auto':
			val = uppernorm_func(np.max(np.abs(self.opdm)))
		else: 
			val = uppernorm
		
		norm = matplotlib.colors.Normalize(vmin=-val, vmax=val)
		sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
		if ax is None:
			fig,ax = plt.subplots()
		
		ax.imshow(np.real(self.opdm),cmap=cmap,vmin=-val,vmax=val)
		
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax)
			#cbar.set_label(r'$\langle n \rangle$',rotation=0,y=cbar_label_y)
			cbar.ax.locator_params(nbins=5)
			
		#ax.set_xticks([])
		#ax.set_yticks([])
		#ax.set_title(r'$Re(\langle \Psi |b_i^{\dag} b_j|\Psi \rangle)$')
		ax.set_title(r'$Re(\rho_{ij})$')
		ax.set_xlabel('j')
		ax.set_ylabel('i',rotation=0)
		
		if plot:
			plt.show()
	
	
	#Miscellaneous functions	
	def save_psi(self,times,filename):
		#Calculate psi and density at all of the specified times, then save the resulting arrays (of shape len(times) x N_states 
		#and len(times) x N_sites respectively) in a .npz file
		#Also save the array of times, and all the system parameters
		psi_save = np.zeros((np.size(times),self.N),dtype='complex128')
		dens_save = np.zeros((np.size(times),self.N_sites),dtype='float64')
		
		if self.evolution_method == 'propagator':
			self.psi = self.state(t=times[0])
			psi_save[0,:] = self.psi
			dens_save[0,:] = self.density()
			for i in range(1,np.size(times)):
				dt = times[i] - times[i-1]
				self.psi = self.evolve_psi(dt)
				psi_save[i,:] = self.psi
				dens_save[i,:] = self.density()
				print(f'Progress: {np.round(100*i/np.size(times),2)}%',end='\r')
			print('Progress: 100.0%')
				
		elif self.evolution_method == 'eigenvector':
			for i,t in enumerate(times):
				self.psi = self.state(t=t)
				psi_save[i,:] = self.psi
				dens_save[i,:] = self.density()
				print(f'Progress: {np.round(100*i/np.size(times),2)}%',end='\r')
			print('Progress: 100.0%')
		
		np.savez(filename,t=times,psi=psi_save,density=dens_save,Lx=self.Lx,Ly=self.Ly,N_sites=self.N_sites,N_states=self.N,N_particles=2,U=self.U,J=self.J,bc=self.bc)


def site_tile(site,a,color):
	#Returns a mpl patch centred on the specified site, which is the region of the lattice closer to that site than any other
	#Used to plot density etc.
	if site.sublattice == 'A':
		ul = np.array([site.x-a/4,site.y+np.sqrt(3)*a/4])
		ur = np.array([site.x+a/4,site.y+a/(4*np.sqrt(3))])
		lr = np.array([site.x+a/4,site.y-np.sqrt(3)*a/4])
		ll = np.array([site.x-a/4,site.y-a/(4*np.sqrt(3))])	
	elif site.sublattice == 'B':
		ul = np.array([site.x-a/4,site.y+a/(4*np.sqrt(3))])	
		ur = np.array([site.x+a/4,site.y+np.sqrt(3)*a/4])
		lr = np.array([site.x+a/4,site.y-a/(4*np.sqrt(3))])	
		ll = np.array([site.x-a/4,site.y-np.sqrt(3)*a/4])
	elif site.sublattice == 'C':
		ul = np.array([site.x-a/2,site.y])	
		ur = np.array([site.x,site.y+a/(2*np.sqrt(3))])
		lr = np.array([site.x+a/2,site.y])	
		ll = np.array([site.x,site.y-a/(2*np.sqrt(3))])
	else:
		print('Error: unknown sublattice')
	tile = matplotlib.patches.Polygon(np.array([ul,ur,lr,ll]),facecolor=color)
	return tile

		
def uppernorm_func(max_val):
	#Given the maximum value (e.g. of density) over the whole lattice, return a well-chosen upper value for the colorbar normalization
	if max_val >= 1:
		vmax = np.ceil(np.round(max_val,1))
	elif max_val >= 0.1:
		vmax = np.ceil(10*max_val)/10
	elif max_val >= 0.01:
		vmax = np.ceil(100*max_val)/100
	else:
		vmax = 0.01
		print('Warning: low psi_max, colorbar scale might be awkward')
	return vmax


def system_from_file(filename,psi_0=None,initial_state_idx=None):
	#Build and return a 2-particle Kagome system from a file containing the Hamiltonian and its eigenvectors and eigenvalues
	#Optionally, psi_0 can be either specified explicitly, or set to only a single basis state being occupied
	data = np.load(filename)
	L = data['L']
	N_sites = int(3*L**2)
	N_states = int(0.5*N_sites*(N_sites+1))
	
	if psi_0 is None:
		psi_0 = np.zeros(N_states)
	if initial_state_idx is not None:
		psi_0[initial_state_idx] = 1
	
	system = Kagome2(psi_0=psi_0,Lx=L,Ly=L,J=data['J'],U=data['U'],skip_H=True,skip_diag=True)
	
	system.H = data['H']
	system.eigvects = data['eigvects']
	system.eigvals = data['eigvals']
	system.w = system.eigvals/system.hbar
	system.c_0 = system.psi_0 @ system.eigvects
	
	return system
	
	
def double_occupied_site_system(L=10,U=1.,a=1.,skip_diag=True,skip_H=False,evolution_method='propagator',bc='periodic',initial_site_idx=None,skip_k=False):
	#Builds and returns a 2-particle Kagome system with LxL unit cells
	#Optionally, if initial_site_idx specified, psi_0 will be the basis state with that site doubly-occupied
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	if initial_site_idx is not None:
		initial_state_idx = int(initial_site_idx*(N_sites - 0.5*initial_site_idx + 0.5))
		psi_0[initial_state_idx] = 1
	system = Kagome2(psi_0,Lx=L,Ly=L,U=U,a=a,skip_diag=skip_diag,skip_H=skip_H,evolution_method=evolution_method,bc=bc,skip_k=skip_k)
	return system
	
	
def single_state_system(L=10,U=1.,a=1.,skip_diag=True,skip_H=False,evolution_method='propagator',bc='periodic',initial_site_idx1=None,initial_site_idx2=None,skip_k=False):
	#Builds and returns a 2-particle Kagome system with LxL unit cells
	#Optionally, if initial_site_idx specified, psi_0 will be the basis state with that site doubly-occupied
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	if initial_site_idx1 is not None and initial_site_idx2 is not None:
		if initial_site_idx1 > initial_site_idx2:
			initial_site_idx1,initial_site_idx2 = initial_site_idx2,initial_site_idx1
		initial_state_idx = int(initial_site_idx1*(N_sites - 0.5*initial_site_idx1 - 0.5) + initial_site_idx2)
		psi_0[initial_state_idx] = 1
	system = Kagome2(psi_0,Lx=L,Ly=L,U=U,a=a,skip_diag=skip_diag,skip_H=skip_H,evolution_method=evolution_method,bc=bc,skip_k=skip_k)
	return system


def double_occupied_hexagon_5x5_system(U=1.):
	#Builds and returns a 2-particle Kagome system with 5x5 unit cells, and an initial state of the six doublon states
	#making up a hexagon, with alternating signs of coefficients round the hexagon
	#Demonstrates that 2-particle states cannot be inferred by double-occupying equivalent single-particle states, but instead 
	#are direct products of the single-particle states
	L = 5
	N_sites = 75
	N_states = 2850
	psi_0 = np.zeros(N_states)
	A = 1/np.sqrt(6)
	
	psi_0[1419] = A
	psi_0[1472] = -A
	psi_0[1524] = A
	psi_0[1575] = -A
	psi_0[2147] = A
	psi_0[2070] = -A
	
	system = Kagome2(psi_0=psi_0,Lx=L,Ly=L,U=U,skipH=False)
	
	return system
	
	
def direct_product(psi1,psi2):
	#Given two single-particle states (both of size N_sites), calculate and return the 2-particle state formed from the direct product of these states
	N_sites = np.size(psi1)
	if np.size(psi1) != np.size(psi2):
		print('Warning: psi1 and psi2 not of same size')
		
	N_states = int(0.5*N_sites*(N_sites+1))
	psi = np.zeros(N_states,dtype=np.complex128)
	
	for i in range(N_sites):
		if psi1[i] != 0:
			for j in range(N_sites):
				if psi2[j] != 0:	
					if i < j:
						state_idx = int(i*(N_sites - 0.5*i - 0.5) + j)
						psi[state_idx] += psi1[i]*psi2[j]
					elif i > j:
						state_idx = int(j*(N_sites - 0.5*j - 0.5) + i)
						psi[state_idx] += psi1[i]*psi2[j]
					else:		#i = j, need to normalise correctly
						state_idx = int(i*(N_sites - 0.5*i - 0.5) + j)
						psi[state_idx] += psi1[i]*psi2[j]*np.sqrt(2)
						
	psi = psi / np.linalg.norm(psi)
	
	return psi
	
	
def hexagon(idx,Ly):			
	#Return single-particle hexagon state on a lattice with Ly unit cells in each 'column'
	#idx is index of lower-left site of the hexagon; should be B site
	N_sites = 3*Ly**2
	A = 1/np.sqrt(6)
	
	idxs = [idx,idx+1,idx+2,idx+3,idx+3*Ly+1,idx+3*Ly-1]
	
	psi = np.zeros(N_sites)
	n=0
	for i in idxs:
		psi[i] = A*(-1)**n
		n += 1
	
	return psi

		
def two_hexagons_system(idx1,idx2,U=0.,L=5,skip_diag=True,evolution_method='propagator',bc='periodic',skip_k=False):
	#Build and return a system with psi_0 determined as the direct product of two single-particle hexagon states
	#idx1 and idx2 are lower-left site indices of the two hexagons; should both be B sites
	#Can have idx1=idx2 to have double-occupied hexagon
	hex1 = hexagon(idx=idx1,Ly=L)
	hex2 = hexagon(idx=idx2,Ly=L)
	
	psi_0 = direct_product(hex1,hex2)
	
	system = Kagome2(psi_0,Lx=L,Ly=L,U=U,skip_diag=skip_diag,evolution_method=evolution_method,bc=bc,skip_k=skip_k)

	return system
	
	
def singlesite_hexagon_system(idx1,idx2,U=0.,L=5,skip_diag=False,evolution_method='eigenvector'):
	#Build and return a system with psi_0 the direct product of one single-particle single-site state, and one single-particle hexagon
	#idx1 is the single site index, and idx2 the lower-left hexagon site index
	psi1 = np.zeros(3*L**2)
	psi1[idx1] = 1
	
	hex2 = hexagon(idx2,L)
	
	psi_0 = direct_product(psi1,hex2)
	
	system = Kagome2(psi_0,Lx=L,Ly=L,U=U,skip_diag=skip_diag,evolution_method=evolution_method)
	
	return system


if __name__ == '__main__':
	#system = singlesite_hexagon_system(idx1=34,idx2=37,U=1.,L=5)
	#system = double_occupied_site_system(initial_site_idx=165,U=1.,L=10,skip_k=False,skip_diag=True,evolution_method='eigenvector',bc='periodic')
	#system = double_occupied_site_system(initial_site_idx=133,U=100.,L=10,skip_k=True,skip_diag=True,evolution_method='propagator',bc='periodic')
	#system.save_psi(times=np.arange(0,100.1,0.5),filename='N2_L10x10_SingleSite_Periodic_T100_dt0.5_J1_U100.npz')
	#system = double_occupied_site_system(initial_site_idx=133,U=10.,L=10,skip_k=True,skip_diag=True,evolution_method='propagator',bc='periodic')
	#system.save_psi(times=np.arange(0,100.1,0.5),filename='N2_L10x10_SingleSite_Periodic_T100_dt0.5_J1_U10.npz')
	#system = double_occupied_site_system(initial_site_idx=133,U=5.,L=10,skip_k=True,skip_diag=True,evolution_method='propagator',bc='periodic')
	#system.save_psi(times=np.arange(0,100.1,0.5),filename='N2_L10x10_SingleSite_Periodic_T100_dt0.5_J1_U5.npz')
	#system = double_occupied_site_system(initial_site_idx=133,U=1.,L=10,skip_k=True,skip_diag=True,evolution_method='propagator',bc='periodic')
	#system.save_psi(times=np.arange(0,100.1,0.5),filename='N2_L10x10_SingleSite_Periodic_T100_dt0.5_J1_U1.npz')
	#system = double_occupied_site_system(initial_site_idx=133,U=0.1,L=10,skip_k=True,skip_diag=True,evolution_method='propagator',bc='periodic')
	#system = single_state_system(initial_site_idx1=103,initial_site_idx2=163,U=100,L=10,skip_k=True,skip_diag=True,evolution_method='propagator',bc='periodic')
	#system.save_psi(times=np.arange(0,100.1,0.1),filename='N2_L10x10_SeparateSites_Periodic_T100_dt0.1_J1_U100.npz')
	#system = single_state_system(initial_site_idx1=103,initial_site_idx2=163,U=0,L=10,skip_k=True,skip_diag=True,evolution_method='propagator',bc='periodic')
	#system.save_psi(times=np.arange(0,100.1,0.1),filename='N2_L10x10_SeparateSites_Periodic_T100_dt0.5_J1_U0.npz')
	
	
	system = two_hexagons_system(idx1=133,idx2=133,L=10,U=0.1,skip_diag=True,evolution_method='propagator',bc='periodic',skip_k=False)
	
	P = system.fb_projector
	print(sp.linalg.ishermitian(P))
	
	I = np.conjugate(P).T @ P
	
	fig,ax = plt.subplots()
	ax.imshow(np.real(I))
	plt.show()
	
	'''
	psi = np.zeros(75)
	psi[21] = 1
	
	c = psi @ system.fb_psi_k_sp
	
	print(np.sum(np.abs(c)**2))
	
	fig,ax = plt.subplots()
	ax.plot(np.arange(np.size(c)),np.abs(c))
	ax.set_xlabel('Idx')
	ax.set_ylabel('|c|')
	plt.show()
	'''
	
	
	
	'''
	fb_evals,fb_evects = sp.sparse.linalg.eigsh(system.H,k=300,return_eigenvectors=True,which='LA')
	high_evects = fb_evects[:,fb_evals>4.011]
	high_c = high_evects.T @ system.psi
	high_occupancy = np.sum(np.abs(high_c)**2)
	low_evects = fb_evects[:,fb_evals<=4.011]
	low_c = low_evects.T @ system.psi
	low_occupancy = np.sum(np.abs(low_c)**2)
	
	print(f'High occupancy: {high_occupancy}')
	print(f'Low occupancy: {low_occupancy}')
	print(f'Other occupancy: {1-high_occupancy-low_occupancy}')
	'''
	
	
	
	#system = single_state_system(initial_site_idx1=22,initial_site_idx2=51,U=10.,L=5,skip_k=True,skip_diag=False,evolution_method='eigenvector',bc='periodic')
	#system = single_state_system(initial_site_idx1=103,initial_site_idx2=163,U=1.,L=10,skip_k=False,skip_diag=True,evolution_method='eigenvector',bc='periodic')
	
	#system = two_hexagons_system(idx1=133,idx2=133,L=10,U=0.1,skip_diag=True,evolution_method='propagator',bc='periodic',skip_k=False)
	#system.plot_density_evolution(times=[0,100,200,500])
	#system.save_psi(times=np.arange(0,100,1),filename='test.npz')
	#system.rdf()
	#system.save_psi(times=np.arange(0,1001,20),filename='N2_L10x10_SameHexagon_Periodic_T1000_dt20_J1_U100.npz')
	
	#system.plot_BZ()
	#fig,ax = plt.subplots()
	#system.plot_density(fig=fig,ax=ax)
	#plt.show()
	#system.plot_density_evolution(times=[0,200,400,600])
	
	'''
	print(f'Expected Separation = {system.exp_separation(psi=np.ones(system.N)/np.sqrt(system.N))}')
	
	r,g,counts = system.rdf(psi=np.ones(system.N)/np.sqrt(system.N),return_counts=True)
	fig,ax=plt.subplots()
	ax.plot(r,g,color='b',marker='x',ls='')
	ax.set_xlabel(r'$r$ / $a$')
	ax.set_ylabel(r'$g(r)$',rotation=0)
	ax.set_title('Site Density Correlations, uniform psi')
	plt.show()
	
	fig,ax=plt.subplots()
	ax.plot(r,counts,color='b',marker='x',ls='')
	ax.set_xlabel(r'$r$ / $a$')
	ax.set_ylabel(r'$N(r)$',rotation=0)
	ax.set_title('Count of states with each value r')
	plt.show()
	
	fig,ax=plt.subplots()
	ax.plot(r,np.divide(g,counts),color='b',marker='x',ls='',label='Data')
	ax.set_xlabel(r'$r$ / $a$')
	ax.set_ylabel(r'$\frac{g(r)}{N(r)}$',rotation=0)
	ax.set_title('Normalized Site Density Correlations, uniform psi')
	ax.axhline(1/system.N,color='r',ls=':',label='No Correlations')
	ax.legend()
	plt.show()
	'''
	

	

	
	#OPDM = system.OPDM()
	#evals,evects = np.linalg.eigh(OPDM)
	#print(np.sum(evals))
	
	'''
	L = 5
	N_sites = 3*L**2
	N_states = int(0.5*N_sites*(N_sites+1))
	psi_0 = np.zeros(N_states)
	idx1 = int(33*(N_sites-0.5*33+0.5))
	idx2 = int(33*(N_sites-0.5*33-0.5)+36)
	psi_0[idx1] = 1/np.sqrt(2)
	psi_0[idx2] = -1/np.sqrt(2)
	#psi_0 /= np.linalg.norm(psi_0)
	system = Kagome2(psi_0,Lx=L,Ly=L,U=1.,a=1.,skip_diag=True,skip_H=False,evolution_method='eigenvector',bc='periodic',skip_k=False)
	
	fig,axs = plt.subplots(1,2)
	#system.plot_density(plot=True)
	system.plot_opdm(fig=fig,ax=axs[0],plot=False,plot_cbar=False)
	
	
	opdm = system.calc_opdm()
	proj_opdm = system.opdm_fb_projection(opdm=opdm)
	
	system.plot_opdm(fig=fig,ax=axs[1],opdm=proj_opdm,plot=False)
	
	axs[0].set_title('OPDM')
	axs[1].set_title('Projected OPDM')
	plt.show()
	
	
	
	
	
	fig,axs = plt.subplots(1,2)
	system.plot_density(fig,axs[0])
	axs[0].set_title('Density')
	system.plot_amplitude(fig,axs[1])
	axs[1].set_title('Amplitude')
	plt.suptitle('2 Particle')
	plt.show()
	#system.flatband_projection_kspace()
	#fig,ax = plt.subplots()
	#system.plot_density(fig,ax)
	#plt.show()
	#system.save_psi(times=np.arange(0,2000,10),filename='N2_L10x10_SameHexagon_Periodic_T2000_dt10_J1_U0.1.npz')
	
	system1 = Kag.Kagome(psi_0=system.OPDM(),Lx=10,Ly=10)
	fig,axs=plt.subplots(1,2)
	system1.plot_state_tiled(fig,axs[0])
	system1.plot_re_wavefunction_tiled(fig,axs[1])
	axs[0].set_title('Density')
	axs[1].set_title('Amplitude')
	plt.suptitle('OPDM')
	plt.show()
	'''
	'''
	system1 = Kag.Kagome(psi_0=np.zeros(75),Lx=5,Ly=5)
	psi = system.hex(idx=64)
	system1.psi = psi
	
	fig,ax = plt.subplots()
	system1.plot_re_wavefunction_tiled(fig,ax)
	plt.show()
	
	
	evects = system.eigvects[:,-1:-26:-1]
	
	N = int(0.5*25*26)
	psi_ij = np.zeros((2850,325))
	n = 0
	for i in range(25):
		psi_i = evects[:,i]
		for j in range(i,25):
			psi_j = evects[:,j]
			psi_ij[:,n] = direct_product(psi_i,psi_j)
			n += 1
	
	fig,axs = plt.subplots(1,2)		
	axs[0].imshow(psi_ij.T @ psi_ij)
	axs[0].set_title('Eigenvector Matrix Product')
	axs[1].set_title('Identity')
	axs[1].imshow(np.eye(N))
	plt.show()
	'''	
	
	
	#system.psi = system.psi_h(idx1=19,idx2=37)
	#fig,ax = plt.subplots()
	#system2.plot_re_wavefunction_tiled(fig,ax)
	#plt.show()
	#ax.plot(np.arange(system.N_sites),np.real(system.amplitude()),'b-')
	
	#system.plot_density(fig,ax)
	#plt.show()
	#system.plot_BZ()
	
	#system.plot_BZ()
	#psi_projected = system.flatband_projection()
	#print(np.sum(np.abs(psi_projected)**2))
	
	
	#system = double_occupied_site_system(L=10,U=1,skip_diag=True,evolution_method='eigenvector',bc='periodic',initial_site_idx=299)
	

	#system.save_psi(times=np.arange(0,2000,20),filename='N2_L5x5_SameHexagon_Periodic_T2000_dt20_J1_U0.1.npz')







