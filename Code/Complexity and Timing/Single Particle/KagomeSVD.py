import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import cProfile


class Lattice_Site:
	def __init__(self,r,sublattice=None):
		self.r = r
		self.x = self.r[0]
		self.y = self.r[1]
		self.nn = []
		self.sublattice = sublattice
	
	
	def add_nn(self,neighbour):
		if neighbour not in self.nn:
			self.nn.append(neighbour)
		if self not in neighbour.nn:
			neighbour.nn.append(self)
		
		#self.nn.append(neighbour)
		

		

#class Unit_Cell: Could implement this later, can't really see the advantage at the moment
	
		
		
class Kagome:
	def __init__(self,psi_0,Lx,Ly,a=1.,J=1.,U=1.,hbar=1.):
		self.psi_0 = psi_0
		self.psi = np.copy(psi_0)
		self.Lx = Lx
		self.Ly = Ly
		self.a = a
		self.J = J
		self.U = U
		self.hbar = hbar
		self.N = 3*self.Lx*self.Ly
		if np.size(self.psi_0) != self.N:
			print('Warning: psi_0 not of correct dimensions for lattice')
		self.H = np.zeros((self.N,self.N))
		
		#Lattice Vectors
		self.a1 = self.a*np.array([1.,0.])
		self.a2 = self.a*np.array([0.5,np.sqrt(3)/2])
		
		#Build lattice
		self.sites = []
		for i in range(self.Lx):
			for j in range(self.Ly):
				#num_previous = 3*(Ly*i+j)				#Number of sites already added
				r_uc = i*self.a1 + j*self.a2		#Unit cell location
				
				A = Lattice_Site(r_uc,sublattice='A')
				B = Lattice_Site(r_uc + 0.5*self.a1,sublattice='B')
				C = Lattice_Site(r_uc + 0.5*self.a2,sublattice='C')
				
				A.add_nn(B)
				A.add_nn(C)
				B.add_nn(C)
				
				if j > 0:			
					A.add_nn(self.sites[3*(Ly*i+j) - 1])						#Connection between current A site and previous C site
					
				if i > 0:
					A.add_nn(self.sites[3*(Ly*(i-1)+j) + 1])			#Connection between current A site and B site on previous 'column'
					if j < Ly - 1:
						C.add_nn(self.sites[3*(Ly*(i-1)+j) + 4])			#Between current C site and B site on previous column and up one row
					
				self.add_site(A)
				self.add_site(B)
				self.add_site(C)
				
		self.calc_H()
		
		#Calculate eigenvalues and eigenvectors
		self.eigvects,s,vh = np.linalg.svd(self.H,hermitian=True)
		self.eigvals = s
		self.w = self.eigvals/self.hbar
		self.c_0 = self.psi_0 @ self.eigvects
				
				
		
		
	def add_site(self,site):
		self.sites.append(site)
		
		
	def plot_lattice(self,ax,color='blue',plot_sites=True,label_sites=False,fontsize=10):
		marker = 'o' if plot_sites else None
		dx = self.a*self.Lx/100
		dy = self.a*self.Ly/100
		for i,site in enumerate(self.sites):
			for nn in site.nn:
				ax.plot(np.array([site.x,nn.x]),np.array([site.y,nn.y]),color=color,marker=marker)
			if label_sites:
				if site.sublattice == 'A':
					ax.text(site.x-dx,site.y+dy,str(i),horizontalalignment='right',verticalalignment='bottom',fontsize=fontsize)
				elif site.sublattice == 'B':
					ax.text(site.x+dx,site.y+dy,str(i),horizontalalignment='left',verticalalignment='bottom',fontsize=fontsize)
				elif site.sublattice == 'C':
					ax.text(site.x-dx*np.sqrt(2),site.y,str(i),horizontalalignment='right',verticalalignment='center',fontsize=fontsize)
				
	def plot_lattice_nn(self,ax,central_site,color='blue',site_color='yellow',neighbour_color='red',plot_sites=False):
		marker = 'o' if plot_sites else None
		for site in self.sites:
			for nn in site.nn:
				ax.plot(np.array([site.x,nn.x]),np.array([site.y,nn.y]),color=color,marker=marker)
		ax.plot(central_site.x,central_site.y,color=site_color,marker='o')
		for nn in central_site.nn:
			ax.plot(nn.x,nn.y,color=neighbour_color,marker='o')
			
			
	def plot_sublattices(self,ax,lattice_color='blue',sublattice_colors={'A':'red','B':'green','C':'yellow'}):
		self.plot_lattice(ax,color=lattice_color,plot_sites='False')
		for site in self.sites:
			ax.plot(site.x,site.y,marker='o',color=sublattice_colors[site.sublattice])
		
	
	def calc_H(self):			#Single particle case
		#H[i,j] annhilates particle in state j and creates in i.
		for i in range(self.Lx):
			for j in range(self.Ly):
				#Hopping within unit cell:
				self.add_uc_hopping(i,j)
				
				#Hopping to neighbouring cells:
				if i > 0:
					self.add_hopping(i,j,'A',i-1,j,'B')
				if j > 0:
					self.add_hopping(i,j,'A',i,j-1,'C')
				if i < self.Lx-1:
					self.add_hopping(i,j,'B',i+1,j,'A')
				if j > 0 and i < self.Lx-1:
					self.add_hopping(i,j,'B',i+1,j-1,'C')
				if j < self.Ly-1:
					self.add_hopping(i,j,'C',i,j+1,'A')
				if j < self.Ly-1 and i > 0:
					self.add_hopping(i,j,'C',i-1,j+1,'B')

	
	def add_uc_hopping(self,i,j):
		self.add_hopping(i,j,'A',i,j,'B')
		self.add_hopping(i,j,'A',i,j,'C')
		self.add_hopping(i,j,'B',i,j,'A')
		self.add_hopping(i,j,'B',i,j,'C')
		self.add_hopping(i,j,'C',i,j,'A')
		self.add_hopping(i,j,'C',i,j,'B')
		
			
	def add_hopping(self,i0,j0,L0,i1,j1,L1):
		#Adds hopping element from state lattice site (i0,j0,L0) to (i1,j1,L1)
		#Index of site (A site) at bottom left of unit cell (i,j) is 3(Ly*i + j); +1 for B site, +2 for C site
		if L0 == 'A':
			L0 = 0
		elif L0 == 'B':
			L0 = 1
		elif L0 == 'C':
			L0 = 2
		else:
			print('L0 error')
		if L1 == 'A':
			L1 = 0
		elif L1 == 'B':
			L1 = 1
		elif L1 == 'C':
			L1 = 2
		else:
			print('L1 error')
		self.H[3*(self.Ly*i1+j1)+L1,3*(self.Ly*i0+j0)+L0] = -self.J
	
	
	def evolve(self,t):
		exp = np.cos(self.w*t) - 1j*np.sin(self.w*t)
		c = self.psi @ self.eigvects
		self.psi = np.transpose(self.eigvects @ np.transpose(np.multiply(c,exp)))
		
	def state(self,t):
		exp = np.cos(self.w*t) - 1j*np.sin(self.w*t)
		self.psi = np.transpose(self.eigvects @ np.transpose(np.multiply(self.c_0,exp)))
		return self.psi
		
	def abs_psi2(self):
		return np.real(self.psi)**2 + np.imag(self.psi)**2
	
	
	def plot_state(self,t,ax,cmap=plt.cm.rainbow,norm='auto',plot_cbar=True,markersize=5):
		#Plot lattice as a 'heatmap', with colour of lattice site indicating value of wavefunction there
		self.psi = self.state(t)
		if norm == 'auto':
			norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(self.abs_psi2()))
		elif norm == '0to1':
			norm = matplotlib.colors.Normalize(vmin=0, vmax=1.)
		sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
		self.plot_lattice(ax,color='k',plot_sites=False)
		for i,site in enumerate(self.sites):
			ax.plot(site.x,site.y,marker='o',color=cmap(norm(self.abs_psi2()[i])),markersize=markersize)
		
		if plot_cbar:
			cbar=plt.colorbar(mappable=sm,ax=ax)
			cbar.set_label(r'$|\Psi^2|$',rotation=0)
			
			
	def plot_current_state(self,ax,cmap=plt.cm.rainbow,norm='auto',plot_cbar=True,markersize=5):
		#Plot lattice as a 'heatmap', with colour of lattice site indicating value of wavefunction there
		#self.psi = self.state(t)
		if norm == 'auto':
			norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(self.abs_psi2()))
		elif norm == '0to1':
			norm = matplotlib.colors.Normalize(vmin=0, vmax=1.)
		sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
		self.plot_lattice(ax,color='k',plot_sites=False)
		for i,site in enumerate(self.sites):
			ax.plot(site.x,site.y,marker='o',color=cmap(norm(self.abs_psi2()[i])),markersize=markersize)
		
		if plot_cbar:
			cbar=plt.colorbar(mappable=sm,ax=ax)
			cbar.set_label(r'$|\Psi^2|$',rotation=0)
	
	
	def plot_evolution(self,times,markersize=5):
		fig,axs = plt.subplots(2,2)
		for i,t in enumerate(times):
			ax = axs[0 if i<2 else 1,i%2]
			self.plot_state(t,ax,markersize=markersize)
			ax.set_title(f't={np.round(t,2)}')
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_aspect('equal')
			
		plt.tight_layout()
		plt.show()
		
		
	def plot_stat_psi(self,times):
		vals = np.zeros(np.size(times),dtype='complex128')
		for i,t in enumerate(times):
			self.psi = self.state(t)
			vals[i] = np.vdot(self.psi_0,self.psi)		#Overlap with initial wavefunction
		plt.plot(times,np.real(vals)**2 + np.imag(vals)**2,'b-')
		plt.title('Density in original site')
		plt.ylabel(r'$|\Psi|^2$')
		plt.xlabel('t')
		plt.show()
	
	
	def animate(self,i,ax,dt=1.):
		self.evolve(dt)
		im = self.plot_current_state(ax,norm=matplotlib.colors.Normalize(vmin=0, vmax=0.1),plot_cbar=False)
		#im = self.plot_state(i*dt,ax,norm=matplotlib.colors.Normalize(vmin=0, vmax=0.1),plot_cbar=False)
		ax.set_title(f't={np.round(i*dt,2)}')
		return im,
	
	

	def animate_evolution(self,T):
		fig,ax = plt.subplots()

		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		
		self.psi = self.state(0)
		
		im = self.plot_state(0,ax,norm=matplotlib.colors.Normalize(vmin=0, vmax=0.1))
		
	
		ani = animation.FuncAnimation(fig, self.animate, frames=T, fargs=(ax,), interval=100, blit=False)
	
		plt.show()
		
		
	def save_psi(self,times,filename):
		psi_save = np.zeros((np.size(times),self.N),dtype='complex128')
		for i,t in enumerate(times):
			psi_save[i,:] = self.state(t)
			print(f'Progress: {np.round(100*i/np.size(times),2)}%',end='\r')
		print('\nDone')
		np.savez(filename,t=times,psi=psi_save)
		
		
if __name__ == '__main__':
	Lx=5
	Ly=5
	psi_0 = np.zeros(3*Lx*Ly)
	psi_0[24] = 1
	system = Kagome(psi_0=psi_0,Lx=Lx,Ly=Ly)
	times = np.arange(0,4,1)
	
	print(system.H @ system.eigvects[:,10])
	print(system.eigvals[10] * system.eigvects[:,10])

	system.plot_evolution(times)
	#system.save_psi(times,'Test4')

'''
#Stationary state:
psi_0[24] = 1/np.sqrt(6)
psi_0[25] = -1/np.sqrt(6)
psi_0[38] = 1/np.sqrt(6)
psi_0[36] = -1/np.sqrt(6)
psi_0[22] = 1/np.sqrt(6)
psi_0[23] = -1/np.sqrt(6)
'''


	


#system.animate_evolution(100)

#t = np.linspace(0,100,1000)
#system.plot_stat_psi(t)

'''
fig,ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
system.plot_state(0,ax,markersize=5,norm=matplotlib.colors.Normalize(vmin=0, vmax=0.1))
ax.set_aspect('equal')
ax.set_title('t=0')
plt.show()

fig,ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
system.plot_state(1000,ax,markersize=5,norm=matplotlib.colors.Normalize(vmin=0, vmax=0.1))
ax.set_aspect('equal')
ax.set_title('t=1000')
plt.show()
'''

'''
fig,ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
system.plot_lattice(ax,label_sites=True)
ax.set_aspect('equal')
plt.show()
'''


#times = np.linspace(0,30,4)
#system.plot_evolution(times)


'''
fig,ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
system.plot_lattice(ax,label_sites=True)
ax.set_aspect('equal')
plt.show()

fig,ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
system.plot_state(0,ax)
ax.set_aspect('equal')
plt.show()

fig,ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
system.plot_state(1,ax)
ax.set_aspect('equal')
plt.show()
'''



