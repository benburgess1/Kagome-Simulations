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
		self.r_lv = np.zeros(2)
	
	
	def add_nn(self,neighbour):
		if neighbour not in self.nn:
			self.nn.append(neighbour)
		if self not in neighbour.nn:
			neighbour.nn.append(self)
		
		#self.nn.append(neighbour)
		
		
	def set_r_lv(self,r):		#Position expressed in basis vectors of the lattice
		self.r_lv[:] = r		
		

		

#class Unit_Cell: Could implement this later, can't really see the advantage at the moment
	
		
		
class Kagome:
	def __init__(self,psi_0,Lx,Ly,a=1.,J=1.,U=1.,hbar=1.,bc='open',skip_diag=False):
		self.psi_0 = psi_0
		self.psi = np.copy(psi_0)
		self.Lx = Lx
		self.Ly = Ly
		self.a = a
		self.J = J
		self.U = U
		self.hbar = hbar
		self.bc = bc
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
				
				A.set_r_lv([i,j])
				B.set_r_lv([i+0.5,j])
				C.set_r_lv([i+0.25,j+np.sqrt(3)/4])
				
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
				
		self.calc_H()
		
		#Calculate eigenvalues and eigenvectors
		if not skip_diag:
			self.eigvals,self.eigvects = np.linalg.eigh(self.H)
			self.w = self.eigvals/self.hbar
			self.c_0 = self.psi_0 @ self.eigvects
			
		else:
			print('Warning: ensure eigenvalues/vectors input explicitly if required')
				
				
		
		
	def add_site(self,site):
		self.sites.append(site)
		
		
	def plot_lattice(self,ax,color='blue',plot_sites=True,label_sites=False,fontsize=10,thickness=1,markersize=5,plot_sublattices=False,sublattice_colors={'A':'red','B':'green','C':'yellow'},plot_lattice_vectors=False):
		marker = 'o' if plot_sites else None
		dx = self.a*self.Lx/100
		dy = self.a*self.Ly/100
		for i,site in enumerate(self.sites):
			for nn in site.nn:
				if np.linalg.norm(site.r-nn.r) < self.a:
					ax.plot(np.array([site.x,nn.x]),np.array([site.y,nn.y]),color=color,marker=marker,markersize=markersize,linewidth=thickness)
			if label_sites:
				if site.sublattice == 'A':
					ax.text(site.x-dx,site.y+dy,str(i),horizontalalignment='right',verticalalignment='bottom',fontsize=fontsize)
				elif site.sublattice == 'B':
					ax.text(site.x+dx,site.y+dy,str(i),horizontalalignment='left',verticalalignment='bottom',fontsize=fontsize)
				elif site.sublattice == 'C':
					ax.text(site.x-dx*2,site.y,str(i),horizontalalignment='right',verticalalignment='center',fontsize=fontsize)
		if plot_sublattices:
			for site in self.sites:
				ax.plot(site.x,site.y,marker='o',color=sublattice_colors[site.sublattice])
			pointA = matplotlib.lines.Line2D([0],[0],label='A Site',color=sublattice_colors['A'],marker='o',ls='')
			pointB = matplotlib.lines.Line2D([0],[0],label='B Site',color=sublattice_colors['B'],marker='o',ls='')
			pointC = matplotlib.lines.Line2D([0],[0],label='C Site',color=sublattice_colors['C'],marker='o',ls='')
			ax.legend(handles=[pointA,pointB,pointC])
			
		if plot_lattice_vectors:
			ax.arrow(0,0,self.a1[0],self.a1[1],length_includes_head=True,facecolor='r',edgecolor='r',width=0.02,head_length=self.a/10)
			ax.arrow(0,0,self.a2[0],self.a2[1],length_includes_head=True,color='r',width=0.02,head_length=self.a/10)
	
		
			
				
	def plot_lattice_nn(self,ax,central_site,color='blue',site_color='yellow',neighbour_color='red',plot_sites=False):
		marker = 'o' if plot_sites else None
		for site in self.sites:
			for nn in site.nn:
				if np.linalg.norm(site.r-nn.r) < self.a:
					ax.plot(np.array([site.x,nn.x]),np.array([site.y,nn.y]),color=color,marker=marker)
		ax.plot(central_site.x,central_site.y,color=site_color,marker='o')
		for nn in central_site.nn:
			ax.plot(nn.x,nn.y,color=neighbour_color,marker='o')
			
			
	def plot_sublattices(self,ax,lattice_color='blue',sublattice_colors={'A':'red','B':'green','C':'yellow'}):
		self.plot_lattice(ax,color=lattice_color,plot_sites='False')
		for site in self.sites:
			ax.plot(site.x,site.y,marker='o',color=sublattice_colors[site.sublattice])
		
	
	def calc_H(self):			#(Single particle case)
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
					
				if self.bc == 'periodic':
					if i == 0:
						self.add_hopping(i,j,'A',self.Lx-1,j,'B')
						if j < self.Ly-1:
							self.add_hopping(i,j,'C',self.Lx-1,j+1,'B')
						else:
							self.add_hopping(i,j,'C',self.Lx-1,0,'B')
						
					if i == self.Lx - 1:
						self.add_hopping(i,j,'B',0,j,'A')
						if j > 0:
							self.add_hopping(i,j,'B',0,j-1,'C')
						else:
							self.add_hopping(i,j,'B',0,self.Ly-1,'C')
							
					if j == 0:
						self.add_hopping(i,j,'A',i,self.Ly-1,'C')
						if i < self.Lx-1:		
							self.add_hopping(i,j,'B',i+1,self.Ly-1,'C')
					
					if j == self.Ly - 1:
						self.add_hopping(i,j,'C',i,0,'A')
						if i > 0:
							self.add_hopping(i,j,'C',i-1,0,'B')

	
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
		
	def density(self,t):		#Same as abs_psi2, but just with t argument
		self.psi = self.state(t)
		return np.real(self.psi)**2 + np.imag(self.psi)**2
		
	def abs_psi2(self):
		return np.real(self.psi)**2 + np.imag(self.psi)**2
	
	
	def plot_state(self,fig,ax,t=None,cmap=plt.cm.rainbow,norm='auto',plot_cbar=True,markersize=5):
		#Plot lattice as a 'heatmap', with colour of lattice site indicating value of wavefunction there
		if t is not None:
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
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax)
			cbar.set_label(r'$|\Psi^2|$',rotation=0)
	
	
	def plot_state_tiled(self,fig,ax,t=None,cmap=plt.cm.Blues,norm='auto',plot_lattice=True,lattice_thickness=1,plot_cbar=True,cbar_label_y=0.5,cbar_labelpad=0):
		if t is not None:
			self.psi = self.state(t)

		if norm == 'auto':
			norm = matplotlib.colors.Normalize(vmin=0, vmax=uppernorm(np.max(self.abs_psi2())))
		elif norm == '0to1':
			norm = matplotlib.colors.Normalize(vmin=0, vmax=1.)
		sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
		for i,site in enumerate(self.sites):
			tile = site_tile(site,a=self.a,color=cmap(norm(self.abs_psi2()[i])))
			ax.add_patch(tile)
		
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar = plt.colorbar(mappable=sm,cax=cax)
			cbar.set_label(r'$\langle n \rangle$',rotation=0,y=cbar_label_y,labelpad=cbar_labelpad)
			cbar.ax.locator_params(nbins=5)
			
		if plot_lattice:
			self.plot_lattice(ax,color='k',plot_sites=False,thickness=lattice_thickness)
			
		ax.set_xticks([])
		ax.set_yticks([])
			
			
	def plot_re_wavefunction_tiled(self,fig,ax,t=None,cmap=plt.cm.bwr,norm='auto',plot_lattice=True,lattice_thickness=1,plot_cbar=True,cbar_label_y=0.5):
		if t is not None:
			self.psi = self.state(t)
		
		cbar_ticks = None
		auto_ticks = True
		if norm == 'auto':
			val = uppernorm(np.sqrt(np.max(self.abs_psi2())))
			norm = matplotlib.colors.Normalize(vmin=-val, vmax=val)
			if val == 0.15:		#Only case where automatic ticks are awkward with nbins=5
				cbar_ticks = np.arange(-0.15,0.16,0.05)
				auto_ticks = False
		elif norm == '1to1':
			norm = matplotlib.colors.Normalize(vmin=-1., vmax=1.)
		sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
		for i,site in enumerate(self.sites):
			tile = site_tile(site,a=self.a,color=cmap(norm(np.real(self.psi[i]))))
			ax.add_patch(tile)
		
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax,ticks=cbar_ticks)
			#cbar.set_label(r'$Re(\Psi)$',rotation=0,y=cbar_label_y)
			if auto_ticks:
				cbar.ax.locator_params(nbins=5)
			
		if plot_lattice:
			self.plot_lattice(ax,color='k',plot_sites=False,thickness=lattice_thickness)
			
		ax.set_xticks([])
		ax.set_yticks([])
	
		
	def plot_im_wavefunction_tiled(self,fig,ax,t=None,cmap=plt.cm.bwr,norm='auto',plot_lattice=True,lattice_thickness=1,plot_cbar=True,cbar_label_y=1.1):
		if t is not None:
			self.psi = self.state(t)
		
		if norm == 'auto':
			val = uppernorm(np.sqrt(np.max(self.abs_psi2())))
			norm = matplotlib.colors.Normalize(vmin=-val, vmax=val)
		elif norm == '1to1':
			norm = matplotlib.colors.Normalize(vmin=-1., vmax=1.)
		sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
		
		for i,site in enumerate(self.sites):
			tile = site_tile(site,a=self.a,color=cmap(norm(np.imag(self.psi[i]))))
			ax.add_patch(tile)
		
		if plot_cbar:
			cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
			cbar=plt.colorbar(mappable=sm,cax=cax)
			#cbar.set_label(r'$Im(\Psi)$',rotation=0,y=cbar_label_y)
			cbar.ax.locator_params(nbins=5)
			
		if plot_lattice:
			self.plot_lattice(ax,color='k',plot_sites=False,thickness=lattice_thickness)
			
		ax.set_xticks([])
		ax.set_yticks([])
			
	
	
	def plot_evolution(self,times,markersize=5):
		fig,axs = plt.subplots(2,2)
		for i,t in enumerate(times):
			ax = axs[0 if i<2 else 1,i%2]
			self.plot_state(t,fig,ax,markersize=markersize)
			ax.set_title(f't={np.round(t,2)}')
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_aspect('equal')
			
		plt.tight_layout()
		plt.show()
	
	
	def plot_evolution_tiled(self,times,plot_lattice=True,lattice_thickness=1,norm='auto',cbar_label_y=1.1):
		fig,axs = plt.subplots(2,2)
		for i,t in enumerate(times):
			ax = axs[0 if i<2 else 1,i%2]
			self.plot_state_tiled(fig,ax,t=t,plot_lattice=plot_lattice,lattice_thickness=lattice_thickness,norm=norm,cbar_label_y=cbar_label_y)
			ax.set_title(f't={np.round(t,2)}')
			ax.set_xticks([])
			ax.set_yticks([])
			#ax.set_aspect('equal')
			
		#plt.tight_layout()
		plt.show()
		
	def plot_wf_evolution_tiled(self,times,plot_lattice=True,lattice_thickness=1,norm='auto',cbar_label_y=1.1):
		fig,axs = plt.subplots(2,2)
		for i,t in enumerate(times):
			ax = axs[0 if i<2 else 1,i%2]
			self.plot_re_wavefunction_tiled(fig,ax,t=t,plot_lattice=plot_lattice,lattice_thickness=lattice_thickness,norm=norm,cbar_label_y=cbar_label_y)
			ax.set_title(f't={np.round(t,2)}')
			ax.set_xticks([])
			ax.set_yticks([])
			#ax.set_aspect('equal')
			
		#plt.tight_layout()
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
		np.savez(filename,t=times,psi=psi_save,density=np.abs(psi_save)**2,Lx=self.Lx,Ly=self.Ly,N=self.N,N_particles=1,J=self.J)


def site_tile(site,a,color):
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

		
def uppernorm(max_val):
	if max_val >= 0.2:
		vmax = np.ceil(10*max_val)/10
	if max_val >= 0.1:
		vmax = np.ceil(20*max_val)/20
	elif max_val >= 0.01:
		vmax = np.ceil(100*max_val)/100
	else:
		vmax = 0.01
		print('Warning: low psi_max, colorbar scale might be awkward')
	return vmax


		
if __name__ == '__main__':
	L = 20
	psi_0 = np.zeros(3*L**2)
	psi_0[690] = 1
	system = Kagome(psi_0,Lx=L,Ly=L,J=1.,bc='open')
	
	system.save_psi(times=np.arange(0,10000.1,1),filename='N1_L20x20_SingleSite_Open_T10000_dt1.npz')
	
	'''
	low_c = system.psi @ system.eigvects[:,system.eigvals<-1]
	#high_c = system.psi @ system.eigvects[:,system.eigvals>=-1 and system.eigvals < 2]
	flat_c = system.psi @ system.eigvects[:,system.eigvals>=1.99999]
	
	print(np.sum(np.abs(low_c)**2))
	print(1- np.sum(np.abs(low_c)**2)-np.sum(np.abs(flat_c)**2))
	print(np.sum(np.abs(flat_c)**2))
	#system.save_psi(times=np.arange(0,5001,5),filename='N1_L10x10_Periodic_T5000_dt5_J-0.02.npz')
	'''
	'''
	Lx=10
	Ly=10
	psi_1 = np.zeros(3*Lx*Ly)
	psi_2 = np.zeros(3*Lx*Ly)
	A = 1/np.sqrt(6)
	
	psi_1[133] = A
	psi_1[134] = -A
	psi_1[135] = A
	psi_1[136] = -A
	psi_1[164] = A
	psi_1[162] = -A
	
	psi_2[160] = A
	psi_2[161] = -A
	psi_2[162] = A
	psi_2[163] = -A
	psi_2[191] = A
	psi_2[189] = -A
	
	psi_0 = psi_1 - psi_2 
	psi_0 /= np.linalg.norm(psi_0)
	
	system = Kagome(psi_0=psi_0,Lx=Lx,Ly=Ly,J=1.,bc='periodic')
	
	system.plot_evolution_tiled(times=[0,10,20,30])
	'''
	#system.save_psi(times=np.arange(0,2001,10),filename='N1_L10x10_SingleSite_Open_T2000_dt10.npz')
	
	#fig,ax = plt.subplots()
	#ax.plot(np.arange(system.N),system.eigvals,'bx')
	#plt.show()
	
	#times = np.arange(0,5001,5)
	#system.save_psi(times,'N1_L5x5_T5000_dt5_J-0.02.npz')

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



