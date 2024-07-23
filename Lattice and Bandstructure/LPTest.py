import numpy as np
import matplotlib.pyplot as plt



class Lattice_Site:
	def __init__(self,r,sublattice=None):
		self.r = r
		self.x = self.r[0]
		self.y = self.r[1]
		self.nn = []
		self.sublattice = sublattice
	
	
	def add_nn(self,neighbour):
		'''
		if neighbour not in self.nn:
			self.nn.append(neighbour)
		if self not in neighbour.nn:
			neighbour.nn.append(self)
		'''
		self.nn.append(neighbour)
		
		
A = Lattice_Site(np.array([0,0]),sublattice='A')
B = Lattice_Site(np.array([1,0]),sublattice='B')
C = Lattice_Site(np.array([0.5,np.sqrt(3)/2]),sublattice='C')

sites = [A,B,C]

A.add_nn(B)



print(f'Sites: {sites}')
print(f'A near neighbours: {A.nn}')
print(f'B near neighbours: {B.nn}')
print(f'C near neighbours: {C.nn}')










