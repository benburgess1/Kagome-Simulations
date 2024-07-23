import numpy as np
import matplotlib.pyplot as plt


def sing_particle():
	build_data = np.load('BuildTime.npz')
	build_L = build_data['L']
	build_N = build_data['N']
	build_means = build_data['means']
	build_stdevs = build_data['stdevs']

	state_data = np.load('StateTime.npz')
	state_L = state_data['L']
	state_N = state_data['N']
	state_means = state_data['means']
	state_stdevs = state_data['stdevs']


	fig,ax = plt.subplots()
	ax.set_title('Function execution time vs system size',fontsize=10)
	ax.set_xlabel(r'$N$')
	ax.set_ylabel('Time / s')
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	
	ax.errorbar(build_N,build_means,build_stdevs,marker='x',color='b',label='Build Lattice')
	ax.errorbar(state_N,state_means,state_stdevs,marker='x',color='r',label='Calculate State')
	
	
	x = np.linspace(state_N[0],state_N[-1],100)
	quad_fit = 1e-8*x**2
	cub_fit = 0.3e-9*x**3
	ax.plot(x,quad_fit,'k--',label=r'$\mathcal{O}(N^2)$')
	ax.plot(x,cub_fit,'k:',label=r'$\mathcal{O}(N^3)$')
	
	ax.legend()
	plt.show()
	
sing_particle()
