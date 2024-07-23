import numpy as np
import matplotlib.pyplot as plt

L = np.arange(1,21,1)
N_states = 4.5*L**4

#Coefficients obtained from L=5 data, and known scalings/complexities
t_diag = 8.77e-9 * L**12
t_state = 1.52e-7 * L**8

f = lambda x: 4.5*x**4
g = lambda x: (x/4.5)**0.25

fig,ax1 = plt.subplots()
ax2 = ax1.secondary_xaxis('top',functions=(f,g))

ax1.plot(L,t_diag,'k-',label=r'$t_{diag}$')
ax1.plot(L,t_state,'b-',label=r'$t_{state}$')
ax1.plot(L,100*t_state,'r-',label=r'$100 t_{state}$')
ax1.plot(L,1000*t_state,'g-',label=r'$1000 t_{state}$')
#ax2.plot(N_states,t_diag,'b-')

#ax1.set_xlim(0,20)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('L')
ax1.set_ylabel('Time / s')
ax2.set_xlabel(r'$N_{states}$')

ax1.axhline(60,color='k',ls=':')
ax1.axhline(3600,color='k',ls=':')
ax1.axhline(86400,color='k',ls=':')

A = 1.5
ax1.text(1,60*A,'1 Minute')
ax1.text(1,3600*A,'1 Hour')
ax1.text(1,86400*A,'1 Day')

ax1.legend()

plt.show()
