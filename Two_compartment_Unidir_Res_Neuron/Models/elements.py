# -*- coding: utf-8 -*-

#############################Basic Elements Class##############################
import numpy as np

class resistor_old:#Use for multicompartments
    def __init__(self, R_value=1, n_res=1, V_L=0, V_R=0):
        self.value=R_value
        self.n_res=n_res
        self.V_L=V_L #initial values
        self.V_R=V_R #initial values
        self.I_res=np.zeros((n_res, 1))
    
    def __call__(self, V_LR):
        self.V_L=V_LR[:,0]
        self.V_R=V_LR[:,1]
        I=(self.V_L-self.V_R)/self.value
        self.I_res=I
        return I

class resistor:
    def __init__(self, R_value=1, n_res=1, n_branch=1):
        self.value=R_value
        self.n_res=n_res
        self.n_branch=n_branch
        #self.V_diff=V_diff #initial values
        self.I_res=np.zeros((self.n_res, self.n_branch))
    
    def __call__(self, V_diff):
        I=(V_diff)/self.value
        self.I_res=I
        return I

class resistor_unidir:
    def __init__(self, R_value=1, n_res=1, n_branch=1):
        self.value=R_value
        self.n_res=n_res
        self.n_branch=n_branch
        #self.V_diff=V_diff #initial values
        self.I_res=np.zeros((self.n_res, self.n_branch))
    
    def __call__(self, V_diff):
        I=np.clip((V_diff)/self.value,0,10000)
        self.I_res=I
        return I
    
class capacitor_old:
    def __init__(self, C_value=.1, n_cap=1, V_init=0, n_branch=1, dt=1e-4):
        self.value=C_value
        self.n_cap=n_cap
        self.n_branch=n_branch
        self.dt=dt
        self.V_init=V_init
        self.V_out=self.V_init*np.ones((self.n_cap, self.n_branch))
    
    def __call__(self, I_in):
        V_out=self.V_init+(1/self.value)*I_in*self.dt
        self.V_init=V_out
        return V_out

class capacitor:
    def __init__(self, C_value=.1, n_cap=1, V_init=0, n_branch=1, dt=1e-4):
        self.value=C_value
        self.n_cap=n_cap
        self.n_branch=n_branch
        self.dt=dt
        self.V_init=V_init
        self.V_out=self.V_init*np.ones((self.n_cap, self.n_branch))
               
    
    def __call__(self, I_in):
        k1 = self.dt*I_in
        k2 = self.dt*(I_in + 0.5*k1)
        k3 = self.dt*(I_in + 0.5*k2)
        k4 = self.dt*(I_in + k3)
        I_inj=(k1 + 2*k2 + 2*k3 + k4) / 6
        V_out=self.V_init+(1/self.value)*I_inj
        self.V_init=V_out
        return V_out

    
#class Sw_Cap:
#    def __init__(self, N, n_in, IO=8.52E-13, KOUP=27, dt=1e-6, Vdd=0.2):
#        self.n_in=n_in
#        self.N=N
#        self.dt=dt
#        self.I_mos=np.zeros((self.n_in, self.N))
#        self.cap=capacitor(C_value=6e-15, dt=self.dt)
#        self.IO=IO
#        self.KOUP=KOUP
#        self.UT=0.025
#        self.Vdd=Vdd
#        self.Vmem=0.1
#        self.Esyn=0
#        self.V_cap=self.Vmem
#    
#    def __call__(self, Vg_a):
#        Vg_b=self.Vdd-Vg_a
#        I_mos_a=self.IO*np.exp(self.KOUP(self.Vdd-Vg_a))*(np.exp(-(self.Vdd-self.Esyn)/self.UT)-np.exp(-(self.Vdd-self.V_cap)/self.UT))
#        I_mos_b=self.IO*np.exp(self.KOUP(self.Vdd-Vg_b))*(np.exp(-(self.Vdd-self.V_cap)/self.UT)-np.exp(-(self.Vdd-self.Vmem)/self.UT))
#        I_cap=I_mos_a+I_mos_b
#        self.V_cap=self.cap(I_cap)
#        return I_mos_b
