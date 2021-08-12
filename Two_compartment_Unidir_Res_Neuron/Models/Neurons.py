# -*- coding: utf-8 -*-

import numpy as np

class Masq_SRM:
    def __init__(self, N, n_in, dt=1e-3, tc_m=10e-3, tc_s=2.5e-3, tref=1e-3,
                 vrest=0, vthr=500, init_exc_W=None):
        self.N = N
        self.n_in=n_in
        self.dt = dt
        self.tref = tref
        self.tc_m = tc_m 
        self.vrest = vrest
        self.vthr = vthr
        self.td=tc_s
        self.syn_weight=init_exc_W
        self.v = self.vrest*np.ones((1, N)) #I changed from(N) to (N,1)
        self.r=self.vrest*np.ones((self.n_in, self.N))
        self.e = self.vrest*np.ones((self.n_in, self.N))
        self.epsp= self.vrest*np.ones((self.n_in, self.N))
        self.e_s= self.vrest*np.ones((1, N))
        self.r_s= self.vrest*np.ones((1, N))
        self.v_ = None
        self.tlast = 0
        self.tcount = 0
        self.t_cut=7#7
        self.flag_in=False
        self.K1=2
        self.K2=4
        self.s=0
        self.coeff=1/(1-(self.td/self.tc_m))
        self.tlast_pre=np.ones((self.n_in, self.N))
        
    def __call__(self, spike, weight):
        self.s=0
        self.r=(self.r+(self.dt*(-self.r)/self.td+spike)*((self.dt*self.tcount) > (self.tlast + self.tref)))*((self.tcount*self.dt-self.tlast_pre)<(self.t_cut*self.tc_m))
        #r = (self.r*(1-self.dt/self.td)+spike)*((self.dt*self.tcount) > (self.tlast + self.tref))
        #self.r=r
        #e = self.r*(1-self.dt/self.td)+spike
        self.e =(self.e+(self.dt*(self.vrest-self.e)/self.tc_m+spike)*((self.dt*self.tcount) > (self.tlast + self.tref)))*((self.tcount*self.dt-self.tlast_pre)<(self.t_cut*self.tc_m))
        #self.e=e
        #if self.v<=self.vthr:
        self.weighted_epsp=weight*(self.e-self.r)
        #self.epsp=np.sum(self.weighted_epsp)
        self.epsp=np.sum(self.weighted_epsp)*self.coeff
        if (self.v>self.vthr and self.flag_in==False):
            self.flag_in=True
            self.tlast=self.dt*self.tcount
            #print(self.tlast, self.epsp)
            self.s=1
            self.r=np.zeros((self.n_in, self.N))
            self.e=np.zeros((self.n_in, self.N))
            self.e_s=np.zeros((1, self.N))
            self.r_s=np.zeros((1, self.N))
        
        self.e_s=self.e_s*(1-self.dt/self.tc_m)+self.s
        self.r_s=self.r_s*(1-self.dt/self.td)+self.s
        kernel=2*(self.e_s)-3*self.coeff*(self.e_s-self.r_s)
        self.eeta=(self.vthr*kernel)*((self.tcount*self.dt-self.tlast)<(self.t_cut*self.tc_m))
        #self.eeta=(self.vthr*(self.K1*self.e_s-self.K2*(self.e_s-self.r_s)))*((self.tcount*self.dt-self.tlast)<(self.t_cut*self.tc_m))
        v=self.epsp+self.eeta    
        self.v=v      
        if v<self.vthr:
            self.flag_in=False
            #print(self.epsp)
        self.tcount += 1 #serves as the for loop iterator i in the LIF_single.py code
        self.tlast_pre=spike*self.dt*self.tcount+(1-spike)*self.tlast_pre
        return self.s, v
    
    
class Analog_Silicon_Neuron:
    def __init__(self, N=1, Mv=0.52, DELTAv=0.42, Ev=160000, Rv21=1.0, Rv20=1.0, 
                Ia_v=0.068, C_v=0.0006, Mn=0.6, DELTAn=0.485, En=1600, Rn21=2.0, 
                Rn20=1.0, Ia_n=0.065, C_n=0.0009, Er=250000, Rr21=1.0, Rr20=1.0, 
                KOUP=27.0, IOP=0.000852, solver="RK4", dt=1e-4):
        self.N=N
        self.Mv=Mv 
        self.DELTAv=DELTAv
        self.Ev=Ev 
        self.Rv21=Rv21 
        self.Rv20=Rv20
        self.Ia_v=Ia_v
        self.C_v=C_v
        self.Mn=Mn 
        self.DELTAn=DELTAn
        self.En=En
        self.Rn21=Rn21
        self.Rn20=Rn20
        self.Ia_n=Ia_n
        self.C_n=C_n
        self.Er=Er
        self.Rr21=Rr21
        self.Rr20=Rr20
        self.KOUP=KOUP
        self.IOP=IOP
        
        self.dt=dt
        self.solver=solver
        
        self.variables=0.3*np.ones((self.N, 2)) #Initail values
        self.v_prev=np.zeros(self.N)
        self.spike_count=0
        self.I_syn=None
        
    def Solvers(self, eqn, x, dt):
        if self.solver=="RK4":
            k1 = dt*eqn(x)
            k2 = dt*eqn(x + 0.5*k1)
            k3 = dt*eqn(x + 0.5*k2)
            k4 = dt*eqn(x + k3)
            return x + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        elif self.solver == "Euler": 
            return x + dt*eqn(x)
        else:
            return None
        
    def fv_v(self, v):
        return self.Mv / (1 + np.exp(-self.KOUP * (v - self.DELTAv)))
    def gv_v(self, v):
        return self.IOP * np.sqrt(self.Rv20 * self.Ev / (1 + self.Rv21 * self.Ev * np.exp(-self.KOUP * v)))
    def fn_v(self, v):
        return self.Mn / (1 + np.exp(-self.KOUP * (v - self.DELTAn)))
    def gn_v(self, v):
        return self.IOP * np.sqrt(self.Rn20 * self.En / (1 + self.Rn21 * self.En * np.exp(-self.KOUP * v)))
    def rn_n(self, n):
        return self.IOP * np.sqrt(self.Rr20 * self.Er / (1 + self.Rr21 * self.Er * np.exp(-self.KOUP * n)))
    def soe(self, variables):
        v=variables[:,0] 
        n=variables[:,1]
        
        dvdt=(self.fv_v(v)-self.gv_v(v)+self.Ia_v-self.rn_n(n)+self.I_syn)/self.C_v
        dndt=(self.fn_v(v)-self.gn_v(v)+self.Ia_n-self.rn_n(n))/self.C_n
        
        derivs=np.zeros([self.N, 2])
        derivs[:,0]=dvdt
        derivs[:,1]=dndt
        return derivs
    
    def __call__(self, I_inj):
        self.I_syn=I_inj
        variables=self.Solvers(self.soe, self.variables, self.dt)
        self.variables=variables
        #s=(self.variables[:,0]>0.4 and self.v_prev<0.4)*1
        s=np.logical_and(self.variables[:,0]>0.4, self.v_prev<0.4)*1
        """if (self.variables[:,0]>0.4 and self.v_prev<0.4): #0.4 is chosen as the threshold voltage to count spikes
            s=1
            self.spike_count+=1
        else:
            s=0"""
        self.v_prev=variables[:,0]
        return s, variables[:,0]
    
    
#Analog Silicon Neuron Model used in this study, configured to spike in Fast Spiking Class 1 mode in the Hodgkin's classification.    
class Analog_Silicon_Neuron_Rev_Pot_block: 
    def __init__(self, N=1, Mv=0.52, DELTAv=0.42, Ev=160000, Rv21=1.0, Rv20=1.0, 
                Ia_v=0.068, C_v=0.0006, Mn=0.6, DELTAn=0.485, En=100, Rn21=2.0, 
                Rn20=1.0, Ia_n=0.065, C_n=0.0009, Er=250000, Rr21=1.0, Rr20=1.0, 
                KOUP=27.0, IOP=0.000852, gEsyn=0.4, solver="RK4", rev_flag=1, dt=1e-4):
        self.N=N
        self.Mv=Mv*np.ones(self.N) 
        self.DELTAv=DELTAv*np.ones(self.N)
        self.Ev=Ev*np.ones(self.N)
        self.Rv21=Rv21*np.ones(self.N)
        self.Rv20=Rv20*np.ones(self.N)
        self.Ia_v=Ia_v*np.ones(self.N)
        self.C_v=C_v*np.ones(self.N)
        self.Mn=Mn*np.ones(self.N) 
        self.DELTAn=DELTAn*np.ones(self.N)
        self.En=En*np.ones(self.N)
        self.Rn21=Rn21*np.ones(self.N)
        self.Rn20=Rn20*np.ones(self.N)
        self.Ia_n=Ia_n*np.ones(self.N)
        self.C_n=C_n*np.ones(self.N)
        self.Er=Er*np.ones(self.N)
        self.Rr21=Rr21*np.ones(self.N)
        self.Rr20=Rr20*np.ones(self.N)
        self.KOUP=KOUP
        self.IOP=IOP
        self.gEsyn=gEsyn*np.ones(self.N)
        self.scale_factor=10
        self.rev_flag=rev_flag*np.ones(self.N)
        self.dt=dt
        self.solver=solver
        
        self.variables=0.3*np.ones((self.N, 2)) #Initail values
        self.v_prev=np.zeros(self.N)
        self.spike_count=0
        self.I_syn=None
        
    def Solvers(self, eqn, x, dt):
        if self.solver=="RK4":
            k1 = dt*eqn(x)
            k2 = dt*eqn(x + 0.5*k1)
            k3 = dt*eqn(x + 0.5*k2)
            k4 = dt*eqn(x + k3)
            return x + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        elif self.solver == "Euler": 
            return x + dt*eqn(x)
        else:
            return None
        
    def fv_v(self, v):
        return self.Mv / (1 + np.exp(-self.KOUP * (v - self.DELTAv)))
    def gv_v(self, v):
        return self.IOP * np.sqrt(self.Rv20 * self.Ev / (1 + self.Rv21 * self.Ev * np.exp(-self.KOUP * v)))
    def fn_v(self, v):
        return self.Mn / (1 + np.exp(-self.KOUP * (v - self.DELTAn)))
    def gn_v(self, v):
        return self.IOP * np.sqrt(self.Rn20 * self.En / (1 + self.Rn21 * self.En * np.exp(-self.KOUP * v)))
    def rn_n(self, n):
        return self.IOP * np.sqrt(self.Rr20 * self.Er / (1 + self.Rr21 * self.Er * np.exp(-self.KOUP * n)))
    def Isyn_v(self, v):
        return self.I_syn
        
    def soe(self, variables):
        v=variables[:,0] 
        n=variables[:,1]
        
        dvdt=(self.fv_v(v)-self.gv_v(v)+self.Ia_v-self.rn_n(n)+self.Isyn_v(v))/self.C_v
        dndt=(self.fn_v(v)-self.gn_v(v)+self.Ia_n-self.rn_n(n))/self.C_n
        
        derivs=np.zeros([self.N, 2])
        derivs[:,0]=dvdt
        derivs[:,1]=dndt
        return derivs
    
    def __call__(self, I_inj):
        self.I_syn=I_inj
        variables=self.Solvers(self.soe, self.variables, self.dt)
        self.variables=variables
        #s=(self.variables[:,0]>0.4 and self.v_prev<0.4)*1
        s=np.logical_and(self.variables[:,0]>0.4, self.v_prev<0.4)*1
        """if (self.variables[:,0]>0.4 and self.v_prev<0.4): #0.4 is chosen as the threshold voltage to count spikes
            s=1
            self.spike_count+=1
        else:
            s=0"""
        self.v_prev=variables[:,0]
        return s, variables[:,0]
    
    
    
    
#class Analog_Silicon_Neuron_Rev_Pot_block_three_var:
#    def __init__(self, N=1, Mv=0.52, DELTAv=0.42, Ev=160000, Rv21=1.0, Rv20=1.0, 
#                Ia_v=0.1, C_v=0.0006, Mn=0.6, DELTAn=0.485, En=1600, Rn21=2.0, 
#                Rn20=1.0, Ia_n=0.065, C_n=0.0009, Er=250000, Rr21=1.0, Rr20=1.0, 
#                Mq=0.028, DELTAq=0.15, Eq=900, Rq21=1.0, Rq20=1.0, Ia_q=0, C_q=0.0024,
#                KOUP=27.0, IOP=0.000852, gEsyn=0.4, solver="RK4", rev_flag=1, dt=1e-4):
#        self.N=N
#        self.Mv=Mv*np.ones(self.N) 
#        self.DELTAv=DELTAv*np.ones(self.N)
#        self.Ev=Ev*np.ones(self.N)
#        self.Rv21=Rv21*np.ones(self.N)
#        self.Rv20=Rv20*np.ones(self.N)
#        self.Ia_v=Ia_v*np.ones(self.N)
#        self.C_v=C_v*np.ones(self.N)
#        self.Mn=Mn*np.ones(self.N) 
#        self.DELTAn=DELTAn*np.ones(self.N)
#        self.En=En*np.ones(self.N)
#        self.Rn21=Rn21*np.ones(self.N)
#        self.Rn20=Rn20*np.ones(self.N)
#        self.Ia_n=Ia_n*np.ones(self.N)
#        self.C_n=C_n*np.ones(self.N)
#        self.Er=Er*np.ones(self.N)
#        self.Rr21=Rr21*np.ones(self.N)
#        self.Rr20=Rr20*np.ones(self.N)
#        self.Mq=Mq*np.ones(self.N) 
#        self.DELTAq=DELTAq*np.ones(self.N)
#        self.Eq=Eq*np.ones(self.N)
#        self.Rq21=Rq21*np.ones(self.N)
#        self.Rq20=Rq20*np.ones(self.N)
#        self.Ia_q=Ia_q*np.ones(self.N)
#        self.C_q=C_q*np.ones(self.N)
#        
#        self.KOUP=KOUP
#        self.IOP=IOP
#        self.gEsyn=gEsyn*np.ones(self.N)
#        self.scale_factor=10
#        self.rev_flag=rev_flag*np.ones(self.N)
#        self.dt=dt
#        self.solver=solver
#        
#        self.variables=0.3*np.ones((self.N, 3)) #Initail values
#        self.v_prev=np.zeros(self.N)
#        self.spike_count=0
#        self.I_syn=None
#        
#    def Solvers(self, eqn, x, dt):
#        if self.solver=="RK4":
#            k1 = dt*eqn(x)
#            k2 = dt*eqn(x + 0.5*k1)
#            k3 = dt*eqn(x + 0.5*k2)
#            k4 = dt*eqn(x + k3)
#            return x + (k1 + 2*k2 + 2*k3 + k4) / 6
#        
#        elif self.solver == "Euler": 
#            return x + dt*eqn(x)
#        else:
#            return None
#        
#    def fv_v(self, v):
#        return self.Mv / (1 + np.exp(-self.KOUP * (v - self.DELTAv)))
#    def gv_v(self, v):
#        return self.IOP * np.sqrt(self.Rv20 * self.Ev / (1 + self.Rv21 * self.Ev * np.exp(-self.KOUP * v)))
#    def fn_v(self, v):
#        return self.Mn / (1 + np.exp(-self.KOUP * (v - self.DELTAn)))
#    def gn_v(self, v):
#        return self.IOP * np.sqrt(self.Rn20 * self.En / (1 + self.Rn21 * self.En * np.exp(-self.KOUP * v)))
#    def rn_n(self, n):
#        return self.IOP * np.sqrt(self.Rr20 * self.Er / (1 + self.Rr21 * self.Er * np.exp(-self.KOUP * n)))
#    def fq_v(self, v):
#        return self.Mq / (1 + np.exp(-self.KOUP * (v - self.DELTAq)))
#    def rq_q(self, q):
#        return self.IOP * np.sqrt(self.Rq20 * self.Eq / (1 + self.Rq21 * self.Eq * np.exp(-self.KOUP * q)))
#
#    def Isyn_v(self, v):
#        return self.I_syn
#        
#    def soe(self, variables):
#        v=variables[:,0] 
#        n=variables[:,1]
#        q=variables[:,2]
#        
#        dvdt=(self.fv_v(v)-self.gv_v(v)+self.Ia_v-self.rn_n(n)-self.rq_q(q)+self.Isyn_v(v))/self.C_v
#        dndt=(self.fn_v(v)-self.gn_v(v)+self.Ia_n-self.rn_n(n))/self.C_n
#        dqdt=(self.fq_v(v)+self.Ia_q-self.rq_q(q))/self.C_q
#        
#        derivs=np.zeros([self.N, 3])
#        derivs[:,0]=dvdt
#        derivs[:,1]=dndt
#        derivs[:,2]=dqdt
#        return derivs
#    
#    def __call__(self, I_inj):
#        self.I_syn=I_inj
#        variables=self.Solvers(self.soe, self.variables, self.dt)
#        self.variables=variables
#        #s=(self.variables[:,0]>0.4 and self.v_prev<0.4)*1
#        s=np.logical_and(self.variables[:,0]>0.4, self.v_prev<0.4)*1
#        """
#        if (s==1 && self.extra_flag==0):
#            self.extra_flag=1
#        if (self.extra_flag==1):
#            s_count=np.logical_and(self.variables[:,0]<0.25, self.v_prev>0.25)*1
#        if (self.variables[:,0]>0.4 and self.v_prev<0.4): #0.4 is chosen as the threshold voltage to count spikes
#            s=1
#            self.spike_count+=1
#        else:
#            s=0"""
#        self.v_prev=variables[:,0]
#        return s, variables[:,0]
