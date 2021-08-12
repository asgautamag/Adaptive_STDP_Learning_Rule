# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:44:53 2020

@author: Ashish Gautam
"""

import numpy as np

##############################tpost_adaptation_in_steps##########################################
class tpost_step_calc:
    def __init__(self, N=1, td=5, tpost=14e-3, tpost_delta_a=7.8e-3, tpost_delta_b=10.7e-3, tpost_delta_c=5.7e-3, tpost_delta_d=1e-3, tpost_delta_e=-4.2e-3, tpost_delta_f=-11.6e-3 , tpost_delta_g=-11.6e-3, tpost_delta_h=-11.6e-3 , tpost_delta_i=-11.6e-3, dt=1e-5):
        self.N=N
        self.dt=dt
        self.strength=np.ones((self.N))
        self.tpost=tpost
        self.tpost_delta_a=tpost_delta_a
        self.tpost_delta_b=tpost_delta_b
        self.tpost_delta_c=tpost_delta_c
        self.tpost_delta_d=tpost_delta_d
        self.tpost_delta_e=tpost_delta_e
        self.tpost_delta_f=tpost_delta_f
        self.tpost_delta_g=tpost_delta_g
        self.tpost_delta_h=tpost_delta_h
        self.tpost_delta_i=tpost_delta_i
        self.tclip_a=2*td
        self.tclip_b=3*td
        self.tclip_c=4*td
        self.tclip_d=5*td
        self.tclip_e=6*td
        self.tclip_f=140*td
        self.tclip_g=150*td
        self.tclip_h=151*td
        self.tclip_i=152*td
        self.tcount=0
        
    def __call__(self):
        #self.strength=self.strength*(1-s_in)+self.strength*(1-self.dt/self.td)*s_in
        #self.strength=(self.tcount*self.dt<self.tclip)
        self.tcalc=(self.tcount*self.dt<=self.tclip_a)*self.tpost_delta_a+(self.tclip_a<self.tcount*self.dt<=self.tclip_b)*self.tpost_delta_b+(self.tclip_b<self.tcount*self.dt<=self.tclip_c)*self.tpost_delta_c+(self.tclip_c<self.tcount*self.dt<=self.tclip_d)*self.tpost_delta_d+(self.tclip_d<self.tcount*self.dt<=self.tclip_e)*self.tpost_delta_e+(self.tclip_e<self.tcount*self.dt<=self.tclip_f)*self.tpost_delta_f+(self.tclip_f<self.tcount*self.dt<=self.tclip_g)*self.tpost_delta_g+(self.tclip_g<self.tcount*self.dt<=self.tclip_h)*self.tpost_delta_h+(self.tclip_h<self.tcount*self.dt<=self.tclip_i)*self.tpost_delta_i
        self.tcount=self.tcount+1
        return self.tpost-self.tcalc
        
##########Rectangular###########################
class rect_stdp_doublet_rate_pre:
    def __init__(self, N=1, n_in=128, tpre=6e-3, dt=1e-4):
        self.N=N
        self.n_in=n_in
        self.dt=dt
        self.s_state_pre=np.zeros((self.n_in, self.N))
        self. tcount_pre=0
        self.tpre=tpre
        self.last_pre_spike=np.zeros((self.n_in, self.N)) #last presynaptic spike time
        
    def __call__(self, s_in, s_out, tpre):
        self.tpre=tpre
        if s_out==0:
            self.s_state_pre=s_in+((self.tcount_pre*self.dt-self.last_pre_spike)<self.tpre)*self.s_state_pre*(1-s_in)
                        
        elif s_out==1:
            self.s_state_pre=np.zeros((self.n_in, self.N))
        else:
            print("Illegal s_out")
        self.last_pre_spike=self.dt*self.tcount_pre*s_in+ self.last_pre_spike*(1-s_in)
        self.tcount_pre=self.tcount_pre+1    
        return self.s_state_pre

class rect_stdp_doublet_rate_post:
    def __init__(self, N=1, n_in=128, tpost=7e-3, dt=1e-4):
        self.N=N
        self.n_in=n_in
        self.dt=dt
        self.s_state_post=np.zeros((self.n_in, self.N))
        self.depressed_state=np.zeros((self.n_in, self.N))
        self. tcount_post=0
        self.tpost=tpost
        self.last_post_spike=np.zeros((self.n_in, self.N))
        
    def __call__(self, s_in, s_out, tpost):
        self.tpost=tpost
        if s_out==1:
            self.s_state_post=np.ones((self.n_in, self.N))
        else:
            self.s_state_post=np.clip(((((self.tcount_post*self.dt-self.last_post_spike)<self.tpost)*self.s_state_post)*self.depressed_state),0,1)
        self.last_post_spike=self.dt*self.tcount_post*s_out+ self.last_post_spike*(1-s_out)
        self.tcount_post=self.tcount_post+1
        #self.depressed_state=np.clip((s_out*np.ones((self.n_in, self.N))+((self.tcount_post*self.dt-self.last_post_spike)<self.tpost)*self.s_state_post*(1-s_out))*(1-s_in)+s_in*(-1)*np.ones((self.n_in, self.N)),0,1)
        self.depressed_state=s_in*(-1)*np.ones((self.n_in, self.N))+(1-s_in)
        return self.s_state_post

class rect_stdp_doublet_tpost_calc:
    def __init__(self, N=1, n_in=128, tpost=7e-3, tpost_delta=1.5e-3, tclip=50e-3, dt=1e-4):
        self.N=N
        self.n_in=n_in
        self.dt=dt
        self.state=0#np.zeros((self.n_in, self.N))
        self. tcount=0
        self.tpost=tpost
        self.tpost_updated=self.tpost-tpost_delta
        self.last_spike=-1#np.zeros((self.n_in, self.N))
        self.tclip=tclip
        
    def __call__(self, s_out):
        if s_out==1:
            self.state=s_out
        else:
            self.state=s_out+((self.tcount*self.dt-self.last_spike)<self.tclip)*self.state*(1-s_out)
        self.last_spike=self.dt*self.tcount*s_out+ self.last_spike*(1-s_out)
        self.tcount=self.tcount+1    
        return self.state*self.tpost_updated+(1-self.state)*self.tpost
    
###############################Corrected STDP Functions#######################
##########Rectangular###########################
class rect_stdp_doublet_pre:
    def __init__(self, N=1, n_in=128, tpre=6e-3, dt=1e-4):
        self.N=N
        self.n_in=n_in
        self.dt=dt
        self.s_state_pre=np.zeros((self.n_in, self.N))
        self. tcount_pre=0
        self.tpre=tpre
        self.last_pre_spike=np.zeros((self.n_in, self.N)) #last presynaptic spike time
        
    def __call__(self, s_in, s_out):
        if s_out==0:
            self.s_state_pre=s_in+((self.tcount_pre*self.dt-self.last_pre_spike)<self.tpre)*self.s_state_pre*(1-s_in)
                        
        elif s_out==1:
            self.s_state_pre=np.zeros((self.n_in, self.N))
        else:
            print("Illegal s_out")
        self.last_pre_spike=self.dt*self.tcount_pre*s_in+ self.last_pre_spike*(1-s_in)
        self.tcount_pre=self.tcount_pre+1    
        return self.s_state_pre

class rect_stdp_doublet_post:
    def __init__(self, N=1, n_in=128, tpost=7e-3, dt=1e-4):
        self.N=N
        self.n_in=n_in
        self.dt=dt
        self.s_state_post=np.zeros((self.n_in, self.N))
        self.depressed_state=np.zeros((self.n_in, self.N))
        self. tcount_post=0
        self.tpost=tpost
        self.last_post_spike=np.zeros((self.n_in, self.N))
        
    def __call__(self, s_in, s_out):
        if s_out==1:
            self.s_state_post=np.ones((self.n_in, self.N))
        else:
            self.s_state_post=np.clip(((((self.tcount_post*self.dt-self.last_post_spike)<self.tpost)*self.s_state_post)*self.depressed_state),0,1)
        self.last_post_spike=self.dt*self.tcount_post*s_out+ self.last_post_spike*(1-s_out)
        self.tcount_post=self.tcount_post+1
        #self.depressed_state=np.clip((s_out*np.ones((self.n_in, self.N))+((self.tcount_post*self.dt-self.last_post_spike)<self.tpost)*self.s_state_post*(1-s_out))*(1-s_in)+s_in*(-1)*np.ones((self.n_in, self.N)),0,1)
        self.depressed_state=s_in*(-1)*np.ones((self.n_in, self.N))+(1-s_in)
        return self.s_state_post
##############Exponential#########################################
class exponential_stdp_doublet_pre_clipped:
    def __init__(self, N, n_in, dt, td):
        self.N=N
        self.n_in = n_in
        self.dt = dt
        self.td = td
        self.tclip=7*self.td #117ms
        self.tcount_pre=0
        self.s_state_pre = np.zeros((self.n_in, self.N))
        self.scaling_factor=td
        self.last_pre_spike=np.zeros((self.n_in, self.N))
        
    def __call__(self, s_in, s_out):
        if s_out==0:
            self.s_state_pre = (self.s_state_pre*(1-self.dt/self.td)*(1-s_in) + (s_in*self.scaling_factor)/self.td)*((self.tcount_pre*self.dt-self.last_pre_spike)<self.tclip)

        elif s_out==1:
            self.s_state_pre=np.zeros((self.n_in, self.N))
        else:
            print("Illegal s_out")
        self.last_pre_spike=self.dt*self.tcount_pre*s_in+ self.last_pre_spike*(1-s_in)
        self.tcount_pre=self.tcount_pre+1
        return self.s_state_pre
    
class exponential_stdp_doublet_post_clipped:
    def __init__(self, N, n_in, dt, td):
        self.N=N
        self.n_in = n_in
        self.dt = dt
        self.td = td
        self.tclip=7*self.td #117ms
        self.tcount_post=0
        self.s_state_post=np.zeros((self.n_in, self.N))
        self.depressed_state=np.zeros((self.n_in, self.N))
        self.scaling_factor=td
        self.last_post_spike=np.zeros((self.n_in, self.N))
        
    def __call__(self, s_in, s_out):
        if s_out==1:
            self.s_state_post=np.ones((self.n_in, self.N))
        else:
            self.s_state_post = np.clip((((self.s_state_post*(1-self.dt/self.td))*((self.tcount_post*self.dt-self.last_post_spike)<self.tclip))*self.depressed_state),0,1)
        self.last_post_spike=self.dt*self.tcount_post*s_out+ self.last_post_spike*(1-s_out)
        self.tcount_post=self.tcount_post+1
        self.depressed_state=s_in*(-1)*np.ones((self.n_in, self.N))+(1-s_in)
        return self.s_state_post

###############################################################################
class stdp_pre:
    def __init__(self, N=1, n_in=128, tpre=6e-3, dt=1e-4, aplus=1e-4):
        self.N=N
        self.n_in=n_in
        self.dt=dt
        self.s_state_pre=np.zeros((self.n_in, self.N))
        self. tcount_pre=0
        self.tpre=tpre
        self.aplus=aplus
        self.last_pre_spike=np.zeros((self.n_in, self.N)) #last presynaptic spike time
        
    def __call__(self, s_in):
        self.s_state_pre=self.aplus*s_in+((self.tcount_pre*self.dt-self.last_pre_spike)<self.tpre)*self.s_state_pre*(1-s_in)
        self.last_pre_spike=self.dt*self.tcount_pre*s_in+ self.last_pre_spike*(1-s_in)
        self.tcount_pre=self.tcount_pre+1
        return self.s_state_pre
    
class stdp_post:
    def __init__(self, N=1, n_in=128, tpost=7e-3, dt=1e-4, aminus=-1e-4):
        self.N=N
        self.n_in=n_in
        self.dt=dt
        self.s_state_post=np.zeros((1, self.N))
        self. tcount_post=0
        self.tpost=tpost
        self.aminus=aminus
        self.last_post_spike=0
        
    def __call__(self, s_in):
        self.s_state_post=self.aminus*s_in+((self.tcount_post*self.dt-self.last_post_spike)<self.tpost)*self.s_state_post*(1-s_in)
        self.last_post_spike=self.dt*self.tcount_post*s_in+ self.last_post_spike*(1-s_in)
        self.tcount_post=self.tcount_post+1
        return self.s_state_post
    
############ Exponential STDP Parameters, with history of all spikes taken into account###############
class exponential_stdp_post:
    def __init__(self, N, n_in, dt, td):
        """
        Args:
            td (float):Synaptic decay time
        """
        self.N=N
        self.n_in = n_in
        self.dt = dt
        self.td = td
        self.r = np.zeros((1, self.N))
        self.scaling_factor=td
        
    def initialize_states(self):
        self.r = np.zeros(self.N)

    def __call__(self, spike):
        r = self.r*(1-self.dt/self.td) + (spike*self.scaling_factor)/self.td
        self.r = r
        return r

############## Exponential STDP Parameters, only last spike taken into account#####################
class exponential_stdp_last_spike_pre_aplus:
    def __init__(self, N, n_in, dt, td, aplus=1e-4):
        self.N=N
        self.n_in = n_in
        self.dt = dt
        self.td = td
        self.aplus=aplus
        self.r = np.zeros((self.n_in, self.N))
        self.scaling_factor=td
        
    def __call__(self, spike):
        r = self.r*(1-self.dt/self.td)*(1-spike) + (self.aplus*spike*self.scaling_factor)/self.td
        self.r = r
        return r

class exponential_stdp_last_spike_pre:
    def __init__(self, N, n_in, dt, td):
        self.N=N
        self.n_in = n_in
        self.dt = dt
        self.td = td
        #self.aplus=aplus
        self.r = np.zeros((self.n_in, self.N))
        self.scaling_factor=td
        
    def __call__(self, spike):
        r = self.r*(1-self.dt/self.td)*(1-spike) + (spike*self.scaling_factor)/self.td
        self.r = r
        return r

class exponential_stdp_last_spike_post:
    def __init__(self, N, n_in, dt, td):
        self.N=N
        self.n_in = n_in
        self.dt = dt
        self.td = td
        self.r = np.zeros((1, self.N))
        self.scaling_factor=td
        
    def __call__(self, spike):
        r = self.r*(1-self.dt/self.td)*(1-spike) + (spike*self.scaling_factor)/self.td
        self.r = r
        return r
    
class exponential_stdp_last_spike_pre_clipped:
    def __init__(self, N, n_in, dt, td):
        self.N=N
        self.n_in = n_in
        self.dt = dt
        self.td = td
        self.tpre=7*self.td #117ms
        self.tcount_pre=0
        self.r = np.zeros((self.n_in, self.N))
        self.scaling_factor=td
        self.last_pre_spike=np.zeros((self.n_in, self.N))
        
    def __call__(self, spike):
        r = (self.r*(1-self.dt/self.td)*(1-spike) + (spike*self.scaling_factor)/self.td)*((self.tcount_pre*self.dt-self.last_pre_spike)<self.tpre)
        self.last_pre_spike=self.dt*self.tcount_pre*spike+ self.last_pre_spike*(1-spike)
        self.tcount_pre=self.tcount_pre+1
        self.r = r
        return r
    
class exponential_stdp_last_spike_post_clipped:
    def __init__(self, N, n_in, dt, td):
        self.N=N
        self.n_in = n_in
        self.dt = dt
        self.td = td
        self.tpre=7*self.td #117ms
        self.tcount_pre=0
        self.r = np.zeros((1, self.N))
        self.scaling_factor=td
        self.last_pre_spike=np.zeros((1, self.N))
        
    def __call__(self, spike):
        r = (self.r*(1-self.dt/self.td)*(1-spike) + (spike*self.scaling_factor)/self.td)*((self.tcount_pre*self.dt-self.last_pre_spike)<self.tpre)
        self.last_pre_spike=self.dt*self.tcount_pre*spike+ self.last_pre_spike*(1-spike)
        self.tcount_pre=self.tcount_pre+1
        self.r = r
        return r
