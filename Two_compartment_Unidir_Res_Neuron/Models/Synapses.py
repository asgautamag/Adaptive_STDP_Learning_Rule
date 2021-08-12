# -*- coding: utf-8 -*-

import numpy as np
"""Scaling Table for Double exp synapse
    tr          td          scale_factor
    1e-3        2e-3        3.9e-3
    1e-3        5e-3        7.41e-3
    1e-3        10e-3       12.8e-3
    1e-3        100e-3      104.8e-3
    1e-3        200e-3      204.8e-3"""
class SingleExponentialSynapse:
    def __init__(self, N, n_in, dt=1e-4, td=5e-3):
        """
        Args:
            td (float):Synaptic decay time
        """
        self.N=N
        self.n_in = n_in
        self.dt = dt
        self.td = td
        self.r = np.zeros((self.n_in, self.N))
        self.scaling_factor=td

    def initialize_states(self):
        self.r = np.zeros((self.n_in, self.N))

    def __call__(self, spike):
        r = self.r*(1-self.dt/self.td) + (spike*self.scaling_factor)/self.td
        self.r = r
        return r
         

class DoubleExponentialSynapse_old:
    def __init__(self, N, n_in, dt=1e-4, td=5e-2, tr=1e-3, scale_factor=7.41e-3):
        """
        Args:
            td (float):Synaptic decay time
            tr (float):Synaptic rise time
        """
        self.N = N
        self.n_in=n_in
        self.dt = dt
        self.td = td
        self.tr = tr
        self.r = np.zeros((self.n_in, self.N))
        self.hr = np.zeros((self.n_in, self.N))
        self.scaling_factor=scale_factor
    
    def initialize_states(self):
        self.r = np.zeros((self.n_in, self.N))
        self.hr = np.zeros((self.n_in, self.N))
        
    def __call__(self, spike):
        r = self.r*(1-self.dt/self.tr) + self.hr*self.dt 
        hr = self.hr*(1-self.dt/self.td) + (spike*self.scaling_factor)/(self.tr*self.td)
        
        self.r = r
        self.hr = hr
        
        return r
    
class DoubleExponentialSynapse:
    def __init__(self, N, n_in, dt=1e-4, td=5e-2, tr=1e-3, scale_factor=7.41e-3):
        """
        Args:
            td (float):Synaptic decay time
            tr (float):Synaptic rise time
        """
        self.N = N
        self.n_in=n_in
        self.dt = dt
        self.td = td
        self.tr = tr
        self.r = np.zeros((self.n_in, self.N))
        self.hr = np.zeros((self.n_in, self.N))
        self.scaling_factor=scale_factor
    
    def initialize_states(self):
        self.r = np.zeros((self.n_in, self.N))
        self.hr = np.zeros((self.n_in, self.N))
        
    def __call__(self, spike):
        k1 = self.dt
        k2 = self.dt*(1 + 0.5*k1)
        k3 = self.dt*(1 + 0.5*k2)
        k4 = self.dt*(1 + k3)
        r_dt=(k1 + 2*k2 + 2*k3 + k4) / 6
        r = self.r*(1-r_dt/self.tr) + self.hr*r_dt 
        hr = self.hr*(1-r_dt/self.td) + (spike*self.scaling_factor)/(self.tr*self.td)
        
        self.r = r
        self.hr = hr
        
        return r

