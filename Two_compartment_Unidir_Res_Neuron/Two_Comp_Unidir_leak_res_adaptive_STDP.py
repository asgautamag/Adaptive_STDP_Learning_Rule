# -*- coding: utf-8 -*-
"""
Main script corresponding to the simulation results presented in:
An Adaptive STDP Learning Rule for Neuromorphic Systems submitted to Frontiers in Neuroscience - Neuromorphic Engineering.
As input, this script requires two files, input spike train comprising spike patterns to be detected and time where spike patterns are present.

Parameters Units:
Currents: (nA) 
capacitance: (nF)
Resistance: (GOhm)
Voltage: (V)
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
#import pandas as pd
from Models.elements import resistor, capacitor, resistor_unidir
from Models.Neurons import Analog_Silicon_Neuron_Rev_Pot_block
from Models.Synapses import DoubleExponentialSynapse
from Models.Rect_STDP import exponential_stdp_doublet_post_clipped, exponential_stdp_doublet_pre_clipped, rect_stdp_doublet_rate_pre, rect_stdp_doublet_rate_post, tpost_step_calc
np.random.seed(0)

def result_analysis(weight_final):
    pot=[0]
    dep=[0]
    chance=[0]
    vthresh_high=0.012#represent 12 pA
    vthresh_low=0.002#represents 2pA

    for i in range(len(weight_final)):
        if weight_final[i]>=vthresh_high:
            pot.append(i)
        elif weight_final[i]<=vthresh_low:
            dep.append(i)
        else:
            chance.append(i)
    pot.pop(0)
    dep.pop(0)
    chance.pop(0)
    return pot, dep, chance

class AnalogSiliconNeuronNetwork:
    def __init__(self, n_in=128, N=1, n_syn_group=4, n_branch=4, n_cap=8, dt=1e-4, init_exc_W=None, init_inhib_W=None):
        self.N=N
        self.n_cap=n_cap
        self.n_res=n_cap-1
        self.n_branch=n_branch
        self.n_syn_group=n_syn_group
        self.dt=dt
        self.neuron=Analog_Silicon_Neuron_Rev_Pot_block(N=self.N, dt=self.dt, Ia_v=0.07, solver="RK4", rev_flag=0)
        #self.hidden_neuron=Analog_Silicon_Neuron_Rev_Pot_block(N=self.N, dt=self.dt, Ia_v=0.086, solver="RK4", rev_flag=0)
        self.neuron.En[0]=1600
        self.neuron.Mv[0]=0.57
        self.neuron.Mn[0]=0.68
        self.v_m=self.neuron.variables[:,0]
        #self.res_N=resistor(R_value=2, n_res=1)
        self.res_N_unidir=resistor_unidir(R_value=2, n_res=1) #Unidirectional resistor-current into somatic comaprtment
        self.res_Tx_unidir=resistor_unidir(R_value=2, n_res=1)#Unidirectional resistor-current out of somatic compartment
        self.res_leak=resistor(R_value=0.08, n_res=self.n_res)#Leak resistor in dendritic compartment
        self.cap_Den=capacitor(C_value=0.012, n_cap=1, V_init=self.v_m, n_branch=1, dt=self.dt)#Capacitor in the dendritic compartment
        self.n_in=n_in 
        self.V_den_comp=self.v_m
        self.I_Neuron=self.V_den_comp-self.v_m
        self.input_synapse = DoubleExponentialSynapse(N=self.N, n_in=self.n_in, dt=self.dt, td=3e-3, tr=1e-3, scale_factor=5.1e-3)
        #self.shunt_synapse = DoubleExponentialSynapse(N=self.N, n_in=1, dt=self.dt, td=1.5e-3, tr=1e-3, scale_factor=3.3e-3)
        #self.I_shunt=0
        #self.shunt_weight=0e-1
        self.initW = init_exc_W 
        self.weight=self.initW
        self.I_leak=0
        self.E_leak=0.315
        self.wmax=15e-3#Corresponds to 15pA, maximum amplitude of synaptic current
        self.wmin=0
        # Exponential STDP Parameters, with history of all spikes taken into account
        self.aplus_exp=float(self.wmax/32)
        self.aminus_exp=-self.aplus_exp*0.85
        #self.output_synaptictrace= exponential_stdp_pre(N=self.N, n_in=self.n_in, dt=self.dt, td=self.tpost)
        #Rectanguar STDP Parameters
        self.aplus_rect=float(self.wmax/15) # Corresponds to 1pA
        self.aminus_rect=-self.aplus_rect   #Corresponds to -1pA
        #self.tpre=10e-3 #18.2
        #self.tpost=20e-3 #33
        self.tpre=10e-3
        self.tpost=24.0e-3
        self.tadap=3
        self.tpost_delta=13.7e-3
        #self.tclip=50e-3
        
        self.pre_stdp=rect_stdp_doublet_rate_pre(N=self.N, n_in=self.n_in, dt=self.dt, tpre=self.tpre)
        self.post_stdp=rect_stdp_doublet_rate_post(N=self.N, n_in=self.n_in, dt=self.dt, tpost=self.tpost)
        #self.tpost_calculator=rect_stdp_doublet_tpost_calc(N=self.N, n_in=self.n_in, dt=self.dt, tpost=self.tpost, tpost_delta=self.tpost_delta, tclip=self.tclip)
        #self.ltd=LTD(N=self.N, td=15, dt=self.dt)#10
        self.tpost_calculator=tpost_step_calc(N=self.N, td=self.tadap, tpost=self.tpost, tpost_delta_a=self.tpost_delta, tpost_delta_b=10.7e-3, tpost_delta_c=5.7e-3, tpost_delta_d=1e-3, tpost_delta_e=-4.2e-3, tpost_delta_f=-14.6e-3 , tpost_delta_g=-14.6e-3,  dt=self.dt)


        # Exponential STDP trace, only last spike taken into account
        self.input_synaptictrace = exponential_stdp_doublet_pre_clipped(N=self.N, n_in=self.n_in, dt=self.dt, td=self.tpre)
        self.output_synaptictrace= exponential_stdp_doublet_post_clipped(N=self.N, n_in=self.n_in, dt=self.dt, td=self.tpost)
        self.s_out=0

        #Plot initial weights
        self.plot_weights(self.initW)
        #self.gEsyn=0.4
        #self.gEsyn_shunt=0.3


        
        
    def plot_weights(self, S_weights):
        W=S_weights
        plt.rcParams.update({'font.size': 10})
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)

        plt.figure(figsize=(8, 2))
        plt.xlabel("Synaptic Weights")
        plt.plot(W)
        plt.close()
        
    
    
    def __call__(self, spike_in, stdp=True, stdpType="Rect"):
        spike_in=np.expand_dims(spike_in, axis=1)
        #####Next block of code is used to generate copy of input pattern for N neurons, if N>1######
        temp=spike_in
        for i in range (self.N-1):
            if i==0:
                spike_in=np.append(spike_in, spike_in, axis=1)
            else:
                spike_in=np.append(spike_in, temp, axis=1)
        ###########Spike copy for N Neurons generated###############################################               
        if (stdp==True and stdpType=="Rect"):
            apre=self.pre_stdp(spike_in, self.s_out, self.tpre)*self.aplus_rect
        if (stdp==True and stdpType=="Exp"):
            apre=self.input_synaptictrace(spike_in, self.s_out)*self.aplus_exp
        
        if (stdp==True and stdpType=="Rect"):
            apost=self.post_stdp(spike_in, self.s_out, self.tpost)*self.aminus_rect
        if (stdp==True and stdpType=="Exp"):
            apost=self.output_synaptictrace(spike_in, self.s_out)*self.aminus_exp
                   
        Isyn_raw=self.input_synapse(spike_in)
        Isyn_W=np.multiply(Isyn_raw, self.weight)
        #Isyn_dend_in=np.multiply(Isyn_W, (self.gEsyn-self.V_den_comp))*self.gEsyn_scale_factor
        #Isyn_dend_in=Isyn_W*(self.gEsyn-self.V_den_comp)*self.gEsyn_scale_factor
        self.V_den_comp=self.cap_Den(np.sum(Isyn_W)-self.I_leak)
        self.I_leak=self.res_leak(self.V_den_comp-self.E_leak)
        V_diff_N=self.V_den_comp-self.v_m
        self.I_Neuron_lost=self.res_Tx_unidir(-V_diff_N)
        self.I_Neuron=self.res_N_unidir(V_diff_N)-self.I_Neuron_lost

        self.s_out, v_m=self.neuron(self.I_Neuron)
        self.v_m=v_m
        #self.I_shunt=self.shunt_synapse(self.s_out)*(self.gEsyn_shunt-self.V_den_comp)*self.shunt_weight
        #self.I_node[0]=-I_res[0]
        #I_node=-np.diff(I_res, axis=0)
        #self.I_node[1:self.n_res]=I_node
        #self.I_node[self.n_cap-1]=I_res[self.n_res-1]-I_den
        self.tpost=self.tpost_calculator()
        
        #s_out=np.expand_dims(s_out, axis=1)
        ###############################Update spikes for inhibitory weights##########
        #for i in range(self.N):
            #self.s_out_inhib[:,i]=np.delete(s_out, i)
        #############################################################################
        if stdp==True:
            #print(apre.shape, s_out.shape, apost.shape, spike_in.shape)
            #dW=apre*s_out+apost*spike_in
            dW_pos=apre*self.s_out
            self.weight=np.clip(self.weight+dW_pos, self.wmin, self.wmax)
            dW_neg=apost*spike_in
            self.weight=np.clip(self.weight+dW_neg, self.wmin, self.wmax)

        return self.s_out, v_m, self.weight, self.tpost, self.I_Neuron, self.V_den_comp       



if __name__ == '__main__':
    name= ["00", "01","02","03", "04","05", "06", "07", "08", "09", "10", "11", "12"] #These numbers represent the initial seed value used in spike train generation.
    name= ["01"]
    Results=np.zeros((len(name),12))
    for n_run in range(len(name)):
        file_name="ihg_matlab_nosp_nojitter_1000_onefourth_225sec_rand_0"+name[n_run]+".csv"  #Input Spike Train, a.csv file with two columns, index of the afferent and its spike time.
        #file_name="pp_50sec_60Hz_"+name[n_run]+".csv"
        file_name_time="ihg_matlab_128_onefourth_225sec_time_rand_0"+name[n_run]+".csv"       #File with time points where spike pattern exists, used for verification.
        #image_name="pp_50sec_60Hz_"+name[n_run]
        image_name="ihg_matlab_256_onefourth_225sec_time_rand_0"+name[n_run]                 
        T=450                    #Total run duration seconds
        T_pat=225                #Input spike train duration
        num_runs=int(T/T_pat)
        epoch_time=45            #Duration of one epoch
        dt=1e-5                  #Time step
        nt=round(T/(dt))         
        epoch_nt=round(epoch_time/(dt))
        T_pat_nt=round(T_pat/(dt))
        N=1                      #Number of Neurons
        n_in=256                 #Number of Afferents: 256, 1024, 2048
        #n_syn_group=4           #Number of synapses in a single node
        #n_cap=16                #Number of cap nodes in a branch (n_cap*n_syn_group synapses)
        #n_branch=4              #Number of branches
        stdp_state=True
        init_exc_W=np.expand_dims(np.array([(1e-3)*(15/15)*7]), axis=0)*np.ones((n_in, N)) #initial synaptic weights, current of 7pA
        #init_inhib_W=-0*np.ones((N-1, N))
        flag_close=1             #inline plotting flag
        #######################INPUT SPIKES GENERATION#############################
        
        NIST={} #Neurons Index spike time, a dictionary which will have Neuron index as the key and spike times as the content of the dictionary.
        for i in range(n_in):
            NIST[i]=[0]
        with open(file_name, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader) #Skip the first row
            for row in reader:
                if (int(float(row[1]))<n_in):
                    NIST[int(float(row[1]))].append(float(row[0]))
        #####################INPUT SPIKES GENERATED################################
        for i in range(n_in):
            NIST[i].pop(0)

        epoch=int(T_pat/epoch_time)
        print("Number of epochs", epoch)
        for n_rep in range(num_runs):
            for i in range(epoch):
                PP=np.zeros((epoch_nt, n_in))
                for j in range(n_in):
                    spiketime_per_Neuron=NIST[j]
                    spiketime_per_Neuron=[int(round(j/dt))  for j in spiketime_per_Neuron]
                    #print(int(((epoch_time)*(i))/dt), int(((epoch_time)*(i+1))/dt))
                    spiketime_per_Neuron=list(filter(lambda num: int(round(((epoch_time)*(i))/dt))<= num < int(round(((epoch_time)*(i+1))/dt)), spiketime_per_Neuron)) #This line makes 
                    #the pattern even for T value less than 50.2
                    spiketime_per_Neuron=[k-(int(round(i*(epoch_time/dt)))) for k in spiketime_per_Neuron ]
                    PP[spiketime_per_Neuron, j]=1
                print("Poisson Pattern generation of epoch", i, "finishes here, array shape is", PP.shape, np.sum(PP))

                if (n_rep==0 and i==0):
                    network=AnalogSiliconNeuronNetwork(n_in=n_in, N=N, dt=dt, init_exc_W=init_exc_W )
                    spikes_=np.zeros((nt, N))
                    vmem_=np.zeros((nt, N))
                    tpost_=np.zeros((nt, N))
                    spike_time=[]
                    Isyn_=np.zeros((nt, N))
                    V_den_comp_=np.zeros((nt, N))
                    I_shunt_=np.zeros((nt, N))
                for j in tqdm(range(epoch_nt)):
                    if ((j+(i*epoch_nt)+(n_rep*T_pat_nt))*dt)>450: #Can be used to stop STDP learning at any point in time.
                        stdp_state=True
                    spike_out, vmem, syn_W, tpost, Isyn, V_den_comp=network(PP[j], stdp=stdp_state, stdpType="Rect")
                    spikes_[j+(i*epoch_nt)+(n_rep*T_pat_nt)]=spike_out
                    vmem_[j+(i*epoch_nt)+(n_rep*T_pat_nt)]=vmem
                    tpost_[j+(i*epoch_nt)+(n_rep*T_pat_nt)]=tpost
                    weight_final=syn_W
                    Isyn_[j+(i*epoch_nt)+(n_rep*T_pat_nt)]=Isyn
                    V_den_comp_[j+(i*epoch_nt)+(n_rep*T_pat_nt)]=V_den_comp
                    #I_shunt_[j+(i*epoch_nt)+(n_rep*T_pat_nt)]=I_Shunt
                    if (spike_out==1):
                        spike_time.append((j+(i*epoch_nt)+(n_rep*T_pat_nt))*dt)
                    if ((j+(i*epoch_nt)+(n_rep*T_pat_nt))*dt)==1:
                        spike_sum_sec=np.sum(spikes_)
                                    
        spike_sum=np.sum(spikes_)
        #spikes_=0 #free memory
        print("total spikes:", spike_sum)
        time = np.arange(0.0, T, dt)
        plt.rcParams.update({'font.size': 14})
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
    #####Plot Membrane Potential, Synaptic Current and histogram of final synaptic weights#################
        plt.figure(figsize=(10, 14))
        plt.subplot(3,1,1)
        plt.xlabel('Time (s)')
        plt.ylabel("Vmem (V)")
        plt.plot(time, vmem_, label=image_name)
        plt.legend(loc='upper right')
        plt.subplot(3,1,2)
        plt.xlabel('Time (s)')
        plt.ylabel("tpost (msec)")
        plt.plot(time, tpost_)
        plt.subplot(3,1,3)
        plt.xlabel("Synaptic Weights")
        plt.ylabel("Number of Synapses")
        #plt.hist(weight_eve[0, nt-1])
        plt.hist(weight_final*1000, bins=16)
#        plt.subplot(4,1,4)
#        plt.xlabel('Time (s)')
#        plt.ylabel("Isyn (nA)")
#        plt.plot(time, Isyn_)
        fig=plt.gcf()
        fig.savefig(image_name+"_bimodal", dpi=100)
        if (flag_close==1):
            plt.close()
        plt.figure(figsize=(8, 4))
        plt.xlabel('Time(s)')
        plt.ylabel("Vden(V)")
        plt.plot(time, V_den_comp_)
        fig=plt.gcf()
        fig.savefig(image_name+"_v_compartment", dpi=100)
        if (flag_close==1):
            plt.close()
  ####Create a list PT to store pattern times from the input file containing time when 50ms pattern end############# 
        PT=[]
        with open(file_name_time, mode='r') as infile:#c,g,i, j
            reader = csv.reader(infile)
            #next(reader) #Skip the first row
            for row in reader:
                #NIST[int(float(row[1]))].append(float(row[0]))
                PT.append(round(float(row[0]),4))  
        ind_old=0
        for ind in range(len(PT), 2*len(PT),1):
            PT.append(225+PT[ind_old])
            ind_old=ind_old+1
            
        plot_time_list=[] 
        a=round(T/1)
        for i in range(1):
            plot_time_list.append(a*i)
        plot_time_list.append(T-1)
  ####Plot Membrane potential at specific time slots for 1 second duration to see results################        
        for i in plot_time_list:
            plt.figure(figsize=(10,12))
            time_steps=int(i/dt)
            Pat_time=list(filter(lambda num: i<= num < i+1, PT))
            plt.subplot(3,1,1)
            plt.xlabel('Time(s)')
            plt.ylabel("Vmem(V)")
            plt.plot(time[time_steps:time_steps+100000], vmem_[time_steps:time_steps+100000], label=str(Pat_time))
            plt.legend(loc='upper right')
            plt.subplot(3,1,2)
            plt.xlabel('Time(s)')
            plt.ylabel("Isyn(nA)")
            plt.plot(time[time_steps:time_steps+100000], Isyn_[time_steps:time_steps+100000])
            plt.subplot(3,1,3)
            plt.xlabel('Time(s)')
            plt.ylabel("Vden(V)")
            plt.plot(time[time_steps:time_steps+100000], V_den_comp_[time_steps:time_steps+100000])
            fig=plt.gcf()
            fig.savefig(image_name+"_spikes", dpi=100)
            if (flag_close==1):
                plt.close()
        #########################Performance Verification#########################
        PT=list(filter(lambda num: 0<= num <= T, PT))
        pattern_slot_spike_count=[]
        pattern_slot_spike_time=[]
        missed_spikes=[]
        patt_time=[]#List with time where the pattern starts.
        for i in range(len(PT)):
            spiketime=list(filter(lambda num: PT[i]-0.05<= num < PT[i], spike_time))
            patt_time.append(PT[i]-0.05)
            pattern_slot_spike_count.append(len(spiketime))
            if (len(spiketime)==0):
                missed_spikes.append(PT[i])
                pattern_slot_spike_time.append([PT[i]-0.03]) #For missed spikes latency value is fixed at 20ms.
            else:
                pattern_slot_spike_time.append(spiketime)   #When neuron spikes inside the pattern.
                
        latency=[]
        for i in range(len(PT)):
            latency.append(pattern_slot_spike_time[i][0]-patt_time[i])
        #plt.plot(latency)
        extra_slot_spike_count=[]
        extra_slot_spike_time=[]        
        extra_spikes=[]        
        for i in range(len(PT)):
            if(i>0):
                spiketime=list(filter(lambda num: PT[i-1]<= num < PT[i]-0.05, spike_time))
                extra_slot_spike_time.append(spiketime)
                extra_slot_spike_count.append(len(spiketime))
                if (len(spiketime)>0):
                    extra_spikes.append(PT[i])
            
        print("Number of pattern missed", len(missed_spikes))
        print("Hit rate:", (len(PT)-len(missed_spikes))/len(PT))
        plt.figure(figsize=(10,12))
        plt.subplot(3,1,1)
        plt.xlabel('Number of Patterns')
        plt.ylabel("Number of spikes")
        plt.plot(pattern_slot_spike_count, label="1sec:"+str(spike_sum_sec)+",Iav:"+str(network.neuron.Ia_v)+",missed "+str(len(missed_spikes))+"out of "+str(len(PT))+", hit_rate="+str((len(PT)-len(missed_spikes))/len(PT))) #Calculated over entire 450 sec.
        plt.legend(loc='upper right')
        plt.subplot(3,1,2)
        plt.xlabel('Number of Patterns')
        plt.ylabel("Number of spikes")
        plt.plot(extra_slot_spike_count, label="extra spikes: "+str(len(extra_spikes))+" false_alarm:"+str(np.sum(extra_slot_spike_count[-1000:])))#False alarms calculated for last 1000 pattern presentation.
        plt.legend(loc='upper right')
        plt.subplot(3,1,3)
        plt.xlabel('Number of Patterns')
        plt.ylabel("Latency(ms)")
        lat_ms=[i*1000 for i in latency]
        plt.plot(lat_ms, label="latency: "+str(latency[-1]*1000)+"ms")
        plt.legend(loc='lower right')
        fig=plt.gcf()
        fig.savefig(image_name+"_performance", dpi=100)
        if (flag_close==1):
            plt.close()
        ####################################Weight Analysis########################################
        pot, dep, chance=result_analysis(weight_final)
        PIST=list(filter(lambda num: 5 <= num <6, PT))
        pot_pattern={}
        pot_pat=[]
        pot_pat_len=[]
        for i in pot:
            pot_pattern[i]=NIST[i]
            temp=list(filter(lambda num: PIST[0]-0.05 <= num <PIST[0], pot_pattern[i]))
            pot_pattern[i]=temp
            pot_pat.append(temp)
            pot_pat_len.append(len(temp))
        flattened_pot_pat = [y for x in pot_pat for y in x]
        flattened_pot_pat.sort()
    
        dep_pattern={}
        dep_pat=[]
        dep_pat_len=[]
        for i in dep:
            dep_pattern[i]=NIST[i]
            temp=list(filter(lambda num: PIST[0]-0.05 <= num <PIST[0], dep_pattern[i]))
            dep_pattern[i]=temp
            dep_pat.append(temp)
            dep_pat_len.append(len(temp))
        flattened_dep_pat = [y for x in dep_pat for y in x]
        flattened_dep_pat.sort()
#        plt.figure(figsize=(10,12))
#        plt.subplot(2,1,1)
#        plt.plot(flattened_pot_pat, label=str(len(pot)))
#        #plt.scatter(flattened_pot_pat,[i for i in range(len(flattened_pot_pat))])
#        plt.legend(loc='upper right')
#        plt.subplot(2,1,2)
#        plt.plot(flattened_dep_pat, label=str(len(dep)))
#        plt.legend(loc='upper right')
#        fig=plt.gcf()
#        fig.savefig(image_name+"POT_DEP_Synapses", dpi=100)
#        plt.close()
        print("Potentiated synapses:", len(pot))
        print("Depressed synapses:", len(dep))
        print("Chance synapses:", len(chance))
        Results[n_run]=[name[n_run], spike_sum_sec, len(PT), len(missed_spikes), len(extra_spikes), (len(PT)-len(missed_spikes))/len(PT), np.average(pattern_slot_spike_count[-50:]), np.sum(extra_slot_spike_count[-100:]), latency[-1], len(pot), len(dep), len(chance)]
        print("Completed"+str(name[n_run]))
        print("Extra spikes: "+str(np.sum(extra_slot_spike_count[-1000:])))

    #pd.DataFrame(Results).to_csv("ihg_256_onefourth_51_99.csv")
