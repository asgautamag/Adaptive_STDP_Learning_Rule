Scripts corresponding to the simulation results presented in: An Adaptive STDP Learning Rule for Neuromorphic Systems. Submitted to Frontiers in Neuroscience - Neuromorphic Engineering.

1. The folder Spike_train_generation_matlab contains Matlab scripts to generate Poisson spike trains with hidden spike patterns to be detected. 
a) These Matlab scripts are slightly modified versions of the scripts in the study: Masquelier T, Guyonneau R, Thorpe SJ (2008). Competitive STDP-based Spike Pattern Learning. Neural Computation: in press. Only the scripts related to spike train generation are present in this folder. 
b) The script generates a .mat file (ex:afferent.rand001.mat) file comprising spike trains. 
c) The script "looped_pattern_gen.m" can be used to generate spike trains with different seed states. 
d) A file containing the time where spike patterns are present can be generated using the following three lines: 
param_mod %(Edit the variable PARAM.randomState in this file to the desired seed value) 
pat_time=PARAM.posCopyPaste{1}*0.05; 
csvwrite('ihg_matlab_128_onetenth_225sec_time_rand_001.csv', transpose(pat_time)) 
e) Copy the .mat and the .csv file containing the details of spike train and pattern times, respectively, to the folder Two_compartment_Unidir_Res_Neuron.

2. The folder Two_compartment_Unidir_Res_Neuron contains the main python scripts to perform the spike pattern detection task. A biologically plausible unidirectional two-compartment neuron model is used in this study. 
a) The script "mat_to_csvblock.py" requires the "afferent_**.mat file" as input, and generates a .csv file comprising the details of the spike trains. 
b) This file along with the file containing time where patterns are present are used as inputs in the main script: "Two_Comp_Unidir_leak_res_adaptive_STDP.py". 
c) Results are visually represented by four images similar to the ones present in the Results folder. i)image_name_bimodal.png: Displays the evolution of membrane potential of the somatic compartment, the adaptation of tpost, and the bimodal distribution of synaptic weights after learning. ii) image_name_performance: Displays hit rate, false alarms, and the latency to spike. iii) image_name_spikes: Displays the last 1 second of the somatic compartment membrane potential, the synaptic current flowing into the somatic compartment, and the membrane potential of the dendritic compartment. iv) image_name_v_compartment: Evolution of membrane potential of the dendritic compartment.

