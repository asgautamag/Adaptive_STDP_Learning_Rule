function param_mod(random_seed_state)
global PARAM

PARAM.goOn = 0;

if PARAM.goOn
    PARAM.nRun = 1;
%     PARAM.stdp_a_pos = 2^-4; % avoid > 2^-4
% %     PARAM.stdp_a_neg = -.9* PARAM.stdp_a_pos; % at least .85
%     PARAM.stdp_a_neg = -.875* PARAM.stdp_a_pos; % at least .85
% %     PARAM.threshold = Inf;
% %     neuron.nextFiring = Inf;
% %     PARAM.fixedFiringMode = false;
else % new computation
    %clear all %commented
    global PARAM
    PARAM.goOn = false;
    
    PARAM.dump=false;
    PARAM.beSmart=true; % save time by not computing the potential when it is estimated that the threshold cannot be reached. Rigorous only when no inhibition, and no PSP Kernel. Otherwise use at your own risks...
 
    PARAM.fixedFiringMode = false;
    PARAM.fixedFiringLatency = 10e-3;
    PARAM.fixedFiringPeriod = 150e-3;

    n=0; % spike train number
    PARAM.nRun = 3; % number of times spike train is propagated
    
    % Random generators
    PARAM.randomState = random_seed_state;%Change this for generating different spike trains
%     d = dir('../mat/rand*.mat');
%     if isempty(d)
%         PARAM.randomState=0;
%     else
%         last = d(end).name;
%         PARAM.randomState = str2num(last(5:7))+1;
%     end
%     % just to warn other threads that this random is done
%     tmp=0;
%     save(['../mat/rand' sprintf('%03d',PARAM.randomState) '.mat'],'tmp');
   
    %********
    %* STDP *
    %********
    PARAM.stdp_t_pos = 16.8e-3; %(source: Bi & Poo 2001)
    PARAM.stdp_t_neg = 33.7e-3; %(source: Bi & Poo 2001)
    PARAM.stdp_a_pos = 2^-5; % avoid > 2^-5
    PARAM.stdp_a_neg = -.85 * PARAM.stdp_a_pos; % at least .85
    PARAM.stdp_cut = 7;
    PARAM.minWeight = 0.0;

    %***************
    %* EPSP Kernel *
    %***************
    PARAM.tm = 10e-3; % membrane time constant (typically 10-30ms)
    PARAM.ts = 2.5e-3; % synapse time constant
    PARAM.epspCut = 7;% specifies after how many ms we neglect the epsp
    PARAM.tmpResolution = 1e-3;
    % Double exp (Gerstner 2002)
    %PARAM.epspKernel = pspKernel(0:PARAM.tmpResolution:PARAM.epspCut*PARAM.tm,PARAM.ts,PARAM.tm);
    % Simple exp
    PARAM.epspKernel = exp( -[0:PARAM.tmpResolution:PARAM.epspCut*PARAM.tm]/PARAM.tm );

    
    [m idx] = max(PARAM.epspKernel);
    PARAM.epspKernel = PARAM.epspKernel/m;
    PARAM.epspMaxTime = (idx-1)*PARAM.tmpResolution;
    
    % post synaptic spike kernel
    PARAM.usePssKernel = true;
    % time constant: tm
    %commented next three lines
    %PARAM.pssKernel =   0*pspKernel(0:PARAM.tmpResolution:PARAM.epspCut*PARAM.tm,PARAM.ts/10,PARAM.tm/10) ...
                    %-   3*pspKernel(0:PARAM.tmpResolution:PARAM.epspCut*PARAM.tm,PARAM.ts,PARAM.tm) ...
                    %+   2*exp(-[0:PARAM.tmpResolution:PARAM.epspCut*PARAM.tm]/PARAM.tm);
%     % time constant: tm/2
%     PARAM.pssKernel =   -3*pspKernel(0:PARAM.tmpResolution:PARAM.epspCut*PARAM.tm/2,PARAM.ts/2,PARAM.tm/2) ...
%                     +   2*exp(-[0:PARAM.tmpResolution:PARAM.epspCut*PARAM.tm/2]/(PARAM.tm/2));
%     PARAM.pssKernel =   0*PARAM.epspKernel;

    PARAM.refractoryPeriod = 5e-3;
    % inhibitory postsynaptic potential (positive by convention, scaled so that max is 1)
    PARAM.ipspKernel = PARAM.epspKernel;
    PARAM.inhibStrength = 0.25; % inhibition strength (in fraction of threshold)
    
%     % Simple exp
%     PARAM.epspKernel = exp(-[0:PARAM.tmpResolution:PARAM.epspCut*PARAM.tm]/PARAM.tm);
%     PARAM.epspMaxTime = 0;
%     disp(['Neglecting EPSP when below ' num2str(PARAM.epspKernel(end))])

    % figure;plot(0:PARAM.tmpResolution:PARAM.epspCut*PARAM.tm,PARAM.epspKernel);
    % return


    %***************
    %* Spike Train *
    %***************
    PARAM.nPattern = 1;
    PARAM.nAfferent = 256; %256, 1024 or 2048
    PARAM.oscillations = false;
    PARAM.nCopyPasteAfferent = round( 1 * PARAM.nAfferent ); %1 or 0.5
    PARAM.dt = 1e-3;
    PARAM.maxFiringRate = 90;
    PARAM.spontaneousActivity = 0;%0 or 10
    PARAM.copyPasteDuration = 50e-3;
    PARAM.jitter=0;%0 or 1e-3
    PARAM.spikeDeletion=.0;
    PARAM.maxTimeWithoutSpike = PARAM.copyPasteDuration;
    PARAM.patternFreq = 1/10; %Pattern frequency: 1/4 or 1/10
    for idx = 1:PARAM.nPattern
        PARAM.posCopyPaste{idx} = [];
    end
    PARAM.T = 225%3*(500/PARAM.patternFreq)*PARAM.copyPasteDuration;
    if PARAM.patternFreq>0
        rand('state',PARAM.randomState); %seed
        skip = false;
        for p = 1:round( PARAM.T / PARAM.copyPasteDuration )%1:4500, here 4500 is the number of slots of 50ms in T
            if skip
                skip = false;
            else
                if rand<1/(1/PARAM.patternFreq-1)
                    idx = ceil(rand * PARAM.nPattern);
                    PARAM.posCopyPaste{idx} = [PARAM.posCopyPaste{idx} p];
                    skip = true; % skip next
                end
            end
        end
    end
%     PARAM.nCopyPaste = length(PARAM.posCopyPaste);
    
    %**********
    %* Neuron *
    %**********
    % The threshold corresponds roughly to the number of coincindent spikes we want to detect
    % Should be as big as possible (avoids FA), avoiding to get "stuck" in low
    % spike density zone of the copy-pasted pattern.
    % Appropriate value depends on number of afferent. Eg around 125 for 500
    % afferents.
    % Then initial weights should be tuned so as to reach threshold.
    
    PARAM.nNeuron = 9;
%     PARAM.threshold = 65;
% 	ratio=900/2000;
%    PARAM.threshold = ratio*PARAM.nCopyPasteAfferent;
%     PARAM.threshold = Inf;
    PARAM.threshold = .55*(1-PARAM.spikeDeletion)*PARAM.nCopyPasteAfferent;
%    PARAM.nuThr = round( 2.0 * PARAM.maxFiringRate/2*PARAM.nAfferent*PARAM.tm); % trace parameter for thr computation use inf not to use
    PARAM.nuThr = Inf;
%     if PARAM.goOn && PARAM.threshold==Inf
%         warning('"Going on" in infinite threshold mode has no sense. Setting goOn = false');
%         PARAM.goOn = false;
%     end
    %****************
    %* Neural codes *
    %****************
    PARAM.realValuedPattern = false;
%     PARAM.codingMethod = 4; % 1 for poisson, 2 for LIF, 3 for intensity to phase, 4 for LIF with oscillatory drive
%     PARAM.gammaFreq = 50; % freq of oscillatory drive
%     PARAM.oscillMagnitude = 1;% magnitude of oscillatory drive
%     PARAM.oscillPhase = rand*2*pi;% phase of oscillatory drive
%     PARAM.resetPeriod = Inf;%100e-3;
%     PARAM.resetStd = 25e-3;
%     if PARAM.resetPeriod<Inf
%         if PARAM.resetStd==0
%             PARAM.resetTimes = PARAM.resetPeriod * [1:ceil(PARAM.T/PARAM.resetPeriod)];
%         else
%             PARAM.resetTimes = cumsum( PARAM.resetStd * randn(1,round(1.1*PARAM.T/PARAM.resetPeriod)) + PARAM.resetPeriod );
%         end
%     else
%         PARAM.resetTimes = [];
%     end
%     % LIF model for afferents
%     PARAM.R = 1;
%     PARAM.Vthr = 1.72;
%     PARAM.Vreset = 0;
%     PARAM.Vrest = 0;
% 	PARAM.Imin = 0;
% 	PARAM.Imax = 0.8;
%     
%     PARAM.interStimuliInterval = 1 / 16;
%     PARAM.interPatternInterval = 1 / 2;

    
    %*************
    %* Reporting *
    %*************
%     PARAM.plottingPeriod = [ [0 1]  [5 6] PARAM.nRun*PARAM.T+[-1 0] ];
%     PARAM.plottingPeriod = [ [-1 -1] [-1 -1] [-1 -1] ];
end
