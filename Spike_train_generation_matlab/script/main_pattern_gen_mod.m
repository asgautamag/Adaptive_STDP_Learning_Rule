% Main script. The current configuration corresponds to the main simulation in:
% Masquelier T, Guyonneau R, Thorpe SJ (2008). Competitive STDP-based Spike Pattern Learning. Neural Computation: in press.
% Feel free to use and modify but please cite us.
% The main function is the mex file STDPContinuous.
% The script prepares the spike trains and the neurons, calls STDPContinuous and analyzes the results.
% timothee.masquelier@alum.mit.edu

% run ../../setPath
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Script modified to generated spike patterns only.

if ~exist('PARAM','var')
    global PARAM
end

%param

if ~PARAM.goOn
    
    timedLogLn(['RANDOM STATE = ' int2str(PARAM.randomState) ]);
    
    % generate spike train
%     rand('state',PARAM.randomState);
%     randn('state',PARAM.randomState);    
    if exist(['../mat/afferent.rand' sprintf('%03d',PARAM.randomState) '.mat'],'file')
        load(['../mat/afferent.rand' sprintf('%03d',PARAM.randomState) '.mat'])
%         if ~exist('spikeList','var') % 'false' file
%             [spikeList afferentList] = generateSpikeTrain;
%             save(['../mat/afferent.rand' sprintf('%03d',PARAM.randomState) '.mat'],'spikeList','afferentList')
%         end
    else
        timedLogLn('Generating spike train...');   
        if PARAM.realValuedPattern
            [spikeList afferentList patternPeriod, values, times] = generateSpikeTrainWithRealValuedPattern2;
            save(['../mat/afferent.rand' sprintf('%03d',PARAM.randomState) '.mat'],'spikeList','afferentList','patternPeriod','values','times')
        else
            [spikeList afferentList] = generateSpikeTrain;
            save('-v7.3',['../mat/afferent.rand' sprintf('%03d',PARAM.randomState) '.mat'],'spikeList','afferentList')
        end
            
    end
end