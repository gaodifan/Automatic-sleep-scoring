function autoScores = autoScoreEDF(fileName, rejFrac, skip, channel_score, channel_EEG, channel_EMG, pauseTF)

% MULTIPLE-CLASSIFIER AUTOSCORING OF EDF SLEEP RECORDINGS

% EXAMPLE: autoScoreEDF('C:\Users\Vance\Documents\MATLAB\Sleep\HLU C2_11 wk3 trained.edf', 0.05, 1, 3, 5, 6, true)

%   Read .edf files exported by PAL 8200 or PAL 8400.
%   Uses 1 channel of EEG and 1 channel of EMG as features.
%   Learns from training scores (~720 epochs) provided in the recording.
%   Autsocores the recording using a combination of LDA, SVM, NB, NN, 
%   Random-subspace kNN, Random-subspace Tree, and Bagged Tree.
%   Writes autoscores into new file.

% By: Vance Difan Gao
% Last updated 2019/11/29
% ------------------------------------------------------------------------

% To change this function back to a script, simply delete the function
% header at the top, and manually assign values to the variables in the 
% following section.


%% Description of function's inputs

% fileName:
%   The path and filename to the recording you want to score.

% rejFrac:
%   The fraction of epochs to reject, leaving unscored. For example,
%   rejFrac=0.05 will leave 5% of epochs unscored for manual review. 
%   Set to 0 for full automation.
%   features value between 0 and 1.

% skip:
%   if sampling rate is high, may keep one sample per SKIP samples to speed
%   up the Fourier transform and reduce memory usage.
%   For example, a 1000 Hz file read with skip=4 will have an effective
%   sampling rate of 250 Hz.

% channel_score:
%   The signal channel number for the sleep-stage classification.
%   A full list of all the signal channels in the file will display when 
%   the function is run.  

% channel_EEG:
%   The signal channel number for the EEG signal.
%   A full list of all the signal channels in the file will display when 
%   the function is run. 

% channel_EMG:
%   The signal channel number for the EMG channel. 
%   A full list of all the signal channels in the file will display when 
%   the function is run.

% pauseTF:
%   True-false indicating whether to pause for inspection of settings
%   before continuing. If no option is featuresed, the default is true.
%   Useful to set to false if running a multi-file loop.


%% Read the file

% readSleepEDF function to read file
recording = readSleepEDF( fileName, skip, channel_score, channel_EEG, channel_EMG, pauseTF);
tic

trainingScores = recording.scores;
trainSet = trainingScores==1 | trainingScores==2 | trainingScores==3;

% re-read header to get and save useful info
fileID = fopen( fileName);
headerBuf = fread( fileID, 100000);

headerLength = eval( char( headerBuf(185:192))) / 2; %in shorts ('int16')
nEpochs = eval( char( headerBuf(237:244)));
secPerEp = eval( char( headerBuf(245:252)));
nSignals = eval( char( headerBuf(253:256)));

signalLengths = zeros(nSignals,1);
ind = 257 + 216 * nSignals;
for s = 1:nSignals
    signalLengths(s) = eval( char( headerBuf( (ind:(ind+7)) + (s-1)*8)));
end
epShortsLength = sum(signalLengths);

trueSampPerEp = signalLengths(channel_EEG);
trueSampRate = trueSampPerEp / secPerEp;
Hz = trueSampRate/skip;
sampPerEp = numel(1:skip:trueSampPerEp);

fclose all;

% Display number of training scores and interrupt if not enough
disp(' ')
disp( ['Recording has ' num2str( sum( trainSet)) ' training scores (' ...
    num2str( round( sum( trainSet) * secPerEp / 3600, 2)) ' h).'])

if any( trainSet) == false
    disp('ERROR: No training scores in this file!')
    return
elseif sum( trainSet) <= 360
    disp(['ERROR: Only ' num2str( sum( trainSet)) ' epochs are scored. '...
        'Score at least 360 epochs; at least 720 is preferable.'])
    return
elseif sum( trainingScores==3) < 20
    disp(['ERROR: Only ' num2str( sum( recording.scores==3)) 'REM epochs are scored. '...
        'Score at least 20 REM epochs; at least 30 is preferable.'])
    return
end

clear headerBuf fileID


%% Power Spectral Density for EEG2 and EMG

disp(' ')
disp('Performing short-time Fourier transform:') 

nfft = 2^nextpow2( sampPerEp);

eegP = zeros(  nEpochs, ceil( (nfft+1)/2));
emgP  = zeros( nEpochs, ceil( (nfft+1)/2));

for k = 1:nEpochs
    % update message
    if mod(k, 1000) == 0 || k == nEpochs
        disp(['  PSD: epoch ' num2str(k) '/' num2str(nEpochs)])
    end
    
    % power spectral density
    [eegP(k,:), eegF] = periodogram( double(recording.eeg(k,:)), hamming( sampPerEp), [], Hz);
    [emgP( k,:), emgF ] = periodogram( double(recording.emg( k,:)), hamming( sampPerEp), [], Hz);
end

%total power for later artifact detection
eegTotalP = sum( recording.eeg, 2);     
    

%% Pre-Classification Processing, Feature Extraction

disp(' ')
disp('Cleaning signal and extracting features...')

% split data into logarithmic frequency bands
bands = logspace( log10( 0.5), log10( Hz/2), 21);
features = zeros( nEpochs, 20);

for  b = 1 : numel( bands) - 1
    bandInds = eegF>=bands(b) & eegF<bands(b+1);
    features(:, b) = sum( eegP(:, bandInds), 2);
end

% filter out spikes in certain EMG frequency bands
for k = 1:nEpochs
    epochEmgPsd = emgP(k, emgF>=4 & emgF<40);
    epochMedian = median( epochEmgPsd);
    upperFence = iqr( epochEmgPsd);
    epochEmgPsd( epochEmgPsd > 1.5 * upperFence) = epochMedian; 
    features(k, 21) = sum( epochEmgPsd);
end
    
% normalize using a log transformation and smooth over time
features = conv2( log( features), fspecial( 'gaussian', [5 1], 0.6), 'same');

% clean up artifacts in training scores using outlier fence criteria
iqr_wake = iqr( eegTotalP(recording.scores==1)); 
highFence_wake = quantile( eegTotalP, 0.75) + 3*iqr_wake;
lowFence_wake = quantile( eegTotalP, 0.25) - 3*iqr_wake;

iqr_NREM = iqr( eegTotalP(recording.scores==2)); 
highFence_NREM = quantile( eegTotalP, 0.75) + 3*iqr_NREM;
lowFence_NREM = quantile( eegTotalP, 0.25) - 3*iqr_NREM;

iqr_REM = iqr( eegTotalP(recording.scores==3)); 
highFence_REM = quantile( eegTotalP, 0.75) + 3*iqr_REM;
lowFence_REM = quantile( eegTotalP, 0.25) - 3*iqr_REM;

for k=1:nEpochs
    if trainingScores(k)==1
        if eegTotalP(k) > highFence_wake || eegTotalP(k) < lowFence_wake
            trainingScores(k) = 129; %Wake-X
        end
        
    elseif trainingScores(k)==2
        if eegTotalP(k) > highFence_NREM || eegTotalP(k) < lowFence_NREM
            trainingScores(k) = 255;
        end   
        
    elseif trainingScores(k)==3
        if eegTotalP(k) > highFence_REM || eegTotalP(k) < lowFence_REM
            trainingScores(k) = 255;
        end
    end
end

clear b bandInds


%% Train algorithm and predict remaining sleep scores

disp(' ')
disp('Auto-scoring...')

models = MCStrain( features(trainSet,:), recording.scores(trainSet));

[confidLda, confidNb, confidSvm, confidDtBag, confidDtRS, confidKnnRS] = MCSclassify( models, features);


%% Classifier Fusion

% consensus vote on confidence scores
confidCons = (confidDtBag +confidDtRS +confidKnnRS +confidLda +confidNb +confidSvm) / 6;
[~,autoScoresCons] = max(confidCons,[],2);

% rejection criteria
autoScores = autoScoresCons;

maxConfScores = max(confidCons, [], 2);

confOrder = tiedrank(maxConfScores);
rejectSet = confOrder < rejFrac*nEpochs;
autoScores(rejectSet) = 255;


%% Cleanup

disp(' ')
disp('Cleaning up artifacts...')

% 1) Artifact removal by power thresholds, using same criteria as for
% cleaning up training epochs above
nExcludeNR = 0;
nExcludeW = 0;

if rejFrac ~= 0 %only do this if rejection fraction is not set to zero
    for k=1:nEpochs
        %Wake, set to Wake-X
        if autoScores(k)==1 && ~ismember( trainingScores(k), [1 2 3 129 130 131])
            if eegTotalP(k) > highFence_wake || eegTotalP(k) < lowFence_wake
                autoScores(k)= 129;
                nExcludeW = nExcludeW+1;
            end
        %NREM, set to Unscored
        elseif autoScores(k)==2 && ~ismember( trainingScores(k), [1 2 3 129 130 131])
            if eegTotalP(k) > highFence_NREM || eegTotalP(k) < lowFence_NREM
                autoScores(k)= 255;
                nExcludeNR = nExcludeNR+1;
            end   
        %REM, set to Unscored
        elseif autoScores(k)==3 && ~ismember( trainingScores(k), [1 2 3 129 130 131])
            if eegTotalP(k) > highFence_REM || eegTotalP(k) < lowFence_REM
                autoScores(k)= 255;
                nExcludeNR = nExcludeNR + 1;
            end
        end
    end
else %if rejection fraction is 0
    for k = 1 : nEpochs
        %Wake, set to Wake-X
        if autoScores(k)==1 && ~ismember( trainingScores(k),[ 1 2 3 129 130 131])
            if eegTotalP(k) > highFence_wake || eegTotalP(k) < lowFence_wake
                autoScores(k)= 129; %Wake-X
                nExcludeW = nExcludeW+1;
            end
        %NREM, set to NREM-X
        elseif autoScores(k)==2 && ~ismember( trainingScores(k), [1 2 3 129 130 131])
            if eegTotalP(k) > highFence_NREM || eegTotalP(k) < lowFence_NREM
                autoScores(k)= 130;
                nExcludeNR = nExcludeNR+1;
            end   
        %REM, set to REM-X
        elseif autoScores(k)==3 && ~ismember( trainingScores(k), [1 2 3 129 130 131])
            if eegTotalP(k) > highFence_REM || eegTotalP(k) < lowFence_REM
                autoScores(k)= 131;
                nExcludeNR = nExcludeNR + 1;
            end
        end
    end
end

% 2) Manual training scores override auto-scores
nOverride = 0;

for k = 1 : nEpochs  
     if ismember(trainingScores(k), [0 1 2 3 129 130 131]) && trainingScores(k)~=autoScores(k)
        autoScores(k) = trainingScores(k);
        nOverride = nOverride + 1;
    end
end

% 3) Improbable sequences set to unscored
if rejFrac ~= 0  %only do this if rejection fraction is not set to zero
    autoScoresBuf = autoScores;
    
    for k = 2 : nEpochs 
        %Wake follwed by REM
        if autoScores(k-1)==1 && autoScores(k)==3 ...
                && ~ismember(trainingScores(k), [1 2 3 129 130 131]) ...
                && ~ismember(trainingScores(k-1), [1 2 3 129 130 131])
            autoScores(k) = 255;
            autoScores(k-1) = 255;
        end
        %REM followed by NREM
        if autoScores(k-1)==3  && autoScores(k)==2 ...
                && ~ismember(trainingScores(k), [1 2 3 129 130 131]) ...
                && ~ismember(trainingScores(k-1), [1 2 3 129 130 131])
            autoScores(k) = 255;
            autoScores(k-1) = 255;
        end
    end
    nImprobable = sum( autoScores ~= autoScoresBuf);

else
    nImprobable = 0;
end

% 4) Another scan for Wake artifacts
% Wake-X if filtered signal doesn't cross 0 for more than 0.6 s
thres = sampPerEp / secPerEp * 0.6;

for k = 1 : nEpochs
    %update message
    if mod(k, 1000) == 0 || k == nEpochs
        disp(['  Scan for Wake artifacts: ' num2str(k) '/' num2str(nEpochs)])
    end
    
    %skip epoch if not a Wake epoch
    if autoScores(k) ~= 1
        continue
    end
    
    % initial scan on unfiltered signal (filtering is slow)
    count=1;
    signs = sign( recording.eeg(k,:));
    for s = 2 : sampPerEp
        if signs(s) == signs(s-1)
            count = count + 1;
        else
            count = 1;
        end
        
        if count > thres
            break
        end
    end
    
    % if is marked artifact on unfiltered signal, test on filtered signal
    if count > thres
        filtered = highpass( recording.eeg(k,:), 0.5, 256);
        signs = sign( filtered);
        count = 1;
        
        for s = 2 : sampPerEp
            if signs(s) == signs(s-1)
                count = count + 1;
            else
                count = 1;
            end

            if count > thres
                autoScores(k) = 129;
                nExcludeW = nExcludeW + 1;
                break
            end
        end
    end
end

clear autoScoresBuf iqr_wake iqr_NREM iqr_REM highFence_wake highFence_NREM... 
    highFence-REM lowFence_wake lowFence_NREM lowFence_REM eegRMS thres s;


%% Write auto-scores into new .edf copy

disp(' ')
disp('Writing new file...')

% create new .edf, append data epoch by epoch
fileID = fopen(fileName);
fileOutName = [fileName(1:length(fileName)-4) ' ' num2str(round(1-rejFrac,2)) 'Auto.edf'];
if rejFrac~=0
    fileOutName = [fileOutName(1:end-4) ' needsFill.edf'];
end
fopen(fileOutName, 'w');
fileOutID = fopen(fileOutName, 'a');

dataBuf = fread(fileID, headerLength, '*int16');
fwrite(fileOutID, dataBuf, 'int16', 'l');
ind = sum( signalLengths(1:channel_score-1)) + 1;

for k=1:nEpochs
    if mod(k,1000)==0 || k == nEpochs
        disp(['  Writing epoch ' num2str(k) '/' num2str(nEpochs)]);
    end
    
    dataBuf = fread(fileID, epShortsLength, '*int16');
    dataBuf(ind) = autoScores(k);
    fwrite(fileOutID, dataBuf, 'int16', 'l');
    
end

%create supplemental files with training scores and confidence values
supplFileName = [fileName(1:(end-4)) ' training.csv'];
dlmwrite(supplFileName, trainingScores, ',')

supplFileName = [fileName(1:(end-4)) ' confidences.csv'];
dlmwrite(supplFileName, [autoScores confidCons], ',')

fclose all;
clear dataBuf headerLegnth epShortsLength fileID file2ID k


%% Closing Messages

disp(' ')
disp('Done!')
disp(' ')
disp('Training file was:')
disp(['   ' fileName])
disp(' ');
disp('Autoscored file written to:')
disp(['   ' fileOutName])
disp(' ')
disp(['Finished running at ' datestr(now, 'HH:MM on mmmm dd, yyyy.')])
toc
disp(' ')
disp(['Number of training scores: ' num2str( sum( trainSet))])
disp(['Rejection rate: ' num2str( rejFrac * 100) '%'])
disp(['Number of Unscored epochs: ' num2str( sum( autoScores == 255)) '/' num2str( nEpochs) ...
    ' (' num2str( round( sum(autoScores == 255) / nEpochs, 4) * 100) '%)'])
if rejFrac == 0
    disp(['   Number of possible NREM/REM artifacts set to NREM-X/REM-X: ' num2str( nExcludeNR)])
else
    disp(['   Number of possible NREM/REM artifacts set to Unscored: ' num2str( nExcludeNR)])
end
disp(['   Number of possible Wake artifacts set to Wake-X: ' num2str( nExcludeW)])
disp(['   Number of improbable sequence epochs set to Unscored: ' num2str( nImprobable)])
disp(['   Number of training scores overriding autoscores: ' num2str( nOverride)])
