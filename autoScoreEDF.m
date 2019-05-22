function [] = autoScoreEDF(fileName, rejFrac, skip, channel_score, channel_EEG, channel_EMG, pauseTF)

% MULTIPLE-CLASSIFIER AUTOSCORING OF EDF SLEEP RECORDINGS

% EXAMPLE: autoScoreEDF('C:\Users\Vance\Documents\MTLAB\Sleep\HLU C2_11 wk3 trained.edf', 0.05, 1, 3, 5, 6, true)

%   Read .edf files exported by PAL 8200 or PAL 8400.
%   Uses 1 channel of EEG and 1 channel of EMG as features.
%   Learns from training scores (~720 epochs) provided in the recording.
%   Autsocores the recording using a combination of LDA, SVM, NB, NN, 
%   Random-subspace kNN, Random-subspace Tree, and Bagged Tree.
%   Writes autoscores into new file.

% Vance Gao 
% Last edited 2019-05-22
% ------------------------------------------------------------------------

% To change this function back to a script, simply delete the function
% header at the top, the "end" at the bottom, and manually assign values to
% the variables in the following section.


%% Description of function's featuress

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

% https://www.edfplus.info/specs/edf.html
% .EDF file structure, using hex reader (PAL 8200 format, 1000 Hz):
% byte 0-2047: Header (ascii format)
% byte 2048-2051: unix time (4 bytes, 2051 2050 2049 2048)
% byte 2052-2053: gain (almost always 1)
% byte 2054-2055: 1st epoch score
% byte 2056-22055: 1st epoch EEG1
% byte 22056-42055: 1st epoch EEG2
% byte 42056-62055: 1st epoch EMG
% byte 62056-62855: zeroes (annotations)
% byte 62856-62859: time
% byte 62860-62861: gain
% byte 62862-62863: 2nd epoch score
% byte 62864-etc.: 2nd epoch EEG1...

% data is little-endian, twos-complement signed 16-bit integers ('shorts')
% Read into Matlab, bytes -> shorts: 
% Matlab short index *2 -2 = byte index (first of two bytes)

commandwindow

fileID = fopen(fileName);

% read recording info in header
headerBuf = fread(fileID, 100000);
frewind(fileID);

headerLength = eval( char( headerBuf(185:192))) / 2; %in shorts
nEpochs = eval( char( headerBuf(237:244)));
secPerEp = eval( char( headerBuf(245:252)));
nSignals = eval( char( headerBuf(253:256)));

signalLengths = zeros(nSignals,1);
ind = 257 + 216 * nSignals;
for s = 1:nSignals
    signalLengths(s) = eval( char( headerBuf( (ind:(ind+7)) + (s-1)*8)));
end
epShortsLength = sum(signalLengths);

% user check
disp(' ')
disp('-------------------------------------------------------------------')
disp(' ')
disp(datetime('now'))
disp(['Autoscoring file ' '''' fileName '''']);
disp(' ')
disp(['Percent of epochs to be rejected: ' num2str(rejFrac * 100) '%'])
disp(' ')
disp('Full list of signal channels detected in file:')

for s=1:nSignals
    disp([' ' num2str(s) '. ' char( headerBuf((257:272)+16*(s-1)))'])
end
disp(' ')

disp('Assigning the following signal channels to be featuress:')
disp([' Score: ' num2str(channel_score) '   EEG: ' num2str(channel_EEG) '   EMG: ' num2str(channel_EMG)])
if any([channel_score > nSignals, channel_EEG > nSignals, channel_EMG > nSignals])
    error('*ERROR*: Assigned channel numbers are higher than number of channels! \n%s',...
         'Please assign correct signal channels in this function''s arguments.')   
else
    disp(' If channel assignments are incorrect, change settings to correct signal channels in')
    disp(' this function''s arguments.') 
end
disp(' ')

trueSampPerEp = signalLengths(channel_EEG);
trueSampRate = trueSampPerEp / secPerEp;
Hz = trueSampRate/skip;
sampPerEp = numel(1:skip:trueSampPerEp);

disp(['True sampling rate: ' num2str(trueSampRate)])
disp(['Skip: ' num2str(skip)])
disp(['Effective sampling rate: ' num2str(Hz)])
disp(' ')

disp('Does setup look correct? Ctrl+C if no, any key if yes:');
if pauseTF == true
    pause
end
tic

% pre-allocate recording structure
recording.eeg2 = zeros(nEpochs, sampPerEp);
recording.emg = zeros(nEpochs, sampPerEp);
recording.scores = zeros(nEpochs,1);

% read file
disp(' ')
disp('Reading the .EDF file...')
disp(['- Shorts in header: ' num2str(headerLength)]);
disp(['- Shorts per epoch: ' num2str(epShortsLength)]);
disp(' ');

fread( fileID, headerLength, '*int16');

for k=1:nEpochs
    if mod(k,1000)==0
        disp(['Reading epoch ' num2str(k) '/' num2str(nEpochs)]);
    end
    
    epochBuf = fread( fileID, epShortsLength, '*int16');
    
    ind = sum( signalLengths(1:channel_score-1)) + 1; 
    recording.scores(k) = epochBuf(ind); %1 Wake; 2 NREM; 3 REM; 129 Wake-Ex; 130 NREM-Ex; 131 REM-Ex; 0 Artifact; 255 Unscored   
    
    ind = sum( signalLengths(1:channel_EEG-1)) + 1; 
    recording.eeg2(k,:) = epochBuf(ind: skip: ind+signalLengths(channel_EEG)-1);      
    
    ind = sum( signalLengths(1:channel_EMG-1)) + 1; 
    recording.emg(k,:) = epochBuf(ind: skip: ind+signalLengths(channel_EEG)-1);
    
end
fclose(fileID);

% Interrupt if missing scores
if any( ismember( recording.scores, [1 2 3])) == false
    disp('ERROR: No training scores in this file!')
    return
elseif sum( ismember( recording.scores, [1 2 3])) <= 360
    disp(['ERROR: Only ' num2str( sum( ismember( recording.scores, [1 2 3]))) ' epochs are scored! '...
        'Score at least 360 epochs; at least 720 is preferable.'])
    return
elseif sum( recording.scores==3) < 20
    disp(['ERROR: Only ' num2str( sum( recording.scores==3)) 'REM epochs are scored! '...
        'Score at least 20 REM epochs; at least 30 is preferable.'])
    return
end

clear k j s fileID epochBuf fileInfo skip headerBuf ind


%% Power Spectral Density for EEG2 and EMG, hour by hour

disp(' ')
disp('Performing short-time Fourier transform:') 

nfft = 2^nextpow2( sampPerEp);

eeg2P = zeros(  nEpochs, ceil( (nfft+1)/2));
emgP  = zeros( nEpochs, ceil( (nfft+1)/2));
eegRMS = zeros(nEpochs,1);

for k = 1:nEpochs
    if mod(k, 1000) == 0
        disp(['PSD: epoch ' num2str(k) '/' num2str(nEpochs)])
    end
    
    %Power spectral density
    [eeg2P(k,:), eeg2F] = periodogram( recording.eeg2(k,:), hamming(sampPerEp), [], Hz);
    [emgP( k,:), emgF ] = periodogram( recording.emg( k,:), hamming(sampPerEp), [], Hz);
    
    %total power for later artifact detection
    eegRMS(k) = rms(recording.eeg2(n,:)); 
    
end

trainingScores = recording.scores;


%% Pre-Classification Processing, Feature Extraction

% split data into logarithmic frequency bands
bands = logspace(log10(0.5), log10(Hz/2), 21);
features = zeros(nEpochs, 20);

for  b = 1:20 
    bandInds = eeg2F>=bands(b) & eeg2F<bands(b+1);
    features(:, b) = sum( eeg2P(:, bandInds), 2);
end
features(:,21) = sum( emgP(:, emgF>=4 & emgF<40), 2);

% normalize using a log transformation and smooth over time
features = conv2( log( features), fspecial( 'gaussian', [5 1], 0.75), 'same');

% clean up artifacts in training scores using outlier fence criteria
iqr_wake = quantile( eegRMS, 0.75) - quantile( eegRMS, 0.25); 
highFence_wake = quantile( eegRMS, 0.75) + 3*iqr_wake;
lowFence_wake = quantile( eegRMS, 0.25) - 3*iqr_wake;

iqr_NREM = quantile( eegRMS, 0.75) - quantile( eegRMS, 0.25); 
highFence_NREM = quantile( eegRMS, 0.75) + 3*iqr_NREM;
lowFence_NREM = quantile( eegRMS, 0.25) - 3*iqr_NREM;

iqr_REM = quantile( eegRMS, 0.75) - quantile( eegRMS, 0.25); 
highFence_REM = quantile( eegRMS, 0.75) + 3*iqr_REM;
lowFence_REM = quantile( eegRMS, 0.25) - 3*iqr_REM;

for k=1:nEpochs
    if trainingScores(k)==1
        if eegRMS(k) > highFence_wake || eegRMS(k) < lowFence_wake
            trainingScores(k) = 129; %Wake-X
        end
        
    elseif trainingScores(k)==2
        if eegRMS(k) > highFence_NREM || eegRMS(k) < lowFence_NREM
            trainingScores(k) = 255;
        end   
        
    elseif trainingScores(k)==3
        if eegRMS(k) > highFence_REM || eegRMS(k) < lowFence_REM
            trainingScores(k) = 255;
        end
    end
end

clear b bandInds


%% Train algorithm and predict remaining sleep scores

disp(' ')
disp('Auto-scoring...')

trainSet = trainingScores==1 | trainingScores==2 | trainingScores==3;

models = MCStrain( features(trainSet,:), recording.scores(trainSet));

[confidLda, confidNb, confidSvm, confidDtBag, confidDtRS, confidKnnRS] = MCSclassify( models, features);


%% Classifier Fusion

% consensus vote on confidence scores
confidCons = (confidDtBag +confidDtRS +confidKnnRS +confidLda +confidNb +confidSvm +confidNn) /7;
[~,autoScoresCons] = max(confidCons,[],2);

% rejection criteria
autoScores = autoScoresCons;

maxConfScores = max(confidCons, [], 2);

confOrder = tiedrank(maxConfScores);
rejectSet = confOrder < rejFrac*nEpochs;
autoScores(rejectSet) = 255;


%% Cleanup

% 1) Artifact removal by power thresholds, using same criteria as for
% cleaning up training epochs above
nExcludeNR = 0;
nExcludeW = 0;

if rejFrac ~= 0 %only do this if rejection fraction is not set to zero
    for k=1:nEpochs  
        if autoScores(k)==1 && ~ismember(trainingScores(k),[1 2 3 129 130 131])
            if eegRMS(k) > highFence_wake || eegRMS(k) < lowFence_wake
                autoScores(k)= 129; %Wake-X
                nExcludeW = nExcludeW+1;
            end

        elseif autoScores(k)==2 && ~ismember(trainingScores(k),[1 2 3 129 130 131])
            if eegRMS(k) > highFence_NREM || eegRMS(k) < lowFence_NREM
                autoScores(k)= 255;
                nExcludeNR = nExcludeNR+1;
            end   

        elseif autoScores(k)==3 && ~ismember(trainingScores(k),[1 2 3 129 130 131])
            if eegRMS(k) > highFence_REM || eegRMS(k) < lowFence_REM
                autoScores(k)= 255;
                nExcludeNR = nExcludeNR + 1;
            end
        end
    end
else
    for k=1:nEpochs
        if autoScores(k)==1 && ~ismember(trainingScores(k),[1 2 3 129 130 131])
            if eegRMS(k) > highFence_wake || eegRMS(k) < lowFence_wake
                autoScores(k)= 129; %Wake-X
                nExcludeW = nExcludeW+1;
            end

        elseif autoScores(k)==2 && ~ismember(trainingScores(k),[1 2 3 129 130 131])
            if eegRMS(k) > highFence_NREM || eegRMS(k) < lowFence_NREM
                autoScores(k)= 130;
                nExcludeNR = nExcludeNR+1;
            end   

        elseif autoScores(k)==3 && ~ismember(trainingScores(k),[1 2 3 129 130 131])
            if eegRMS(k) > highFence_REM || eegRMS(k) < lowFence_REM
                autoScores(k)= 131;
                nExcludeNR = nExcludeNR + 1;
            end
        end
    end
end

% 2) Manual training scores override auto-scores
nOverride = 0;

for k=1:nEpochs  
     if ismember(trainingScores(k), [0 1 2 3 129 130 131]) && trainingScores(k)~=autoScores(k)
        autoScores(k) = trainingScores(k);
        nOverride = nOverride + 1;
    end
end

% 3) Improbable sequences set to unscored
if rejFrac ~= 0  %only do this if rejection fraction is not set to zero
    autoScoresBuf = autoScores;

    for k = 2:nEpochs 
        if autoScores(k-1)==1 && autoScores(k)==3 ...
                && ~ismember(trainingScores(k), [1 2 3 129 130 131]) ...
                && ~ismember(trainingScores(k-1), [1 2 3 129 130 131])
            autoScores(k) = 255;
            autoScores(k-1) = 255;
        end

        if autoScores(k-1)==3  && autoScores(k)==2 ...
                && ~ismember(trainingScores(k), [1 2 3 129 130 131]) ...
                && ~ismember(trainingScores(k-1), [1 2 3 129 130 131])
            autoScores(k) = 255;
            autoScores(k-1) = 255;
        end
    end
end

nImprobable = sum(autoScores~=autoScoresBuf);

clear autoScoresBuf iqr_wake iqr_NREM iqr_REM highFence_wake highFence_NREM... 
    highFence-REM lowFence_wake lowFence_NREM lowFence_REM eegRMS;


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
    if mod(k,1000)==0
        disp(['Writing epoch ' num2str(k) '/' num2str(nEpochs)]);
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
disp('features file was:')
disp(['   ' fileName])
disp(' ');
disp('Autoscored file written to:')
disp(['   ' fileOutName])
disp(' ')
disp(['Finished running at ' datestr(now, 'HH:MM on mmmm dd, yyyy.')])
toc
disp(' ')
disp(['Number of training scores: ' num2str( sum(trainSet))])
disp(['Rejection rate: ' num2str(rejFrac * 100) '%'])
disp(['Number of Unscored epochs: ' num2str( sum(autoScores == 255)) '/' num2str( nEpochs) ...
    ' (' num2str( round( sum(autoScores == 255) / nEpochs, 4) * 100) '%)'])
disp(['   Number of possible NREM/REM artifacts set to Unscored: ' num2str( nExcludeNR)])
disp(['   Number of possible Wake artifacts set to Wake-X: ' num2str( nExcludeW)])
disp(['   Number of improbable sequence epochs set to Unscored: ' num2str( nImprobable)])
disp(['   Number of training scores overriding autoscores: ' num2str( nOverride)])


end