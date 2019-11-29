function [recording, varargout] = readSleepEDF( fileName, skip, channel_score, channel_EEG, channel_EMG, pauseTF)

% READ .EDF FORMAT FILES AND STORE SLEEP SCORES, EEG, AND EMG

% EXAMPLE: readSleepEDF('C:\Users\Vance\Documents\MATLAB\Sleep\HLU C2_11 wk3 trained.edf', 1, 3, 5, 6, true)

%   Read .edf files exported by PAL 8200 or PAL 8400.
%   Stores sleep recordings, one channel of EEG, and one channel of EMG
%   Optionally output header as well [recording, header] = readSleepEDF(...)

% Details of the EDF format at https://www.edfplus.info/specs/edf.html

% Example .EDF file structure, using hex reader (PAL 8200 format, 1000 Hz):
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
% Read into MATLAB, bytes -> shorts: 
% MATLAB short index *2 -2 = byte index (first of two bytes)

% By: Vance Difan Gao
% Last updated 2019/11/29


%% Description of function's inputs

% fileName:
%   The path and filename to the recording you want to score.

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


%% read recording info in header

commandwindow

fileID = fopen(fileName);
header = fread(fileID, 100000);
frewind(fileID);

headerLength = eval( char( header(185:192))) / 2; %in shorts
nEpochs = eval( char( header(237:244)));
secPerEp = eval( char( header(245:252)));
nSignals = eval( char( header(253:256)));

signalLengths = zeros(nSignals,1);
ind = 257 + 216 * nSignals;
for s = 1:nSignals
    signalLengths(s) = eval( char( header( (ind:(ind+7)) + (s-1)*8)));
end
epShortsLength = sum(signalLengths);

header = header(1:headerLength*2);
varargout{1}  = header;


%% user check
disp(' ')
disp('-------------------------------------------------------------------')
disp(' ')
disp('Full list of signal channels detected in file:')

for s=1:nSignals
    disp([' ' num2str(s) '. ' char( header((257:272)+16*(s-1)))'])
end
disp(' ')

disp('Assigning the following signal channels to be featuress:')
disp([' Score: ' num2str(channel_score) '   EEG: ' num2str(channel_EEG) '   EMG: ' num2str(channel_EMG)])
if any([channel_score > nSignals, channel_EEG > nSignals, channel_EMG > nSignals])
    error('*ERROR*: Assigned channel numbers are higher than number of channels! \n%s',...
         'Please assign correct signal channels in this function''s arguments.')   
else
    disp(' ') 
    disp(' If channel assignments are incorrect, change settings to correct signal channels in')
    disp(' this function''s arguments.') 
end
disp(' ')

trueSampPerEp = signalLengths(channel_EEG);
trueSampRate = trueSampPerEp / secPerEp;
Hz = trueSampRate/skip;
sampPerEp = numel( 1 : skip : trueSampPerEp);

disp(['True sampling rate: ' num2str(trueSampRate)])
disp(['Skip: ' num2str(skip)])
disp(['Effective sampling rate: ' num2str(Hz)])
disp(' ')

disp('Does setup look correct? Ctrl+C if no, any key if yes:');
if pauseTF == true
    pause
end


%% Read the recording

% pre-allocate recording structure
recording.eeg = zeros(nEpochs, sampPerEp);
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
    if mod(k,1000)==0 || k == nEpochs
        disp(['Reading epoch ' num2str(k) '/' num2str(nEpochs)]);
    end
    
    epochBuf = fread( fileID, epShortsLength, '*int16');
    
    ind = sum( signalLengths(1:channel_score-1)) + 1; 
    recording.scores(k) = epochBuf(ind); %1 Wake; 2 NREM; 3 REM; 129 Wake-Ex; 130 NREM-Ex; 131 REM-Ex; 0 Artifact; 255 Unscored   
    
    ind = sum( signalLengths(1:channel_EEG-1)) + 1; 
    recording.eeg(k,:) = epochBuf(ind: skip: ind+signalLengths(channel_EEG)-1);      
    
    ind = sum( signalLengths(1:channel_EMG-1)) + 1; 
    recording.emg(k,:) = epochBuf(ind: skip: ind+signalLengths(channel_EEG)-1);
    
end
fclose(fileID);
