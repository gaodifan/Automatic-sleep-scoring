function [confidLda, confidNb, confidSvm, confidDtBag, confidDtRS, confidKnnRS] = MCSclassify( models, features)

% USE MCS TO MAKE CLASSIFICATIONS

%   As input, takes models structure outputted by MCStrain.m and sample
%   features.
%   Output are confidence scores for each individual classification method.
%   Can sum and max confidence scores to find overall class classification.

% By: Vance Difan Gao
% Last updated 2019/12/18


%% Prediction by individual methods

disp(' ')
disp('Predicting classifications...')

classes = models.classes;
nClasses = numel( classes);

nSamples = size( features, 1);
nLearners = numel( models.DtRS);

% Linear Discriminant Analysis
disp('  Linear Discriminant Analysis')
[~, confidLda] = predict( models.Lda, features);

%Naive Bayes
disp('  Naive Bayes')
[~, confidNb] = predict( models.Nb, features);

% Support Vector Machine
disp('  Support Vector Machine')

confidSvm = zeros(nSamples, nClasses);

for c = 1:nClasses
    [~, confidSvmBuf] = predict( models.Svm{c}, features);
    confidSvm(:,c) = confidSvmBuf(:,1); 
end
    
for k = 1:nSamples
    confidSvm(k,:) = confidSvm(k,:) / sum( confidSvm(k,:));
end

%% Prediction by ensemble methods
% Bagged Decision Tree
disp(['  Bagged Decision Tree (ensemble of ' num2str( nLearners) '):'])
[~, confidDtBag] = predict( models.DtBag, features);


%Random Subspace Decision Tree and K-nearest Neighbors
disp(['  Random Subspace Tree and k-NN (ensemble of ' num2str( nLearners) '):']) 

votesDtRS = zeros( nSamples, nLearners);
votesKnnRS = zeros( nSamples, nLearners);

for learner = 1:nLearners
    if mod(learner,5) == 0
        disp(['  Predicting using subspace ensemble: ' num2str(learner) ' / ' num2str( nLearners)])
    end
    
    votesDtRS(:,learner) = str2num( predict( models.DtRS{learner}, features(:, models.subspaceList(learner,:))));
    votesKnnRS(:,learner) = predict( models.KnnRS{learner}, features(:, models.subspaceList(learner,:)));
end

confidDtRS = zeros(nSamples, nClasses);
confidKnnRS = zeros(nSamples, nClasses);

for c = 1:nClasses
    confidDtRS(:,c) = sum( votesDtRS==classes(c), 2); 
    confidKnnRS(:,c) = sum( votesKnnRS==classes(c), 2); 
end

confidDtRS = confidDtRS / nLearners;
confidKnnRS = confidKnnRS / nLearners;
