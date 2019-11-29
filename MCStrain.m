function models = MCStrain( features, labels) 

% TRAIN A MULTIPLE CLASSIFIER SYSTEM

%   Trains a MCS classifier given training data, and outputs the prediction
%   models in a structure.
%   Use in conjunction with MCSclassify.m.

% By: Vance Difan Gao
% Last updated 2019/11/29


%% Set parameters for enesemble learning

nLearners = 100; %number of individual classifiers to use in ensemble methods
subFeaturesFrac = 0.5; %in subspace method, what proportion of features to use for each individual classifier
featuresFixed = 21; %in subspace method, what features to always use


%% Classification by Individual Methods

disp(' ')
disp('Training models...')

% Linear Discriminant Analysis
disp('  Linear Discriminant Analysis')
modelLda = fitcdiscr( features, labels);

% Naive Bayes
disp('  Naive Bayes')
modelNb = fitcnb( features,labels);

% Support Vector Machine -- one against all
disp('  Support Vector Machine')

classes = unique( labels);
nClasses = length( classes);
outLabel = max( classes) + 1;

modelSvm = cell(nClasses, 1);

for c = 1 : nClasses
    class = classes(c);
    
    labelsSvm = labels;
    labelsSvm(labelsSvm ~= class) = outLabel;
    modelSvm{c} = fitPosterior( fitcsvm( features, labelsSvm, 'BoxConstraint', 1));
end 


%% Classification by Ensemble Methods

% Bagged Decision Tree
disp(['  Bagged Decision Tree (ensemble of ' num2str( nLearners) '):'])
modelDtBag = fitensemble( features, num2str(labels), 'Bag', nLearners, 'tree', 'type', 'classification');

% Random Subspace Decision Tree and k-NN
disp(['  Random Subspace Tree and k-NN (ensemble of ' num2str( nLearners) '):'])

featuresInds = 1:size(features, 2);
nFeaturesFixed = numel( featuresFixed);
featuresUnfixed = setdiff( featuresInds, featuresFixed);
nFeaturesUnfixed = numel( featuresUnfixed);
nFeaturesToSelect = round( subFeaturesFrac * nFeaturesUnfixed); 
nSubFeatures = nFeaturesToSelect + nFeaturesFixed;

modelDtRS = cell( nLearners, 1);
modelKnnRS = cell( nLearners, 1);
subspaceList = zeros( nLearners, nSubFeatures);

for learner = 1:nLearners
    if mod( learner, 5) == 0
        disp(['Training subspace ensemble: ' num2str(learner) ' / ' num2str( nLearners)])
    end
    
    subspace = sort( [randsample( featuresUnfixed, nFeaturesToSelect, false) featuresFixed]);
    
    modelDtRS{learner} = fitctree( features(:,subspace), num2str( labels));        
    modelKnnRS{learner} = fitcknn( features(:,subspace), labels);

    subspaceList(learner, :) = subspace;
    
end


%% Put in output structure
    
models.classes = classes;
models.Lda = modelLda;
models.Nb = modelNb;
models.Svm = modelSvm;
models.DtBag = modelDtBag;
models.DtRS = modelDtRS;
models.KnnRS = modelKnnRS;
models.subspaceList = subspaceList;
