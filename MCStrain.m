function models = MCStrain( features, labels) 
% Description

%% Set parameters for enesemble learning

nLearners = 100;
subFeaturesFrac = 0.5;
featuresFixed = 21;


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
    if mod(learner,5) == 0
        disp(learner)
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

    
end