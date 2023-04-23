clear all
clc 

%Data Input
matlabroot = 'C:\Users\flex 5\OneDrive\Documents\Kopsi 2023\CNN'
digitDatasetPath = fullfile(matlabroot,'dataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

imds = splitEachLabel(imds, minSetCount, 'randomized');
countEachLabel(imds);

%Class Categorial
Acne = find(imds.Labels == 'Acne', 1);
Hairloss = find(imds.Labels == 'Hairloss',1);
NailFungus = find(imds.Labels == 'Nail Fungus',1);
Normal = find(imds.Labels == 'Normal',1);
SkinAllergy = find(imds.Labels == 'Skin Allergy',1);

figure
subplot(2,2,1);
imshow(readimage(imds,Acne));
subplot(2,2,2);
imshow(readimage(imds, Hairloss));
subplot(2,2,3);
imshow(readimage(imds, NailFungus));
subplot(2,2,4);
imshow(readimage(imds, SkinAllergy));
%Bentuk Arsitektur Resnet-50 dalam CNN
net = resnet50();
figure
plot(net)
title('Arsiktetur Dari ResNet-50')
set(gca, 'YLim', [150 170]);

net.Layers(1);
net.Layers(end);

numel(net.Layers(end).ClassNames);
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomized');

imageSize = net.Layers(1).InputSize;

augmentedTrainingDatastore = augmentedImageDatastore(imageSize,... 
    trainingSet, "ColorPreprocessing", 'gray2rgb' );
augmentedTestDatastore = augmentedImageDatastore(imageSize,... 
    testSet, "ColorPreprocessing", 'gray2rgb' );
W1 = net.Layers(2).Weights;
W1 = mat2gray(W1);

figure
montage(W1)
title('Konvolusi Pertama')

featureLayer = 'fc1000';
trainingFeatures = activations(net,...
    augmentedTrainingDatastore, featureLayer, 'MiniBatchSize',...
    32, 'OutputAs', 'columns');

trainingLables = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLables,...
    'Learners','Linear', 'Coding','onevsall',...
    'ObservationsIn', 'columns');
testFeatures = activations(net,...
    augmentedTestDatastore, featureLayer, 'MiniBatchSize',...
    32, 'OutputAs', 'columns');

PredictLabels = predict(classifier, testFeatures,...
    'ObservationsIn', 'columns');
%hitungan confusion matrix dari image
testLables = testSet.Labels;
confMatrix1 = confusionmat(testLables, PredictLabels);
confMatrix = bsxfun(@rdivide, confMatrix1, sum(confMatrix1,2));
  
accuracy = mean(diag(confMatrix))

% Presisi (precision)
precision = diag(confMatrix) ./ sum(confMatrix, 1)'

% Sensitivitas (recall atau true positive rate)
sensitivity = diag(confMatrix) ./ sum(confMatrix, 2)

% F1-score
f1Score = 2 * (precision .* sensitivity) ./ (precision + sensitivity)

figure
tabel_matrix = confusionchart(round(confMatrix1), categories(testLables));
tabel_matrix.NormalizedValues
tabel_matrix.Title = 'Skin Disease Classification Using CNN';
tabel_matrix.RowSummary = 'row-normalized';
tabel_matrix.ColumnSummary = 'column-normalized';

%section mengklasifikasi image dari iput gambar yang serupa dengan- 
%penyakit yang sudah terdapat datanya
newImage = imread(fullfile('acne.jpeg'));

ds = augmentedImageDatastore(imageSize,... 
    newImage, "ColorPreprocessing", 'gray2rgb' );

imageFeatures = activations(net,...
    ds, featureLayer, 'MiniBatchSize',...
    32, 'OutputAs', 'columns');

Label = predict(classifier, imageFeatures,...
    'ObservationsIn', 'columns');
sprintf('The loaded image belongs to %s class', Label)
save('trained_classifier.mat')



