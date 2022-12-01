clearvars
close all

%% Load from the Digits Dataset
% [XTrain,~,~] = digitTrain4DArrayData;
XTrain = processImagesMNIST('train-images-idx3-ubyte.gz');
numImages = size(XTrain, 4);

noise_gaussian = zeros(28,28,1,numImages);
noise_sandp = zeros(28,28,1,numImages);
noise_speckle = zeros(28,28,1,numImages);

%% Loop over all images
for i = 1:numImages
    data = XTrain(:,:,:,i);
    %add 3 types of noise
    gaussian = imnoise(data, 'gaussian', 0.15, 0.04);
    sandp = imnoise(data, 'salt & pepper', 0.2);
    speck = imnoise(data, 'speckle', 0.25);
    %save images
    noise_gaussian(:,:,:,i) = gaussian;
    noise_sandp(:,:,:,i) = sandp;
    noise_speckle(:,:,:,i) = speck;
end

%% Save data 
save 'data/original.mat' XTrain
save 'data/gaussian.mat' noise_gaussian
save 'data/sandp.mat' noise_sandp
save 'data/speckle.mat' noise_speckle