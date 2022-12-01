clearvars

%load our data
load('gaussian.mat')
%load('sandp.mat')
%load('speckle.mat')
load('original.mat')

% specify layers for our neural net
layers = [...
    % input layer
    imageInputLayer([28 28 1], 'Normalization', 'none')

    % encoder - last pooling layer has an output of 5x5, which is the
    % encoded format of the 28x28 image
    convolution2dLayer([3 3], 32, 'Padding', 'same')
    leakyReluLayer
    maxPooling2dLayer([2 2], 'Stride', 2, 'Padding', 'same')

    convolution2dLayer([3 3], 32, 'Padding', 'same')
    leakyReluLayer
    maxPooling2dLayer([2 2], 'Stride', 2, 'Padding', 'same')

    %decoder
    transposedConv2dLayer([3 3], 32, 'Stride', 2, 'Cropping', 'same')
    leakyReluLayer

    transposedConv2dLayer([3 3], 32, 'Stride', 2, 'Cropping', 'same')
    leakyReluLayer

    convolution2dLayer([3 3], 1, 'Padding', 'same')
    sigmoidLayer('Name', 'output')
    ];

% turn layers list into a dlnet object for training
lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);

%train network
trained_net = Train_Net(dlnet, noise_gaussian, XTrain, 150, 10, 0.02, 0.015);

%show an example
figure()
subplot(2,1,1)
imshow(noise_gaussian(:,:,:,42365))
out = predict(trained_net,  dlarray(noise_gaussian(:,:,:,42365), "SSCB"));
subplot(2,1,2)
imshow(extractdata(out))

