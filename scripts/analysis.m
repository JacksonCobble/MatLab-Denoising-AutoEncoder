clearvars

% load original dataset
load('original.mat')

% load info for generic autoencoder
load("autoencoder.mat")
% plot examples
plotNetEx(trained_net, XTrain, XTrain, "Generic Autoencoder Model");

clearvars -except XTrain

% load info for gaussian autoencoder
load("gaussian.mat")
load("gaussian_net.mat")
% plot examples
plotNetEx(trained_net, XTrain, noise_gaussian, "Gaussian Noise Model");

clearvars -except XTrain

% load info for salt and pepper autoencoder
load("sandp.mat")
load("sandp_net.mat")
% plot examples
plotNetEx(trained_net, XTrain, noise_sandp, "Salt and Pepper Noise Model Noise Model");

clearvars -except XTrain

% load info for speckle autoencoder
load("speckle.mat")
load("speckle_net.mat")
% plot examples
plotNetEx(trained_net, XTrain, noise_speckle, "Speckle Noise Model");

