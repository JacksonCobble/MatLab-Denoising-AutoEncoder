function plotNetEx(net, orig, noise, figureTitle)
% Inputs:
% net - trained network
% orig - original, unedited data
% noise - data with noise

% Outputs: 
% none

% get random indexes for our 4 examples to plot
numImages = size(orig, 4);
exIdx = randi([0 numImages], 1, 4);

figure('Name', figureTitle)
hold on


% loop over groups of subplots
for i = 1:numel(exIdx)

    % plot original image data
    subplot(4,3,(3*i-2))
    imshow(orig(:,:,:,exIdx(i)))
    title(strcat('Original Image #', int2str(i)))

    % plot noisy image
    subplot(4,3,(3*i-1))
    imshow(noise(:,:,:,exIdx(i)))
    title(strcat('Autoencoder Input #', int2str(i)))

    %use neural net to predict output for noisy image, plot it
    subplot(4,3,3*i)
    output = predict(net, dlarray(noise(:,:,:,exIdx(i)), "SSCB"));
    imshow(extractdata(output));
    title(strcat('Reconstructed Image Output #', int2str(i)));

end
end