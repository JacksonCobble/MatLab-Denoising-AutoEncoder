clearvars

% load image, convert to grayscale
myimg = rgb2gray(im2double(imread('test.png')));

%load net
load('gaussian_net.mat');
myimg_gauss = imnoise(myimg, 'gaussian', 0.15, 0.04);

%predict and plot
myimg_out = predict(trained_net, dlarray(myimg_gauss(:,:), "SSCB"));
figure()
subplot(1,3,1)
imshow(myimg)
title('original image')
subplot(1,3,2)
imshow(myimg_gauss)
title('noise added')
subplot(1,3,3)
imshow(extractdata(myimg_out))
title('reconstruction')