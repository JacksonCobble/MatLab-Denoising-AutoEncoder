function [loss, gradients] = loss_func(net, X, T)
% Inputs:
% net - dlnetwork object to be fed inputs
% X - training data
% T - truth data

% Outputs:
% loss - loss of our net output vs truth data using binary crossentropy
% gradients - gradient of loss (partial derivative)

%feed input to neural net, get output
Output = forward(net, X);

%find loss of output using binary crossentropy
loss = crossentropy(Output, T, 'TargetCategories', 'independent');

%compute gradient of loss 
gradients = dlgradient(loss, net.Learnables);
end