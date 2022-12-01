function net = Train_Net(dlnet,X, Y, batchsize, numepochs, inital_learn_rate, decay)
% Inputs:
% dlnet - dlnetwork object to be trained
% X - training data: inputs
% Y - training data, expected outputs (ground truth data)
% batchsize - number of inputs to be fed into the neural network 
% before updating its internals
% numepochs - number of times we will train over the entire
% dataset before finishing

% Outputs:
% net - trained neural network

% monitor graph for training
monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");

% number of batches that we will have over the course of 
% training over 1 epoch
iters_per_epoch = floor(size(X, 4)/batchsize);

% current TOTAL iteration of training we are on
current_iter = 0;

% total amout of iterations over entire training
total_iter = iters_per_epoch * numepochs;

% average gradients and average squared gradients 
% of our losses for use with out update function
avgGrad = [];
avgGradSq = [];

% loop over number of epochs
for epoch = 1:numepochs
    
    % determine learning rate base on schedule params
    lrate = inital_learn_rate * exp(-decay*epoch);

    % shuffle data
    idx = randperm(size(X,4));
    X = X(:,:,:,idx);
    Y = Y(:,:,:,idx);
    
    % loop over the number of batches
    for batch = 1:iters_per_epoch
        % increment iteration
        current_iter = current_iter + 1;

        % figure out indexes of the current batch
        batchidx = ((batch-1) * batchsize + 1):(batch*batchsize);

        % convert to dlarray
        dlarr_in = dlarray(X(:,:,:,batchidx), "SSCB");
        dlarr_out = dlarray(Y(:,:,:,batchidx), "SSCB");

        % calculate loss of current net
        [loss, grads] = dlfeval(@loss_func, dlnet, dlarr_in, dlarr_out);

        % update internals of network using adam optimizer
        [dlnet, avgGrad, avgGradSq] = adamupdate(dlnet, grads, avgGrad, avgGradSq, current_iter, lrate, 0.9, 0.999);
 
        % show loss on our monitor
        recordMetrics(monitor, current_iter, Loss=loss);
        updateInfo(monitor, Epoch=epoch + " of " + numepochs);
        monitor.Progress = 100 * current_iter/total_iter;
    end
end

%set our output net to the new dlnet
net = dlnet;

end