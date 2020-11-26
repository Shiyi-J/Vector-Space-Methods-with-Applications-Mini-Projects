function [X_tr, X_te, y_tr, y_te] = extract_and_split(mnist, d, test_size)
% Arguments:
%     mnist -- the MNIST data set read from csv file
%     d -- digit needs to be extracted, can be 0, 1, ..., 9
%     test_size -- the fraction of testing set
% Returns:
%     X_tr -- training set features, a matrix with 785 columns
%             each row corresponds the feature of a sample
%     y_tr -- training set labels, 1d-array
%             each element corresponds the label of a sample
%     X_te -- testing set features, a matrix with 785 columns 
%             each row corresponds the feature of a sample
%     y_te -- testing set labels, 1d-array
%             each element corresponds the label of a sample
    n = size(mnist, 1);
    count = 0;
    for i = 1:n
        if mnist(i, 1) == d
            count = count + 1; % find number of samples of a specific digit
        end
    end
    L = mnist(:, 1) == d; % filter index
    X = zeros(count, 785);
    X = mnist(L, 2:785); % get features
    X(:,785) = (-1) * ones(count, 1); % append -1 at the end
    y = d * ones(count, 1); % generate labels
    c = cvpartition(count, 'HoldOut', test_size);
    tr_idx = training(c);
    te_idx = test(c);
    X_tr = X(tr_idx,:);
    X_te = X(te_idx,:);
    y_tr = y(tr_idx);
    y_te = y(te_idx);
end

function [err_tr, err_te] = mnist_pairwise_LS(mnist, a, b, test_size, verbose)
% Pairwise experiment for applying least-square to classify digit a, b
% Arguments:
%     mnist -- the MNIST data set read from csv file
%     a, b -- digits to be classified
%     test_size -- the fraction of testing set
%     verbose -- whether to print and plot results
% Returns:
%     err_tr -- training set classification error
%     err_te -- testing set classification error

    % Find all samples labeled with digit a and split into train/test sets
    [Xa_tr, Xa_te, ya_tr, ya_te] = extract_and_split(mnist, a, test_size);
    
    % Find all samples labeled with digit b and split into train/test sets
    [Xb_tr, Xb_te, yb_tr, yb_te] = extract_and_split(mnist, b, test_size);
    
    % Construct the full training set and map a labels to -1 and b labels to +1
    X_tr = [Xa_tr; Xb_tr];
    y_tr = [(-1) * ones(size(ya_tr, 1), 1); ones(size(yb_tr, 1), 1)];
    
    % Construct the full testing set
    X_te = [Xa_te; Xb_te];
    y_te = [(-1) * ones(size(ya_te, 1), 1); ones(size(yb_te, 1), 1)];
    
    % Run least-square on training set
    z_hat = X_tr \ y_tr; % trained classifier
    
    % Compute estimation and misclassification on training set
    % Compute estimation and misclassification on testing set
    y_hat_tr = sign(X_tr * z_hat); % elements in labels either -1 or +1
    y_hat_te = sign(X_te * z_hat);
    
    err_tr = mean(y_tr ~= y_hat_tr);
    err_te = mean(y_te ~= y_hat_te);
    
    if verbose
        fprintf('Pairwise experiment, mapping %d to -1, mapping %d to 1\n', a, b)
        fprintf('training error = %.2f%%, testing error = %.2f%%\n', 100 * err_tr, 100 * err_te)
        
        % Compute confusion matrix
        cm_te = confusionmat(y_te, y_hat_te);
        cm_tr = confusionmat(y_tr, y_hat_tr);
        
        fprintf('Confusion matrix:\n')
        disp(cm_te)
        disp(cm_tr)
        
        % Compute the histogram of the function output separately for each 
        % class, then plot the two histograms together
        ya_te_hat = Xa_te * z_hat;
        yb_te_hat = Xb_te * z_hat;
        
        % Remove outlier to make pretty histogram
        ya_te_hat = remove_outlier(ya_te_hat);
        yb_te_hat = remove_outlier(yb_te_hat);
        
        figure(2)
        hold on
        histogram(ya_te_hat, 50, 'facecolor', 'red')
        histogram(yb_te_hat, 50, 'facecolor', 'blue')
        title(sprintf('histogram of pairwise experiment on %d and %d', a, b))
    end
end

function [err_tr, err_te] = mnist_onehot_LS(mnist, test_size)
% Experiment for applying least-square to classify all digits using one-hot 
% encoding
% Arguments:
%     mnist -- the MNIST data set read from csv file
%     test_size -- the fraction of testing set
% Returns:
%     err_tr -- training set classification error
%     err_te -- testing set classification error
    % Split into training/testing set
    N = size(mnist, 1);
    c = cvpartition(N, 'HoldOut', test_size);
    % Construct the training set
    tr_idx = training(c);
    X_tr = zeros(N * (1-test_size), 785);
    X_tr = mnist(tr_idx, 2:785);
    X_tr(:, 785) = (-1) * ones(N * (1-test_size), 1); % append -1 at the end
    y_tr = mnist(tr_idx, 1);
    % Construct the testing set
    te_idx = test(c);
    X_te = zeros(N * test_size, 785);
    X_te = mnist(te_idx, 2:785);
    X_te(:, 785) = (-1) * ones(N * test_size, 1); % append -1 at the end
    y_te = mnist(te_idx, 1);
    % Apply one-hot encoding to training labels
    y_tr = y_tr + ones(N * (1-test_size), 1); % increase label by 1 to satisfy matrix indexing
    y_te = y_te + ones(N * test_size, 1);
    Y_tr_t = full(ind2vec(y_tr', 10));
    % Run least-square on training set
    Z = X_tr \ Y_tr_t';
    % Compute estimation and misclassification on training set
    [~, I_tr] = max(transpose(X_tr * Z));
    y_hat_tr = transpose(I_tr);
    err_tr = mean(y_tr ~= y_hat_tr);
    
    % Compute estimation and misclassification on training set
    [~, I_te] = max(transpose(X_te * Z));
    y_hat_te = transpose(I_te);
    err_te = mean(y_te ~= y_hat_te);
    
    fprintf('training error = %.2f%%, testing error = %.2f%%\n', 100 * err_tr, 100 * err_te)
    % Compute confusion matrix
    cm_tr = confusionmat(y_tr, y_hat_tr);
    cm_te = confusionmat(y_te, y_hat_te);
    fprintf('Confusion matrix:\n')
    disp(cm_tr)
    disp(cm_te)
    figure(4)
    imagesc(cm_tr)
    imagesc(cm_te)
    axis('equal')
    title('Confusion matrix of multi-class classification')
end
