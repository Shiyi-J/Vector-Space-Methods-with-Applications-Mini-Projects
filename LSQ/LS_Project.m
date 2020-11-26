% MATLAB Mini-Project Stub for Least-Squares
%   See function stubs after script

    clc
    close all
    warning('off')
    
    % Load data and avoid reloading every time
    if ~exist('mnist_load','var')
        mnist = csvread('mnist_train.csv', 1, 0);
        mnist_load=1;
    end

    %% Problem 2.1
    % When $n=1$, we can fit a degree-$m$ polynomial by choosing 
    % $f_j(x)=x^{j-1}$ and $M=m+1$. In this case, it follows that 
    % $A_{i,j}=x_{i}^{j-1}$ and the matrix $A$ is called a Vandermonde 
    % matrix.
    %
    % Write a function to create Vandermonde matrix **(5pt)**
    x = 1:10;
    fprintf('\nProblem 2.1\n')
    fprintf('Vandermonte matrix for 1:10 up to degree 3 is\n')
    disp(create_vandermonde(x', 3))

    %% Problem 2.2
    % Write a function to solve least-square problem **(5pt)**
    % Using the setup in the previous example, try fitting the points 
    % (1,2),(2,3),(3,5),(4,7),(5,11),(6,13) to a degree-2 polynomial.
    % 
    % Compute the minimum squared error. **(5pt)**
    % 
    % Plot this polynomial (for $x\in[0,7]$) along with the data points to 
    % see the quality of fit. **(5pt)**
    fprintf('\nProblem 2.2\n')
    x = [1, 2, 3, 4, 5, 6]';
    y = [2, 3, 5, 7, 11, 14]';
    m = 2;
    
    A = create_vandermonde(x, m);
    z_hat = solve_linear_LS(A, y);
    y_hat = A * z_hat;
    mse = sum((y - y_hat).^2) / size(y, 1); % ### compute the minimum squared error
    
    result = sprintf('polynomial fit is %.4f x^2 + %.4f x + %.4f, mse = %.4f', ...
        z_hat(3), z_hat(2), z_hat(1), mse);
    disp(result)
    
    xx = x; % ### generate x values for plotting fitted polynomial
    yy = y_hat; % ### generate y values for plotting fitted polynomial
    
    figure(1)
    hold on
    scatter(x, y, 'ro')
    plot(xx, yy, 'b-')
    legend('data points', 'polynomial fit')
    title(result)
    
    %% Problem 3.2
    % Extract the all samples labeled with digit $d$ and randomly separate 
    % the samples into equal-sized training and testing groups. **(10pt)**
    % Pairwise experiment for applying least-square to classify digit a, b
    % Follow the given steps in the template and implement the function for 
    % pairwise experiment **(25pt)**
    fprintf('\nProblem 3.2\n')
    mnist_pairwise_LS(mnist, 0, 1, 0.5, true);
    
    %% Problem 3.3
    % Repeat the above problem for all pairs of digits. For each pair of 
    % digits, report the classification error rates for the training and 
    % testing sets. The error rates can be formatted nicely into a 
    % triangular matrix. **(15pt)**
    %
    % For example, you can put all testing error in the lower triangle and 
    % all training error in the upper triangle.
    % You may run the classification several times to get an average error 
    % rate over different sample split.
    fprintf('\nProblem 3.3\n')
    num_trial = 1;
    err_matrix = zeros(10);
    for i = 1:10
       for j = 1:10
           if i < j
               % decrease the index by 1 to get the real labels
               [err_tr, err_te] = mnist_pairwise_LS(mnist, i - 1, j - 1, 0.5, false);
               err_matrix(i, j) = err_tr;
               err_matrix(j, i) = err_te;
           end
       end
    end
    
    fprintf('The error matrix is\n')
    disp(err_matrix)
    figure(3)
    imagesc(err_matrix)
    axis('equal')
    title('upper triangle: training error; lower triangle: testing error')
    
    %% Problem 3.4
    % But, what about a multi-class classifier for MNIST digits? 
    % For multi-class linear classification with d classes, one standard 
    % approach is to learn a linear mapping f: R^n -> R^d where the 
    % ?$y$?-value for the $i$-th class is chosen to be the standard basis 
    % vector $ \underline{e}_i \in \mathbb{R}^d $. 
    % This is sometimes called one-hot encoding. 
    % Using the same $A$ matrix as before and a matrix $Y$, defined by 
    % $Y_{i,j}$ if observation $i$ in class $j$ and $Y_{i,j} = 0$ otherwise, 
    % we can solve for the coefficient matrix $Z \in R^d$ coefficients .
    % Then, the classifier maps a vector $\underline{x}$ to class $i$ if 
    % the $i$-th element of $Z^T \underline{x}$ is the largest element in 
    % the vector. 
    % 
    % Follow the given steps in the template and implement the function for 
    % multi-class classification experiment **(30pt)**
    fprintf('\nProblem 3.4\n')
    mnist_onehot_LS(mnist, 0.5);

function A = create_vandermonde(x, m)
% Arguments:
%     x -- 1d-array of (x_1, x_2, ..., x_n)
%     m -- a non-negative integer, degree of polynomial fit
% Returns:
%     A -- an n x (m+1) matrix where A_{ij} = x_i^{j-1}
   n = size(x, 1);
   A = zeros(n, m+1);
   for i = 1:n
      for j = 1:m+1
          A(i, j) = x(i) ^ (j - 1);
      end
   end
end

function z_hat = solve_linear_LS(A, y)
% Arguments:
%     A -- an m x n matrix
%     y -- an n x d matrix
% Returns:
%     z_hat -- m x d matrix, LS solution
    z_hat = A \ y;
end

function [X_tr, X_te, y_tr, y_te] = extract_and_split(mnist, d, test_size)
% extract the samples with given lables and randomly separate the samples 
% into equal-sized training and testing groups, extend each vector to 
% length 785 by appending a ?1
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

function x_filtered = remove_outlier(x)
% returns points that are not outliers to make histogram prettier
% reference: https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting/11886564
% Arguments:
%     x -- 1d-array, points to be filtered
% Returns:
%     x_filtered -- 1d-array, filtered points after dropping outlier
    modified_z_score = 0.6745 * abs(x - median(x));
    x_filtered = x(modified_z_score <= 3.5);
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
