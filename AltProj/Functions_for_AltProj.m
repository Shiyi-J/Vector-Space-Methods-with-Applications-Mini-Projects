function [v, err] = altproj(A, B, v0, n)
% Arguments:
%     A -- matrix whose column span is vector space U
%     B -- matrix whose column span is vector space W
%     v0 -- initialization vector
%     n -- number of sweeps for alternating projection
% Returns:
%     v -- the output after 2n steps of alternating projection
%     err -- the error after each full pass
    % Construct projection matrix
    PU = A * inv(A' * A) * A';
    PW = B * inv(B' * B) * B';
    % Compute the exact solution
    basis_UintW = [A B] * null([A -B], 'r');
    P_UintW = basis_UintW * inv(basis_UintW' * basis_UintW) * basis_UintW';
    v_star = P_UintW * v0; 
    % Apply n full pass of alternating projection
    v = v0;
    err = zeros(1, n);
    for t = 1:n
        v = PU * v;
        v = PW * v;    
        err(t) = max(v - v_star);
    end
end
  
function [X, err] = kaczmarz(A, b, I)
% Arguments:
%     A -- matrix defines the LHS of linear equation
%     b -- vector defines the RHS of linear equation
%     I -- number of full passes through the Kaczmarz algorithm
% Returns:
%     X -- the output of all I full passes
%     err -- the error after each full pass
    [m, n] = size(A);
    v = zeros(n, 1);
    X = zeros(n, I);
    err = zeros(1, I);
    xmin = A' * inv(A * A') * b; % true minimum norm solution
    for i = 1:I
        for j = 1:m
            aj = A(mod(j, m) + 1,:);
            bj = b(mod(j, m) + 1,:);
            v = v - (dot(v, aj) - bj) * aj' / norm(aj) ^ 2;
        end
        X(:, i) = v;
        err(i) = max(X(:, i) - xmin);
    end
end

function [v, err] = lp_altproj(A, b, I)
% Find a feasible solution for A v >= b using alternating projection
% Arguments:
%     A -- matrix defines the LHS of linear equation
%     b -- vector defines the RHS of linear equation
%     I -- number of full passes through the alternating projection
% Returns:
%     v -- the output after I full passes
%     err -- the error after each full pass
    [m, n] = size(A);
    % Apply I sweeps of alternating projection
    v = zeros(n, 1);
    err = zeros(1, I);
    
    for i = 1:I
        for j = 1:m
            aj = A(mod(j,m) + 1,:);
            bj = b(mod(j,m) + 1,:);
            if aj * v < bj
                v = v - (aj * v - bj) * aj' / (norm(aj) ^ 2);
            end
        end
        err(i) = max(b - A*v);
        if err(i) < 0
            err(i) = 0;
        end
    end
end

function [z_hat, err_tr, err_te] = mnist_pairwise_altproj(mnist, a, b, solver, test_size, verbose)
% Pairwise experiment for applying alternating projection to classify digit 
% a, b
% Arguments:
%     mnist -- the MNIST data set read from csv file
%     a, b -- digits to be classified
%     solver -- solver function to return coefficients of linear classifier
%     test_size -- the fraction of testing set
%     verbose -- whether to print and plot results
% Returns:
%     z_hat -- coefficients of linear classifier
%     err_tr -- training set classification error
%     err_te -- testing set classification error

    % Find all samples labeled with digit a and split into train/test sets
    [Xa_tr, Xa_te, ya_tr, ya_te] = extract_and_split(mnist, a, test_size);
    
    % Find all samples labeled with digit b and split into train/test sets
    [Xb_tr, Xb_te, yb_tr, yb_te] = extract_and_split(mnist, b, test_size);
    
    % Construct the full training set
    X_tr = [Xa_tr; Xb_tr];
    y_tr = [(-1) * ones(size(ya_tr, 1), 1); ones(size(yb_tr, 1), 1)];
    
    % Construct the full testing set
    X_te = [Xa_te; Xb_te];
    y_te = [(-1) * ones(size(ya_te, 1), 1); ones(size(yb_te, 1), 1)];
    
    % Run solver on training set to get linear classifier
    A_tilde = y_tr.*X_tr;
    z_hat = solver(A_tilde, y_tr);
    
    % Compute estimation and misclassification on training set
    y_hat_tr = sign(X_te * z_hat); % elements in labels either -1 or +1
    err_tr = mean(y_tr ~= y_hat_tr);
    
    % Compute estimation and misclassification on training set
    y_hat_te = sign(X_tr * z_hat);
    err_te = mean(y_te ~= y_hat_te);
    
    if verbose
        fprintf('Pairwise experiment, mapping %d to -1, mapping %d to 1\n', a, b)
        fprintf('training error = %.2f%%, testing error = %.2f%%\n', 100 * err_tr, 100 * err_te)
        
        % Compute confusion matrix for training set
        cm_tr = confusionmat(y_tr, y_hat_tr);
        fprintf('Confusion matrix for training set:\n')
        disp(cm_tr)
        
        % Compute confusion matrix for testing set
        cm_te = confusionmat(y_te, y_hat_te);
        fprintf('Confusion matrix for testing set:\n')
        disp(cm_te)
        
        % Compute the histogram of the function output separately for each 
        % class, then plot the two histograms together
        ya_te_hat = Xa_te * z_hat;
        yb_te_hat = Xb_te * z_hat;
        
        % Remove outlier to make pretty histogram
        ya_te_hat = remove_outlier(ya_te_hat);
        yb_te_hat = remove_outlier(yb_te_hat);
        
        figure()
        hold on
        histogram(ya_te_hat, 50, 'facecolor', 'red')
        histogram(yb_te_hat, 50, 'facecolor', 'blue')
        title(sprintf('histogram of pairwise experiment on %d and %d', a, b))
    end
end

function [Z, err_tr, err_te] = mnist_multiclass_altproj(mnist, solver, test_size)
% Experiment for applying least-square to classify all digits using one-hot 
% encoding
% Arguments:
%     mnist -- the MNIST data set read from csv file
%     solver -- solver function to return coefficients of linear classifier
%     test_size -- the fraction of testing set
% Returns:
%     Z -- coefficients of linear classifier
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
    
    % Run alternating projection on training set for each digit
    [m, n] = size(X_tr);
    A_tilde = zeros(9 * m, 10 * n);
    for i = 1:m
        xi = (-1) * X_tr(i,:);
        count = 1;
        for j = 1:10
            if j - 1 == y_tr(i)
                continue
            else
                A_tilde((i-1)*9+count, (j-1)*785+1 : j*785) = xi;
                A_tilde((i-1)*9+count, y_tr(i)*785+1 : (y_tr(i)+1)*785) = X_tr(i,:);
                count = count +1;
            end
        end
    end
    b_tilde = zeros(size(A_tilde, 1), 1);
    Z = solver(A_tilde, b_tilde);
    % Reshape Z to 785 x 10
    Z = reshape(Z, [785,10]);
    
    % Compute estimation and misclassification on training set
    [~, I_tr] = max(transpose(X_tr * Z));
    y_hat_tr = transpose(I_tr) - 1;
    err_tr = mean(y_tr ~= y_hat_tr);
    
    % Compute estimation and misclassification on testing set
    [~, I_te] = max(transpose(X_te * Z));
    y_hat_te = transpose(I_te) - 1;
    err_te = mean(y_te ~= y_hat_te);
    
    fprintf('training error = %.2f%%, testing error = %.2f%%\n', 100 * err_tr, 100 * err_te)
    % Compute confusion matrix
    cm_tr = confusionmat(y_tr, y_hat_tr);
    fprintf('Confusion matrix for training set:\n')
    disp(cm_tr)
    
    cm_te = confusionmat(y_te, y_hat_te);
    fprintf('Confusion matrix for testing set:\n')
    disp(cm_te)
    
    figure()
    subplot(1, 2, 1)
    imagesc(cm_tr)
    axis('equal')
    title('Confusion matrix of training set')
    subplot(1, 2, 2)
    imagesc(cm_te)
    axis('equal')
    title('Confusion matrix of testing set')
end
