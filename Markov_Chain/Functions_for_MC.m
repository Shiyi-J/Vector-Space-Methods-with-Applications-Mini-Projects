function [Phi_list, ET] = compute_Phi_ET(P, ns)
% Arguments:
%     P -- n x n, transition matrix of the Markov chain
%     ns -- largest step to consider
% Returns:
%     Phi_list -- n x n x (ns + 1), the Phi matrix for time 0, 1, ...,ns
%     ET -- n x n, expectedd hitting time approxiamated up to step ns

    % Try to compute following quantities:
    % Phi_list(i, j, m) = phi_{i,j}^{(m)} = Pr( T_{i, j} <= m )
    % ET(i, j) = E[ T_{i, j} ] ~ \sum_{m=1}^ns m Pr( T_{i, j} = m )
n = size(P, 1);
Phi_list = zeros(n, n, ns);
ET = zeros(n, n);
for i = 1:ns
    if i == 1
       Phi_list(:,:,i) = eye(n) + (ones(n) - eye(n)).*(P*eye(n));
       ET = i * (Phi_list(:,:,i) - eye(n)) + ET;
    else
       Phi_list(:,:,i) = eye(n) + (ones(n) - eye(n)).*(P*Phi_list(:,:,i-1));
       ET = i * (Phi_list(:,:,i) - Phi_list(:,:,i-1)) + ET;
    end
end
end

function [T] = simulate_hitting_time(P, states, nr)
% Arguments:
%     P -- n x n, transition matrix of the Markov chain
%     states -- the list [start state, end state], index starts from 1
%     nr -- largest step to consider
% Returns:
%     T -- nr x 1, the hitting time of all realizations
    src = states(1);
    dst = states(2);
    n = size(P, 1);
    if src == dst
        T = zeros(nr, 1);
    else
        T = zeros(nr, 1); % initialize the matrix
        for i = 1:nr
            count = 0; % count number of steps to end state 
            current_s = src; 
            while current_s ~= dst
                count = count + 1; % count+1 when next state is not dst
                next_s = 1:n; % possible next state to choose
                prob_s = P(current_s,:); % prob from current to next possible states
                j = randsrc(1, 1, [next_s; prob_s]); % single-element matrix indicating next state
                current_s = j(1,1);
            end
            T(i) = count; % T(i) = hitting time of the i-th realization
        end
    end
end

function P = GetMatrix()
P = zeros(20);
for i = 1:20
    for j = 1:20
        if i ~= 20
            if (j == i + 1) || (j == i + 2) || (j == i + 3) || (j == i + 4)
               P(i, j) = 1/4;
               if j == 4
                  P(i, j) = 0;
                  P(i, 8) = 1/4;
               end
               if j == 14
                  P(i, j) = 0;
                  P(i, 19) = 1/4;
               end
               if j == 13
                  P(i, j) = 0;
                  P(i, 2) = 1/4;
               end
               if j == 17
                  P(i, j) = 0;
                  P(i, 6) = 1/4;
               end
            end
            if (i == 13) || (i == 17) || (i == 4) || (i == 14)
               P(i,:) = zeros(1, 20); 
            end
            if i == 18
               P(i, i) = 1/2; 
            end
            if i == 19
               P(i, i) = 3/4; 
            end
        else
            if i == j
               P(i, j) = 1;
            end
        end
    end
end
end

 function pi_sd = stationary_distribution(P)
% Arguments:
%     P -- n x n, transition matrix of the Markov chain
% 
% Returns:
%     pi_sd -- n x 1, stationary distribution of the Markov chain
    
    % Think pi_sd as column vector, solve linear equations:
    %     P^T pi_sd = pi_sd
    %     sum(pi_sd) = 1
n = size(P,1);
A = transpose(P) - eye(n);
X = null(A);
pi_sd = transpose(X / norm(X,1));
 end