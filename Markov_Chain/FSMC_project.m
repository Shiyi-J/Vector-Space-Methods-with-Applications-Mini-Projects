% Fill in all lines with "###"
% Functions after %%%%%% need to be implemented
clc
clear all
close all
%% Exercise 2.1
% What is the distribution of the number of fair coin tosses before one 
% observes 3 heads in a row? To solve this, consider a 4-state Markov chain
% with transition probability matrix
% $$
%     P = 
%     \begin{bmatrix}
%         0.5 & 0.5 & 0 & 0 \\
%         0.5 & 0 & 0.5 & 0 \\
%         0.5 & 0 & 0 & 0.5 \\
%         0 & 0 & 0 & 1
%     \end{bmatrix}
% $$
% where $ X_t = 1 $ if the previous toss was tails, $ X_t = 2 $ if the last
% two tosses were tails then heads, $ X_t = 3 $ if the last three tosses 
% were tails then heads twice, and $ X_t = 4 $ is an absorbing state that 
% is reached when the last three tosses are heads. 

%% 2.1.a
% Write a computer program (e.g., in Python, Matlab, ...) to compute 
% $ \Pr(T_{1,4} = m) $ for $ m = 1, 2, \ldots, 100 $ and use this to 
% estimate expected number of tosses $ \mathbb{E}[T_{1,4}] $. 

P = [0.5, 0.5, 0, 0; 0.5, 0, 0.5, 0; 0.5, 0, 0 ,0.5; 0, 0, 0, 1];
% Compute Phi probabilities and expectation of hitting time
[Phi_list, ET] = compute_Phi_ET(P, 100);

m = 1:100; % ### steps to be plotted
preP = Phi_list(1, 4, :);
Pr = zeros(1, 100); % ### \Pr(T_{1,4} = m) for all m
for i = 1:100
   if i == 1
       Pr(i) = preP(i);
   else
       Pr(i) = preP(i) - preP(i - 1);
   end
end
E = ET(1, 4); % ### \mathbb{E}[T_{1,4}]

figure()
stem(m, Pr)
xlabel('$ m $', 'Interpreter', 'latex')
ylabel('$ \Pr(T_{1,4}=m) $', 'Interpreter', 'latex')
title(sprintf('$ \\mathbf{E}[T_{1,4}] = %f $', E), 'Interpreter', 'latex')

%% 2.1.b
% Write a computer program that generates 500 realizations from this Markov
% chain and uses them to plots a histogram of $ T_{1,4} $.

T = simulate_hitting_time(P, [1, 4], 500);

figure()
hist(T, (0:max(T)-1) + 0.5);
title(sprintf('mean of $ T_{1,4} = $ %f', mean(T)), 'Interpreter', 'latex')

%% Exercise 2.2
% Consider the miniature chutes and ladders game shown in Figure 1. Assume 
% a player starts on the space labeled 1 and plays by rolling a fair
% four-sided die and then moves that number of spaces. If a player lands on
% the bottom of a ladder, then they automatically climb to the top. If a
% player lands at the top of a slide, then they automatically slide to the 
% bottom. This process can be modeled by a Markov chain with $ n = 16 $ 
% states where each state is associated with a square where players can 
% start their turn (e.g., players never start at the bottom of a ladder or 
% the top of a slide). To finish the game, players must land exactly on 
% space 20 (moves beyond this are not taken). 

%% 2.2.a
% Compute the transition probability matrix $ P $ of the implied Markov
% chain.
% Didn't use templated function.
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
            if i == 17
               P(i, i) = 1/4;
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

figure()
imshow(P, 'InitialMagnification', 'fit');

%% 2.2.b
% For this Markov chain, write a computer program (e.g., in Python, Matlab,
% ...) to compute the cumulative distribution of the number turns a player 
% takes to finish (i.e., the probability $ \Pr(T_{1, 20} \le m) $ where 
% $ T_{1, 20} $ is the hitting time from state 1 to state 20).

[Phi_list, ET] = compute_Phi_ET(P, 100);

m = 1:100; % ### steps to be plotted
preP = Phi_list(1, 20, :);
Pr = reshape(preP, 1, 100); % ### \Pr(T_{1,20} <= m) for all m 
E = ET(1, 20); % ### \mathbb{E}[T_{1,20}]

figure()
plot(m, Pr)
xlabel('$ m $', 'Interpreter', 'latex')
ylabel('$ \Pr(T_{1,20} \le m) $', 'Interpreter', 'latex')
title(sprintf('$ \\mathbf{E}[T_{1,20}] = %f $', E), 'Interpreter', 'latex')

%% 2.2.c
% Write a computer program that generates 500 realizations from this Markov 
% chain and uses them to plot a histogram of $ T_{1, 20} $.
T = simulate_hitting_time(P, [1, 20], 500);

figure()
hist(T, (0:max(T)-1) + 0.5);
title(sprintf('mean of $ T_{1,20} = $ %f', mean(T)), 'Interpreter', 'latex')

%% 2.2.d
% Optional Challenge: If the ?rst player rolls 4 and climbs the ladder to 
% square 8, then what is the probability that the second player will win.
% Pr_win = 0;
% ### compute Pr_win 
% fprintf('The probability that the second player will win is %f', Pr_win)

%% Exercise 2.3
% In a certain city, it is said that the weather is rainy with a 90% 
% probability if it was rainy the previous day and with a 50% probability 
% if it not rainy the previous day. If we assume that only the previous 
% day?s weather matters, then we can model the weather of this city by a 
% Markov chain with $ n = 2 $ states whose transitions are governed by
% $$
%     P = 
%     \begin{bmatrix}
%         0.9 & 0.1 \\
%         0.5 & 0.5
%     \end{bmatrix}
% $$
% Under this model, what is the steady-state probability of rainy weather?
P = [0.9, 0.1; 0.5, 0.5];
fprintf('steady-state probability of rainy weather\n')
disp(stationary_distribution(P)')

%% Exercise 2.4
%% 2.4.a
% Consider a game where the gameboard has 8 different spaces arranged in a 
% circle. During each turn, a player rolls two 4-sided dice and moves 
% clockwise by a number of spaces equal to their sum. Define the transition 
% matrix for this 8-state Markov chain and compute its stationary 
% probability distribution.
P = [1/16, 0,1/16, 1/8, 3/16, 1/4, 3/16, 1/8;
    1/8, 1/16, 0, 1/16, 1/8, 3/16, 1/4, 3/16;
    3/16, 1/8, 1/16, 0, 1/16, 1/8, 3/16, 1/4;
    1/4, 3/16, 1/8, 1/16, 0, 1/16, 1/8, 3/16;
    3/16, 1/4, 3/16, 1/8, 1/16, 0, 1/16, 1/8;
    1/8, 3/16, 1/4, 3/16, 1/8, 1/16, 0, 1/16;
    1/16, 1/8, 3/16, 1/4, 3/16, 1/8, 1/16, 0;
    0, 1/16, 1/8, 3/16, 1/4, 3/16, 1/8, 1/16]; % ### construct the transition matrix
fprintf('steady-state probability of the first game\n')
disp(stationary_distribution(P)')

%% 2.4.b
P = [1/4, 0,1/4, 0, 1/4, 0, 1/4, 0;
     1/8, 1/16, 0, 1/16, 1/8, 3/16, 1/4, 3/16;
     5/16, 0, 1/16, 0, 1/16, 1/8, 3/16, 1/4;
     1/2 1/16, 0, 1/16, 0, 1/16, 1/8, 3/16;
     11/16, 0, 1/16, 0, 1/16, 0, 1/16, 1/8;
     3/4, 1/16, 0, 1/16, 0, 1/16, 0, 1/16;
     13/16, 0, 1/16, 0, 1/16, 0, 1/16, 0;
     3/4, 1/16, 0, 1/16, 0, 1/16, 0, 1/16]; % ### construct the transition matrix
fprintf('steady-state probability of the second game\n')
disp(stationary_distribution(P)')

%%%%%%

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
