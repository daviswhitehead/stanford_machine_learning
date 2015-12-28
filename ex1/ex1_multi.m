%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear regression exercise.
%
%  You will need to complete the following functions in this
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%
%
%% Initialization
%
%% ================ Part 1: Feature Normalization ================
%
%% Clear and Close Figures
clear ; close all; clc
%
fprintf('Loading data ...\n');
%
%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
%
% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
%
fprintf('Program paused. Press enter to continue.\n');
pause;
%
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
%
[X mu sigma] = featureNormalize(X, 0, 0);
%
% Add intercept term to X
X = [ones(m, 1) X];
%
%
%% ================ Part 2: Gradient Descent ================
%
% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha).
%
%               Your task is to first make sure that your functions -
%               computeCost and gradientDescent already work with
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%
%
fprintf('Running gradient descent ...\n');
%
% Choose some alpha value
num_alphas = 20;
alpha_multiplier = 1.4;
num_iters = 400;
alphas = 0.001 * ones(1, num_alphas);
thetas = zeros(3, num_alphas);
J_historys = zeros(num_iters, num_alphas);
colors = hsv(num_alphas);
figure(); hold on;
for a = [1:1:num_alphas]
	if a > 1
		alphas(a) = alphas(a - 1) * alpha_multiplier;
	endif
%
	% Init Theta and Run Gradient Descent
	theta = zeros(3, 1);
	[thetas(:,a), J_historys(:,a)] = gradientDescentMulti(X, y, theta, alphas(a), num_iters);
	% Plot the convergence graph
	plot(1:numel(J_historys(:,a)), J_historys(:,a), 'Color', colors(a,:)); hold on;
	xlabel('Number of iterations');
	ylabel('Cost J');
	legendInfo{a} = ['alpha = ' num2str(alphas(a)) '  ']; % or whatever is appropriate
endfor
legend(legendInfo)
hold off;
pause
%
% Found alpha of .03 to be best
alpha = 0.03;
%
% Init Theta and Run Gradient Descent
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
%
% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
%
% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
%
% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this
X_test = [1650 3];
m = 1;
%
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
%
[X_test mu sigma] = featureNormalize(X_test, mu, sigma);
%
% Add intercept term to X_test
X_test = [ones(m, 1) X_test];
price = X_test * theta;
% ============================================================
%
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
%
fprintf('Program paused. Press enter to continue.\n');
pause;
%
%% ================ Part 3: Normal Equations ================
%
fprintf('Solving with normal equations...\n');
%
% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form
%               solution for linear regression using the normal
%               equations. You should complete the code in
%               normalEqn.m
%
%               After doing so, you should complete this code
%               to predict the price of a 1650 sq-ft, 3 br house.
%
%
%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% testing
sq_foot_lin = min(data(:,1)):1:max(data(:,1));
sq_foot_mean = mean(data(:,1));
sq_foot_m = length(sq_foot_lin);
br_lin = min(data(:,2)):1:max(data(:,2));
br_mean = mean(data(:,2));
br_m = length(br_lin);
X_sq_foot = [ones(sq_foot_m, 1) [sq_foot_lin;(ones(sq_foot_m, 1)' * br_mean)]'];
X_br = [ones(br_m, 1) [(ones(br_m, 1)' * sq_foot_mean);br_lin]'];

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this
m = 1;
X_test = [ones(m, 1) 1650 3];
price = X_test * theta;


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

br_price = X_br * theta;
sq_foot_price = X_sq_foot * theta;
colors = hsv(2);
clf; hold on;
subplot(2, 1, 1), plot(X_br(:,3), br_price, 'Color', colors(1,:), X(:,3), y, 'o', 'Color', colors(2,:));
legendInfo{1} = ['predicted' num2str(alphas(a)) '  ']; % or whatever is appropriate
subplot(2, 1, 2), plot(X_sq_foot(:,2), sq_foot_price, 'Color', colors(1,:), X(:,2), y, 'o', 'Color', colors(2,:));
hold off;
pause;



