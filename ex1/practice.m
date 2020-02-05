%%% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
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
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%

%% Initialization
clear ; close all; clc

%% ======================= Part 2: Plotting =======================
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y);

%% =================== Part 3: Cost and Gradient descent ===================

X = [ones(m, 1), data(:,1)];
theta = [6;3]; 
iterations = 1500;
alpha = 0.01;


%% ============= Part 4: Visualizing J(theta_0, theta_1) =============

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;  
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
for i=1:10,
  theta = gradientDescent(X, y, theta, alpha, 100);
  plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
end;