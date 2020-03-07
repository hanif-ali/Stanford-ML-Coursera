function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

% Find the predictions using the given theta
h_theta = X*theta;
% Devitations from correct labels
errors = h_theta - y;
J_unreg = (1/(2*m))*sum(errors.^2, 1);

J = J_unreg + (lambda/(2*m))*sum(theta(2:end).^2, 1); % Add the regularization terms

% Multiply each error with the corresponding x_i value
for i=1:m;
  X(i, :) = X(i, :) .* errors(i);
end;
 
grad = (1/m) * sum(X, 1)';
% Add the regularizations terms to all except theta0
grad(2:end) = grad(2:end) + (lambda/m) * theta(2:end);




% =========================================================================

grad = grad(:);

end
