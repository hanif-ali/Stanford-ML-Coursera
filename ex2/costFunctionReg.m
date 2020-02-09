function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Finding cost
n = size(theta, 1);
predictions = sigmoid(X * theta);
J_unreg = -(1/m)*sum(y.*log(predictions) + (1-y).*log(1-predictions)); % Unregularized

J = J_unreg + (lambda/(2*m))*sum(theta(2:n).^2); % Add regularization term to theta=2 to n

% Finding gradient
errors = predictions - y;
grad_unreg = (1/m) * (sum(errors .* X))';
grad = grad_unreg + (lambda/m)*[0;theta(2:n)];

% =============================================================

end
