function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  predictions = X * theta;
  errors = predictions - y;
  
  %sigma_term = zeros(2, 1);
  %features = X';
  %for i=1:m,
   % sigma_term = sigma_term + (predictions(i) - y(i))*features(:, i);
  %delta = sigma_term ./ m;
  %theta = theta - (alpha * delta);
  
  delta = zeros(2,1);
  delta(1) = sum(errors);
  delta(2) = sum(errors.*X(:, 2));
  theta = theta - (alpha/m)*delta;
  

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
J_history(iter) = computeCost(X, y, theta);

end
end
