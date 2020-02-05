function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
no_features = size(X, 2);


for iter = 1:num_iters

    predictions = X*theta;
    errors = predictions - y;
    delta = zeros(no_features,1);
    for i=1:no_features,
       delta(i) = sum(errors.*X(:, i));
    end;
    theta = theta - (alpha/m)*delta;
    


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
printf("Final Cost: %d\n", computeCostMulti(X, y, theta));
end
