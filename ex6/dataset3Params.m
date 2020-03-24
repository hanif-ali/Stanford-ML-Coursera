function [C, sigma] = dataset3Params(X, y, Xval, yval)
  % NOTE: Use this function only to compute the optimum values for 
  % C and sigma using the cross validation sets by training and cross-validating
  % 64 possible models using different combinations of C and sigma
  % I have already computed the optimum values and commented out the line calling this 
  % function in ex6.m and instead hard coded the values there.
  
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Train and set error for initial C and sigma
model_temp= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model_temp, Xval);
min_error = mean(double(predictions ~= yval));  % Mean classification error

% Values that C and sigma will loop over
multiples = [0.01 0.03 0.1 0.3 1 3 10 30]

disp("Computing Best Value for C and Sigma");

for C_temp=multiples;
  for sigma_temp=multiples;   
    model_temp = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    predictions = svmPredict(model_temp, Xval);
    error_current = mean(double(predictions ~= yval));
    if error_current < min_error;
      min_error = error_current;
      C = C_temp;
      sigma = sigma_temp;
    endif;
    printf("Training Model with C=%d and sigma=%d", C_temp, sigma_temp);
  endfor;
endfor;
printf("\nThe optimum Values are: C=%d, sigma=%d\n", C, sigma);


% =========================================================================

end