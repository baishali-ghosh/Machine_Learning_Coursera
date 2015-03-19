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
%

z = X*theta;
val = (z-y).^2;
J = sum(val)*1/(2*m);

%computing regularized parameters.
val2=sum((theta(2:size(theta)).^2));
val2=(val2*lambda)/(2*m);
J = J + val2;

%gradient computation
grad = 1/m*(X' * (z-y));
grad(2:size(theta)) = grad(2:size(theta)) + (lambda/m)*theta(2:size(theta));

% =========================================================================

grad = grad(:);

end
