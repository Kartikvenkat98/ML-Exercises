function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;


#for i = 1:m
	#cost(i) = (theta(1) + theta(2)*X(i, 2) - y(i))^2;

J = sum((theta(1) + theta(2)*X(:, 2) - y(:)).^2)/2*m;
	

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end

#J = sum(cost)/2*m
