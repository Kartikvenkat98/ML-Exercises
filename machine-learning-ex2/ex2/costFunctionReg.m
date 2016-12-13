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
div = 2*m;

for i = 1:m
	cost(i) = (-(y(i))*log(sigmoid(X(i, :)*theta)) - (1 - y(i))*log(1 - sigmoid(X(i, :)*theta)))/m;

end

for i = 2:size(theta)
	reg(i) = lambda/div*((theta(i))^2);

end

J = sum(cost) + sum(reg);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

for i = 1:size(theta)
	for j = 1:m
		temp(j) = ((sigmoid(X(j,:)*theta)) - y(j))*X(j,i)/m;
	end

	if i == 1
		grad(i) = sum(temp);
	else
		grad(i) = sum(temp) + lambda/m*theta(i);
	end
end

end
