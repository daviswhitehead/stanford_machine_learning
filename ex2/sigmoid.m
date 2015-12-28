function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for col = [1:size(z, 2)]
	for row = [1:size(z, 1)]
		g(row, col) = (1 / (1 + exp(-z(row, col))));
	endfor
endfor

% =============================================================

end
