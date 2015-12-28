function [X_norm, mu, sigma] = featureNormalize(X, mu, sigma)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
if size(mu, 2) ~= size(X, 2)
	mu = zeros(1, size(X, 2));
	update_mu = 1;
else
	update_mu = 0;
end
if size(sigma, 2) ~= size(X, 2)
	sigma = zeros(1, size(X, 2));
	update_sigma = 1;
else
	update_sigma = 0;
end

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma.
%
%               Note that X is a matrix where each column is a
%               feature and each row is an example. You need
%               to perform the normalization separately for
%               each feature.
%
% Hint: You might find the 'mean' and 'std' functions useful.
%
for col = [1:size(X_norm, 2)]
	if update_mu == 1
		mu(col) = mean(X_norm(:,col));
	endif
	if update_sigma == 1
		sigma(col) = std(X_norm(:,col));
	endif
	for row = [1:size(X_norm, 1)]
		X_norm(row, col) = ((X_norm(row, col) - mu(col)) / sigma(col));
	endfor
endfor
% ============================================================

end
