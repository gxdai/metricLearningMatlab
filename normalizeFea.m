function [nFea, mFea, sigFea] = normalizeFea(feaSet)

mFea = mean(feaSet);
sigFea = std(feaSet);

nFea = (feaSet - repmat(mFea, size(feaSet, 1), 1)) ./ repmat(sigFea, size(feaSet, 1), 1);
index = find(isnan(nFea) == 1);
nFea(index) = 0;

