%% Example code %%
NNpara = './Exp_Result/nn_parameter.mat';
resultsDir = './Exp_Result/performance.mat';
net.Struct = [200 100 50 10];
net.activation_function = 'sigmoid';
net.learningRate = -0.015;              %%% update as the oppositive direction of gradient%%
net.epoch = 200;
net.momentum = 0.1;
net.batchSize = 30;
h = 1;
normFlag = 1;
initNetFlag = 1;
pretrainedWeightFile = nan;     %% no pretrained weights
%% ALl feature and labels are fake (I just make up)
trainFeaMatrix = randn(1000, 200);
trainLabelMatrix = reshape(mod(randperm(1000), 10) + 1, 1000, 1);

testFeaMatrix = randn(500, 200);
testLabelMatrix = reshape(mod(randperm(500), 10) + 1, 500, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
metric_learning(net, trainFeaMatrix, trainLabelMatrix, testFeaMatrix, testLabelMatrix, NNpara, resultsDir, h, normFlag, initNetFlag, pretrainedWeightFile)

