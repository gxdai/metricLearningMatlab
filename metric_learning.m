%% This function is designed for metric learning %%
%% metric learning use pairwise learning strategy %%

function metric_learning(net, trainFeaMatrix, trainLabelMatrix, testFeaMatrix, testLabelMatrix, NNpara, resultsDir, h, normFlag, initNetFlag, pretrainedWeightFile)
    
%%% net:                            the network structure and paraemters 
%%% trainFeaMatrix:                 N*K matrix of input training features, where N is the number of features, K is the feature size
%%% trainLabelMatrix:               N*1 vector of input training label, where N is the number labels, label start from 1.
%%% testFeaMatrix:                  M*K matrix of input training features, where M is the number of features, K is the feature size
%%% testLabelMatrix:                M*1 vector of input training label, where M is the number labels, label start from 1.
%%% paraDir:                        Directory for saving the "net"
%%% h:                              The threshod value for hinge loss
%%% normFlag:                       1 for normalize input feature, 0 for not.
%%% initNetFlag:                    1 for training from scratch, 0 for loading from pre-trained weights
%%% pretrainedWeightFile:           The path of pre-trained weights file


    %%% init network weights and biases  %%%%%%%
    if initNetFlag                                  
        net = initNet(net, 1, 1);                                                 %%% randomly initialize parameters
    else
        load(pretrainedWeightFile);
    end

    %% init network hyperparameters %%%
    learningRate = net.learningRate;
    MaxIter = net.epoch;
    momentum = net.momentum;
    batchSize = net.batchSize;
    printInterval = 20;                                                             %% every 20 iterations print out the performance


    %% preprocessing for features 
    numExample = size(trainFeaMatrix, 1);                                           %% Get the number of training samples
    if normFlag
        [trainFeaMatrix, meanValue, sigValue] = normalizeFea(trainFeaMatrix);      %% normalize the training features 
    end
    
    errSum = 0; 
    rand('state', 0);

    for iter = 1 : MaxIter

        index1 = randperm(numExample);                  %% Get a shuffled list of training data
        index2 = randperm(numExample);                  %% Get a second shuffled list of training data, and pairwise training data from two list
    
        for loopIndex = 1 : batchSize : numExample - batchSize
            %%% a small batch of features and labels from first list 
            FeaTemp1         =   trainFeaMatrix(index1(loopIndex : loopIndex+batchSize-1), :);
            LabelTemp1       =   trainLabelMatrix(index1(loopIndex : loopIndex+batchSize-1), :);
            %%% a small batch of features and labels from second list 
            FeaTemp2         =   trainFeaMatrix(index2(loopIndex : loopIndex+batchSize-1), :);  
            LabelTemp2       =   trainLabelMatrix(index2(loopIndex : loopIndex+batchSize-1), :);    

            %% Forward and Backward  calculation %%
            [net_, errNorm]   =   netbp(FeaTemp1, LabelTemp1, FeaTemp2, LabelTemp2, net, h);
            
            %% update weights %%
            net = deltaWeight(net, net_, momentum);
            net = updateWeight(net, learningRate);

            %% summation of error  %%
            errSum = errSum + errNorm;
        end

        fprintf('Total error of %d-th iteration is %5g\n', iter, errSum); 
        errSum = 0;

        %%% uncomment if you want weight decay after certain number of iterations
%      if iter > 100
%          learningRate = learningRate * 0.96;
%      end
      

        if mod(iter, printInterval) == 0
            if normFlag
                [trainFea, ~, ~] = normalizeFea(trainFeaMatrix);
                [testFea, ~, ~] = normalizeFea(testFeaMatrix);
            else
                trainFea = trainFeaMatrix;
                testFea = testFeaMatrix;
            end
            net1 = netff(trainFea, net);
            trainFea = net1.ff{end};
            net2 = netff(testFea, net);
            testFea = net2.ff{end};

            %% There are two different types of retrieval evaluation %%
            %% 1)       test sampe as query, find the relavent objects in the training set.
            %% 2)       leave one out. only use samples from testing set. Use one sample as query, find relavent object in rest set.
            

            fprintf('This is the retrieval results for mode 1\n');
            test_mode = 1;

            C_depth = zeros(size(testLabelMatrix, 1), 1);
            unique_labels = unique(testLabelMatrix);
            sample_num = zeros(length(unique_labels),1);
            for i = 1 : length(unique_labels)
                sample_num(unique_labels(i)) = sum(trainLabelMatrix == unique_labels(i));
            end

            for i = 1 : size(testLabelMatrix, 1)
                C_depth(i) = sample_num(testLabelMatrix(i));
            end

            simti = pdist2(testFea, trainFea);
            depth_label = testLabelMatrix;
            model_label = trainLabelMatrix;
            [NN_av,FT_av,ST_av,dcg_av,E_av,Mean_Av_Precision,P_points, pre, rec] = RetrievalEva( C_depth, simti, model_label, depth_label, test_mode);
            %%%%%%%%%% Plot the results %%%%%%%%%%%
            figure(1);
            plot(rec,pre,'r','LineWidth',2);
            xlim([0 1]);
            ylim([0 1]);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            save(resultsDir, 'NN_av', 'FT_av', 'ST_av', 'dcg_av', 'E_av', 'Mean_Av_Precision', 'P_points', 'pre', 'rec');
            fprintf('The NN_av is %g\n', NN_av);
            fprintf('The FT_av is %g\n', FT_av);
            fprintf('The ST_av is %g\n', ST_av);
            fprintf('The dcg_av is %g\n', dcg_av);
            fprintf('The E_av is %g\n', E_av);
            fprintf('The Mean_Av_Precision is %g\n', Mean_Av_Precision);

            %% The is separate line %%%% 
            fprintf('***********************************************************\n\n');
            fprintf('***********************************************************\n\n');
            fprintf('This is the retrieval results for mode 2\n');
            test_mode = 2;

            C_depth = zeros(size(testLabelMatrix, 1), 1);
            unique_labels = unique(testLabelMatrix);
            sample_num = zeros(length(unique_labels),1);
            for i = 1 : length(unique_labels)
                sample_num(unique_labels(i)) = sum(testLabelMatrix == unique_labels(i));
            end

            for i = 1 : size(testLabelMatrix, 1)
                C_depth(i) = sample_num(testLabelMatrix(i));
            end

            simti = pdist2(testFea, testFea);
            depth_label = testLabelMatrix;
            [NN_av,FT_av,ST_av,dcg_av,E_av,Mean_Av_Precision,P_points, pre, rec] = RetrievalEva( C_depth, simti, depth_label, depth_label, test_mode);
            %%%%%%%%%% Plot the results %%%%%%%%%%%
            figure(1);
            plot(rec,pre,'r','LineWidth',2);
            xlim([0 1]);
            ylim([0 1]);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            save(resultsDir, 'NN_av', 'FT_av', 'ST_av', 'dcg_av', 'E_av', 'Mean_Av_Precision', 'P_points', 'pre', 'rec');
            fprintf('The NN_av is %g\n', NN_av);
            fprintf('The FT_av is %g\n', FT_av);
            fprintf('The ST_av is %g\n', ST_av);
            fprintf('The dcg_av is %g\n', dcg_av);
            fprintf('The E_av is %g\n', E_av);
            fprintf('The Mean_Av_Precision is %g\n', Mean_Av_Precision);

        end
        
        
        if (mod(iter, 100) == 0)        %% every 100 iteratioin save the weights
            save(NNpara,'net');
        end
    end

    
