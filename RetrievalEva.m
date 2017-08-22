%%% This function is used to calculate the standard metrics for retrieval %%%
function [ NN_av,FT_av,ST_av,dcg_av,E_av,Mean_Av_Precision,P_points, pre, rec] = RetrievalEva( C_depth, simti, model_label, depth_label, test_mode )
  
  %%% test_mode = 1:    For using example in the test set as query, find relevant examples in the training set. This is common in SHREC contest; 
  %%% test_mode = 2:    For evaulation only with examples in the test set, which is the evalution of ModelNet. Basicaly, it's leave one out. Use one example as query, find relavent examples in the remaining exmaples.
  %%% C_depth:          The number of relevant exmaples for each query.
  %%% simti:            The distance matrix
  %%% model_label:      The labels of examples in the database
  %%% depth_label:      The labels of query examples.
  
  if test_mode == 1                          
    C=C_depth;
    recall = zeros(size(simti));        %%% matrix to store all the recall values 
    precision = zeros(size(simti));     %%% matrix to store all the precision values
  elseif test_mode == 2                 
    C = C_depth -1 ;                     %%% update C_depth, recall, precision, to ignore the first returned examples(itself)
    recall = zeros(size(simti,1), size(simti,2)-1);   
    precision = zeros(size(simti,1), size(simti,2)-1);
  end

  number_of_queries=length(C);                
  P_points=zeros(number_of_queries,max(C));         %% P measurement
  Av_Precision=zeros(1,number_of_queries);          %% average precison
  NN=zeros(1,number_of_queries);                    %% nearest neightbor
  FT=zeros(1,number_of_queries);                    %% first tier
  ST=zeros(1,number_of_queries);                    %% second tier
  dcg=zeros(1,number_of_queries);                   %% discounted cumulated gain
  E=zeros(1,number_of_queries);                     %% E measure



  for qqq=1:number_of_queries  
%     [tempx,R] = sort(simti(qqq,:),'descend');
    [tempx,R] = sort(simti(qqq,:));                 %% sort distance
    if test_mode == 1
        model_label_l=model_label(R);               %% For the qqq-th query, get the labels of the ranked retrieved examples.
        numRetrieval = size(simti,2);               %% The the number of retrieved exmaples
        G=zeros(1,numRetrieval);                    %% saving the returned result, 
    elseif test_mode == 2
        model_label_l=model_label(R(2:end));
        numRetrieval = size(simti,2)-1;
        G=zeros(1,numRetrieval);
    end
    for r=1:numRetrieval
        if model_label_l(r)==depth_label(qqq)       %% same label means relevant, different labels means irrelevant.
            G(r)=1;
        end
    end; 
    G_sum=cumsum(G);                                %% Cumulative sum of elements
    r1 = G_sum./repmat(C(qqq), [1, numRetrieval]);  %% recall value
    p1 = G_sum./(1:numRetrieval);                   %% precision value.
    R_points=zeros(1,C(qqq));                       %% R measurement
    %fprintf('Here is the %d-th test example:\n', qqq);
    for rec=1:C(qqq)
        R_points(rec)=find((G_sum==rec),1);
    end;
    P_points(qqq,1:C(qqq))=G_sum(R_points)./R_points;
    Av_Precision(qqq)=mean(P_points(qqq,1:C(qqq)));
    NN(qqq)=G(1);                                   %% Nearest neightbor
    FT(qqq)=G_sum(C(qqq))/C(qqq);                   %% First tier
    ST(qqq)=G_sum(2*C(qqq))/C(qqq);                 %% Second tier
    % P_32=G_sum(32)/32;                             
    % R_32=G_sum(32)/C(qqq);

    %% sometimes there are less than 32 examples in the database
    P_32=G_sum(min(32, length(G_sum)))/min(32, length(G_sum));                    R_32=G_sum(min(32, length(G_sum)))/C(qqq);

 
    if (P_32==0)&&(R_32==0);
        E(qqq)=0;
    else
        E(qqq)=2*P_32*R_32/(P_32+R_32);
    end;
    if test_mode == 1
    NORM_VALUE=1+sum(1./log2([2:C(qqq)]));          %% This is DCG normalization.
    dcg_i=(1./log2([2:length(R)])).*G(2:end);
    dcg_i=[G(1);dcg_i(:)];
    dcg(qqq)=sum(dcg_i)/NORM_VALUE;
    recall(qqq,:) = r1;
    precision(qqq,:) = p1;
    elseif test_mode == 2
    NORM_VALUE=1+sum(1./log2([2:C(qqq)]));
    dcg_i=(1./log2([2:length(R(2:end))])).*G(2:end);
    dcg_i=[G(1);dcg_i(:)];
    dcg(qqq)=sum(dcg_i)/NORM_VALUE;
    recall(qqq,:) = r1;
    precision(qqq,:) = p1;
    end
end;
% Get the  averaged results for all the measures
NN_av=mean(NN);                                     
FT_av=mean(FT);
ST_av=mean(ST);
dcg_av=mean(dcg);
E_av=mean(E);
Mean_Av_Precision=mean(Av_Precision);
pre = sum(precision,1)/size(precision,1);
rec = sum(recall,1)/size(recall,1);
end

