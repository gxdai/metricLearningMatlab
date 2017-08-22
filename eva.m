fprintf('This is test mode 2\n')
test_mode = 2;

% LABEL
label = load('C:/Users/gdai/Downloads/testdataLable.mat');		%% Change  path to labels 
label = label.testdataLable;

% Feature
fea = load('C:/Users/gdai/Downloads/testfeats.mat');			%% change path to features
fea = fea.testfeats;

fea = fea';			% Transpose matrix

C_depth = zeros(size(fea,1), 1);
unique_labels = unique(label);

sample_num = zeros(length(unique_labels),1)
for i = 1 : length(unique_labels)
	unique_labels(i)
	sample_num(unique_labels(i)) = sum(label == unique_labels(i));
end



for i = 1 : size(fea, 1)
    C_depth(i) = sample_num(label(i));
end


%sample_num
%C_depth

simti = pdist2(fea,fea); % generate distance matrix
%simti = zeros(size(fea, 1), size(fea,1));

%for i = 1 : size(fea, 1)
%  for j = 1 : size(fea, 1)
%     simti = norm( fea(i,:) - fea(j,:) , 2 );
%  end
% end


[ NN_av,FT_av,ST_av,dcg_av,E_av,Mean_Av_Precision,P_points, pre, rec] = RetrievalEva( C_depth, simti, label, label, test_mode)


%{
depth_label = [1, 2, 1, 1, 2, 3, 1, 2, 3];
model_label  = [2, 2, 1, 1, 3, 1, 2, 1, 2, 3, 1, 2];      
%% randomly generatefeature 
test_fea = rand(length(depth_label), 1000);
simti = pdist2(test_fea,test_fea); % generate distance matrix
C_depth = [4, 3, 4, 4, 3, 2, 4, 3, 2];        		%% counter the number of 1, 2, 3 in model_label. Here, we have 4 1s, 3 2s, 2 3s.


[ NN_av,FT_av,ST_av,dcg_av,E_av,Mean_Av_Precision,P_points, pre, rec] = RetrievalEva( C_depth, simti, model_label, depth_label, test_mode )
%}