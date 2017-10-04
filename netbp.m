function [net, loss] = netbp(fea1, label1, fea2, label2, net, margin, lossType=1)
    
    
    %   Y           : similarity flag, 1 for similar pair, 0 for non-similar pair     
    %   f           : The network transform function: x ----> f(x), where x is network input, f(x) is network output
    %   D(x1, x2)   : The L2 norm of f(x1)-f(x2):       D(x1, x2) = |f(x1) - f(x2)|_{l2}
    % The implementation includes two types of loss function
    %   type 1 (I use)      :           Loss = Y * D(x1, x2)^2 + (1-Y) * max{0, h - D(x1, x2)^2}         
    %   type 2 (standard)   :           Loss = Y * D(x1, x2)^2 + (1-Y) * max{0, h - D(x1, x2)}^2



    %% Function arguments description %%

    % fea1:         input features(x1), N*M matrix, N is sample number, M is feature size
    % label1:       input labels, N*1 matrix, N is sample number 
    % fea2:         input features(x2), N*M matrix, N is sample number, M is feature size
    % label2:       input labels, N*1 matrix, N is sample number
    % net:          Network definition
    % margin:       The margin (h) in the contrastive loss
    % lossType:         two different types of losses, 1 for the one I use, 2 for standard.



    % This is the forward output for fea1:      x1 ---> f(x1)
	net1 = netff(fea1, net);  
    % This is the forward output for fea2:      x2 ---> f(x2)      
	net2 = netff(fea2, net); 

    % Get the diff of f(x1) and f(x2):          diff = f(x1) - f(x2)
	diff = net1.ff{end} - net2.ff{end};         
	typeOfActivation = net.activation_function;


    % Check the pairwise samples are similar pair (Flag: 1) or non-similar pair (Flag: 0) 
	Y = (label1 == label2);        
    % Get the L2 distance norm
	D_x1_x2 = (sum(diff .* diff, 2)).^0.5;  			 
    % batch size
    batch_size = length(Y) 

    % For non-similar pair, do margin check to see if D_x1_x2 has exceeded margin

    if lossType == 1            % Use the loss function I use
        margin_check = double(margin >= D_x1_x2.^2)            % 1 for L2 norm is within the margin, 0 for the L2 distance has exceeded the margin (ignore for bp)
        sample_flags = Y + (1-Y)*(-1) .* margin_check           % Only the useful samples (for negative pair, distance is within margin)
        % dz_dx = (Y + (1-Y)*(-1) .* margin_check)
        % Get the loss
        Loss = sum(Y .* D_x1_x2.^2 + (1-Y) .* (margin - D_x1_x2.^2) . * margin_check, 2) / length(Y)

    else if lossType == 2       % Use the standard loss function 
        margin_check = double(margin >= D_x1_x2)            % 1 for L2 norm is within the margin, 0 for the L2 distance has exceeded the margin (ignore for bp)
        margin_neg   = doulbe(margin - D_x1_x2)             % denotes "h - D(x1, x2)^2"
        epsilon = 1e-5          % A small value for stability, avoiding 0/0
        %sample_flags = Y + (1-Y)*(-1) .* margin_check 
        sample_flags = (Y + (1-Y)*(-1) .* margin_check .* margin_neg ./ (D_x1_x2 + epsiplon))
        %dz_dx = (Y + (1-Y)*(-1) .* margin_check .* margin_neg ./ (D_x1_x2 + epsiplon))
        % Get the loss
        Loss = sum(Y .* D_x1_x2.^2 + (1-Y) .* margin_neg .* margin_check, 2) / length(Y)

    
	delS = diff .* deri_activation(net1.ff{end}, typeOfActivation);  %%% sensitivity for source 
	delT = -diff .* deri_activation(net2.ff{end}, typeOfActivation); %%% sensitivity for target 

	srcLength = length(net1.ff);
	net.dw = cell(1, srcLength-1);
	net.db = cell(1, srcLength-1);
	net.dw{1, end} = (repmat(sample_flags', size(net1.ff{end-1}', 1), 1).*net1.ff{end-1}') * delS + (repmat(sample_flags', size(net2.ff{end-1}', 1), 1).*net2.ff{end-1}') * delT;
	net.db{1, end} = sample_flags' * delS + sample_flags'* delT;
	for i = srcLength : -1 : 3
    	delS = delS * (net.w{i-1})' .* deri_activation(net1.ff{i-1}, typeOfActivation);
    	delT = delT * (net.w{i-1})' .* deri_activation(net2.ff{i-1}, typeOfActivation);
    	net.dw{i-2} = repmat(sample_flags', size(net1.ff{i-2}', 1), 1).*net1.ff{i-2}' * delS + repmat(sample_flags', size(net2.ff{i-2}', 1), 1).*net2.ff{i-2}' * delT;
    	net.db{i-2} = sample_flags'*delS + sample_flags'*delT;
	end
