function [net, errNorm] = netbp(fea1, label1, fea2, label2, net, h)
	
	net1 = netff(fea1, net);  
	net2 = netff(fea2, net); 
	err = net1.ff{end} - net2.ff{end};
	typeOfActivation = net.activation_function;

	labelCheck1 = (label1 == label2);
	err1 = (sum(err .* err, 2)).^0.5;  			%%% the err 
	hAll = repmat(h, length(err1), 1);  
	deltaH1 = (hAll >= err1);  					%%% thresh check 
	coeff1 = labelCheck1 - 0.5;
	newCoeff1 = 2*(coeff1 .* or(labelCheck1, deltaH1));


	%% calculating error
	err2 = sum(err.*err, 2).^0.5;
	errNorm = 2 * err2.*coeff1;
	errNorm = sum(errNorm);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	delS = err .* deri_activation(net1.ff{end}, typeOfActivation);  %%% sensitivity for source 
	delT = -err .* deri_activation(net2.ff{end}, typeOfActivation); %%% sensitivity for target 


	srcLength = length(net1.ff);
	net.dw = cell(1, srcLength-1);
	net.db = cell(1, srcLength-1);
	net.dw{1, end} = (repmat(newCoeff1', size(net1.ff{end-1}', 1), 1).*net1.ff{end-1}') * delS + (repmat(newCoeff1', size(net2.ff{end-1}', 1), 1).*net2.ff{end-1}') * delT;
	net.db{1, end} = newCoeff1' * delS + newCoeff1'* delT;
	for i = srcLength : -1 : 3
    	delS = delS * (net.w{i-1})'.* deri_activation(net1.ff{i-1}, typeOfActivation);
    	delT = delT * (net.w{i-1})'.* deri_activation(net2.ff{i-1}, typeOfActivation);
    	net.dw{i-2} = repmat(newCoeff1', size(net1.ff{i-2}', 1), 1).*net1.ff{i-2}' * delS + repmat(newCoeff1', size(net2.ff{i-2}', 1), 1).*net2.ff{i-2}' * delT;
    	net.db{i-2} = newCoeff1'*delS + newCoeff1'*delT;
	end