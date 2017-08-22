function net = initNet(net, flagW, flagDW)
%% flagW:			flag for initialize (weights)
%% flagDW:			flag for initialize (delta weights)

netLength = length(net.Struct);
for i = 1 : netLength - 1
    if flagW
    	%% randomly initialize the weights and bias
        net.w{i} = 0.1*randn(net.Struct(i), net.Struct(i+1));
        net.b{i} = 0.1*randn(1, net.Struct(i+1));
    end
    if flagDW
        net.dw{i} = zeros(net.Struct(i), net.Struct(i+1));
        net.db{i} = zeros(1, net.Struct(i+1));
    end
end
