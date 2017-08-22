function net = netff(feaSet, net)

netLength = length(net.w);   
net.ff{1} = feaSet;


for i = 1 : netLength
    switch net.activation_function
        case 'sigmoid'
%             net.ff{i+1} = sigmf(net.ff{i} * net.w{i} + repmat(net.b{i}, size(net.ff{i}, 1), 1), [1, 0]);
            net.ff{i+1} = sigmoid(net.ff{i} * net.w{i} + repmat(net.b{i}, size(net.ff{i}, 1), 1));
        case 'tanh'
            net.ff{i+1} = tanh(net.ff{i} * net.w{i} + repmat(net.b{i}, size(net.ff{i}, 1), 1));
        case 'relu'
            net.ff{i+1} = max(net.ff{i} * net.w{i} + repmat(net.b{i}, size(net.ff{i}, 1), 1), 0);
    end
end
   