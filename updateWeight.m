function net = updateWeight(net, learningRate)

    for i = 1 : length(net.w)
        net.w{i} = net.w{i} + learningRate*net.dw{i};
        net.b{i} = net.b{i} + learningRate*net.db{i};
    end