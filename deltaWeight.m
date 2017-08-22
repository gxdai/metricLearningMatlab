function net = deltaWeight(net, net_, momentum)

for i = 1 : length(net.w)
        net.dw{i} = momentum*net.dw{i} + net_.dw{i};
        net.db{i} = momentum*net.db{i} + net_.db{i};
end


