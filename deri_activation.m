function output = deri_activation(x, typeOfActivation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch typeOfActivation
    case 'sigmoid'
        output = x .* ( 1 - x);
    case 'tanh'
        output = 1 - x.*x;
    case 'relu'
        output = (x>0);
end