function [sumRes, S, AIC, BIC] = ZeppelinStickTortuositySSD_IC(x, Avox, bvals, qhat)
    % Extract the parameters
    S0 = x(1)^2;
    diff = x(2)^2;
    f = (1/(1+exp(-x(3))));
    theta = x(4);
    phi = x(5);   
    lambda1 = 5;
    % Synthesize the signals
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*((1-f) + (lambda1-(1-f)).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
    BIC = (6*log(3612)) + (3612*log(sumRes/3612));
    AIC = 12 + (3612*log(sumRes/3612));
end