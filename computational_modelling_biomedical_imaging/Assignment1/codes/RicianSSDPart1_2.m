function [sumRes, S] = RicianSSDPart1_2(x, Avox, bvals, qhat)
    % Extract the parameters
    S0 = x(1)^2;
    diff = x(2)^2;
    f = (1/(1+exp(-x(3))));
    theta = x(4);
    phi = x(5);
    % Synthesize the signals
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
    % Compute the sum of square differences
    sumRes = sum((Avox - sqrt((S'.^2)+40000)).^2)/40000;
    % tot = 0;
    % for i=1:length(Avox)
    %    temp = ((Avox(i)-sqrt(S(i)^2 + 40000))^2)/40000;
    %    tot = tot + temp;
    % end
    % sumRes = tot;
    % tot = 0;
    % for i=1:length(Avox)
    %     temp = 2*log(200) - log((Avox(i)*S(i))/40000) + ((Avox(i)^2+S(i)^2)/80000);
    %     tot = tot + temp;
    % end
    % sumRes = tot;    
end