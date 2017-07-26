% function compute_simple_error.m -> helper function
% to compute simple error. works  
%
% inputs -> ERROR. cell array with number of times a data point is 
% in C but not in S.
%
% outputs -> simple_error for a given run 
%
function simple_error = compute_simple_error(ERROR)
    tot_error = 0; % initialise total error
    tot_points = 150; % total number of points in data set
    for k=1:3
        error_component = sum(ERROR{k}); % error for a given i=1,...k
        tot_error = tot_error + error_component;
    end
    simple_error = tot_error/tot_points; % returns simple error
end