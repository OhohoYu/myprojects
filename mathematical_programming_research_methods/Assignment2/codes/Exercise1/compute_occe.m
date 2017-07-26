% function compute_occe.m -> computes optimistic clustering 
% classification error.
%
% inputs -> batches: these are the cluster groups output by the
% k-means algorithm i.e. partitionings C_1, C_2, C_3.
% true_splits contain the true partitionings S_1, S_2, S_3.
%
% outputs -> occe: the optimistic clustering classification error
% for C_i, S_i. 
%
function occe = compute_occe(batches, true_splits)
    % this part computes permutations of cluster groups
    n = length(batches);
    permutes = 1;
    for nn=2:n
        % helper for recursion
        Psmall = permutes;
        m = size(Psmall,1);
        permutes = zeros(nn*m,nn);
        permutes(1:m, 1) = nn;
        permutes(1:m, 2:end) = Psmall;
        for i = nn-1:-1:1
            reorder = [1:i-1, i+1:nn];
            % assign the next m rows in permutes.
            permutes((nn-i)*m+1:(nn-i+1)*m,1) = i;
            permutes((nn-i)*m+1:(nn-i+1)*m,2:end) = reorder(Psmall);
        end
    end
    % inspiration from MATLAB perms function
    if isequal(batches, 1:n)
        permutes = cast(permutes, 'like', batches);
    else
        permutes = batches(permutes);
    end
    % permutes -> batches (vectors of n rows) have created a matrix
    % of n! rows and n columns with all possible permutations of n
    % elements
    tot_runs = size(permutes,1); % total number of iterations
    simple_error = zeros(tot_runs,1); % vector keeping track of simple error
    % initialise cell array keeping track of cluster permutations
    CLUSTER = cell(3,1);
    % initialise cell array keeping track of errors i.e. this 
    % variable keeps track of the number of times x is an element 
    % of C but not of S.
    ERROR = cell(3,1);
    min_simple_error = 100000000;
    for run = 1:tot_runs   % loop over each iterations
        for k = 1:3   % for each k = 1,..3
            CLUSTER{k} = permutes{run,k}; % compute cluster assignments
            % compute number of times x is in C and not in S
            % row/data point mismatches
            ERROR{k} = ~ismember(CLUSTER{k},true_splits{k},'rows');
        end
        % call function for simple error computation
        simple_error(run) = compute_simple_error(ERROR);
        % keep track of minimum simple error
        if simple_error(run) < min_simple_error
               min_simple_error = simple_error(run);
        end    
    end
    % occe is the optimum minimum error
    occe = min_simple_error;
end