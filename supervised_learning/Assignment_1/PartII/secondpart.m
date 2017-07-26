clear all;
max_dimensions = 80;
n_trials = 50; % number of iterations for averaging
low_dim_limit = 30;
mid_dim_limit = 50;
high_dim_limit = 65;
low_multiplier = 60;
mid_multiplier = 40;
high_multiplier = 30;
higher_multiplier = 20;
rng(45) % set random seed
limit = 0.1; % target generalisation error
init_error = 10^8; % error initialisation 

for dimension = 1:max_dimensions
    if dimension <= low_dim_limit; 
        n_examples = dimension*low_multiplier;
    elseif dimension > low_dim_limit && dimension <=mid_dim_limit 
        n_examples = dimension*mid_multiplier;
    elseif dimension > mid_dim_limit && dimension <=high_dim_limit 
        n_examples = dimension*high_multiplier;
    else
        n_examples = dimension*higher_multiplier;
    end
    x = zeros(n_examples,dimension,n_trials);
    for trial = 1:n_trials    % iterations for averaging
        % matrix generation -1s and 1s
        x(:,:,trial) = ones(n_examples,dimension) ... 
                           - floor(rand(n_examples,dimension)*2)*2;
        % PERCEPTRON
        error = init_error;   % initialisation
        example = 1; % initialisation
        while error>limit
            weights_PERC = ones(1,dimension); % weights init
            x_train = x(1:example,:,trial);
            y_train = x_train(:,1);
            x_test = x(example+1:end,:,trial);
            weights_PERC = perceptron_w(x_train,y_train,weights_PERC);
            y_pred_PERC = perceptron_pred(x_test,weights_PERC);
            error = accuracy(x_test,y_pred_PERC);
            example = example+1;
        end
        % sample complexity
        error_PERC(dimension,trial) = error;
        example_PERC(dimension,trial) = example;
        % winnow 
        error = init_error; % initialisation
        example = 1; % initialisation
        while error>limit
            weights_WINNOW= zeros(1,dimension);  % initialisation
            weights_WINNOW(1,:) = ones(1,dimension); % initialisation
            x_train = x(1:example,:,trial);
            y_train = x_train(:,1);
            x_test = x(example+1:end,:,trial);
            weights_WINNOW_new = winnow_w(x_train,y_train,weights_WINNOW);
            y_pred_WINNOW = winnow_pred(x_test,weights_WINNOW_new);
            error = accuracy(x_test,y_pred_WINNOW);
            example = example+1;
        end
        % SAMPLE COMPLEXITY
        error_winnow(dimension,trial) = error;
        example_WINNOW(dimension,trial) = example;                       
        % LEAST SQUARES
        error = init_error; % initialisation
        example = 1; % initialisation
        while error>limit
            x_train = x(1:example,:,trial);
            y_train = x_train(:,1);
            x_test = x(example+1:end,:,trial);
            weights_LS = least_squares_w(x_train,y_train);
            y_pred_LS = least_squares_pred(x_test,weights_LS);
            error = accuracy(x_test,y_pred_LS);
            example = example+1;
        end
        %sample complexity
        error_LS(dimension,trial) = error;
        example_LS(dimension,trial) = example;
    end
    if mod(dimension,5) == 0
            disp('Generalisation error calculated for # dimensions: ');dimension
    end    
end

dimensions = 1:max_dimensions;

% plots
figure
errorbar(dimensions,mean(example_PERC,2),std(example_PERC'),'-r');
xlabel('Dimensions,n');
ylabel('Sample Complexity,m');
hold on;

figure;
errorbar(dimensions,mean(example_WINNOW,2),std(example_WINNOW'), '-b');
xlabel('Dimensions,n');
ylabel('Sample Complexity,m');
hold on;

figure;
errorbar(dimensions,mean(example_LS,2),std(example_LS'), '-k');
xlabel('Dimensions, n');
ylabel('Sample Complexity, m');

% K-NN
% ntrials = 1;
% max_dimensions = 25
% for dimension = 1:max_dimensions
%   if dimension <= low_dim_limit; 
%        n_examples = dimension*low_multiplier;
%   elseif dimension > low_dim_limit && dimension <=mid_dim_limit 
%        n_examples = dimension*mid_multiplier;
%   elseif dimension > mid_dim_limit && dimension <=high_dim_limit 
%        n_examples = dimension*high_multiplier;
%   else
%        n_examples = dimension*higher_multiplier;
%   end
%   x = zeros(n_examples,dimension,n_trials);
%   for trial = 1:n_trials    % iterations for averaging
        % matrix generation -1s and 1s
%        x(:,:,trial) = ones(n_examples,dimension) ... 
%                           - floor(rand(n_examples,dimension)*2)*2;
       
%        error = 10^8;   % initialisation
%        example = 1; % initialisation
%        while error>0.1
%        x_train = x(1:example,:,trial);
%        y_train = x_train(:,1);
%        x_test = x(example+1:end,:,trial);
%        Mdl = fitcknn(x_train,y_train,'NumNeighbors',1,'Standardize',1);
%        y_pred = predict(Mdl,x_test);
%           
%        error = accuracy(x_test,y_pred);
%        example= m+1;
%     end
%     example_NN(dimension,trial) = example;
%     error_NN(dimension,trial) = error;

% end

% dimensions = 1:25
% figure;
% (dimensions,mean(example_NN,2));
% xlabel('dimensions',n);
% ylabel('Sample Complexity, m');

