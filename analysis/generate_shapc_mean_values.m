% SHAPC-Mean and SHAPC-Var Comparison
setup; % Loads alg list and dataset_configs
       % If not all algs are needed either
       % change the algs listed in setup.m
       % OR uncomment the below line of code

%algs = ["iTAML"];       

dataset = 'cifar10';
config = dataset_configs.(dataset);
num_sessions = config.num_sessions;
num_classes = config.num_classes;
cls_per_task = config.cls_per_task;
samples_per_cls = config.samples_per_cls;

shapc_path = sprintf("%s_shapc_data.mat", dataset);

if isfile(shapc_path)
    sh_nm = fieldnames(load(shapc_path));
    shapc_data = load(shapc_path).(string(sh_nm));
else
    shapc_data = struct();
end
%%
% Load SHAPC values (First and Last 1000)
for i=1:length(algs)
    alg = algs(i);
    if strcmp(dataset, "cifar100")
        save_path = sprintf("%s/%s/shapc_vals_first_last_2000.mat", alg, dataset);
    elseif strcmp(dataset, "imagenet200")
        save_path = sprintf("%s/%s/shapc_vals_first_last_4000.mat", alg, dataset);
    else
        save_path = sprintf("%s/%s/shapc_vals_first_last_1000.mat", alg, dataset);
    end

    if isfile(save_path)
        shapc_struct = load(save_path);
       
        % Load SHAPC values all
        shapc_avgs = [];
        for i=1:num_sessions-1
            pair_str = 'sc' + string(i-1) +string(num_sessions-1);
            shapcs = [];
            sample_list = string(fieldnames(shapc_struct.(pair_str)));
            for k=1:length(fieldnames(shapc_struct.(pair_str)))
                sample_str = sample_list(k);
                shapcs = [shapcs; shapc_struct.(pair_str).(sample_str)];
            end
            shapc_avgs = [shapc_avgs; mean(shapcs)];

        end
        
        % Note the shapc is represented as percentage
        %disp("From averaging all tasks together:")
        shapc_mean_perc = mean(shapc_avgs);
        shapc_mean = mean(shapc_avgs) / 100;
    else
        shapc_mean_perc = NaN;
        shapc_mean = NaN;
    end

    if strcmp(dataset, "cifar100")
        shapc_str = "first_last_2000_shapc";
        time_str = "first_last_2000_time";
    elseif strcmp(dataset, "imagenet200")
        shapc_str = "first_last_4000_shapc";
        time_str = "first_last_4000_time";
    else
        shapc_str = "first_last_1000_shapc";
        time_str = "first_last_1000_time";
    end
    shapc_data.(alg).(shapc_str) = shapc_mean_perc;

    %shapc_data.(alg).first_last_time = shapTimes.(dataset).first_last.(alg);
end
save(shapc_path, "shapc_data")

% Create Table
%shapc_table = table(Y', 'VariableNames', ["SHAPC-Mean"], 'RowNames', X)
%abs_diff = abs(Y(1)-Y(2))
rows = algs;
columns = ["Accuracy (%)" "SHAPC-Mean (%)" "Time (hrs)"];
column_data1 = [];
column_data2 = [];
column_data3 = [];
for i=1:length(algs)
    alg = algs(i);
    if ~isfield(shapc_data.(alg), 'acc')
        shapc_data.(alg).acc = NaN;
    end
    column_data1 = [column_data1; shapc_data.(alg).acc];
    
    column_data2 = [column_data2; shapc_data.(alg).(shapc_str)];

    if ~isfield(shapc_data.(alg), time_str)
        shapc_data.(alg).(time_str) = NaN;
    end
    column_data3 = [column_data3; shapc_data.(alg).(time_str)];
end
first_last_1000_shapcs = column_data2;
first_last_1000_times = column_data3;
shapc_table_first_last_1000 = table(column_data1, column_data2, column_data3, ...
    'VariableNames', columns, 'RowNames', rows);
sorted_shapc_first_last_1000 = sortrows(shapc_table_first_last_1000, {'SHAPC-Mean (%)'}, {'ascend'});
disp(sorted_shapc_first_last_1000) %[output:63fdabc1]


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":34.1}
%---
%[output:63fdabc1]
%   data: {"dataType":"text","outputData":{"text":"              <strong>Accuracy (%)<\/strong>    <strong>SHAPC-Mean (%)<\/strong>    <strong>Time (hrs)<\/strong>\n              <strong>____________<\/strong>    <strong>______________<\/strong>    <strong>__________<\/strong>\n\n    <strong>RPSnet<\/strong>        61.35           23.217          6.9869  \n    <strong>xder  <\/strong>         55.8           25.706            0.44  \n    <strong>foster<\/strong>        71.61           28.713         0.50417  \n    <strong>memo  <\/strong>        88.36           29.583         0.68417  \n    <strong>iTAML <\/strong>        93.45           30.672         0.57083  \n    <strong>icarl <\/strong>        86.45           31.202         0.15694  \n    <strong>dsal  <\/strong>        72.15            36.43          0.1775  \n    <strong>tagfex<\/strong>       90.402           40.791          1.8206  \n    <strong>der   <\/strong>           89           41.909          1.5094  \n\n","truncated":false}}
%---
