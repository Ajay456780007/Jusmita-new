function models_return_metrics(DB, epochs, ok, percents)

    if nargin < 3
        ok = true;
    end

    if nargin < 4 || isempty(percents)
        training_percentages = 0.4;
        % training_percentages = [0.4 0.5 0.6 0.7 0.8 0.9];
    else
        training_percentages = percents;
    end

    % Create model object
    C = Models(epochs, DB,1);

    % Model names
    modelNames = { ...
        'Model1', ...
        'Model2', ...
        'Model3', ...
        'Model4', ...
        'Model5', ...
        'Model6', ...
        'Model7', ...
        'Model8',...
        'Model9'};

    if ok

        for m = 1:length(modelNames)

            modelName = modelNames{m};

            fprintf('\n==== Training model: %s ====\n', modelName);

            all_metrics = cell(length(training_percentages),1);

            for p = 1:length(training_percentages)

                percent = training_percentages(p);

                fprintf('  -> Training %s with %d%% training data...\n', ...
                    modelName, round(percent*100));

                % Split data
                [x_train,x_test,y_train,y_test] = ...
                    train_test_splitter(DB,percent);

                % Call corresponding model
                switch modelName

                    case 'Model1'
                        metrics = C.Model7(x_train,x_test,y_train,y_test);
                    % 
                    % case 'Model2'
                    %     metrics = C.Model2(x_train,x_test,y_train,y_test);
                    % 
                    % case 'Model3'
                    %     metrics = C.Model3(x_train,x_test,y_train,y_test);
                    % 
                    % case 'Model4'
                    %     metrics = C.Model4(x_train,x_test,y_train,y_test);
                    % 
                    % case 'Model5'
                    %     metrics = C.Model5(x_train,x_test,y_train,y_test);
                    % 
                    % case 'Model6'
                    %     metrics = C.Model6(x_train,x_test,y_train,y_test);

                    case 'Model7'
                        metrics = C.Model7(x_train,x_test,y_train,y_test);

                    % case 'Model8'
                    %     metrics = C.Classifier_prop(x_train,x_test,y_train,y_test,false);

                end

                all_metrics{p} = metrics;

            end

            % Create directory
            saveDir = fullfile('Temp',DB,'Comp',modelName);

            if ~exist(saveDir,'dir')
                mkdir(saveDir);
            end

            % Save metrics
            save(fullfile(saveDir,'all_metrics.mat'), ...
                'all_metrics');

            fprintf('✔ Saved all metrics for %s to %s\n', ...
                modelName, ...
                fullfile(saveDir,'all_metrics.mat'));

        end

    end
end