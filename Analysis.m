classdef Analysis
    properties
        DB,save
    end 

    methods

        function obj = Analysis(DB)
            obj.DB = DB;
            obj.save = false;
        end 
        
        function Comparative_Analysis(obj)
        
            % Training percentages
            tr = [0.4 0.5 0.6 0.7 0.8 0.9];
        
            % Train all models
            models_return_metrics(obj.DB,1,true,tr);
        
            % Performance metric names
            perf_names = {'ACC','SEN','SPE','F1score','REC',...
                          'PRE','TPR','FPR','PC1','RE1'};
        
            % Output directory
            outDir = fullfile('Analysis1','Comparative_Analysis',obj.DB);
        
            if ~exist(outDir,'dir')
                mkdir(outDir);
            end
        
            % Output filenames
            files_name = cell(length(perf_names),1);
        
            for i = 1:length(perf_names)
                files_name{i} = fullfile(outDir,[perf_names{i} '_1.mat']);
            end
        
            % Load all models
            A = load(fullfile('Temp',obj.DB,'Comp','Model1','all_metrics.mat'));
            B = load(fullfile('Temp',obj.DB,'Comp','Model2','all_metrics.mat'));
            C = load(fullfile('Temp',obj.DB,'Comp','Model3','all_metrics.mat'));
            D = load(fullfile('Temp',obj.DB,'Comp','Model4','all_metrics.mat'));
            E = load(fullfile('Temp',obj.DB,'Comp','Model5','all_metrics.mat'));
            F = load(fullfile('Temp',obj.DB,'Comp','Model6','all_metrics.mat'));
            G = load(fullfile('Temp',obj.DB,'Comp','Model7','all_metrics.mat'));
            H = load(fullfile('Temp',obj.DB,'Comp','Model8','all_metrics.mat'));
            I = load(fullfile('Temp',obj.DB,'Comp','Model9','all_metrics.mat'));
            
            % Cell array containing all models
            all_models = { ...
                A.all_metrics,...
                B.all_metrics,...
                C.all_metrics,...
                D.all_metrics,...
                E.all_metrics,...
                F.all_metrics,...
                G.all_metrics};
        
            if obj.save
        
                for j = 1:length(perf_names)
        
                    new = [];
        
                    for m = 1:length(all_models)
        
                        model_metrics = all_models{m};
        
                        values = zeros(1,length(model_metrics));
        
                        for k = 1:length(model_metrics)
        
                            values(k) = model_metrics{k}(j);
        
                        end
        
                        new = [new; values];
        
                    end
        
                    Perf = new;
        
                    save(files_name{j},'Perf');
        
                end
        
            end
        
        end

        function Performance_Results = PERF_Analysis(DB)

            epoch = [100, 200, 300, 400, 500];
            Training_Percentage = 40;
        
            fprintf('The Performance Analysis Starts...\n');
        
            % Cell array to store all results
            Performance_Results = cell(6,1);
        
            for i = 1:6
        
                fprintf('[%d/6] Performance Analysis Count\n', i);
        
                % Store results for one training percentage
                output = cell(length(epoch),1);
        
                for j = 1:length(epoch)
        
                    ep = epoch(j);
        
                    % Create classifier object
                    C = Models(ep, DB,6);
        
                    % Split the dataset
                    [x_train, x_test, ...
                        y_train, y_test] = train_test_splitter(DB.DB, Training_Percentage/100,100);
        
                    % Run classifier
                    result = C.Classifier_prop2( ...
                                x_train, ...
                                x_test, ...
                                y_train, ...
                                y_test,0);
        
                    % Store result
                    output{j} = result;
        
                end
        
                % Store all epoch results for this training percentage
                Performance_Results{i} = output;
        
                % Increase training percentage
                Training_Percentage = Training_Percentage + 10;
        
            end
        
            fprintf('The results are saved successfully for Performance Analysis...\n');
            fprintf('[OK] Execution of Performance Analysis Completed.\n');
        
        end

        function KF_Analysis(DB)

            % K-Fold values
            kr = [6 7 8 9 10];
        
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
        
            % Create model object
            epochs = 100;
            C = Models(epochs, DB,6);
        
            %==========================
            % Run K-Fold Analysis
            %==========================
            for m = 1:length(modelNames)
        
                modelName = modelNames{m};
        
                saveDir = fullfile('Temp','KF',DB.DB);
        
                if ~exist(saveDir,'dir')
                    mkdir(saveDir);
                end
        
                saveFile = fullfile(saveDir,[modelName '.mat']);
        
                if exist(saveFile,'file')
                    fprintf('Skipping %s because it already exists.\n',modelName);
                    continue;
                end
        
                output = cell(length(kr),1);
        
                for i = 1:length(kr)
        
                    fprintf('\n%s : %d-Fold Validation\n',modelName,kr(i));
        
                    % Load complete dataset
                    [feat,label] = Load_data(DB.DB);
        
                    % Stratified K-Fold
                    cv = cvpartition(label,'KFold',kr(i));
        
                    fold_metrics = [];
        
                    for k = 1:cv.NumTestSets
        
                        fprintf('   Fold %d / %d\n',k,cv.NumTestSets);
        
                        trainIdx = training(cv,k);
                        testIdx  = test(cv,k);
        
                        x_train = feat(trainIdx,:);
                        y_train = label(trainIdx);
        
                        x_test  = feat(testIdx,:);
                        y_test  = label(testIdx);
        
                        switch modelName
        
                            case 'Model1'
                                metrics = C.Model1(x_train,x_test,y_train,y_test);
        
                            case 'Model2'
                                metrics = C.Model2(x_train,x_test,y_train,y_test);
        
                            case 'Model3'
                                metrics = C.Model3(x_train,x_test,y_train,y_test);
        
                            case 'Model4'
                                metrics = C.Model4(x_train,x_test,y_train,y_test);
        
                            case 'Model5'
                                metrics = C.Model5(x_train,x_test,y_train,y_test);
        
                            case 'Model6'
                                metrics = C.Model6(x_train,x_test,y_train,y_test);
        
                            case 'Model7'
                                metrics = C.Model7(x_train,x_test,y_train,y_test);
        
                            case 'Model8'
                                metrics = C.Classifier_prop(x_train,x_test,y_train,y_test);

                            case 'Model9'
                                metrics = C.Classifier_prop2(x_train,x_test,y_train,y_test);
        
                        end
        
                        % metrics should be a row vector
                        fold_metrics = [fold_metrics; metrics];
        
                    end
        
                    % Mean of all folds
                    output{i} = mean(fold_metrics,1);
        
                end
        
                save(saveFile,'output');
        
                fprintf('Saved %s\n',saveFile);
        
            end
        
            %====================================
            % Collect all model outputs
            %====================================
        
            A = load(fullfile('Temp','KF',DB,'Model1.mat'));
            B = load(fullfile('Temp','KF',DB,'Model2.mat'));
            C1 =load(fullfile('Temp','KF',DB,'Model3.mat'));
            D = load(fullfile('Temp','KF',DB,'Model4.mat'));
            E = load(fullfile('Temp','KF',DB,'Model5.mat'));
            F = load(fullfile('Temp','KF',DB,'Model6.mat'));
            G = load(fullfile('Temp','KF',DB,'Model7.mat'));
            H = load(fullfile('Temp','KF',DB,'Model8.mat'));
        
            A = cell2mat(A.output);
            B = cell2mat(B.output);
            C1 = cell2mat(C1.output);
            D = cell2mat(D.output);
            E = cell2mat(E.output);
            F = cell2mat(F.output);
            G = cell2mat(G.output);
            H = cell2mat(H.output);
        
            perf_names = { ...
                'ACC',...
                'SEN',...
                'SPE',...
                'F1score',...
                'REC',...
                'PRE',...
                'TPR',...
                'FPR'};
        
            analysisDir = fullfile('Analysis1','KF_Analysis',DB);
        
            if ~exist(analysisDir,'dir')
                mkdir(analysisDir);
            end
        
            for i = 1:length(perf_names)
        
                max_out = [
                    A(:,i)'; ...
                    B(:,i)'; ...
                    C1(:,i)'; ...
                    D(:,i)'; ...
                    E(:,i)'; ...
                    F(:,i)'; ...
                    G(:,i)'; ...
                    H(:,i)'
                    ];
        
                save(fullfile(analysisDir,[perf_names{i} '.mat']),...
                    'max_out');
        
            end
        
            fprintf('\nK-Fold Analysis Completed Successfully.\n');
        
        end
    end 
end