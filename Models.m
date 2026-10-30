classdef Models

    properties
        epochs 
        DB
        num_classes = 1;
    end 

    methods
        % Constructor
        function obj = Models(ep,DB,num_classes)
            obj.epochs = ep;
            obj.DB = DB;
            obj.num_classes = num_classes;
        end 

        % Model 1: 1D CNN for Sequence Data
        function metrics = Model1(obj, XTrain, XTest, y_train, y_test)
            
            XTrainCell = cell(size(XTrain,1),1);

            for i = 1:size(XTrain,1)
                XTrainCell{i} = XTrain(i,:)';
            end

            XTestCell = cell(size(XTest,1),1);

            for i = 1:size(XTest,1)
                XTestCell{i} = XTest(i,:)';
            end

            % Number of features
            if iscell(XTrain)
                numFeatures = size(XTrain{1},1);
            else
                numFeatures = size(XTrain,2);
            end
        
            layers = [
            sequenceInputLayer(numFeatures,"Name","input")
        
            convolution1dLayer(3,128,"Padding","causal")
            reluLayer
            dropoutLayer(0.2)
        
            convolution1dLayer(3,64,"Padding","causal")
            reluLayer
            dropoutLayer(0.2)
        
            convolution1dLayer(3,32,"Padding","causal")
            reluLayer
            dropoutLayer(0.2)
        
            globalAveragePooling1dLayer
        
            fullyConnectedLayer(obj.num_classes)
            regressionLayer                 
        ];
        
            options = trainingOptions('adam', ...
                'MaxEpochs',obj.epochs,...
                'MiniBatchSize',32,...
                'InitialLearnRate',0.001,...
                'GradientThreshold',1,...
                'Shuffle','every-epoch',...
                'Verbose',false);
        
            %% Train
        
            % if ~iscategorical(y_train)
            %     y_train = categorical(y_train);
            % end
            
            net = trainNetwork(XTrainCell, y_train, layers, options);
        
            %% Save
            modelFolder = fullfile(pwd, "Models");
    
            if ~exist(modelFolder, "dir")
                mkdir(modelFolder);
            end
            
            save(fullfile(modelFolder, "Model1.mat"), "net");
        
            %% Predict
        
             pred = predict(net, XTestCell);
        
            %% Convert categorical to numeric
        
            pred = double(pred);
            y_test = double(y_test);
        
            %% Performance metrics
        
            metrics = Evaluation_Metrics1(y_test,pred);
        
        end
    
        function metrics = Model2(obj, XTrain, XTest, YTrain, YTest)

            %% Number of Features and Classes
        
            numFeatures = size(XTrain,2);
        
            %% Network
        
            layers = [
                featureInputLayer(numFeatures,'Name','input','Normalization','zscore')
        
                fullyConnectedLayer(128,'Name','fc_block1')
                reluLayer('Name','relu_1')
                dropoutLayer(0.2,'Name','drop_1')
        
                fullyConnectedLayer(64,'Name','fc_block2')
                reluLayer('Name','relu_2')
                dropoutLayer(0.2,'Name','drop_2')
        
                fullyConnectedLayer(32,'Name','fc_block3')
                reluLayer('Name','relu_3')
                dropoutLayer(0.2,'Name','drop_3')
        
                fullyConnectedLayer(obj.num_classes,'Name','fc_out')
                regressionLayer
            ];
        
            %% Training Options
        
            options = trainingOptions('adam', ...
                'MaxEpochs',obj.epochs,...
                'MiniBatchSize',32,...
                'InitialLearnRate',0.001,...
                'GradientThreshold',1,...
                'Shuffle','every-epoch',...
                'Plots','none',...
                'Verbose',false);
        
            %% Train
        
            % if ~iscategorical(YTrain)
            %     YTrain = categorical(YTrain);
            % end
            
            net2 = trainNetwork(XTrain, YTrain, layers, options);
                    
            %% Save
        
            modelFolder = fullfile(pwd,"Models");
        
            if ~exist(modelFolder,"dir")
                mkdir(modelFolder);
            end
        
            save(fullfile(modelFolder,"Model2.mat"),"net2");
        
            %% Predict
        
            pred = predict(net2, XTest);
        
            %% Convert categorical to numeric
        
            % if ~iscategorical(YTest)
            %     YTest = categorical(YTest);
            % end
            
            pred = double(pred);
            y_test = double(YTest);
                    
            %% Performance Metrics
        
            metrics = Evaluation_Metrics1(y_test,pred);
        
        end


        function metrics = Model3(obj, XTrain, XTest, YTrain, YTest)
            
            XTrainCell = cell(size(XTrain,1),1);
            
            for i = 1:size(XTrain,1)
                XTrainCell{i} = XTrain(i,:)';
            end
            
            XTestCell = cell(size(XTest,1),1);
            
            for i = 1:size(XTest,1)
                XTestCell{i} = XTest(i,:)';
            end
            % Number of features
            if iscell(XTrain)
                numFeatures = size(XTrain{1},1);
            else
                numFeatures = size(XTrain,2);
            end
        
            % if ~iscategorical(YTrain)
            %     YTrain = categorical(YTrain);
            % end
            % 
            % if ~iscategorical(YTest)
            %     YTest = categorical(YTest);
            % end
        
            % numClasses = numel(unique(YTrain));
        
            layers = [
                sequenceInputLayer(numFeatures, 'Name', 'input')
        
                lstmLayer(128, 'OutputMode', 'sequence', 'Name', 'rnn_block1')
                reluLayer('Name', 'relu_1')
                dropoutLayer(0.2, 'Name', 'drop_1')
        
                lstmLayer(64, 'OutputMode', 'sequence', 'Name', 'rnn_block2')
                reluLayer('Name', 'relu_2')
                dropoutLayer(0.2, 'Name', 'drop_2')
        
                lstmLayer(32, 'OutputMode', 'last', 'Name', 'rnn_block3')
                reluLayer('Name', 'relu_3')
                dropoutLayer(0.2, 'Name', 'drop_3')
        
                fullyConnectedLayer(obj.num_classes, 'Name', 'fc_out')
                regressionLayer
            ];
        
            options = trainingOptions('adam', ...
                'MaxEpochs', obj.epochs, ...
                'MiniBatchSize', 64, ...
                'InitialLearnRate', 0.001, ...
                'GradientThreshold', 1, ...
                'Shuffle', 'never', ...
                'Plots', 'none', ...
                'Verbose', false);
        
            %% Train
            net3 = trainNetwork(XTrainCell, YTrain, layers, options);
        
            %% Save
            if ~exist('Models','dir')
                mkdir('Models');
            end
            save("Models/Model3.mat","net3");
        
            %% Predict
            pred = predict(net3, XTestCell);
        
            %% Convert categorical to numeric
            pred = grp2idx(pred);
            y_test = grp2idx(YTest);
        
            %% Calculate metrics
            metrics = Evaluation_Metrics1(y_test, pred);
        
        end

        function metrics = Model4(obj, XTrain, XTest, YTrain, YTest)
        
            YTrain = double(YTrain(:));
            YTest = double(YTest(:));
        
            net4 = fitrtree(XTrain, YTrain);
        
            modelFolder = fullfile(pwd,"Models");
        
            if ~exist(modelFolder,"dir")
                mkdir(modelFolder);
            end
        
            save(fullfile(modelFolder,"Model4.mat"),"net4");
        
            pred = predict(net4, XTest);
        
            pred = double(pred(:));
            y_test = YTest;
        
            metrics = Evaluation_Metrics1(y_test, pred);
        
        end

     
        function metrics = Model5(obj, XTrain, XTest, YTrain, YTest)

            %% Convert Training Data to Cell Array
        
            XTrainCell = cell(size(XTrain,1),1);
        
            for i = 1:size(XTrain,1)
                XTrainCell{i} = XTrain(i,:)';
            end
        
            XTestCell = cell(size(XTest,1),1);
        
            for i = 1:size(XTest,1)
                XTestCell{i} = XTest(i,:)';
            end
        
            %% Number of Features
        
            numFeatures = size(XTrain,2);
        
            %% Convert Labels
        
            % if ~iscategorical(YTrain)
            %     YTrain = categorical(YTrain);
            % end
            % 
            % if ~iscategorical(YTest)
            %     YTest = categorical(YTest);
            % end
        
            % numClasses = numel(categories(YTrain));
        
            %% LSTM Network
        
            layers = [
        
                sequenceInputLayer(numFeatures,'Name','input')
        
                lstmLayer(128,'OutputMode','sequence','Name','lstm_block1')
                reluLayer('Name','relu_1')
                dropoutLayer(0.2,'Name','drop_1')
        
                lstmLayer(64,'OutputMode','sequence','Name','lstm_block2')
                reluLayer('Name','relu_2')
                dropoutLayer(0.2,'Name','drop_2')
        
                lstmLayer(32,'OutputMode','last','Name','lstm_block3')
                reluLayer('Name','relu_3')
                dropoutLayer(0.2,'Name','drop_3')
        
                fullyConnectedLayer(obj.num_classes,'Name','fc_out')
                regressionLayer
        
            ];
        
            %% Training Options
        
            options = trainingOptions('adam', ...
                'MaxEpochs',obj.epochs,...
                'MiniBatchSize',32,...
                'InitialLearnRate',0.001,...
                'GradientThreshold',1,...
                'Shuffle','every-epoch',...
                'Plots','none',...
                'Verbose',false);
        
            %% Train
        
            net5 = trainNetwork(XTrainCell,YTrain,layers,options);
        
            %% Save
        
            modelFolder = fullfile(pwd,"Models");
        
            if ~exist(modelFolder,"dir")
                mkdir(modelFolder);
            end
        
            save(fullfile(modelFolder,"Model5.mat"),"net5");
        
            %% Predict
        
            pred = predict(net5,XTestCell);
        
            %% Convert Labels to Numeric
        
            pred = grp2idx(pred);
            y_test = grp2idx(YTest);
        
            %% Performance Metrics
        
            metrics = Evaluation_Metrics1(y_test,pred);
        
        end
        
        function metrics = Model6(obj, XTrain, XTest, YTrain, YTest)

            %% Convert Training Data to Cell Array
        
            XTrainCell = cell(size(XTrain,1),1);
        
            for i = 1:size(XTrain,1)
                XTrainCell{i} = XTrain(i,:)';
            end
        
            XTestCell = cell(size(XTest,1),1);
        
            for i = 1:size(XTest,1)
                XTestCell{i} = XTest(i,:)';
            end
        
            %% Number of Features
        
            numFeatures = size(XTrain,2);
        
            %% Convert Labels
        
            % if ~iscategorical(YTrain)
            %     YTrain = categorical(YTrain);
            % end
            % 
            % if ~iscategorical(YTest)
            %     YTest = categorical(YTest);
            % end
            
            % numClasses = numel(categories(YTrain));
        
            %% Network
        
            layers = [
        
                sequenceInputLayer(numFeatures,'Name','input')
        
                bilstmLayer(64,'OutputMode','sequence','Name','bilstm_block1')
                reluLayer('Name','relu_1')
                dropoutLayer(0.2,'Name','drop_1')
        
                bilstmLayer(32,'OutputMode','sequence','Name','bilstm_block2')
                reluLayer('Name','relu_2')
                dropoutLayer(0.2,'Name','drop_2')
        
                bilstmLayer(16,'OutputMode','sequence','Name','bilstm_block3')
                reluLayer('Name','relu_3')
                dropoutLayer(0.2,'Name','drop_3')
        
%               selfAttentionLayer(4,32,'Name','self_attention')
        
                globalAveragePooling1dLayer('Name','global_pool')
        
                fullyConnectedLayer(obj.num_classes,'Name','fc_out')
                regressionLayer
        
            ];
        
            %% Training Options
        
            options = trainingOptions('adam', ...
                'MaxEpochs',obj.epochs,...
                'MiniBatchSize',128,...
                'InitialLearnRate',0.001,...
                'GradientThreshold',1,...
                'Shuffle','every-epoch',...
                'Plots','none',...
                'Verbose',false);
        
            %% Train
        
            net6 = trainNetwork(XTrainCell,YTrain,layers,options);
        
            %% Save
        
            modelFolder = fullfile(pwd,"Models");
        
            if ~exist(modelFolder,"dir")
                mkdir(modelFolder);
            end
        
            save(fullfile(modelFolder,"Model6.mat"),"net6");
        
            %% Predict
        
            pred = predict(net6,XTestCell);
        
            %% Convert Labels to Numeric
        
            pred = grp2idx(pred);
            y_test = grp2idx(YTest);
        
            %% Performance Metrics
        
            metrics = Evaluation_Metrics1(y_test,pred);
        
        end
        function metrics = Model7(obj, XTrain, XTest, YTrain, YTest)
            %% 1. Standardize Input Sequences into Cell Arrays
            if ~iscell(XTrain)
                XTrainCell = cell(size(XTrain, 1), 1);
                for i = 1:size(XTrain, 1)
                    XTrainCell{i} = XTrain(i, :)';
                end
            else
                XTrainCell = XTrain;
            end
        
            if ~iscell(XTest)
                XTestCell = cell(size(XTest, 1), 1);
                for i = 1:size(XTest, 1)
                    XTestCell{i} = XTest(i, :)';
                end
            else
                XTestCell = XTest;
            end
        
            numFeatures = size(XTrainCell{1}, 1);
        
            %% 2. Architectural Hyperparameters (Tuned for CPU)
            numHeads = 2;          % Reduced heads for low CPU overhead
            gamma = 2; 
            beta = 1;
            reductionRatio = 2; 
        
            %% 3. CPU-Streamlined Layer Graph
            layers = [
                sequenceInputLayer(numFeatures, 'Name', 'input')
        
                % Fast 1D Convolutions with temporal downsampling (Strides reduce sequence length)
                convolution1dLayer(3, 256, 'Padding', 'same', 'Stride', 2, 'Name', 'conv1d_1')
                batchNormalizationLayer('Name', 'batch_norm_1')
                reluLayer('Name', 'relu_conv_1')
        
                convolution1dLayer(3, 128, 'Padding', 'same', 'Stride', 2, 'Name', 'conv1d_2')
                batchNormalizationLayer('Name', 'batch_norm_2')
                reluLayer('Name', 'relu_conv_2')
        
                gruLayer(256, 'OutputMode', 'sequence', 'Name', 'gru_enc_1')
                dropoutLayer(0.2, 'Name', 'gru_drop_1')
                
                gruLayer(128, 'OutputMode', 'sequence', 'Name', 'gru_enc_2')
                dropoutLayer(0.2, 'Name', 'gru_drop_2')
                
                gruLayer(64, 'OutputMode', 'sequence', 'Name', 'gru_enc_3')
        
                % Attention operating on short feature length (64 channels)
                selfAttentionLayer(numHeads, 64, 'Name', 'mha_base')
                
                % CORRECTED: Replaced cross-attention 'attentionLayer' with 'selfAttentionLayer'
                % This enables sequential processing over a single feature stream.
                selfAttentionLayer(4, 64, 'Name', 'mha_secondary') 
                
                ChannelAttentionLayer(reductionRatio, 64, 'Name', 'channel_attn')
                ECA1DLayer(gamma, beta, 'Name', 'eca_block')
                
                % Collapse temporal dimension
                globalAveragePooling1dLayer('Name', 'gap')
                
                fullyConnectedLayer(obj.num_classes, 'Name', 'fc_out')
                regressionLayer('Name', 'regression_output')
            ];
        
            lgraph = layerGraph(layers);
        
            %% 4. Format Target Data
            if iscell(YTrain)
                YTrain = cell2mat(YTrain);
            end
            YTrain = double(YTrain);
        
            if isvector(YTrain) && obj.num_classes > 1
                YTrainMat = full(ind2vec(grp2idx(YTrain)', obj.num_classes))';
            else
                YTrainMat = YTrain;
            end
        
            %% 5. CPU Training Options
            options = trainingOptions('adam', ...
                'MaxEpochs', obj.epochs, ...
                'MiniBatchSize', 16, ...                   % Small batch size optimal for CPU cache
                'ExecutionEnvironment', 'cpu', ...         % Explicitly target CPU
                'InitialLearnRate', 0.001, ...
                'GradientThreshold', 1, ...
                'Shuffle', 'every-epoch', ...
                'Plots', 'none', ...
                'Verbose', false);
        
            %% 6. Train Network
            net7 = trainNetwork(XTrainCell, YTrainMat, lgraph, options);
        
            %% 7. Save Model
            modelFolder = fullfile(pwd, "Models");
            if ~exist(modelFolder, "dir")
                mkdir(modelFolder);
            end
            save(fullfile(modelFolder, "Model7.mat"), "net7");
        
            %% 8. Inference
            pred = predict(net7, XTestCell);
        
            if size(pred, 2) > 1
                [~, pred_idx] = max(pred, [], 2);
            else
                pred_idx = round(pred);
            end
        
            if iscell(YTest)
                YTest = cell2mat(YTest);
            end
            y_test = grp2idx(YTest);
        
            %% 9. Metrics Calculation
            metrics = Evaluation_Metrics1(y_test, pred_idx);
        end

    end
end

% https://www.mathworks.com/matlabcentral/fileexchange/184243-customized-invertible-attention-layer/files/InvertibleAttentionLayer.m