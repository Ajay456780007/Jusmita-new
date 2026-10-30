classdef ChannelAttentionLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        ReductionRatio
    end
    
    properties (Learnable)
        Weights1 % [reducedChannels, inputChannels]
        Bias1    % [reducedChannels, 1]
        Weights2 % [inputChannels, reducedChannels]
        Bias2    % [inputChannels, 1]
    end
    
    methods
        function layer = ChannelAttentionLayer(reductionRatio, inputChannels, args)
            arguments
                reductionRatio (1,1) {mustBePositive}
                inputChannels (1,1) {mustBePositive, mustBeInteger}
                args.Name {mustBeTextScalar} = "ChannelAttention"
            end
            
            layer.Name = args.Name;
            layer.ReductionRatio = reductionRatio;
            layer.Description = "CBAM Channel Attention Mechanism adapted for 1D Sequences (CBT)";
            
            reducedChannels = max(1, round(inputChannels / reductionRatio));
            
            % He / Xavier initialization configuration
            layer.Weights1 = dlarray(randn([reducedChannels, inputChannels], 'single') * sqrt(2 / inputChannels));
            layer.Bias1    = dlarray(zeros([reducedChannels, 1], 'single'));
            layer.Weights2 = dlarray(randn([inputChannels, reducedChannels], 'single') * sqrt(2 / reducedChannels));
            layer.Bias2    = dlarray(zeros([inputChannels, 1], 'single'));
        end
        
        function Z = predict(layer, X)
            % Strip active dimension layout tags to handle calculations manually
            % Expected incoming sequence format layout from prior graph blocks: [Channels, Batch, Time]
            X_raw = stripdims(X);
            [C, B, T] = size(X_raw);
            
            % 1. Sequence Pooling Operations
            % Calculate Global Average and Max Pooling across the sequence Timeline axis (Dim 3)
            avg_pool = mean(X_raw, 3); % Resulting dimensions: [C, B]
            max_pool = max(X_raw, [], 3); % Resulting dimensions: [C, B]
            
            % 2. Shared Multi-Layer Perceptron (FC1 -> ReLU -> FC2)
            % First FC Layer Projection + ReLU non-linear mappings
            avg_h = relu(layer.Weights1 * avg_pool + layer.Bias1);
            max_h = relu(layer.Weights1 * max_pool + layer.Bias1);
            
            % Second FC Layer Output Reconstruction
            avg_out = layer.Weights2 * avg_h + layer.Bias2;
            max_out = layer.Weights2 * max_h + layer.Bias2;
            
            % 3. Merging Activation Filters and Channel Scale Reshaping
            % Apply sigmoid activation to combine channel weights
            scale = sigmoid(avg_out + max_out); % Matrix profile: [C, B]
            
            % Reshape to [C, B, 1] so MATLAB can cleanly broadcast 
            % the scaling mask element-wise across all sequence steps (T)
            scale = reshape(scale, [C, B, 1]);
            
            % 4. Dynamic Channel Recalibration Interaction
            Z_raw = X_raw .* scale;
            
            % Re-attach the canonical network sequence tracking formatting tags
            Z = dlarray(Z_raw, "CBT");
        end
    end
end
