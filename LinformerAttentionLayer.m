classdef LinformerAttentionLayer < nnet.layer.Layer ...
         & nnet.layer.Formattable
    % LinformerAttentionLayer - Extracts dynamic features using low-rank temporal projections.
    % See reference 10.1016/j.neunet.2026.109443 (DOI) for architectural specifics.

    properties
        QueryKeySize    % Feature size of query and key in dot-attention
        ValueSize       % Feature size of value in dot-attention
        NumHeads        % Number of multi-head attention spaces
        TimeProjSize    % Size of low-rank matrix temporal projection (k < T)
    end
    
    properties (Learnable)
        QueryWeights         % Weights of query projection [qkSize, inputSize, numHeads]
        KeyWeigths           % Weights of key projection [qkSize, inputSize, numHeads]
        ValueWeights         % Weights of value projection [vSize, inputSize, numHeads]
        KeyTimeProjWeights   % Weights of low-rank projection for key [k, T, numHeads]
        ValueTimeProjWeights % Weights of low-rank projection for value [k, T, numHeads]
        OutputWeights        % Weights of output projection [numHeads*vSize, numHeads*vSize]
    end

    methods
        function layer = LinformerAttentionLayer(qkSize, vSize, numHeads, timeProjSize, nvargs)
            arguments (Input)
                qkSize (1,1) {mustBePositive,mustBeInteger}
                vSize (1,1) {mustBePositive,mustBeInteger}
                numHeads (1,1) {mustBePositive,mustBeInteger}
                timeProjSize (1,1) {mustBePositive,mustBeInteger}
                nvargs.Name {mustBeTextScalar} = "Linformer"
            end
            layer.QueryKeySize = qkSize;
            layer.ValueSize = vSize;
            layer.NumHeads = numHeads;
            layer.TimeProjSize = timeProjSize;
            layer.Name = nvargs.Name;
            layer.Description = "Attention of Linformer with low-rank time projection";
        end
        
        function layer = initialize(layer, layout)
            % Extract architectural shape dimensions from active layout context
            inputSize = layout.Size(finddim(layout, "C"));
            T = layout.Size(finddim(layout, "T")); % Sequence timeline length
            
            querykeySize = layer.QueryKeySize;
            valueSize = layer.ValueSize;
            H = layer.NumHeads;
            timeLowRankSize = layer.TimeProjSize;
            
            % Memory instantiation allocations using standard normalization scaling
            if isempty(layer.QueryWeights)
                layer.QueryWeights = dlarray(randn([querykeySize, inputSize, H], "single"));
            end
            if isempty(layer.KeyWeigths)
                layer.KeyWeigths = dlarray(randn([querykeySize, inputSize, H], "single"));
            end
            if isempty(layer.ValueWeights)
                layer.ValueWeights = dlarray(randn([valueSize, inputSize, H], "single"));
            end
            if isempty(layer.KeyTimeProjWeights)
                layer.KeyTimeProjWeights = dlarray(randn([timeLowRankSize, T, H], "single"));
            end
            if isempty(layer.ValueTimeProjWeights)
                layer.ValueTimeProjWeights = dlarray(randn([timeLowRankSize, T, H], "single"));
            end
            if isempty(layer.OutputWeights)
                layer.OutputWeights = dlarray(randn(H * valueSize, H * valueSize, "single"));
            end
        end
        
        function Y = predict(layer, X)
            queryWeights = layer.QueryWeights;
            keyWeights = layer.KeyWeigths;
            valueWeights = layer.ValueWeights;
            timeProjE = layer.KeyTimeProjWeights;
            timeProjF = layer.ValueTimeProjWeights;
            WO = layer.OutputWeights;
            
            % Setup multi-head optimization handle loop
            oneHeadFun = @(k) oneHeadLinFormer(X, queryWeights(:,:,k), keyWeights(:,:,k), ...
                                               valueWeights(:,:,k), timeProjE(:,:,k), timeProjF(:,:,k));
            
            multiHeads = arrayfun(oneHeadFun, 1:layer.NumHeads, "UniformOutput", false);
            catHeads = cat(1, multiHeads{:}); % Concatenate channel steps along "C"
            
            % Process structural output via final projection linear maps
            catHeads_raw = stripdims(catHeads);
            Y_raw = fullyconnect(catHeads_raw, WO, zeros(size(WO, 1), 1, "like", WO), 'DataFormat', 'CBT');
            Y = dlarray(Y_raw, "CBT");
        end
    end
end


function head = oneHeadLinFormer(X, WQ, WK, WV, E, F)
    % Strip active dimension layout tags to bypass validation drops
    X_raw = stripdims(X); % Format layout: [Channels, Batch, Time]
    
    % 1. Linearly project the Query matrix across channel domains
    query_raw = fullyconnect(X_raw, WQ, zeros(size(WQ, 1), 1, "like", WQ), 'DataFormat', 'CBT');
    query = dlarray(query_raw, 'CBT'); % Dimensions: [qkSize, Batch, Time]
    
    % 2. Apply Low-Rank Projections (E and F) along the Time axis (T -> k)
    % E is [k, T]. X_raw is [C, B, T]. 
    % We need to multiply E by the transpose of the time dimension.
    % To do this safely for all batch instances, we permute X to [Time, Channels, Batch]
    X_spatial = permute(X_raw, [3, 1, 2]); 
    
    % FIX: Use pagemtimes instead of pagefun to natively multiply dlarrays
    % Performs: [k, T] x [T, Channels] for each Batch page
    keyTimeProj = pagemtimes(E, X_spatial); % Resulting Dimensions: [k, Channels, Batch]
    valTimeProj = pagemtimes(F, X_spatial); % Resulting Dimensions: [k, Channels, Batch]
    
    % Restore the canonical layout format: [Channels, Batch, k]
    keyLowRank = permute(keyTimeProj, [2, 3, 1]);
    valueLowRank = permute(valTimeProj, [2, 3, 1]);
    
    % 3. Linearly map down-sampled dense relational projections
    key_raw = fullyconnect(keyLowRank, WK, zeros(size(WK, 1), 1, "like", WK), 'DataFormat', 'CBT');
    key = dlarray(key_raw, 'CBT'); % Dimensions: [qkSize, Batch, k]
    
    value_raw = fullyconnect(valueLowRank, WV, zeros(size(WV, 1), 1, "like", WV), 'DataFormat', 'CBT');
    value = dlarray(value_raw, 'CBT'); % Dimensions: [vSize, Batch, k]
    
    % 4. Call native multi-head dot-product attention routine
    head = attention(query, key, value, 1); % Resulting Dimensions: [vSize, Batch, Time]
end
