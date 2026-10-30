classdef InvertibleAttentionLayer < nnet.layer.Layer ...
         & nnet.layer.Formattable 

    properties
        % Declare layer properties here.
        Dim1 % Splited dimension of disjoint part-1 (channel dimension)
        Dim2 % Splited dimension of disjoint part-2 (channel dimension)
        EmbedDim % Embedding dimension for self-attention calculation (h)
        HiddenUnits % Vector of hidden units for MLP network
    end
    properties (Learnable)
        % (Optional) Layer learnable parameters.
        % Declare learnable parameters here.
        InputQueryWeights % Query weight for input data [h,h]
        InputKeyWeights % Key weight for input data [h,h]
        InputValueWeights % Value weight for input data [h,h]
        InputQueryBias % Query bias for input data [h,1]
        InputKeyBias % Key bias for input data [h,1]
        InputValueBias % Value bias for input data [h,1]
        InputEmbedWeights % Input embedding weights [d1,h]
        
        CondKeyWeights % Key weights for conditional data [h,h] 
        CondValueWeights % Value weights for conditional data [h,h]
        CondKeyBias % Key bias for conditional data [h,1]
        COndValueBias % Value bias for conditional data [h,1]
        CondEmbedWeights % Condition embedding weights [d_c,h]
        AlhpaWeights % Importance weights for X1 aggregation [d1,d1]
        AlhpaBias % Importance bias for X1 aggregation [d1,1]
    end
    properties (State)
        % (Optional) Layer state parameters.
        % Declare state parameters here.
        LogJacobianDet % Logarithm of the absolute determinant of the Jacobian matrix [1,B]
    end
    properties (Learnable, State)
        % (Optional) Nested dlnetwork objects with both learnable
        % parameters and state parameters.
        % Declare nested networks with learnable and state parameters here.
        MLPNetwork % MLP network for scaling and shifting vectors
    end
    methods
        function layer = InvertibleAttentionLayer(dim1,dim2,embedDim,hiddenUnits,nvargs)
            % This function must have the same name as the class.
            % Define layer constructor function here.
            arguments (Input)
                dim1 (1,1) {mustBeInteger,mustBePositive}
                dim2 (1,1) {mustBePositive,mustBeInteger}
                embedDim (1,1) {mustBeInteger,mustBePositive}
                hiddenUnits {mustBeColumn,mustBePositive,mustBeInteger}
                nvargs.Name {mustBeTextScalar}="IALLayer" 
            end
            layer.Dim1 = dim1;
            layer.Dim2 = dim2;
            layer.HiddenUnits = hiddenUnits;
            layer.EmbedDim = embedDim;
            layer.Name = nvargs.Name;
            layer.Description = "Invertible attention layer for IA-cINNs";
            layer.NumInputs = 2; % There are two inputs (dataInput+condInput)
            layer.InputNames = {'Data','Condition'};
            % Initialize nested dlnetwork in constructor function
            hiddenLayers = arrayfun(@(hiddenUnit) [fullyConnectedLayer(hiddenUnit);preluLayer],layer.HiddenUnits,"UniformOutput",false);
            hiddenLayers = [hiddenLayers{:}]; % <--- Fixed object unpacking replacement
            layersMLP = [featureInputLayer(2*layer.EmbedDim,"Name","st-input")
                hiddenLayers(:)
                fullyConnectedLayer(2*layer.Dim2,"Name","st-output")];
            layer.MLPNetwork = dlnetwork(layersMLP);

        end
        function layer = initialize(layer,layoutData,layoutCond)
            
            % Define layer initialization function here.
            % Find the dimension of data input
            dataDim = layoutData.Size(finddim(layoutData,"C"));
            assert(layer.Dim1+layer.Dim2==dataDim,"Splited dimensions must be compatible!");
            % Find the dimension of conditional input
            condDim = layoutCond.Size(finddim(layoutCond,"C")); % d_c
            % Initialize learnable parameters
            if isempty(layer.InputQueryWeights)
                layer.InputQueryWeights = dlarray(randn(layer.EmbedDim,"single"));
            end
            if isempty(layer.InputKeyWeights)
                layer.InputKeyWeights = dlarray(randn(layer.EmbedDim,"single"));
            end
            if isempty(layer.InputValueWeights)
                layer.InputValueWeights = dlarray(randn(layer.EmbedDim,"single"));
            end
            if isempty(layer.InputQueryBias)
                layer.InputQueryBias = dlarray(zeros(layer.EmbedDim,1,"single"));
            end
            if isempty(layer.InputKeyBias)
                layer.InputKeyBias = dlarray(zeros(layer.EmbedDim,1,"single"));
            end
            if isempty(layer.InputValueBias)
                layer.InputValueBias = dlarray(zeros(layer.EmbedDim,1,"single"));
            end
            if isempty(layer.InputEmbedWeights)
                layer.InputEmbedWeights = dlarray(randn(layer.Dim1,layer.EmbedDim,"single"));
            end
            if isempty(layer.CondKeyWeights)
                layer.CondKeyWeights = dlarray(randn(layer.EmbedDim,"single"));
            end
            if isempty(layer.CondValueWeights)
                layer.CondValueWeights = dlarray(randn(layer.EmbedDim,"single"));
            end
            if isempty(layer.CondKeyBias)
                layer.CondKeyBias = dlarray(zeros(layer.EmbedDim,1,"single"));
            end
            if isempty(layer.COndValueBias)
                layer.COndValueBias = dlarray(zeros(layer.EmbedDim,1,"single"));
            end
            if isempty(layer.CondEmbedWeights)
                layer.CondEmbedWeights = dlarray(randn(condDim,layer.EmbedDim,"single"));
            end
            if isempty(layer.AlhpaWeights)
                layer.AlhpaWeights = dlarray(randn(layer.Dim1,"single"));
            end
            if isempty(layer.AlhpaBias)
                layer.AlhpaBias = dlarray(zeros(layer.Dim1,1,"single"));
            end
            
            % Initialize states
            if isempty(layer.LogJacobianDet)
                layer.LogJacobianDet = 0; % 
            end
  
              
        end
        
        function [Y,stateLogJacDet,stateMLP] = predict(layer,X,C)
            % Strip active dimension layout tags to bypass validation constraint drops
            X_raw = stripdims(X);
            C_raw = stripdims(C);
            
            % Spliting data input X into disjoint parts X1 and X2
            X1_raw = X_raw(1:layer.Dim1,:); % X1 part
            X2_raw = X_raw(layer.Dim1+1:end,:); % X2 part
            
            X1 = dlarray(X1_raw, "CB");
            X2 = dlarray(X2_raw, "CB");
            
            % Data input embedding with size [h,B,d1] "CBT"
            dataH = dlarray(X1_raw.*reshape(layer.InputEmbedWeights,layer.Dim1,1,[]),"TBC"); 
            dataQuery = fullyconnect(dataH,layer.InputQueryWeights,layer.InputQueryBias);
            dataKey = fullyconnect(dataH,layer.InputKeyWeights,layer.InputKeyBias);
            dataValue = fullyconnect(dataH,layer.InputValueWeights,layer.InputValueBias);
            Z = attention(dataQuery,dataKey,dataValue,1); % [h,B,d1] "CBT"
            
            % Use unformatted variables to bypass data layout checks safely
            fc_out = fullyconnect(X1_raw, layer.AlhpaWeights, layer.AlhpaBias, 'DataFormat', 'CB');
            importanceWeigths = softmax(fc_out, 'DataFormat', 'CB');
            
            fInput = sum(dlarray(stripdims(importanceWeigths),"TBC").*Z,3); % [C,B,1] "CBT"
            fInput = dlarray(stripdims(fInput),"CB"); % [h,B] "CB" format
            
            % Condition input embedding with size [h,B,dc] "CBT"
            condH = dlarray(C_raw.*reshape(layer.CondEmbedWeights,size(C_raw,1),1,[]),"TBC");
            condKey = fullyconnect(condH,layer.CondKeyWeights,layer.CondKeyBias);
            condValue = fullyconnect(condH,layer.CondValueWeights,layer.COndValueBias);
            Zcross = attention(dataQuery,condKey,condValue,1); % [h,B,d1] "CBT"
            fCross = sum(dlarray(stripdims(importanceWeigths),"TBC").*Zcross,3); % [C,B,1]
            fCross = dlarray(stripdims(fCross),"CB"); % [h,B] "CB" format
            
            % Forward pass through MLP for scaling and shifting vectors
            [st,stateMLP] = forward(layer.MLPNetwork,[fInput;fCross]);
            
            % Spliting output st into scaling and shifting vectors
            s = st(1:layer.Dim1,:); % splits scaling part
            t = st(layer.Dim1+1:end,:); % splits shifting part
            boudingS = 0.5 * tanh(s); % scaling vector is bounded for numerical stability 
            stateLogJacDet = stripdims(sum(boudingS,1)); % [1,B] 
            
            Y1 = X1;
            Y2 = X2 .* exp(boudingS) + t;
            Y = [Y1;Y2]; % return cat Y1 and Y2      
        end

        function layer = resetState(layer)
            % (Optional) Reset layer state.
            % Define reset state function here.
            layer.LogJacobianDet = 0;
        end

        % end
        function X = inverse(layer,Y,C)
            % inverse function for reconstructing X from Y and C
            Y_raw = stripdims(Y);
            C_raw = stripdims(C);
            
            X1_raw = Y_raw(1:layer.Dim1,:); % [C,B] 
            Y2_raw = Y_raw(layer.Dim1+1:end,:); % [C,B]
            
            X1 = dlarray(X1_raw, "CB");
            Y2 = dlarray(Y2_raw, "CB");
            
            % Data input embedding with size [h,B,d1] "CBT"
            dataH = dlarray(X1_raw.*reshape(layer.InputEmbedWeights,layer.Dim1,1,[]),"TBC"); 
            dataQuery = fullyconnect(dataH,layer.InputQueryWeights,layer.InputQueryBias);
            dataKey = fullyconnect(dataH,layer.InputKeyWeights,layer.InputKeyBias);
            dataValue = fullyconnect(dataH,layer.InputValueWeights,layer.InputValueBias);
            Z = attention(dataQuery,dataKey,dataValue,1); % [h,B,d1] "CBT"
            
            fc_out_inv = fullyconnect(X1_raw, layer.AlhpaWeights, layer.AlhpaBias, 'DataFormat', 'CB');
            importanceWeigths = softmax(fc_out_inv, 'DataFormat', 'CB');
            
            fInput = sum(dlarray(stripdims(importanceWeigths),"TBC").*Z,3); % [C,B,1] "CBT"
            fInput = dlarray(stripdims(fInput),"CB"); % [h,B] "CB" format
            
            % Condition input embedding with size [h,B,dc] "CBT"
            condH = dlarray(C_raw.*reshape(layer.CondEmbedWeights,size(C_raw,1),1,[]),"TBC");
            condKey = fullyconnect(condH,layer.CondKeyWeights,layer.CondKeyBias);
            condValue = fullyconnect(condH,layer.CondValueWeights,layer.COndValueBias);
            Zcross = attention(dataQuery,condKey,condValue,1); % [h,B,d1] "CBT"
            fCross = sum(dlarray(stripdims(importanceWeigths),"TBC").*Zcross,3); % [C,B,1]
            fCross = dlarray(stripdims(fCross),"CB"); % [h,B] "CB" format
            
            % Forward pass through MLP for scaling and shifting vectors
            st = forward(layer.MLPNetwork,[fInput;fCross]);
            
            % Spliting output st into scaling and shifting vectors
            s = st(1:layer.Dim1,:); % splits scaling part
            t = st(layer.Dim1+1:end,:); % splits shifting part
            boudingS = 0.5 * tanh(s); % scaling vector is bounded for numerical stability 
            X2 = (Y2 - t).*exp(-boudingS); % "CB"
            X = [X1;X2]; % return X by cat X1 and X2
        end
    end
end