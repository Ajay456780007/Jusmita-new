classdef ECA1DLayer < nnet.layer.Layer ...
         & nnet.layer.Formattable  
        % & nnet.layer.Acceleratable % (Optional)
    properties
        % Declare layer properties here.
        kernelSize % Conv1d kernelSize
        Gamma %
        Beta %
    end
    properties (Learnable)
      % Declare learnable parameters here.
      convWeights % 1-D convolution weights 
    end
    properties (State)
        % (Optional) Layer state parameters.
        % Declare state parameters here.
    end
    % properties (Learnable, State)
    %     % (Optional) Nested dlnetwork objects with both learnable
    %     % parameters and state parameters.
    % 
    %     % Declare nested networks with learnable and state parameters here.
    % end
    methods
        function layer = ECA1DLayer(gamma,beta,args)
            % This function must have the same name as the class.
            % Define layer constructor function here.
            arguments (Input)
                gamma (1,1) {mustBeInteger,mustBePositive}
                beta (1,1) {mustBeInteger,mustBePositive}
                args.Name {mustBeTextScalar}="ECA1DLayer" 
            end
                layer.Name = args.Name;
                layer.Gamma = gamma;
                layer.Beta = beta;
                layer.Description = "Efficient Channel Attention Layer" + ...
                                    " Named as " + layer.Name;
                 
        end
        function layer = initialize(layer,layout)
            % (Optional) Initialize layer learnable and state parameters.
            %
            % Inputs:
            %         layer  - Layer to initialize
            %         layout - Data layout, specified as a networkDataLayout
            %                  object
            %
            % Outputs:
            %         layer - Initialized layer
            %
            %  - For layers with multiple inputs, replace layout with 
            %    layout1,...,layoutN, where N is the number of inputs.
            
            % Define layer initialization function here.
            idxC = finddim(layout,"C");
            inChannel = layout.Size(idxC);
            k = floor(log2(inChannel)/layer.Gamma + layer.Beta);
            if mod(k,2) == 0
                k = k+1; % convert to odd integer
            end
            layer.kernelSize = k;
            if isempty(layer.convWeights)
                layer.convWeights = dlarray(randn(k,1,1,"single"));
            end
            
        end
        
        function Y = predict(layer,X)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Outputs:
            %         Y     - Output of layer forward function
            %         state - (Optional) Updated layer state
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Y with 
            %    Y1,...,YM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.
            % Define layer predict function here.
            GAP = mean(X,3); % global average pooling [C,B,1]
            
            w = dlconv(dlarray(GAP,"TBC"),layer.convWeights,0,...
                       "Padding",floor(layer.kernelSize / 2),"WeightsFormat","TCU");
            alpha = sigmoid(dlarray(w,"TBC"));
            Y = dlarray(X .* alpha,"CBT");
        end
        % function [Y,state,memory] = forward(layer,X)
        %     % (Optional) Forward input data through the layer at training
        %     % time and output the result, the updated state, and a memory
        %     % value.
        %     %
        %     % Inputs:
        %     %         layer - Layer to forward propagate through 
        %     %         X     - Layer input data
        %     % Outputs:
        %     %         Y      - Output of layer forward function 
        %     %         state  - (Optional) Updated layer state 
        %     %         memory - (Optional) Memory value for custom backward
        %     %                  function
        %     %
        %     %  - For layers with multiple inputs, replace X with X1,...,XN, 
        %     %    where N is the number of inputs.
        %     %  - For layers with multiple outputs, replace Y with 
        %     %    Y1,...,YM, where M is the number of outputs.
        %     %  - For layers with multiple state parameters, replace state 
        %     %    with state1,...,stateK, where K is the number of state 
        %     %    parameters.
        % 
        %     % Define layer forward function here.
        % end
        % function layer = resetState(layer)
        %     % (Optional) Reset layer state.
        % 
        %     % Define reset state function here.
        % end
        % function [dLdX,dLdW,dLdSin] = backward(layer,X,Y,dLdY,dLdSout,memory)
        %     % (Optional) Backward propagate the derivative of the loss
        %     % function through the layer.
        %     %
        %     % Inputs:
        %     %         layer   - Layer to backward propagate through 
        %     %         X       - Layer input data 
        %     %         Y       - Layer output data 
        %     %         dLdY    - Derivative of loss with respect to layer 
        %     %                   output
        %     %         dLdSout - (Optional) Derivative of loss with respect 
        %     %                   to state output
        %     %         memory  - Memory value from forward function
        %     % Outputs:
        %     %         dLdX   - Derivative of loss with respect to layer input
        %     %         dLdW   - (Optional) Derivative of loss with respect to
        %     %                  learnable parameter 
        %     %         dLdSin - (Optional) Derivative of loss with respect to 
        %     %                  state input
        %     %
        %     %  - For layers with state parameters, the backward syntax must
        %     %    include both dLdSout and dLdSin, or neither.
        %     %  - For layers with multiple inputs, replace X and dLdX with
        %     %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
        %     %    the number of inputs.
        %     %  - For layers with multiple outputs, replace Y and dLdY with
        %     %    Y1,...,YM and dLdY,...,dLdYM, respectively, where M is the
        %     %    number of outputs.
        %     %  - For layers with multiple learnable parameters, replace 
        %     %    dLdW with dLdW1,...,dLdWP, where P is the number of 
        %     %    learnable parameters.
        %     %  - For layers with multiple state parameters, replace dLdSin
        %     %    and dLdSout with dLdSin1,...,dLdSinK and 
        %     %    dLdSout1,...,dldSoutK, respectively, where K is the number
        %     %    of state parameters.
        % 
        %     % Define layer backward function here.
        % end
    end
end