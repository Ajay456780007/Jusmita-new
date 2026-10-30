function Read_data()
    % Define file paths
    filePath = fullfile('Dataset', 'VRFB', 'data', ...
        '2. Raw data_ Research article 1_Interdigitated flow fields_Life cycle at 90J SF7_THTCp_1500 cm2 area.csv');
    outputDir = 'data_loader';
    
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % 1. Import Data
    opts = detectImportOptions(filePath, 'HeaderLines', 13);
    rawTable = readtable(filePath, opts);

    % Clean variable names to standard MATLAB identifiers
    rawTable.Properties.VariableNames = matlab.lang.makeValidName(rawTable.Properties.VariableNames);

    % 2. Identify Key Columns for SOC Calculation
    varNames = lower(rawTable.Properties.VariableNames);
    
    timeIdx    = find(contains(varNames, {'time', 'testtime', 'step_time'}), 1);
    currentIdx = find(contains(varNames, {'current', 'curr', 'i_'}), 1);
    voltIdx    = find(contains(varNames, {'volt', 'voltage', 'v_'}), 1);
    capIdx     = find(contains(varNames, {'capacity', 'coulomb', 'ah', 'mah'}), 1);

    % 3. Calculate SOC (Labels)
    if ~isempty(capIdx)
        % Case A: Dataset already has cumulative charge / capacity
        cap = rawTable{:, capIdx};
        cap = fillmissing(cap, 'linear');
        SOC = (cap - min(cap)) / (max(cap) - min(cap) + eps);
    elseif ~isempty(timeIdx) && ~isempty(currentIdx)
        % Case B: Coulomb Counting Integration from I and t
        t = rawTable{:, timeIdx};
        I = rawTable{:, currentIdx};
        
        t = fillmissing(t, 'linear');
        I = fillmissing(I, 'linear');
        
        % Ensure time is in seconds
        dt = [0; diff(t)];
        dt(dt < 0) = 0; % Reset on cycle boundary steps
        
        chargeAh = cumsum(I .* dt) / 3600;
        SOC = (chargeAh - min(chargeAh)) / (max(chargeAh) - min(chargeAh) + eps);
    elseif ~isempty(voltIdx)
        % Case C: Voltage-window normalization fallback
        V = rawTable{:, voltIdx};
        V = fillmissing(V, 'linear');
        SOC = (V - min(V)) / (max(V) - min(V) + eps);
    else
        error('Could not identify Time, Current, Capacity, or Voltage columns to compute SOC.');
    end

    % Ensure SOC is bounded to [0, 1]
    SOC = max(0, min(1, SOC));
    Labels = SOC;

    % 4. Feature Selection & Imputation
    % Separate strict numeric columns
    isNumericCol = varfun(@(x) isnumeric(x) || islogical(x), rawTable, 'OutputFormat', 'uniform');
    featureTable = rawTable(:, isNumericCol);
    
    % Remove capacity/SOC derivative columns from predictors to prevent target leakage
    leakageCols = contains(lower(featureTable.Properties.VariableNames), {'soc', 'stateofcharge'});
    if ~isempty(capIdx) && ismember(rawTable.Properties.VariableNames{capIdx}, featureTable.Properties.VariableNames)
        leakageCols = leakageCols | strcmp(featureTable.Properties.VariableNames, rawTable.Properties.VariableNames{capIdx});
    end
    featureTable(:, leakageCols) = [];

    featureMatrix = table2array(featureTable);

    % Step A: KNN Imputation (k = 5)
    imputedMatrix = fillmissing(featureMatrix, 'knn', 5);

    % Step B: Fallback cleanup for edge/neighbor NaN residual columns
    if any(isnan(imputedMatrix), 'all')
        imputedMatrix = fillmissing(imputedMatrix, 'linear', 1);
        imputedMatrix = fillmissing(imputedMatrix, 'nearest', 1);
    end

    Features = imputedMatrix;
    FeatureNames = featureTable.Properties.VariableNames;

    % 5. Save Artifacts
    save(fullfile(outputDir, 'Features.mat'), 'Features', 'FeatureNames', '-v7.3');
    save(fullfile(outputDir, 'Labels.mat'), 'Labels', '-v7.3');

    fprintf('Saved %d samples with %d features to "%s".\n', size(Features, 1), size(Features, 2), outputDir);
end