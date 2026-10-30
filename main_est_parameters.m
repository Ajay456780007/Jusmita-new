function metrics = main_est_parameters(y_true, pred)
%MAIN_EST_PARAMETERS Computes classification performance metrics
%
% Inputs:
%   y_true - Ground truth labels (Nx1)
%   pred   - Predicted labels (Nx1)
%
% Output:
%   metrics = [Acc, Sen, Spe, F1score, Rec, Pre, TPR, FPR]

    % Compute confusion matrix
    cm = confusionmat(y_true, pred);

    % For multiclass problems, sum one-vs-rest confusion matrices
    if size(cm,1) > 2
        classes = unique(y_true);
        mcm = zeros(2,2,length(classes));

        for i = 1:length(classes)
            c = classes(i);

            TP = sum((y_true == c) & (pred == c));
            FP = sum((y_true ~= c) & (pred == c));
            FN = sum((y_true == c) & (pred ~= c));
            TN = sum((y_true ~= c) & (pred ~= c));

            mcm(:,:,i) = [TP FP;
                          FN TN];
        end

        cm = sum(mcm,3);
    end

    % Extract values
    TP = cm(1,1);
    FP = cm(1,2);
    FN = cm(2,1);
    TN = cm(2,2);

    % Avoid divide-by-zero
    Acc = (TP + TN) / max((TP + TN + FP + FN), eps);
    Sen = TP / max((TP + FN), eps);
    Spe = TN / max((TN + FP), eps);
    Pre = TP / max((TP + FP), eps);
    Rec = TP / max((TP + FN), eps);
    F1score = 2 * (Pre * Rec) / max((Pre + Rec), eps);
    TPR = TP / max((TP + FN), eps);
    FPR = FP / max((FP + TN), eps);

    % Return metrics
    metrics = [Acc, Sen, Spe, F1score, Rec, Pre, TPR, FPR];

end