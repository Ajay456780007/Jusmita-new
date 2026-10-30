function metrics = Evaluation_Metrics1(y, y_pred)
    % Ensure both inputs are column vectors for consistent calculation
    y = y(:);
    y_pred = y_pred(:);

    % 1. Mean Squared Error (MSE)
    mse = mean((y - y_pred).^2);

    % 2. Root Mean Squared Error (RMSE)
    rmse = sqrt(mse);

    % 3. Mean Absolute Error (MAE)
    mae = mean(abs(y - y_pred));

    % 4. R-squared (Coefficient of Determination)
    % Calculated as: 1 - (Sum of Squared Residuals / Total Sum of Squares)
    ss_res = sum((y - y_pred).^2);
    ss_tot = sum((y - mean(y)).^2);
    r2 = 1 - (ss_res / ss_tot);

    % 5. Correlation (Square root of R2)
    cor = sqrt(r2);

    % Return all metrics as a row vector (similar to a Python list)
    metrics = [mse, rmse, mae, r2, cor];
end
