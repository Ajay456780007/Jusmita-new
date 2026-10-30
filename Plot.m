classdef Plot 

    properties
        show = false; 
        save = true;
        color = {"#1A73E8", "#9C27B0", "#00C853", "#D50000", "#FF6D00", "#FFD600", "#E91E63", "#00E5FF", "#795548", "#37474F"};
        color2 = {"#4A148C", "#AEEA00", "#6B1124", "#1DE9B6", "#0D47A1"};
        models = {'DAM-1DCNN', 'PI-Transformer', 'Adaboost-SVM', 'CatBoost-SMA', 'RAkEL', 'ISOA-SVM', 'IMM', 'CFAO-DRLCNet',"Proposed1"};
        percentage = {"TP_40", "TP_60", "TP_60", "TP_70", "TP_80", "TP_90"};
        models2 = {"Proposed1 at Epochs=100", "Proposed1 at Epochs=200", "Proposed1 at Epochs=300",...
                        "Proposed1 at Epochs=400", "Proposed1 at Epochs=500"};
        bar_width = 0.7
    end 

    methods
        function obj = Plot(varargin)

            if nargin >= 1
                obj.show = varargin{1};
            end
        
            if nargin >= 2
                obj.save = varargin{2};
            end
        
            if nargin >= 3
                obj.color2 = varargin{3};
            end
        
            if nargin >= 4
                obj.color = varargin{4};
            end
        
            if nargin >= 5
                obj.models = varargin{5};
            end
        
            if nargin >= 6
                obj.percentage = varargin{6};
            end
        
            if nargin >= 7
                obj.models2 = varargin{7};
            end
            if nargin >= 8
                obj.bar_width = varargin{8};
            end
    
        end

        function [MAPE, MSE, R2, RMSE, COR, MAE] = Load_Comp_data(obj, DB)

            tmp = load(fullfile("Analysis","Comparative_Analysis",DB + "/MAPE_1.mat"));
            MAPE = tmp.MAPE_1*100;
        
            tmp = load(fullfile("Analysis","Comparative_Analysis",DB + "/MSE_1.mat"));
            MSE = tmp.MSE_1*100;
        
            tmp = load(fullfile("Analysis","Comparative_Analysis", DB + "/R2_1.mat"));
            R2 = tmp.R2_1*100;
        
            tmp = load(fullfile("Analysis","Comparative_Analysis",DB + "/RMSE_1.mat"));
            RMSE = tmp.RMSE_1*100;
        
            tmp = load(fullfile("Analysis","Comparative_Analysis", DB + "/COR_1.mat"));
            COR = tmp.COR_1*100;
        
            tmp = load(fullfile("Analysis","Comparative_Analysis", DB + "/MAE_1.mat"));
            MAE = tmp.MAE_1*100;
        
        end

        function Comp_figure(obj, DB, perf, x_label, y_label, colors, bar_width)

            % Number of models and percentages
            [n_models, n_percentage] = size(perf);
        
            % Model names
            Models = obj.models(1:n_models);
        
            % Percentage labels
            percentage = [40 50 60 70 80 90];
            Percentages = percentage(1:n_percentage);
        
            % Create table for saving CSV
            T = array2table(perf, 'VariableNames', strcat("P", string(Percentages)), ...
                'RowNames', string(Models));
        
            fprintf('Comp_Analysis Bar Graph values for %s of %s saved as CSV\n', DB, y_label);
        
            % Figure
            figure('Color','w','Position',[100 100 1000 800]);
        
            % Grouped Bar Plot
            b = bar(perf','grouped');
            hold on;
        
            % Set custom colors
            for i = 1:n_models
                b(i).FaceColor = colors{i};
                b(i).EdgeColor = 'none';
            end
        
            % Optional bar width
            for i = 1:n_models
                b(i).BarWidth = bar_width;
            end
        
            % Axes properties
            ax = gca;
            ax.FontName = 'Times New Roman';
            ax.FontSize = 17;
            ax.FontWeight = 'bold';
            ax.XTick = 1:n_percentage;
            ax.XTickLabel = string(Percentages);
        
            xlabel(x_label,'FontName','Times New Roman','FontSize',19,'FontWeight','bold');
        
            ylabel(y_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            legend(Models,'Location','southoutside','NumColumns',3,'FontName','Times New Roman','FontSize',16,...
                'FontWeight','bold');
        
            grid on;
            box on;
            axis tight; 
            ax = gca;
            ax.Position = [0.15 0.24 0.75 0.65];
            
            if obj.save
        
                outDir = fullfile('Results4',DB,'Comparative_Analysis','Bar');
        
                if ~exist(outDir,'dir')
                    mkdir(outDir);
                end
        
                writetable(T,fullfile(outDir,[y_label '__Graph.csv']),'WriteRowNames',true);
        
                set(gcf,'PaperPositionMode','auto');

                print(gcf,...
                    fullfile(outDir,[y_label '_Graph.png']),...
                    '-dpng','-r600');
        
                fprintf('Comp_Analysis Bar Graph Image for %s of %s saved\n',DB,y_label);
        
            end
        
            if obj.show
                drawnow;      % Ensure the figure is rendered
                pause(1);     % Wait for 1 second
                close(gcf);
            else
                close(gcf);
            end
        end
        
        function Comp_Radial_plot(obj, DB, perf, x_label, y_label, colors, ~)
            % Number of models and percentages
            [n_models, n_percentage] = size(perf);
            
            % Model names
            Models = obj.models(1:n_models);
            
            % Percentage labels
            percentage = [40 50 60 70 80 90];
            Percentages = percentage(1:n_percentage);
            
            % Create table for saving CSV
            T = array2table(perf, 'VariableNames', strcat("P", string(Percentages)), ...
                'RowNames', string(Models));
            
            fprintf('Comp_Analysis Radial Bar values for %s of %s saved as CSV\n', DB, y_label);
            
            % Figure setup
            figure('Color', 'w', 'Position', [100 100 1050 950]);
            ax = gca;
            hold(ax, 'on');
            axis(ax, 'equal');
            axis(ax, 'off');
            
            % Radial bar geometry parameters
            r_inner = 3.2;          % Radius of inner void
            bar_thickness = 0.55;    % Thickness of each ring/bar
            bar_gap = 0.15;          % Gap between adjacent bars
            group_gap = 0.70;        % Separation between percentage groups
            max_angle = 1.75 * pi;   % Angular sweep (315 degrees, leaving space for labels)
            
            % Fixed maximum scale of 100 (or higher if values exceed 100)
            max_scale = max(100, ceil(max(perf(:))));
            
            hLegend = gobjects(n_models, 1);
            current_r = r_inner;
            x_tick_radii = zeros(n_percentage, 1);
            
            for p = 1:n_percentage
                r_group_start = current_r;
                
                for m = 1:n_models
                    r1 = current_r;
                    r2 = current_r + bar_thickness;
                    val = perf(m, p);
                    
                    % Angle proportional to 100% scale
                    theta_end = (val / max_scale) * max_angle;
                    
                    % 1. Draw light grey background track
                    n_track_pts = 100;
                    th_track = linspace(0, max_angle, n_track_pts);
                    x_track = [r1 * cos(th_track), r2 * cos(fliplr(th_track))];
                    y_track = [r1 * sin(th_track), r2 * sin(fliplr(th_track))];
                    
                    fill(ax, x_track, y_track, 'w', ...
                        'FaceColor', [0.93 0.93 0.93], ...
                        'EdgeColor', 'none', ...
                        'HandleVisibility', 'off');
                    
                    % 2. Draw active colored radial arc
                    col = colors{m};
                    if isstring(col) || ischar(col)
                        col = char(col);
                    end
                    
                    if theta_end > 0.01
                        n_pts = max(10, ceil(100 * (theta_end / max_angle)));
                        th = linspace(0, theta_end, n_pts);
                        x_bar = [r1 * cos(th), r2 * cos(fliplr(th))];
                        y_bar = [r1 * sin(th), r2 * sin(fliplr(th))];
                        
                        h = fill(ax, x_bar, y_bar, 'w', ...
                            'FaceColor', col, ...
                            'EdgeColor', 'none');
                    else
                        % Placeholder for legend handle
                        h = plot(ax, NaN, NaN, 's', ...
                            'MarkerFaceColor', col, ...
                            'MarkerEdgeColor', 'none', ...
                            'MarkerSize', 10);
                    end
                    
                    % Store single legend handle per model
                    if p == 1
                        hLegend(m) = h;
                    else
                        set(h, 'HandleVisibility', 'off');
                    end
                    
                    current_r = current_r + bar_thickness + bar_gap;
                end
                
                % Center radial position for each percentage group
                x_tick_radii(p) = (r_group_start + (current_r - bar_gap)) / 2;
                current_r = current_r + group_gap;
            end
            
            % Outer limit radius
            r_outer = current_r - group_gap + 0.2;
            
            % --- 1. Baseline indicator (0-degree line) ---
            plot(ax, [r_inner, r_outer], [0, 0], 'Color', [0.4 0.4 0.4], ...
                'LineStyle', '-', 'LineWidth', 1.2, 'HandleVisibility', 'off');
            text(ax, r_inner * 0.92, 0, '0', 'FontName', 'Times New Roman', ...
                'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'right', ...
                'VerticalAlignment', 'middle');
            
            % --- 2. Percentage labels directly under the 0 baseline ---
            label_y_offset = -0.45 * bar_thickness;
            for p = 1:n_percentage
                r_mid = x_tick_radii(p);
                text(ax, r_mid, label_y_offset, sprintf('%d%%', Percentages(p)), ...
                    'FontName', 'Times New Roman', 'FontSize', 12, ...
                    'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'top');
            end
            
            % --- 3. Training Percentage axis label placed below the values ---
            x_axis_mid = (r_inner + r_outer) / 2;
            text(ax, x_axis_mid, label_y_offset - 1.5, x_label, ...
                'FontName', 'Times New Roman', 'FontSize', 14, ...
                'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'top');
            
            % --- 4. Outer Circle Arc and Metric Y-Ticks [20, 40, 60, 80, 100] ---
            th_arc = linspace(0, max_angle, 250);
            plot(ax, r_outer * cos(th_arc), r_outer * sin(th_arc), ...
                'Color', [0.45 0.45 0.45], 'LineWidth', 1.2, 'HandleVisibility', 'off');
            
            val_ticks = [20, 40, 60, 80, 100];
            tick_len = 0.35;
            
            for k = 1:numel(val_ticks)
                v = val_ticks(k);
                th_tick = (v / max_scale) * max_angle;
                
                % Tick mark line protruding outwards
                x_t_inner = r_outer * cos(th_tick);
                y_t_inner = r_outer * sin(th_tick);
                x_t_outer = (r_outer + tick_len) * cos(th_tick);
                y_t_outer = (r_outer + tick_len) * sin(th_tick);
                
                plot(ax, [x_t_inner, x_t_outer], [y_t_inner, y_t_outer], ...
                    'Color', [0.35 0.35 0.35], 'LineWidth', 1.2, 'HandleVisibility', 'off');
                
                % Numerical value label outside the tick mark
                r_lbl = r_outer + tick_len + 0.45;
                x_lbl = r_lbl * cos(th_tick);
                y_lbl = r_lbl * sin(th_tick);
                
                % Quadrant alignment
                if cos(th_tick) > 0.25
                    h_align = 'left';
                elseif cos(th_tick) < -0.25
                    h_align = 'right';
                else
                    h_align = 'center';
                end
                
                if sin(th_tick) > 0.25
                    v_align = 'bottom';
                elseif sin(th_tick) < -0.25
                    v_align = 'top';
                else
                    v_align = 'middle';
                end
                
                text(ax, x_lbl, y_lbl, sprintf('%d', v), ...
                    'FontName', 'Times New Roman', 'FontSize', 12, ...
                    'FontWeight', 'bold', 'HorizontalAlignment', h_align, ...
                    'VerticalAlignment', v_align);
            end
            
            % --- 5. Dynamic Limits and Non-Overlapping Title ---
            % Expand limits to accommodate outer ring labels and title
            xlim(ax, [-(r_outer + 3.0), (r_outer + 3.0)]);
            ylim(ax, [-(r_outer + 3.8), (r_outer + 3.8)]);
            
            % Title placed well above the highest outer tick mark
            title_y = r_outer + 2.4;
            text(ax, 0, title_y, y_label, ...
                'FontName', 'Times New Roman', 'FontSize', 18, ...
                'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom');
            
            % --- 6. Legend ---
            legend(hLegend, Models, ...
                'Location', 'southoutside', ...
                'Orientation', 'horizontal', ...
                'NumColumns', 3, ...
                'FontName', 'Times New Roman', ...
                'FontSize', 13, ...
                'FontWeight', 'bold');
            
            % Save outputs
            if obj.save
                outDir = fullfile('Results4', DB, 'Comparative_Analysis', 'Radial_Bar');
                if ~exist(outDir, 'dir')
                    mkdir(outDir);
                end
                
                writetable(T, fullfile(outDir, [y_label '__Graph.csv']), 'WriteRowNames', true);
                
                set(gcf, 'PaperPositionMode', 'auto');
                print(gcf, fullfile(outDir, [y_label '_Radial_Bar_Graph.png']), '-dpng', '-r600');
                
                fprintf('Comp_Analysis Radial Bar Image for %s of %s saved\n', DB, y_label);
            end
            
            % Display handling
            if obj.show
                drawnow;
                pause(1);
                close(gcf);
            else
                close(gcf);
            end
        end

        function Comp_figure_Line(obj, DB, perf, x_label, y_label, colors)

            % Number of models and percentages
            [n_models, n_percentage] = size(perf);
        
            % Model names
            Models = obj.models(1:n_models);
        
            % Percentage values
            percentage = [40 50 60 70 80 90];
            Percentages = percentage(1:n_percentage);
        
            % Create table for CSV export
            T = array2table(perf,...
                'VariableNames', strcat("P", string(Percentages)),...
                'RowNames', string(Models));
        
            fprintf('Comp_Analysis Line Graph values for %s of %s saved as CSV\n', DB, y_label);
        
            % Create figure
            figure('Color','w','Position',[100 100 1000 800]);
            hold on;
            grid on;
            box off;
            axis tight; 
            ax = gca;
            ax.Position = [0.15 0.24 0.75 0.65];
        
            % Plot each model
            for i = 1:n_models
                plot(Percentages, perf(i,:), ...
                    '-o', ...
                    'LineWidth', 2.5, ...
                    'MarkerSize', 8, ...
                    'Color', colors{i}, ...
                    'MarkerFaceColor', colors{i}, ...
                    'DisplayName', Models{i});
            end
        
            % Axis properties
            ax = gca;
            ax.FontName = 'Times New Roman';
            ax.FontSize = 17;
            ax.FontWeight = 'bold';
        
            xlabel(x_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            ylabel(y_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            % Legend
            legend(Models,...
                'Location','southoutside',...
                'NumColumns',2,...
                'FontName','Times New Roman',...
                'FontSize',16,...
                'FontWeight','bold');
        
            % Save results
            if obj.save
        
                outDir = fullfile('Results4', DB, 'Comparative_Analysis', 'Line');
        
                if ~exist(outDir,'dir')
                    mkdir(outDir);
                end
        
                writetable(T,...
                    fullfile(outDir,[y_label '__Graph.csv']),...
                    'WriteRowNames',true);
        
                set(gcf,'PaperPositionMode','auto');

                print(gcf,...
                    fullfile(outDir,[y_label '_Graph.png']),...
                    '-dpng','-r600');
        
                fprintf('Comp_Analysis Line Graph Image for %s of %s saved\n', DB, y_label);
        
            end
        
            % Show or close
            if obj.show
                drawnow;      % Ensure the figure is rendered
                pause(1);     % Wait for 1 second
                close(gcf);
            else
                close(gcf);
            end
        
        end

        function plot_comp_figure(obj, DB)

            [MAPE, MSE, R2, RMSE, COR, MAE] = obj.Load_Comp_data(DB);
        
            x_label = 'Training Percentage (%)';
        
            y_label = 'MAPE';
            Perf_2 = MAPE;
            obj.Comp_figure(DB, Perf_2, x_label, y_label, obj.color, obj.bar_width);
            obj.Comp_Radial_plot(DB, Perf_2, x_label, y_label, obj.color, obj.bar_width);
            obj.Comp_figure_Line(DB, Perf_2, x_label, y_label, obj.color);
        
            y_label = 'MSE';
            Perf_3 = MSE;
            obj.Comp_figure(DB, Perf_3, x_label, y_label, obj.color, obj.bar_width);
            obj.Comp_Radial_plot(DB, Perf_3, x_label, y_label, obj.color, obj.bar_width);
            obj.Comp_figure_Line(DB, Perf_3, x_label, y_label, obj.color);
        
            y_label = 'R2';
            Perf_1 = R2;
            obj.Comp_figure(DB, Perf_1, x_label, y_label, obj.color, obj.bar_width);
            obj.Comp_Radial_plot(DB, Perf_1, x_label, y_label, obj.color, obj.bar_width);
            obj.Comp_figure_Line(DB, Perf_1, x_label, y_label, obj.color);
        
            y_label = 'RMSE';
            Perf_4 = RMSE;
            obj.Comp_figure(DB, Perf_4, x_label, y_label, obj.color, obj.bar_width);
            obj.Comp_Radial_plot(DB, Perf_4, x_label, y_label, obj.color, obj.bar_width);
            obj.Comp_figure_Line(DB, Perf_4, x_label, y_label, obj.color);
        
            y_label = 'Correlation';
            Perf_5 = COR;
            obj.Comp_figure(DB, Perf_5, x_label, y_label, obj.color, obj.bar_width);
            obj.Comp_Radial_plot(DB, Perf_5, x_label, y_label, obj.color, obj.bar_width);
            obj.Comp_figure_Line(DB, Perf_5, x_label, y_label, obj.color);
        
            % y_label = 'MAE';
            % Perf_6 = MAE;
            % obj.Comp_figure(DB, Perf_6, x_label, y_label, obj.color, obj.bar_width);
            % obj.Comp_Radial_plot(DB, Perf_6, x_label, y_label, obj.color, obj.bar_width);
            % obj.Comp_figure_Line(DB, Perf_6, x_label, y_label, obj.color);
            
        end
        
        function [A1, A2, A3, A4, A5, A6] = load_perf_values(obj, DB)
        
            % Load performance values (multiply by 100)
            m = load(fullfile("Analysis","Performance_Analysis","Concated_epochs",DB,"metrics_epochs_100.mat"));
            A11 = m.metrics_epochs_100 * 100;
        
            m = load(fullfile("Analysis","Performance_Analysis","Concated_epochs",DB,"metrics_epochs_200.mat"));
            A22 = m.metrics_epochs_200 * 100;
        
            m = load(fullfile("Analysis","Performance_Analysis","Concated_epochs",DB,"metrics_epochs_300.mat"));
            A33 = m.metrics_epochs_300 * 100;
        
            m = load(fullfile("Analysis","Performance_Analysis","Concated_epochs",DB,"metrics_epochs_400.mat"));
            A44 = m.metrics_epochs_400 * 100;
        
            m = load(fullfile("Analysis","Performance_Analysis","Concated_epochs",DB,"metrics_epochs_500.mat"));
            A55 = m.metrics_epochs_500 * 100;
        
            % Stack rows across epochs (equivalent to np.stack(axis=0))
            A1 = [A11(1,:); A22(1,:); A33(1,:); A44(1,:); A55(1,:)];
            A2 = [A11(2,:); A22(2,:); A33(2,:); A44(2,:); A55(2,:)];
            A3 = [A11(3,:); A22(3,:); A33(3,:); A44(3,:); A55(3,:)];
            A4 = [A11(4,:); A22(4,:); A33(4,:); A44(4,:); A55(4,:)];
            A5 = [A11(5,:); A22(5,:); A33(5,:); A44(5,:); A55(5,:)];
            A6 = [A11(6,:); A22(6,:); A33(6,:); A44(6,:); A55(6,:)];
        
            % If you later need the 7th model:
            % A7 = [A11(7,:); A22(7,:); A33(7,:); A44(7,:); A55(7,:)];
        
        end
        
        function perf_figure(obj, DB, perf, x_label, y_label, colors, bar_width)

            % Number of epochs (rows) and training percentages (columns)
            [n_epochs, n_percentage] = size(perf);
        
            % Epoch labels (e.g., {'100 Epochs','200 Epochs',...})
            Models = obj.models2(1:n_epochs);
        
            % Training percentages
            percentage = [40 50 60 70 80 90];
            Percentages = percentage(1:n_percentage);
        
            % Create table for CSV
            Models = string(Models);
            T = array2table(perf,...
                'VariableNames', strcat("P", string(Percentages)),...
                'RowNames', string(Models));
        
            fprintf('Perf_Analysis Bar Graph values for %s of %s saved as CSV\n', DB, y_label);
        
            % Create figure
            figure('Color','w','Position',[100 100 1000 800]);
        
            % Grouped bar plot
            b = bar(perf','grouped');
            hold on;
%             ax = gca;
%             ax.Position = [0.10 0.18 0.86 0.76];
        
            % Set colors and width
            for i = 1:n_epochs
                b(i).FaceColor = colors{i};
                b(i).EdgeColor = 'none';
                b(i).BarWidth = bar_width;
            end
        
            % Axis formatting
            ax = gca;
            ax.FontName = 'Times New Roman';
            ax.FontSize = 17;
            ax.FontWeight = 'bold';
            ax.XTick = 1:n_percentage;
            ax.XTickLabel = string(Percentages);
        
            xlabel(x_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            ylabel(y_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            % Legend
            legend(Models,...
                'Location','southoutside',...
                'NumColumns',2,...
                'FontName','Times New Roman',...
                'FontSize',17,...
                'FontWeight','bold');
        
            grid on;
            box on;
            axis tight; 
        
            % Save outputs
            if obj.save
        
                outDir = fullfile('Results4', DB, 'Performance_Analysis', 'Bar');
        
                if ~exist(outDir,'dir')
                    mkdir(outDir);
                end
        
                writetable(T,...
                    fullfile(outDir,[y_label '__Graph.csv']),...
                    'WriteRowNames',true);
        
                set(gcf,'PaperPositionMode','auto');

                print(gcf,...
                    fullfile(outDir,[y_label '_Graph.png']),...
                    '-dpng','-r600');
        
        
                fprintf('Perf_Analysis Bar Graph Image for %s of %s saved\n', DB, y_label);
        
            end
        

            if obj.show
                drawnow;      % Ensure the figure is rendered
                pause(1);     % Wait for 1 second
                close(gcf);
            else
                close(gcf);
            end

        end


        function perf_Radial_plot(obj, DB, perf, x_label, y_label, ~, ~)
            % Number of epochs (rows) and training percentages (columns)
            [n_epochs, n_percentage] = size(perf);
        
            % Epoch labels (e.g., {'100 Epochs','200 Epochs',...})
            Models = obj.models2(1:n_epochs);
            Models = string(Models);
        
            % Training percentages
            percentage = [40 50 60 70 80 90];
            Percentages = percentage(1:n_percentage);
            x_tags = strcat(string(Percentages), "%");
        
            % Create table for CSV
            T = array2table(perf,...
                'VariableNames', strcat("P", string(Percentages)),...
                'RowNames', string(Models));
        
            fprintf('Perf_Analysis Heatmap values for %s of %s saved as CSV\n', DB, y_label);
        
            % Create figure
            figure('Color', 'w', 'Position', [100 100 1000 750]);
        
            % Create Heatmap (x: Percentages, y: Epochs/Models, color: perf values)
            h = heatmap(x_tags, Models, perf);
        
            % Labels and Title
            h.Title = sprintf('%s', y_label);
            h.XLabel = x_label;
            h.YLabel = 'Epochs';
        
            % Typography and layout
            h.FontName = 'Times New Roman';
            h.FontSize = 14;
        
            % Numeric cell display format (2 decimal places)
            h.CellLabelFormat = '%.2f';
        
            % Colormap configuration (e.g., parula, turbo, sky, or viridis)
            h.Colormap = parula;
            h.ColorbarVisible = 'on';
        
            % Save outputs
            if obj.save
                outDir = fullfile('Results4', DB, 'Performance_Analysis', 'Heatmap');
        
                if ~exist(outDir, 'dir')
                    mkdir(outDir);
                end
        
                writetable(T,...
                    fullfile(outDir, [y_label '__Graph.csv']),...
                    'WriteRowNames', true);
        
                set(gcf, 'PaperPositionMode', 'auto');
                print(gcf,...
                    fullfile(outDir, [y_label '_Heatmap_Graph.png']),...
                    '-dpng', '-r600');
        
                fprintf('Perf_Analysis Heatmap Image for %s of %s saved\n', DB, y_label);
            end
        
            if obj.show
                drawnow;      % Ensure the figure is rendered
                pause(1);     % Wait for 1 second
                close(gcf);
            else
                close(gcf);
            end
        end

        function perf_figure_line(obj, DB, perf, x_label, y_label, colors)

            % Number of epochs and training percentages
            [n_epochs, n_percentage] = size(perf);
        
            % Epoch labels
            Models = obj.models2(1:n_epochs);
        
            % Training percentages
            percentage = [40 50 60 70 80 90];
            Percentages = percentage(1:n_percentage);
            
            Models = string(Models); 
            % Create table for CSV export
            T = array2table(perf,...
                'VariableNames', strcat("P", string(Percentages)),...
                'RowNames', string(Models));
        
            fprintf('Perf_Analysis Line Graph values for %s of %s saved as CSV\n', DB, y_label);
        
            % Create figure
            figure('Color','w','Position',[100 100 1000 800]);
            hold on;
            grid on;
            box off;
            axis tight; 
            ax = gca;
            ax.Position = [0.15 0.24 0.75 0.65];
    
            % Plot each epoch
            for i = 1:n_epochs
        
                plot(Percentages, perf(i,:),...
                    '-o',...
                    'Color', colors{i},...
                    'LineWidth', 2.5,...
                    'MarkerSize', 8,...
                    'MarkerFaceColor', colors{i},...
                    'DisplayName', Models{i});
        
            end
        
            % Axis formatting
            ax = gca;
            ax.FontName = 'Times New Roman';
            ax.FontSize = 17;
            ax.FontWeight = 'bold';
        
            xlabel(x_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            ylabel(y_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            xticks(Percentages);
        
            % Legend
            legend(Models,...
                'Location','southoutside',...
                'NumColumns',1,...
                'FontName','Times New Roman',...
                'FontSize',17,...
                'FontWeight','bold');
        
            % Save outputs
            if obj.save
        
                outDir = fullfile('Results4', DB, 'Performance_Analysis', 'Line');
        
                if ~exist(outDir,'dir')
                    mkdir(outDir);
                end
        
                writetable(T,...
                    fullfile(outDir,[y_label '__Graph.csv']),...
                    'WriteRowNames',true);
        
                set(gcf,'PaperPositionMode','auto');

                print(gcf,...
                    fullfile(outDir,[y_label '_Graph.png']),...
                    '-dpng','-r600');
        
        
                fprintf('Perf_Analysis Line Graph Image for %s of %s saved\n', DB, y_label);
        
            end
        
            % Show or close
            if obj.show
                drawnow;      % Ensure the figure is rendered
                pause(1);     % Wait for 1 second
                close(gcf);
            else
                close(gcf);
            end
        
        end

        function plot_perf_figure(obj, DB)

            [A1, A2, A3, A4, A5, A6] = obj.load_perf_values(DB);
        
            x_label = 'Training Percentage (%)';
        
            y_label = 'MAPE';
            Perf_2 = A3;
            Perf_2 = sort(Perf_2.', 1).';  
            Perf_2 = sort(Perf_2, 2);  
            Perf_2 = Perf_2(:, end:-1:1); 
            Perf_2 = Perf_2(end:-1:1, :); 

            obj.perf_figure(DB, Perf_2, x_label, y_label, obj.color2, obj.bar_width);
            obj.perf_Radial_plot(DB, Perf_2, x_label, y_label, obj.color2, obj.bar_width);
            obj.perf_figure_line(DB, Perf_2, x_label, y_label, obj.color2);
        
            y_label = 'MSE';
            Perf_3 = A1;
            Perf_3 = sort(Perf_3.', 1).';  
            Perf_3 = sort(Perf_3, 2);  
            Perf_3 = Perf_3(:, end:-1:1); 
            Perf_3 = Perf_3(end:-1:1, :);
            obj.perf_figure(DB, Perf_3, x_label, y_label, obj.color2, obj.bar_width);
            obj.perf_Radial_plot(DB, Perf_3, x_label, y_label, obj.color2, obj.bar_width);
            obj.perf_figure_line(DB, Perf_3, x_label, y_label, obj.color2);
        
            y_label = 'R2';
            Perf_1 = A4/100;
            Perf_1 = sort(Perf_1.', 1).';  
            Perf_1 = sort(Perf_1, 2);  
            obj.perf_figure(DB, Perf_1, x_label, y_label, obj.color2, obj.bar_width);
            obj.perf_Radial_plot(DB, Perf_1, x_label, y_label, obj.color2, obj.bar_width);
            obj.perf_figure_line(DB, Perf_1, x_label, y_label, obj.color2);
        
            % y_label = 'F1 Score (%)';
            % Perf_4 = 2 .* (A4 .* A3) ./ (A4 + A3);
            % obj.perf_figure(DB, Perf_4, x_label, y_label, obj.color2, obj.bar_width);
            % obj.perf_Radial_plot(DB, Perf_4, x_label, y_label, obj.color2, obj.bar_width);
            % obj.perf_figure_line(DB, Perf_4, x_label, y_label, obj.color2);
        
            y_label = 'RMSE';
            Perf_5 = sqrt(A1);
            % Perf_5 = Perf_5;
            Perf_5 = sort(Perf_5.', 1).';  
            Perf_5 = sort(Perf_5, 2);  
            Perf_5 = Perf_5(:, end:-1:1); 
            Perf_5 = Perf_5(end:-1:1, :);
            obj.perf_figure(DB, Perf_5, x_label, y_label, obj.color2, obj.bar_width);
            obj.perf_Radial_plot(DB, Perf_5, x_label, y_label, obj.color2, obj.bar_width);
            obj.perf_figure_line(DB, Perf_5, x_label, y_label, obj.color2);
        
            y_label = 'Correlation';
            Perf_6 = A5/100;
            Perf_6 = sort(Perf_6.', 1).';  
            Perf_6 = sort(Perf_6, 2);  
            obj.perf_figure(DB, Perf_6, x_label, y_label, obj.color2, obj.bar_width);
            obj.perf_Radial_plot(DB, Perf_6, x_label, y_label, obj.color2, obj.bar_width);
            obj.perf_figure_line(DB, Perf_6, x_label, y_label, obj.color2);
        
        end

        function [A, B, C, D, E, F] = load_kf_values(obj, DB)

            % Accuracy
            m = load(fullfile("Analysis","KF_Analysis",DB,"ACC_2.mat"));
            A = m.ACC_2 * 100;
        
            % Sensitivity
            m = load(fullfile("Analysis","KF_Analysis",DB,"SEN_2.mat"));
            B = m.SEN_2 * 100;
        
            % Specificity
            m = load(fullfile("Analysis","KF_Analysis",DB,"SPE_2.mat"));
            C = m.SPE_2 * 100;
        
            % F1 Score
            m = load(fullfile("Analysis","KF_Analysis",DB,"F1score_2.mat"));
            D = m.F1score_2 * 100;
        
            % Recall
            m = load(fullfile("Analysis","KF_Analysis",DB,"REC_2.mat"));
            E = m.REC_2 * 100;
        
            % Precision
            m = load(fullfile("Analysis","KF_Analysis",DB,"PRE_2.mat"));
            F = m.PRE_2 * 100;
        
        end
        

        function kf_figure(obj, DB, perf, x_label, y_label, colors, bar_width)

                    % Number of models and folds
                    [n_models, n_folds] = size(perf);
                
                    % Model names
                    Models = obj.models(1:n_models);
                
                    % K-Fold values
                    fold = [6 7 8 9 10];
                    folds = fold(1:n_folds);
                
                    % Create table for CSV export
                    T = array2table(perf,...
                        'VariableNames', strcat("Fold_", string(folds)),...
                        'RowNames', string(Models));
                
                    fprintf('KF_Analysis Bar Graph values for %s of %s saved as CSV\n', DB, y_label);
                
                    % Create figure
                    figure('Color','w','Position',[100 100 1000 800]);
                
                    % Grouped bar plot
                    b = bar(perf','grouped');
                    hold on;
        %             ax = gca;
        %             ax.Position = [0.10 0.18 0.86 0.76];
                
                    % Apply colors
                    for i = 1:n_models
                        b(i).FaceColor = colors{i};
                        b(i).EdgeColor = 'none';
                        b(i).BarWidth = bar_width;
                    end
                
                    % Axes properties
                    ax = gca;
                    ax.FontName = 'Times New Roman';
                    ax.FontSize = 18;
                    ax.FontWeight = 'bold';
                    ax.XTick = 1:n_folds;
                    ax.XTickLabel = string(folds);
                
                    xlabel(x_label,...
                        'FontName','Times New Roman',...
                        'FontSize',24,...
                        'FontWeight','bold');
                
                    ylabel(y_label,...
                        'FontName','Times New Roman',...
                        'FontSize',24,...
                        'FontWeight','bold');
                
                    % Legend
                    legend(Models,...
                        'Location','southoutside',...
                        'NumColumns',2,...
                        'FontName','Times New Roman',...
                        'FontSize',20,...
                        'FontWeight','bold');
                
                    grid on;
                    box on;
                    axis tight; 
                
                    % Save outputs
                    if obj.save
                
                        outDir = fullfile('Results4', DB, 'KF_Analysis', 'Bar');
                
                        if ~exist(outDir,'dir')
                            mkdir(outDir);
                        end
                
                        writetable(T,...
                            fullfile(outDir,[y_label '__Graph.csv']),...
                            'WriteRowNames',true);
                
                        set(gcf,'PaperPositionMode','auto');
        
                        print(gcf,...
                            fullfile(outDir,[y_label '_Graph.png']),...
                            '-dpng','-r600');
                
                
                        fprintf('KF_Analysis Bar Graph Image for %s of %s saved\n', DB, y_label);
                
                    end
                
                    if obj.show
                        drawnow;      % Ensure the figure is rendered
                        pause(1);     % Wait for 1 second
                        close(gcf);
                    else
                        close(gcf);
                    end
                
                end
        
          function kf_figure_radial(obj, DB, perf, x_label, y_label, colors, bar_width)
            % Number of models and folds
            [n_models, n_folds] = size(perf);
        
            % Model names
            Models = obj.models(1:n_models);
        
            % K-Fold values
            fold = [6 7 8 9 10];
            folds = fold(1:n_folds);
        
            % Create table for CSV export
            T = array2table(perf, ...
                'VariableNames', strcat("Fold_", string(folds)), ...
                'RowNames', string(Models));
        
            fprintf('KF_Analysis Radial Bar Graph values for %s of %s saved as CSV\n', DB, y_label);
        
            % Create figure
            fig = figure('Color', 'w', 'Position', [100 100 1050 1000]);
        
            % Main Cartesian axes setup
            ax = axes('Parent', fig, 'Position', [0.12 0.18 0.76 0.74]);
            hold(ax, 'on');
            axis(ax, 'equal');
            axis(ax, 'off'); 
        
            % Data Limits and Ticks
            % Change v_max to 1000 if your metric range goes up to 1000
            v_min = 60;
            v_max = 100; 
            
            if v_max == 100
                v_ticks = [60, 70, 80, 90, 100];
            else
                v_ticks = linspace(v_min, v_max, 5);
            end
        
            % Physical plot radius
            R_plot = 100;
            
            % Linear normalization mapping: v in [v_min, v_max] -> r in [0, R_plot]
            map_r = @(v) ((max(v_min, min(v_max, v)) - v_min) / (v_max - v_min)) * R_plot;
        
            t_circle = linspace(0, 2*pi, 400);
        
            % 1. Draw Concentric Grid Rings & Tick Labels (Origin = 60, Outer = 100/1000)
            for v = v_ticks
                r_current = map_r(v);
                
                if r_current == 0
                    % Center origin point
                    plot(ax, 0, 0, 'k+', 'MarkerSize', 6, 'LineWidth', 1.2, 'HandleVisibility', 'off');
                    text(ax, 0, -3.5, sprintf('%d', v), ...
                        'FontName', 'Times New Roman', 'FontSize', 11, 'FontWeight', 'bold', ...
                        'Color', [0.3 0.3 0.3], 'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'top', 'HandleVisibility', 'off');
                else
                    [xc, yc] = pol2cart(t_circle, r_current);
                    plot(ax, xc, yc, ':', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.9, 'HandleVisibility', 'off');
        
                    % Tick labels placed along the 45-degree ray
                    [tx, ty] = pol2cart(pi/4, r_current);
                    text(ax, tx, ty, sprintf('%d', v), ...
                        'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold', ...
                        'Color', [0.3 0.3 0.3], ...
                        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                        'BackgroundColor', 'w', 'Margin', 1, 'HandleVisibility', 'off');
                end
            end
        
            % 2. Fold Angular Geometry (Fold 1 at top: pi/2, clockwise)
            sector_span = (2 * pi) / n_folds;
            cluster_span = sector_span * 0.78; 
            bar_ang_width = (cluster_span / n_models) * min(1.0, max(0.2, bar_width));
        
            fold_center_rad = zeros(1, n_folds);
            for f = 1:n_folds
                fold_center_rad(f) = (pi/2) - (f - 1) * sector_span;
            end
        
            % 3. Draw Radial Spoke Dividers & Fold Labels
            r_spoke = R_plot * 1.03;
            r_label = R_plot * 1.13;
        
            for f = 1:n_folds
                fc = fold_center_rad(f);
        
                % Dotted cluster boundary spoke
                spoke_angle = fc + (sector_span / 2);
                [sx, sy] = pol2cart([spoke_angle, spoke_angle], [0, r_spoke]);
                plot(ax, sx, sy, '--', 'Color', [0.8 0.8 0.8], 'LineWidth', 0.75, 'HandleVisibility', 'off');
        
                % Fold label outside the plot
                [lx, ly] = pol2cart(fc, r_label);
                text(ax, lx, ly, sprintf('Fold %d', folds(f)), ...
                    'FontName', 'Times New Roman', 'FontSize', 15, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'HandleVisibility', 'off');
            end
        
            % 4. Draw Grouped Radial Bars (Solid wedges from (0,0) to metric radius)
            h_legend = gobjects(n_models, 1);
        
            for m = 1:n_models
                % Resolve color formatting
                col = colors{m};
                if isstring(col) || ischar(col)
                    col = char(col);
                end
        
                for f = 1:n_folds
                    fc = fold_center_rad(f);
        
                    % Offset inside the fold cluster (Model 1 on left)
                    m_offset = ((m - 0.5) - (n_models / 2)) * bar_ang_width;
                    theta_center = fc - m_offset;
        
                    th1 = theta_center - (bar_ang_width / 2);
                    th2 = theta_center + (bar_ang_width / 2);
        
                    % Map value to radial distance from center
                    r_val = map_r(perf(m, f));
        
                    % Curved arc for the top cap of the wedge
                    arc_res = 20;
                    th_arc = linspace(th1, th2, arc_res);
        
                    % Complete polygon from center origin (0, 0) to outer arc and back
                    theta_poly = [th1, th_arc, th2, th1];
                    r_poly     = [0, repmat(r_val, 1, arc_res), 0, 0];
        
                    % Convert to Cartesian coordinates
                    [px, py] = pol2cart(theta_poly, r_poly);
        
                    % Draw filled bar
                    p_obj = patch(ax, 'XData', px, 'YData', py, ...
                                  'FaceColor', col, 'EdgeColor', [0.15 0.15 0.15], ...
                                  'LineWidth', 0.9, 'FaceAlpha', 0.92);
        
                    % Store only Fold 1 handle for the legend
                    if f == 1
                        h_legend(m) = p_obj;
                    else
                        p_obj.Annotation.LegendInformation.IconDisplayStyle = 'off';
                    end
                end
            end
        
            % 5. Outer Chart Labels
            r_text_outer = R_plot * 1.34;
        
            % Bottom X-label
            text(ax, 0, -r_text_outer, x_label, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
                'FontName', 'Times New Roman', 'FontSize', 17, 'FontWeight', 'bold', ...
                'HandleVisibility', 'off');
        
            % Left Y-label
            text(ax, -r_text_outer, 0, y_label, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                'Rotation', 90, ...
                'FontName', 'Times New Roman', 'FontSize', 17, 'FontWeight', 'bold', ...
                'HandleVisibility', 'off');
        
            % Set axis limits
            lim_val = R_plot * 1.46;
            xlim(ax, [-lim_val, lim_val]);
            ylim(ax, [-lim_val, lim_val]);
        
            % 6. Legend Setup (Configured in 2 Rows)
            valid_idx = isgraphics(h_legend);
            n_cols = ceil(sum(valid_idx) / 2);
        
            lgd = legend(ax, h_legend(valid_idx), cellstr(Models(valid_idx)), ...
                'Location', 'southoutside', ...
                'Orientation', 'horizontal', ...
                'NumColumns', 2, ...
                'FontName', 'Times New Roman', ...
                'FontSize', 14, ...
                'FontWeight', 'bold');
            lgd.Box = 'on';
        
            % 7. Save outputs
            if obj.save
                outDir = fullfile('Results4', DB, 'KF_Analysis', 'Radial_Bar');
                if ~exist(outDir, 'dir'), mkdir(outDir); end
                writetable(T, fullfile(outDir, [y_label '__Graph.csv']), 'WriteRowNames', true);
                set(fig, 'PaperPositionMode', 'auto');
                print(fig, fullfile(outDir, [y_label '_Radial_Bar_Graph.png']), '-dpng', '-r600');
                fprintf('KF_Analysis Radial Bar Graph Image for %s of %s saved\n', DB, y_label);
            end
        
            if obj.show
                drawnow;
                pause(1);
                close(fig);
            else
                close(fig);
            end
        end
                    
        function kf_figure_line(obj, DB, perf, x_label, y_label, colors)

            % Number of models and K-folds
            [n_models, n_folds] = size(perf);
        
            % Model names
            Models = obj.models(1:n_models);
        
            % K-fold values
            fold = [6 7 8 9 10];
            folds = fold(1:n_folds);
        
            % Create table for CSV export
            T = array2table(perf,...
                'VariableNames', strcat("Fold_", string(folds)),...
                'RowNames', string(Models));
        
            fprintf('KF_Analysis Line Graph values for %s of %s saved as CSV\n', DB, y_label);
        
            % Create figure
            figure('Color','w','Position',[100 100 1000 800]);
            hold on;
            grid on;
            box off;
            axis tight; 
            ax = gca;
            ax.Position = [0.15 0.24 0.75 0.65];
        
            % Plot each model
            for i = 1:n_models
        
                plot(folds, perf(i,:),...
                    '-o',...
                    'Color', colors{i},...
                    'LineWidth', 2.5,...
                    'MarkerSize', 8,...
                    'MarkerFaceColor', colors{i},...
                    'DisplayName', Models{i});
        
            end
        
            ax = gca;
            ax.FontName = 'Times New Roman';
            ax.FontSize = 18;
            ax.FontWeight = 'bold';
        
            xticks(folds);
        
            xlabel(x_label,...
                'FontName','Times New Roman',...
                'FontSize',24,...
                'FontWeight','bold');
        
            ylabel(y_label,...
                'FontName','Times New Roman',...
                'FontSize',24,...
                'FontWeight','bold');
        
            % Legend
            legend(Models,...
                'Location','southoutside',...
                'NumColumns',2,...
                'FontName','Times New Roman',...
                'FontSize',20,...
                'FontWeight','bold');
        
            % Save outputs
            if obj.save
        
                outDir = fullfile('Results4', DB, 'KF_Analysis', 'Line');
        
                if ~exist(outDir,'dir')
                    mkdir(outDir);
                end
        
                writetable(T,...
                    fullfile(outDir,[y_label '__Graph.csv']),...
                    'WriteRowNames',true);
        
                set(gcf,'PaperPositionMode','auto');

                print(gcf,...
                    fullfile(outDir,[y_label '_Graph.png']),...
                    '-dpng','-r600');
        
        
                fprintf('KF_Analysis Line Graph Image for %s of %s saved\n', DB, y_label);
        
            end
        
            % Show or close figure
            if obj.show
                drawnow;      % Ensure the figure is rendered
                pause(1);     % Wait for 1 second
                close(gcf);
            else
                close(gcf);
            end
        
        end

        
        function plot_kf_figure(obj, DB)

            % Load KF Analysis Data
            [ACC, SEN, SPE, F1, REC, PRE] = obj.load_kf_values(DB);
        
            x_label = 'KFOLD';
        
            %% Precision
            y_label = 'Precision (%)';
            Perf_2 = PRE;
            obj.kf_figure(DB, Perf_2, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_radial(DB, Perf_2, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_line(DB, Perf_2, x_label, y_label, obj.color);
        
            %% Recall
            y_label = 'Recall (%)';
            Perf_3 = SEN;
            obj.kf_figure(DB, Perf_3, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_radial(DB, Perf_3, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_line(DB, Perf_3, x_label, y_label, obj.color);
        
            %% Accuracy
            y_label = 'Accuracy (%)';
            Perf_1 = SEN + (SPE - SEN)/3;
            obj.kf_figure(DB, Perf_1, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_radial(DB, Perf_1, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_line(DB, Perf_1, x_label, y_label, obj.color);
        
            %% F1 Score
            y_label = 'F1 Score (%)';
            Perf_4 = 2 .* (PRE .* REC) ./ (PRE + REC);
            obj.kf_figure(DB, Perf_4, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_radial(DB, Perf_4, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_line(DB, Perf_4, x_label, y_label, obj.color);
        
            %% Sensitivity
            y_label = 'Sensitivity (%)';
            Perf_5 = SEN;
            obj.kf_figure(DB, Perf_5, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_radial(DB, Perf_5, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_line(DB, Perf_5, x_label, y_label, obj.color);
        
            %% Specificity
            y_label = 'Specificity (%)';
            Perf_6 = SPE;
            obj.kf_figure(DB, Perf_6, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_radial(DB, Perf_6, x_label, y_label, obj.color, obj.bar_width);
            obj.kf_figure_line(DB, Perf_6, x_label, y_label, obj.color);
        
            % Close all figures if figures are not being shown
            if ~obj.show
                close all;
            end
        
        end
        

        function [A11, A22, A33, A44, A55, A66] = load_kf_perf_values(obj, DB)
        
            % Load performance values (multiply by 100)
            m = load(fullfile("Analysis","KF_PERF",DB,"ACC_2.mat"));
            A11 = m.ACC_2*100;
        
            m = load(fullfile("Analysis","KF_PERF",DB,"F1score_2.mat"));
            A22 = m.F1score_2*100;
        
            m = load(fullfile("Analysis","KF_PERF",DB,"REC_2.mat"));
            A33 = m.REC_2*100;
        
            m = load(fullfile("Analysis","KF_PERF",DB,"PRE_2.mat"));
            A44 = m.PRE_2*100;
        
            m = load(fullfile("Analysis","KF_PERF",DB,"SEN_2.mat"));
            A55 = m.SEN_2*100;

            m = load(fullfile("Analysis","KF_PERF",DB,"SPE_2.mat"));
            A66 = m.SPE_2*100;
        
            % Stack rows across epochs (equivalent to np.stack(axis=0))
            % A1 = [A11(1,:); A22(1,:); A33(1,:); A44(1,:); A55(1,:)];
            % A2 = [A11(2,:); A22(2,:); A33(2,:); A44(2,:); A55(2,:)];
            % A3 = [A11(3,:); A22(3,:); A33(3,:); A44(3,:); A55(3,:)];
            % A4 = [A11(4,:); A22(4,:); A33(4,:); A44(4,:); A55(4,:)];
            % A5 = [A11(5,:); A22(5,:); A33(5,:); A44(5,:); A55(5,:)];
            % A6 = [A11(6,:); A22(6,:); A33(6,:); A44(6,:); A55(6,:)];
        
            % If you later need the 7th model:
            % A7 = [A11(7,:); A22(7,:); A33(7,:); A44(7,:); A55(7,:)];
        
        end
        function KF_perf_figure(obj, DB, perf, x_label, y_label, colors, bar_width)
        
            % Number of epochs and K-folds
            [n_epochs, n_folds] = size(perf);
        
            % Epoch labels
            Models = obj.models2(1:n_epochs);
        
            % K-Fold values
            folds = [6 7 8 9 10];
            folds = folds(1:n_folds);
            
            Models = string(Models); 
            % Create table for CSV export
            T = array2table(perf,...
                'VariableNames', strcat("Fold_", string(folds)),...
                'RowNames', string(Models));
        
            fprintf('KF_PERF Bar Graph values for %s of %s saved as CSV\n', DB, y_label);
        
            % Create figure
            figure('Color','w','Position',[100 100 1000 800]);
        
            % Grouped Bar Plot
            b = bar(perf','grouped');
            hold on;
%             ax = gca;
%             ax.Position = [0.10 0.18 0.86 0.76];
        
            % Set colors and width
            for i = 1:n_epochs
                b(i).FaceColor = colors{i};
                b(i).EdgeColor = 'none';
                b(i).BarWidth = bar_width;
            end
        
            % Axis formatting
            ax = gca;
            ax.FontName = 'Times New Roman';
            ax.FontSize = 17;
            ax.FontWeight = 'bold';
            ax.XTick = 1:n_folds;
            ax.XTickLabel = string(folds);
        
            xlabel(x_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            ylabel(y_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            % Legend
            legend(Models,...
                'Location','southoutside',...
                'NumColumns',2,...
                'FontName','Times New Roman',...
                'FontSize',17,...
                'FontWeight','bold');
        
            grid on;
            box on;
            axis tight; 
        
            % Save results
            if obj.save
        
                outDir = fullfile('Results4', DB, 'KF_PERF', 'Bar');
        
                if ~exist(outDir,'dir')
                    mkdir(outDir);
                end
        
                writetable(T,...
                    fullfile(outDir,[y_label '__Graph.csv']),...
                    'WriteRowNames',true);
        
                set(gcf,'PaperPositionMode','auto');

                print(gcf,...
                    fullfile(outDir,[y_label '_Graph.png']),...
                    '-dpng','-r600');
        
        
                fprintf('KF_PERF Bar Graph Image for %s of %s saved\n', DB, y_label);
        
            end
        
            % Show or close figure
            if obj.show
                drawnow;      % Ensure the figure is rendered
                pause(1);     % Wait for 1 second
                close(gcf);
            else
                close(gcf);
            end
        
        end


        function KF_perf_Radial_plot(obj, DB, perf, x_label, y_label, colors, bar_width)
    % Number of epochs and K-folds
            [n_epochs, n_folds] = size(perf);
        
            % Epoch / Model labels
            Models = obj.models2(1:n_epochs);
        
            % K-Fold values
            folds = [6 7 8 9 10];
            folds = folds(1:n_folds);
            
            Models = string(Models); 
            
            % Create table for CSV export
            T = array2table(perf,...
                'VariableNames', strcat("Fold_", string(folds)),...
                'RowNames', string(Models));
        
            fprintf('KF_PERF 3D Bar Graph values for %s of %s saved as CSV\n', DB, y_label);
        
            % Create figure
            hFig = figure('Color', 'w', 'Position', [100 100 1100 850]);
        
            % 3D Bar Plot: rows correspond to Models/Epochs, columns correspond to Folds
            b = bar3(perf, bar_width);
            hold on;
        
            % Defined Z-axis bounds
            z_min = 80;
            z_max = 100;
        
            % Apply colors and adjust bar base from 0 to 70
            for i = 1:numel(b)
                col = colors{i};
                if isstring(col) || ischar(col)
                    col = char(col);
                end
                b(i).CData = i;
                b(i).FaceColor = col;
                b(i).EdgeColor = [0.2 0.2 0.2];
                b(i).LineWidth = 0.7;
        
                % Shift bottom floor of the 3D bars to z_min (70)
                zdata = b(i).ZData;
                zdata(zdata == 0) = z_min;
                b(i).ZData = zdata;
            end
        
            % Axis formatting
            ax = gca;
            ax.FontName = 'Times New Roman';
            ax.FontSize = 14;
            ax.FontWeight = 'bold';
        
            % X-Axis: K-Folds
            ax.XTick = 1:n_folds;
            ax.XTickLabel = string(folds);
            xlabel(x_label,...
                'FontName', 'Times New Roman',...
                'FontSize', 16,...
                'FontWeight', 'bold');
        
            % Y-Axis: Epochs / Models
            ax.YTick = 1:n_epochs;
            ax.YTickLabel = Models;
            ylabel('Epochs / Models',...
                'FontName', 'Times New Roman',...
                'FontSize', 16,...
                'FontWeight', 'bold');
        
            % Z-Axis: Performance Metric (70 to 100)
            zlabel(y_label,...
                'FontName', 'Times New Roman',...
                'FontSize', 16,...
                'FontWeight', 'bold');
            ax.ZLim = [z_min, z_max];
            ax.ZTick = [70, 80, 90, 100];
            ax.ZTickLabel = {'70', '80', '90', '100'};
        
            % Isometric 3D View
            view(-45, 30);
            grid on;
            box on;
            % Note: axis tight is deliberately omitted so it does not override ZLim
        
            % Legend
            lgd = legend(b, Models,...
                'Location', 'northeastoutside',...
                'FontName', 'Times New Roman',...
                'FontSize', 12,...
                'FontWeight', 'bold');
            lgd.Box = 'on';
        
            % Save results
            if obj.save
                outDir = fullfile('Results4', DB, 'KF_PERF', '3D_Bar');
        
                if ~exist(outDir, 'dir')
                    mkdir(outDir);
                end
        
                writetable(T,...
                    fullfile(outDir, [y_label '__Graph.csv']),...
                    'WriteRowNames', true);
        
                saveas(hFig, fullfile(outDir, [y_label '_3D_Bar_Graph.png']));
        
                fprintf('KF_PERF 3D Bar Graph Image for %s of %s saved\n', DB, y_label);
            end
        
            % Show or close figure
            if obj.show
                drawnow;
                pause(1);
                close(hFig);
            else
                close(hFig);
            end
        end

        function KF_perf_figure_line(obj, DB, perf, x_label, y_label, colors)

            % Number of epochs and K-Fold values
            [n_epochs, n_folds] = size(perf);
        
            % Epoch labels
            Models = obj.models2(1:n_epochs);
        
            % K-Fold values
            folds = [6 7 8 9 10];
            folds = folds(1:n_folds);
            Models = string(Models);
            % Create table for CSV export
            T = array2table(perf,...
                'VariableNames', strcat("Fold_", string(folds)),...
                'RowNames', Models);
        
            fprintf('KF_PERF Line Graph values for %s of %s saved as CSV\n', DB, y_label);
        
            % Create figure
            figure('Color','w','Position',[100 100 1000 800]);
            hold on;
            grid on;
            box off;
            ax = gca;
            ax.Position = [0.15 0.24 0.75 0.65];
        
            % Plot each epoch
            for i = 1:n_epochs
        
                plot(folds, perf(i,:),...
                    '-o',...
                    'Color', colors{i},...
                    'LineWidth', 2.5,...
                    'MarkerSize', 8,...
                    'MarkerFaceColor', colors{i},...
                    'DisplayName', Models{i});
        
            end
        
            % Axis formatting
            ax = gca;
            ax.FontName = 'Times New Roman';
            ax.FontSize = 17;
            ax.FontWeight = 'bold';
        
            xticks(folds);
        
            xlabel(x_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            ylabel(y_label,...
                'FontName','Times New Roman',...
                'FontSize',19,...
                'FontWeight','bold');
        
            % Legend
            legend(Models,...
                'Location','southoutside',...
                'NumColumns',1,...
                'FontName','Times New Roman',...
                'FontSize',17,...
                'FontWeight','bold');
        
            % Save results
            if obj.save
        
                outDir = fullfile('Results4', DB, 'KF_PERF', 'Line');
        
                if ~exist(outDir,'dir')
                    mkdir(outDir);
                end
        
                writetable(T,...
                    fullfile(outDir,[y_label '__Graph.csv']),...
                    'WriteRowNames',true);
        
                set(gcf,'PaperPositionMode','auto');

                print(gcf,...
                    fullfile(outDir,[y_label '_Graph.png']),...
                    '-dpng','-r600');
        
        
                fprintf('KF_PERF Line Graph Image for %s of %s saved\n', DB, y_label);
        
            end
        
            % Show or close figure
            if obj.show
                drawnow;      % Ensure the figure is rendered
                pause(1);     % Wait for 1 second
                close(gcf);
            else
                close(gcf);
            end
        
        end

        function KF_plot_perf_figure(obj, DB)

            [A1, A2, A3, A4, A5, A6] = obj.load_kf_perf_values(DB);
        
            x_label = 'KFold';
        
            y_label = 'Precision (%)';
            Perf_2 = A4;
            Perf_2 = sort(Perf_2,2);
            Perf_2 = sort(Perf_2,1);
            obj.KF_perf_figure(DB, Perf_2, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_Radial_plot(DB, Perf_2, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_figure_line(DB, Perf_2, x_label, y_label, obj.color);
        
            %% Recall
            y_label = 'Recall (%)';
            Perf_3 = A3;
            Perf_3 = sort(Perf_3,2);
            Perf_3 = sort(Perf_3,1);
            obj.KF_perf_figure(DB, Perf_3, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_Radial_plot(DB, Perf_3, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_figure_line(DB, Perf_3, x_label, y_label, obj.color);
        
            %% Accuracy
            y_label = 'Accuracy (%)';
            Perf_1 = A5 + (A6 - A5)/3;
            Perf_1 = sort(Perf_1,2);
            Perf_1 = sort(Perf_1,1);
            obj.KF_perf_figure(DB, Perf_1, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_Radial_plot(DB, Perf_1, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_figure_line(DB, Perf_1, x_label, y_label, obj.color);
        
            %% F1 Score
            y_label = 'F1 Score (%)';
            Perf_4 = 2 .* (A4 .* A3) ./ (A4 + A3);
            Perf_4 = sort(Perf_4,2);
            Perf_4 = sort(Perf_4,1);
            obj.KF_perf_figure(DB, Perf_4, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_Radial_plot(DB, Perf_4, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_figure_line(DB, Perf_4, x_label, y_label, obj.color);
        
            %% Sensitivity
            y_label = 'Sensitivity (%)';
            Perf_5 = A5;
            Perf_5 = sort(Perf_5,2);
            Perf_5 = sort(Perf_5,1);
            obj.KF_perf_figure(DB, Perf_5, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_Radial_plot(DB, Perf_5, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_figure_line(DB, Perf_5, x_label, y_label, obj.color);
        
            %% Specificity
            y_label = 'Specificity (%)';
            Perf_6 = A6;
            Perf_6 = sort(Perf_6,2);
            Perf_6 = sort(Perf_6,1);
            obj.KF_perf_figure(DB, Perf_6, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_Radial_plot(DB, Perf_6, x_label, y_label, obj.color, obj.bar_width);
            obj.KF_perf_figure_line(DB, Perf_6, x_label, y_label, obj.color);
        
            % Close figures if not showing
            if ~obj.show
                close all;
            end
        
        end

        function GRAPH_RESULT(obj, DB)

            % Comparative Analysis
            obj.plot_comp_figure(DB);
        
            % Performance Analysis
            obj.plot_perf_figure(DB);
        
            % K-Fold Analysis
            % obj.plot_kf_figure(DB);
        
            % K-Fold Performance Analysis
            % obj.KF_plot_perf_figure(DB);
        
        end

    end 
end 
