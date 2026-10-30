clc;
clear;
addpath(genpath(pwd));
warning off;


tt = menu("Which mode of Execution?","Complete Execution","Pre Evaluated Plots");

DB = "VRFB";

if tt ==1
    % Function used for reading the data
    Read_data();
    % Instance created for Analysis
    T = Analysis(DB);
    % Performing Comparative Analysis
    T.Comparative_Analysis();
    % Performing Performance Analysis
    T.PERF_Analysis();
    % Instance created for plotting the grahs
    A = Plot();
    % Function call for plotting the graphs and saving it
    A.GRAPH_RESULT(DB);
else
    DB = "VRFB";
    % Instance created for plot 
    A = Plot();
    % Function call for plotting the graphs and saving it 
    A.GRAPH_RESULT(DB);
end 