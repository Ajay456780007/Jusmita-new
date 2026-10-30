function [feat,label] = Load_data(DB)

F = load(fullfile("data_loader",DB,"Features.mat"));
L = load(fullfile("data_loader",DB,"Labels.mat"));

feat = F.Features;
label = L.Labels;

end