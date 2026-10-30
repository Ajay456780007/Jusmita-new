function [training_sequence, testing_sequence, training_labels, testing_labels] = train_test_splitter(data, percent, num)
    if nargin < 3
        num = 500;
    end

    [feat, label] = Load_data(data);  

%     label = label.EncodedLabels;
    
    unique_classes = unique(label);
    
    selected_indices_per_class = cell(length(unique_classes), 1);

    for i = 1:length(unique_classes)
        cls = unique_classes(i);
        
        class_indices = find(label == cls);
        
        rand_idx = randi(length(class_indices), num, 1);
        selected_class_indices = class_indices(rand_idx);
        
        selected_indices_per_class{i} = selected_class_indices;
    end

    selected_indices = cat(1, selected_indices_per_class{:});

    shuffled_order = randperm(length(selected_indices));
    selected_indices = selected_indices(shuffled_order);

    balanced_feat = feat(selected_indices, :); 
    balanced_label = label(selected_indices);

    data_size = size(balanced_feat, 1);
    split_point = floor(data_size * percent);

    training_sequence = balanced_feat(1:split_point, :);
    training_labels   = balanced_label(1:split_point);
    
    testing_sequence  = balanced_feat(split_point+1:end, :);
    testing_labels    = balanced_label(split_point+1:end);
end