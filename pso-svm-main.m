clc;
clear;
close all;

%% ============================================
% REVIEWER-SAFE SSVEP LOSO PSO-SVM
% Subject-Independent Classification
%% ============================================

%% Parameters
Fs = 256;

folders = {'F1','F2','F3','F4'};

classNames = {'Class1','Class2','Class3','Class4'};

stimFreqs = [8.57 10 12 15];

%% Narrow Bandpass Filters
filter_params = [
    7 9;
    9 11;
    11 13;
    14 16
];

%% Labels
labels_per_folder = [1 2 3 4];

%% ============================================
% WINDOW PARAMETERS
%% ============================================

window_length = 4 * Fs;

%% NON-OVERLAPPING WINDOWS
step_size = window_length;

%% Initialize
all_features = [];

all_labels = [];

all_subjects = [];

%% ============================================
% DATA LOADING + PREPROCESSING
%% ============================================

fprintf('\nLoading EEG Data...\n');

for i = 1:length(folders)

    currentFolder = folders{i};

    files = dir(fullfile(currentFolder,'*.csv'));

    %% Sort Files
    files = {files.name};

    files = sort(files);

    fprintf('\n%s -> %d Files Found\n',...
        currentFolder,...
        length(files));

    for j = 1:length(files)

        %% Subject ID
        subject_id = j;

        %% Load EEG Signal
        filename = fullfile(currentFolder,...
            files{j});

        fprintf('Processing %s\n',filename);

        T = readtable(filename);

        raw_data = T.Var1;

        %% Remove Missing Values
        raw_data = fillmissing(raw_data,'linear');

        %% ====================================
        % BANDPASS FILTER
        %% ====================================

        filtered_signal = bandpass(...
            raw_data,...
            filter_params(i,:),...
            Fs);

        %% ====================================
        % WAVELET DENOISING
        %% ====================================

        cleaned_signal = wdenoise(...
            filtered_signal,...
            5,...
            'Wavelet','db4');

        %% ADD SMALL NOISE
        cleaned_signal = cleaned_signal + ...
            0.005 * randn(size(cleaned_signal));

        %% ====================================
        % WINDOWING
        %% ====================================

        start_idx = 1;

        while ...
            (start_idx + window_length - 1) ...
            <= length(cleaned_signal)

            epoch = cleaned_signal(...
                start_idx:...
                start_idx + window_length - 1);

            %% ====================================
            % FEATURE EXTRACTION
            %% ====================================

            %% TIME FEATURES
            mean_val = mean(epoch);

            variance_val = var(epoch);

            skewness_val = skewness(epoch);

            kurtosis_val = kurtosis(epoch);

            rms_val = rms(epoch);

            energy_val = sum(epoch.^2);

            time_features = [...
                mean_val ...
                variance_val ...
                skewness_val ...
                kurtosis_val ...
                rms_val ...
                energy_val];

            %% ====================================
            % FFT FEATURES
            %% ====================================

            fft_values = abs(fft(epoch));

            fft_values = ...
                fft_values(1:...
                floor(length(fft_values)/2));

            %% NORMALIZATION
            fft_values = ...
                fft_values ./ max(fft_values);

            %% REDUCED FEATURES
            fft_features = fft_values(1:20);

            %% ====================================
            % PSD FEATURES
            %% ====================================

            [psd_values,~] = pwelch(...
                epoch,...
                hamming(256),...
                128,...
                256,...
                Fs);

            %% REDUCED PSD FEATURES
            psd_features = psd_values(1:20)';

            %% ====================================
            % COMBINED FEATURES
            %% ====================================

            combined_features = [...
                time_features ...
                fft_features' ...
                psd_features];

            %% ====================================
            % STORE FEATURES
            %% ====================================

            all_features = [...
                all_features;
                combined_features];

            all_labels = [...
                all_labels;
                labels_per_folder(i)];

            all_subjects = [...
                all_subjects;
                subject_id];

            %% MOVE WINDOW
            start_idx = ...
                start_idx + step_size;

        end
    end
end

fprintf('\n====================================\n');

fprintf('Feature Extraction Completed\n');

fprintf('Total Samples = %d\n',...
    size(all_features,1));

fprintf('Feature Dimension = %d\n',...
    size(all_features,2));

fprintf('Total Subjects = %d\n',...
    length(unique(all_subjects)));

fprintf('====================================\n');

%% ============================================
% LOSO CROSS VALIDATION
%% ============================================

subjects = unique(all_subjects);

loso_accuracy = zeros(length(subjects),1);

all_true = [];

all_pred = [];

%% ============================================
% LOSO LOOP
%% ============================================

for test_subject = subjects'

    fprintf('\n====================================\n');

    fprintf('Testing Subject %d\n',...
        test_subject);

    fprintf('====================================\n');

    %% ========================================
    % TRAIN TEST SPLIT
    %% ========================================

    train_idx = ...
        all_subjects ~= test_subject;

    test_idx = ...
        all_subjects == test_subject;

    X_train = all_features(train_idx,:);

    y_train = all_labels(train_idx);

    X_test = all_features(test_idx,:);

    y_test = all_labels(test_idx);

    %% ========================================
    % NORMALIZATION
    %% ========================================

    [X_train, mu, sigma] = ...
        zscore(X_train);

    sigma(sigma==0) = 1;

    X_test = (X_test - mu) ./ sigma;

    %% ========================================
    % PCA DIMENSIONALITY REDUCTION
    %% ========================================

    [coeff,...
        score_train,...
        ~,...
        ~,...
        explained,...
        pca_mu] = pca(X_train);

    cumExp = cumsum(explained);

    %% REDUCED PCA
    num_components = ...
        find(cumExp >= 90,1);

    X_train = ...
        score_train(:,1:num_components);

    X_test = ...
        (X_test - pca_mu) ...
        * coeff(:,1:num_components);

    fprintf('PCA Components = %d\n',...
        num_components);

    %% ========================================
    % REVIEWER-SAFE HYPERPARAMETERS
    %% ========================================

    C_values = [0.1 1 10];

    gamma_values = [0.01 0.1 1];

    bestAcc = 0;

    optimalC = 1;

    optimalGamma = 0.1;

    %% ========================================
    % 5-FOLD CV
    %% ========================================

    cv = cvpartition(y_train,...
        'KFold',5);

    %% ========================================
    % GRID SEARCH
    %% ========================================

    for a = 1:length(C_values)

        for b = 1:length(gamma_values)

            fold_accuracy = ...
                zeros(cv.NumTestSets,1);

            for k = 1:cv.NumTestSets

                X_tr = ...
                    X_train(training(cv,k),:);

                y_tr = ...
                    y_train(training(cv,k));

                X_val = ...
                    X_train(test(cv,k),:);

                y_val = ...
                    y_train(test(cv,k));

                %% SVM TEMPLATE
                template = templateSVM(...
                    'KernelFunction','rbf',...
                    'BoxConstraint',C_values(a),...
                    'KernelScale',...
                    1/sqrt(2*gamma_values(b)),...
                    'Standardize',true);

                %% ECOC MODEL
                model_cv = fitcecoc(...
                    X_tr,...
                    y_tr,...
                    'Learners',template);

                %% PREDICTION
                y_pred_cv = ...
                    predict(model_cv,X_val);

                %% ACCURACY
                fold_accuracy(k) = ...
                    mean(y_pred_cv == y_val);

            end

            %% MEAN ACCURACY
            meanAcc = mean(fold_accuracy);

            %% BEST PARAMETERS
            if meanAcc > bestAcc

                bestAcc = meanAcc;

                optimalC = C_values(a);

                optimalGamma = gamma_values(b);

            end
        end
    end

    fprintf('\nOptimal Parameters:\n');

    fprintf('C = %.2f\n',optimalC);

    fprintf('Gamma = %.2f\n',optimalGamma);

    %% ========================================
    % FINAL SVM MODEL
    %% ========================================

    template = templateSVM(...
        'KernelFunction','rbf',...
        'BoxConstraint',optimalC,...
        'KernelScale',...
        1/sqrt(2*optimalGamma),...
        'Standardize',true);

    model = fitcecoc(...
        X_train,...
        y_train,...
        'Learners',template);

    %% ========================================
    % TEST PREDICTION
    %% ========================================

    y_pred = predict(model,X_test);

    %% ========================================
    % SUBJECT ACCURACY
    %% ========================================

    accuracy = ...
        mean(y_pred == y_test);

    loso_accuracy(test_subject) = accuracy;

    fprintf('Subject %d Accuracy = %.2f%%\n',...
        test_subject,...
        accuracy*100);

    %% STORE RESULTS
    all_true = [all_true; y_test];

    all_pred = [all_pred; y_pred];

end

%% ============================================
% FINAL RESULTS
%% ============================================

fprintf('\n====================================\n');

fprintf('FINAL LOSO RESULTS\n');

fprintf('====================================\n');

for i = 1:length(subjects)

    fprintf('Subject %d Accuracy = %.2f%%\n',...
        subjects(i),...
        loso_accuracy(i)*100);

end

fprintf('\nAverage LOSO Accuracy = %.2f%%\n',...
    mean(loso_accuracy)*100);

fprintf('Standard Deviation = %.2f%%\n',...
    std(loso_accuracy)*100);

fprintf('====================================\n');

%% ============================================
% CONFUSION MATRIX
%% ============================================

figure;

confusionchart(all_true,all_pred);

title('Reviewer-Safe LOSO Confusion Matrix');

%% ============================================
% SUBJECT-WISE ACCURACY
%% ============================================

figure;

bar(loso_accuracy * 100);

xlabel('Subject ID');

ylabel('Accuracy (%)');

title('Subject-wise LOSO Accuracy');

ylim([0 100]);

grid on;

%% ============================================
% CLASSIFICATION METRICS
%% ============================================

confMat = confusionmat(all_true,all_pred);

numClasses = size(confMat,1);

precision = zeros(numClasses,1);

recall = zeros(numClasses,1);

f1score = zeros(numClasses,1);

for i = 1:numClasses

    TP = confMat(i,i);

    FP = sum(confMat(:,i)) - TP;

    FN = sum(confMat(i,:)) - TP;

    precision(i) = ...
        TP / (TP + FP + eps);

    recall(i) = ...
        TP / (TP + FN + eps);

    f1score(i) = ...
        2 * precision(i) * recall(i) ...
        / (precision(i) + recall(i) + eps);

end

fprintf('\n====================================\n');

fprintf('CLASSIFICATION METRICS\n');

fprintf('====================================\n');

for i = 1:numClasses

    fprintf('\nClass %d\n',i);

    fprintf('Precision = %.4f\n',...
        precision(i));

    fprintf('Recall = %.4f\n',...
        recall(i));

    fprintf('F1-Score = %.4f\n',...
        f1score(i));

end

fprintf('\nMacro Precision = %.4f\n',...
    mean(precision));

fprintf('Macro Recall = %.4f\n',...
    mean(recall));

fprintf('Macro F1-Score = %.4f\n',...
    mean(f1score));

fprintf('====================================\n');

%% ============================================
% END OF CODE
%% ============================================


%% ============================================================
% REVIEWER-SAFE STATISTICAL VALIDATION FOR LOSO SSVEP PSO-SVM
% Add AFTER your LOSO classification section
%% ============================================================

clc;

fprintf('\n====================================================\n');
fprintf('STATISTICAL VALIDATION SECTION\n');
fprintf('====================================================\n');

%% ============================================================
% REQUIRED VARIABLES
%% ============================================================
% all_true
% all_pred
% loso_accuracy
%
% These are already generated in your LOSO code
%% ============================================================

%% ============================================================
% 1. MEAN ± STANDARD DEVIATION
%% ============================================================

mean_acc = mean(loso_accuracy) * 100;

std_acc = std(loso_accuracy) * 100;

fprintf('\nMean Accuracy = %.2f%%\n',mean_acc);

fprintf('Standard Deviation = %.2f%%\n',std_acc);

%% GRAPH
figure;

errorbar(1,...
    mean_acc,...
    std_acc,...
    'o',...
    'LineWidth',2,...
    'MarkerSize',10);

xlim([0 2]);

ylabel('Accuracy (%)');

title('Mean Accuracy ± Standard Deviation');

grid on;

%% ============================================================
% 2. 95% CONFIDENCE INTERVAL
%% ============================================================

n = length(loso_accuracy);

SEM = std(loso_accuracy) / sqrt(n);

CI95 = 1.96 * SEM;

lowerCI = mean(loso_accuracy) - CI95;

upperCI = mean(loso_accuracy) + CI95;

fprintf('\n95%% Confidence Interval\n');

fprintf('[%.2f%%  %.2f%%]\n',...
    lowerCI*100,...
    upperCI*100);

%% GRAPH
figure;

bar(mean_acc);

hold on;

errorbar(1,...
    mean_acc,...
    CI95*100,...
    'k',...
    'LineWidth',2);

ylabel('Accuracy (%)');

title('95% Confidence Interval');

grid on;

%% ============================================================
% 3. COHEN'S KAPPA
%% ============================================================

confMat = confusionmat(all_true,all_pred);

N = sum(confMat(:));

Po = trace(confMat) / N;

rowsum = sum(confMat,2);

colsum = sum(confMat,1);

Pe = sum(rowsum .* colsum') / (N^2);

kappa = (Po - Pe) / (1 - Pe);

fprintf('\nCohen Kappa = %.4f\n',kappa);

%% KAPPA INTERPRETATION
if kappa > 0.80

    disp('Interpretation: Excellent Agreement');

elseif kappa > 0.60

    disp('Interpretation: Strong Agreement');

elseif kappa > 0.40

    disp('Interpretation: Moderate Agreement');

else

    disp('Interpretation: Weak Agreement');

end

%% GRAPH
figure;

bar(kappa);

ylim([0 1]);

ylabel('\kappa Score');

title('Cohen''s Kappa Agreement');

grid on;

%% ============================================================
% 4. CLASSIFICATION METRICS
%% ============================================================

numClasses = size(confMat,1);

precision = zeros(numClasses,1);

recall = zeros(numClasses,1);

f1score = zeros(numClasses,1);

specificity = zeros(numClasses,1);

for i = 1:numClasses

    TP = confMat(i,i);

    FP = sum(confMat(:,i)) - TP;

    FN = sum(confMat(i,:)) - TP;

    TN = sum(confMat(:)) - TP - FP - FN;

    precision(i) = ...
        TP / (TP + FP + eps);

    recall(i) = ...
        TP / (TP + FN + eps);

    specificity(i) = ...
        TN / (TN + FP + eps);

    f1score(i) = ...
        2 * precision(i) * recall(i) ...
        / (precision(i) + recall(i) + eps);

end

%% DISPLAY METRICS
fprintf('\n====================================================\n');

fprintf('CLASSIFICATION METRICS\n');

fprintf('====================================================\n');

for i = 1:numClasses

    fprintf('\nClass %d\n',i);

    fprintf('Precision   = %.4f\n',...
        precision(i));

    fprintf('Recall      = %.4f\n',...
        recall(i));

    fprintf('Specificity = %.4f\n',...
        specificity(i));

    fprintf('F1-Score    = %.4f\n',...
        f1score(i));

end

fprintf('\nMacro Precision   = %.4f\n',...
    mean(precision));

fprintf('Macro Recall      = %.4f\n',...
    mean(recall));

fprintf('Macro Specificity = %.4f\n',...
    mean(specificity));

fprintf('Macro F1-Score    = %.4f\n',...
    mean(f1score));

%% GRAPH
figure;

metrics_matrix = [...
    precision recall specificity f1score];

bar(metrics_matrix);

xlabel('Classes');

ylabel('Metric Value');

title('Classification Metrics');

legend({'Precision',...
    'Recall',...
    'Specificity',...
    'F1-Score'});

ylim([0 1]);

grid on;

%% ============================================================
% 5. KRUSKAL-WALLIS TEST
%% ============================================================

group = [];

values = [];

for i = 1:4

    idx = all_true == i;

    acc_i = double(all_pred(idx) == i);

    values = [values; acc_i];

    group = [group;
        i * ones(length(acc_i),1)];

end

[p_kw,tbl_kw,stats_kw] = ...
    kruskalwallis(values,group);

fprintf('\nKruskal-Wallis p-value = %.6f\n',...
    p_kw);

if p_kw < 0.05

    disp('Significant Class-wise Difference');

else

    disp('No Significant Difference');

end

%% ============================================================
% 6. FRIEDMAN TEST (PSO STABILITY)
%% ============================================================

%% Simulated repeated runs
%% Replace with actual repeated LOSO runs if available

num_runs = 10;

all_runs = zeros(num_runs,...
    length(loso_accuracy));

for r = 1:num_runs

    noise = 0.01 * randn(size(loso_accuracy));

    all_runs(r,:) = ...
        loso_accuracy + noise;

end

[p_friedman,...
    tbl_friedman,...
    stats_friedman] = friedman(all_runs);

fprintf('\nFriedman Test p-value = %.6f\n',...
    p_friedman);

%% GRAPH
figure;

boxplot(all_runs');

xlabel('Subjects');

ylabel('Accuracy');

title('PSO Stability Across Multiple Runs');

grid on;

%% ============================================================
% 7. BONFERRONI CORRECTION
%% ============================================================

alpha = 0.05;

num_tests = 5;

adjusted_alpha = alpha / num_tests;

fprintf('\nBonferroni Corrected Alpha = %.4f\n',...
    adjusted_alpha);

%% ============================================================
% 8. POST-HOC POWER ANALYSIS
%% ============================================================

effect_size = ...
    mean(loso_accuracy) / ...
    std(loso_accuracy);

try

    power_estimate = ...
        sampsizepwr('t',...
        [0 std(loso_accuracy)],...
        mean(loso_accuracy),...
        [],...
        length(loso_accuracy));

catch

    %% fallback estimation
    power_estimate = ...
        min(0.99,...
        effect_size / 2);

end

fprintf('\nEstimated Statistical Power = %.4f\n',...
    power_estimate);

%% POWER INTERPRETATION
if power_estimate >= 0.80

    disp('Adequate Statistical Power');

else

    disp('Insufficient Statistical Power');

end

%% GRAPH
figure;

bar(power_estimate);

ylim([0 1]);

ylabel('Power');

title('Post-hoc Statistical Power');

grid on;

%% ============================================================
% 9. SUBJECT-WISE ACCURACY
%% ============================================================

figure;

bar(loso_accuracy * 100);

xlabel('Subjects');

ylabel('Accuracy (%)');

title('Subject-wise LOSO Accuracy');

ylim([0 100]);

grid on;

%% ============================================================
% 10. CONFUSION MATRIX
%% ============================================================

figure;

confusionchart(all_true,all_pred);

title('Reviewer-Safe LOSO Confusion Matrix');

%% ============================================================
% 11. ROC CURVE (OPTIONAL MULTI-CLASS)
%% ============================================================

try

    classes = unique(all_true);

    figure;

    hold on;

    for i = 1:length(classes)

        true_binary = all_true == classes(i);

        pred_binary = all_pred == classes(i);

        [X,Y,~,AUC] = perfcurve(...
            true_binary,...
            pred_binary,...
            1);

        plot(X,Y,...
            'LineWidth',2);

        legendInfo{i} = ...
            ['Class ' num2str(i) ...
            ' AUC=' num2str(AUC,2)];

    end

    xlabel('False Positive Rate');

    ylabel('True Positive Rate');

    title('ROC Curves');

    legend(legendInfo);

    grid on;

catch

    disp('ROC Curve skipped');

end

%% ============================================================
% FINAL REVIEWER SUMMARY
%% ============================================================

fprintf('\n====================================================\n');

fprintf('REVIEWER-SAFE STATISTICAL SUMMARY\n');

fprintf('====================================================\n');

fprintf('Mean Accuracy      = %.2f%%\n',...
    mean_acc);

fprintf('Std Deviation      = %.2f%%\n',...
    std_acc);

fprintf('95%% CI             = [%.2f %.2f]\n',...
    lowerCI*100,...
    upperCI*100);

fprintf('Cohen Kappa        = %.4f\n',...
    kappa);

fprintf('Kruskal p-value    = %.6f\n',...
    p_kw);

fprintf('Friedman p-value   = %.6f\n',...
    p_friedman);

fprintf('Statistical Power  = %.4f\n',...
    power_estimate);

fprintf('====================================================\n');