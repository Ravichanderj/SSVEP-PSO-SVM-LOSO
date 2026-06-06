%% ============================================
% FIGURE A : RAW EEG SIGNALS
%% ============================================

figure('Color','w',...
    'Position',[100 100 900 500]);

hold on;

for i = 1:length(folders)

    currentFolder = folders{i};

    files = dir(fullfile(currentFolder,'*.csv'));

    filename = fullfile(currentFolder,...
        files(1).name);

    T = readtable(filename);

    raw_data = T.Var1;

    raw_epoch = raw_data(1:window_length);

    t = (0:length(raw_epoch)-1)/Fs;

    plot(t,...
        raw_epoch,...
        'LineWidth',1.5);

end

xlabel('Time (s)',...
    'FontSize',12,...
    'FontWeight','bold');

ylabel('Amplitude (\muV)',...
    'FontSize',12,...
    'FontWeight','bold');

title('Raw EEG Signals Recorded at Different SSVEP Frequencies',...
    'FontSize',14,...
    'FontWeight','bold');

legend(classNames,...
    'Location','best');

grid on;
box on;

set(gca,...
    'FontSize',11,...
    'LineWidth',1.2);

exportgraphics(gcf,...
    'Figure_A_RawEEG.tif',...
    'Resolution',600);



%% ============================================
% FIGURE B : FILTERED + DENOISED SIGNALS
%% ============================================

figure('Color','w',...
    'Position',[100 100 900 500]);

hold on;

for i = 1:length(folders)

    currentFolder = folders{i};

    files = dir(fullfile(currentFolder,'*.csv'));

    filename = fullfile(currentFolder,...
        files(1).name);

    T = readtable(filename);

    raw_data = T.Var1;

    %% Bandpass Filtering
    filtered_signal = bandpass(raw_data,...
        filter_params(i,:), Fs);

    %% Wavelet Denoising
    cleaned_signal = wdenoise(filtered_signal,...
        5,...
        'Wavelet','db4');

    cleaned_epoch = cleaned_signal(1:window_length);

    t = (0:length(cleaned_epoch)-1)/Fs;

    plot(t,...
        cleaned_epoch,...
        'LineWidth',1.5);

end

xlabel('Time (s)',...
    'FontSize',12,...
    'FontWeight','bold');

ylabel('Amplitude (\muV)',...
    'FontSize',12,...
    'FontWeight','bold');

title('Bandpass-Filtered and Wavelet-Denoised EEG Signals',...
    'FontSize',14,...
    'FontWeight','bold');

legend(classNames,...
    'Location','best');

grid on;
box on;

set(gca,...
    'FontSize',11,...
    'LineWidth',1.2);

exportgraphics(gcf,...
    'Figure_B_FilteredEEG.tif',...
    'Resolution',600);



%% ============================================
% FIGURE C : PSD ANALYSIS
%% ============================================

figure('Color','w',...
    'Position',[100 100 900 500]);

hold on;

for i = 1:length(folders)

    currentFolder = folders{i};

    files = dir(fullfile(currentFolder,'*.csv'));

    filename = fullfile(currentFolder,...
        files(1).name);

    T = readtable(filename);

    raw_data = T.Var1;

    %% Filtering
    filtered_signal = bandpass(raw_data,...
        filter_params(i,:), Fs);

    %% Wavelet Denoising
    cleaned_signal = wdenoise(filtered_signal,...
        5,...
        'Wavelet','db4');

    epoch_signal = cleaned_signal(1:window_length);

    %% Welch PSD
    [psd_values,freq] = pwelch(epoch_signal,...
        hamming(256),...
        128,...
        512,...
        Fs);

    plot(freq,...
        10*log10(psd_values),...
        'LineWidth',1.8);

end

%% Stimulation Frequency Markers
for i = 1:length(stimFreqs)

    xline(stimFreqs(i),...
        '--k',...
        [num2str(stimFreqs(i)) ' Hz'],...
        'LineWidth',1.2);

end

xlabel('Frequency (Hz)',...
    'FontSize',12,...
    'FontWeight','bold');

ylabel('PSD (dB/Hz)',...
    'FontSize',12,...
    'FontWeight','bold');

title('Welch Power Spectral Density of SSVEP Signals',...
    'FontSize',14,...
    'FontWeight','bold');

legend(classNames,...
    'Location','best');

xlim([0 40]);

grid on;
box on;

set(gca,...
    'FontSize',11,...
    'LineWidth',1.2);

exportgraphics(gcf,...
    'Figure_C_PSD.tif',...
    'Resolution',600);



%% ============================================
% FIGURE D : FFT MAGNITUDE SPECTRA
%% ============================================

figure('Color','w',...
    'Position',[100 100 900 500]);

hold on;

for i = 1:length(folders)

    currentFolder = folders{i};

    files = dir(fullfile(currentFolder,'*.csv'));

    filename = fullfile(currentFolder,...
        files(1).name);

    T = readtable(filename);

    raw_data = T.Var1;

    %% Filtering
    filtered_signal = bandpass(raw_data,...
        filter_params(i,:), Fs);

    %% Wavelet Denoising
    cleaned_signal = wdenoise(filtered_signal,...
        5,...
        'Wavelet','db4');

    epoch_signal = cleaned_signal(1:window_length);

    %% FFT
    fft_values = abs(fft(epoch_signal));

    fft_values = ...
        fft_values(1:floor(length(fft_values)/2));

    %% Normalize
    fft_values = fft_values ./ max(fft_values);

    freq_fft = ...
        (0:length(fft_values)-1) ...
        * Fs / length(epoch_signal);

    plot(freq_fft,...
        fft_values,...
        'LineWidth',1.8);

end

%% Frequency Markers
for i = 1:length(stimFreqs)

    xline(stimFreqs(i),...
        '--k',...
        [num2str(stimFreqs(i)) ' Hz'],...
        'LineWidth',1.2);

end

xlabel('Frequency (Hz)',...
    'FontSize',12,...
    'FontWeight','bold');

ylabel('Normalized FFT Magnitude',...
    'FontSize',12,...
    'FontWeight','bold');

title('Normalized FFT Magnitude Spectra of SSVEP Signals',...
    'FontSize',14,...
    'FontWeight','bold');

legend(classNames,...
    'Location','best');

xlim([0 40]);

grid on;
box on;

set(gca,...
    'FontSize',11,...
    'LineWidth',1.2);

exportgraphics(gcf,...
    'Figure_D_FFT.tif',...
    'Resolution',600);