function out=PlotEEGEMGFLiP(rawdat_dir, acqn, extracted_dir, frequency, ylims)
% Written by Yao, 4/5/2021
% I ran it under 2021a. Does not work with 2014 or 2012.
% clear the workspace first.
% frequency is the EEG/EMG input frequency, typically 400.


% save ScoringData
% Get your ScoringData this way
ScoringData = [];
ScoringFile=dir(sprintf('%sStatesAcq%d*.npy',extracted_dir,acqn));
for a = 1:length(ScoringFile)
    this_file = sprintf('%s/%s', ScoringFile(a).folder,ScoringFile(a).name);
    data=readNPY(this_file);
    ScoringData = [ScoringData, data'];
end

% Create figure
figure1 = figure;
figure1.Position=[10 10 800 1100];

% load EEG/EMG data, scale them correctly and convert the scale to
% microsecond, and store in ScaledEEGEMG.
ScaledEEGEMG=cell(3,1);
channels = [0 2 3]; %if EEG/EMG channels are 0, 2 and 3
for i=1:3
    EEGfile=sprintf('%sAD%d_%d.mat',rawdat_dir,channels(i),acqn);
    load(EEGfile);
    eval(['ScaledEEGEMG{', int2str(i), ',1}=AD', int2str(channels(i)), '_', int2str(acqn), '.data/5000*10^6;']); %What is a more elegant way to access the data?
end

% load FLiP data
FLiPfile=sprintf('%sAcq%d_analysis.mat',rawdat_dir, acqn);
load(FLiPfile);
TimeRange=max(time); %time range to plot, in seconds, based on FLiP data. We need to specify it because ephys data sometimes exceeded the duration of FliP data.

% Create subplot
subplot1 = subplot(6,1,1,'Parent',figure1);
hold(subplot1,'on');

plot(ScaledEEGEMG{1,1},'Parent',subplot1);
xlim(subplot1,[0 TimeRange*frequency]);

% Create ylabel
ylabel('EEG1 (uV)');

box(subplot1,'on');
hold(subplot1,'off');
% Create subplot
subplot2 = subplot(6,1,2,'Parent',figure1);
hold(subplot2,'on');

% Create plot
plot(ScaledEEGEMG{2,1},'Parent',subplot2);
xlim(subplot2,[0 TimeRange*frequency]);

% Create ylabel
ylabel('EEG2 (uV)');

box(subplot2,'on');
hold(subplot2,'off');
% Create subplot
subplot3 = subplot(6,1,3,'Parent',figure1);
hold(subplot3,'on');

% Create plot
plot(ScaledEEGEMG{3,1},'Parent',subplot3);
xlim(subplot3,[0 TimeRange*frequency]);

% Create ylabel
ylabel('EMG (uV)');

box(subplot3,'on');
hold(subplot3,'off');
% Create subplot
subplot(6,1,4,'Parent',figure1);

% Create heatmap
ScoringData=ScoringData([1:TimeRange/4]);
ScoringHeatMap=heatmap(figure1,ScoringData,'Colormap',[0 1 0;0 0 1;1 0 0;[.5 0 .5]],...
    'GridVisible','off',...
    'FontColor',[0 0 0],...
    'ColorLimits',[1 4],...
    'Title','Scoring (green=wake;blue=NREM;red=REM;purple=Quiet Wake)');
ScoringHeatMap.YDisplayLabels=repmat(' ', 1, 1);
ScoringHeatMap.XDisplayLabels=repmat(' ', length(ScoringData), 1);
    
    
    
% Scoring=heatmap(figure1,ScoringData,'Colormap',[0 1 0;0 0 1;1 0 0],...
% 'GridVisible','off',...
% 'FontColor',[0 0 0],...
% 'ColorLimits',[1 3]);
% Scoring.XDisplayLabels=repmat(' ', 452, 1);
% Scoring.YDisplayLabels=repmat(' ', 1, 1);
% Scoring.Title='Scoring (green=wake;blue=NREM;red=REM)'

% Create subplot
time = time(~isnan(time));
photoncount = photoncount(~isnan(photoncount));

subplot4 = subplot(6,1,5,'Parent',figure1);
hold(subplot4,'on');

% Create plot
plot(time,tau_fit_G,'Parent', subplot4);
% scatter(time(1973:3430),tau(1973:3430), 10, 'filled','b');
xlim(subplot4,[0 max(time)]);

% Create ylabel
ylabel('lifetime (ns)');

% Create xlabel
xlabel(' ');
ylim(ylims);

% Create title
title({'Fluorescence lifetime'});

% Uncomment the following line to preserve the Y-limits of the axes
% ylim(subplot4,[1.77 1.81]);
box(subplot4,'on');
hold(subplot4,'off');
% Create subplot
subplot5 = subplot(6,1,6,'Parent',figure1);
hold(subplot5,'on');

% Create plot
plot(time,photoncount,'Parent', subplot5);
xlim(subplot5,[0 max(time)]);

% Create xlabel
xlabel('time (sec)');

% Create title
title('Photon counts');


% Uncomment the following line to preserve the Y-limits of the axes
% ylim(subplot5,[190000 240000]);
box(subplot5,'on');
hold(subplot5,'off');
% Create textbox
OverallTitle=sprintf('%s, acqn%d',rawdat_dir, acqn);
annotation(figure1,'textbox',...
    [0.282608695652174 0.94695127409373 0.386287614404159 0.0299003316648511],...
    'String',OverallTitle,...
    'LineStyle','none',...
    'FontWeight','bold');
% Insert title for the whole figure;

%save everything
figurefile=sprintf('%s/FLiPEEGEMGStatefig_acqn%d', extracted_dir, acqn);
savefig(figure1, figurefile);
saveas(figure1,strcat(figurefile, '.png'))
Summaryfile=sprintf('%s/FLiPEEGEMGState_acqn%d_summary', extracted_dir, acqn);
save(Summaryfile, 'ScaledEEGEMG', 'ScoringData', 'ScoringHeatMap');


%Next figure
figure2 = figure;
figure2.Position=[10 10 1200 700];

% Create subplot
subplot1 = subplot(3,1,1,'Parent',figure2);
set(subplot1, 'Position', [0.1300 0.8093 0.7750 0.1157])
% Create heatmap
ScoringData=ScoringData([1:TimeRange/4]);
ScoringHeatMap=heatmap(figure2,ScoringData,'Colormap',[0 1 0;0 0 1;1 0 0;[.5 0 .5]],...
    'GridVisible','off',...
    'FontColor',[0 0 0],...
    'ColorLimits',[1 4],...
    'Title','Scoring (green=wake;blue=NREM;red=REM;purple=Quiet Wake)');
ScoringHeatMap.YDisplayLabels=repmat(' ', 1, 1);
ScoringHeatMap.XDisplayLabels=repmat(' ', length(ScoringData), 1);
    
    
    
% Scoring=heatmap(figure1,ScoringData,'Colormap',[0 1 0;0 0 1;1 0 0],...
% 'GridVisible','off',...
% 'FontColor',[0 0 0],...
% 'ColorLimits',[1 3]);
% Scoring.XDisplayLabels=repmat(' ', 452, 1);
% Scoring.YDisplayLabels=repmat(' ', 1, 1);
% Scoring.Title='Scoring (green=wake;blue=NREM;red=REM)'

% Create subplot
time = time(~isnan(time));
photoncount = photoncount(~isnan(photoncount));

subplot2 = subplot(3,1,2,'Parent',figure2);
hold(subplot2,'on');
set(subplot2, 'Position', [0.1300 0.3593 0.7750 0.4157])


% Create plot
plot(time,tau_fit_G,'Parent', subplot2);
% scatter(time(1973:3430),tau(1973:3430), 10, 'filled','b');
xlim(subplot2,[0 max(time)]);

% Create ylabel
ylabel('lifetime (ns)');

% Create xlabel
xlabel(' ');
ylim(ylims);

% Create title
title({'Fluorescence lifetime'});

% Uncomment the following line to preserve the Y-limits of the axes
% ylim(subplot4,[1.77 1.81]);
box(subplot2,'on');
hold(subplot2,'off');
% Create subplot
subplot3 = subplot(3,1,3,'Parent',figure2);
hold(subplot3,'on');


% Create plot
plot(time,photoncount,'Parent', subplot3);
xlim(subplot3, [0 max(time)]);

% Create xlabel
xlabel('time (sec)');

% Create title
title('Photon counts');


% Uncomment the following line to preserve the Y-limits of the axes
% ylim(subplot5,[190000 240000]);
box(subplot3,'on');
hold(subplot3,'off');
% Create textbox
OverallTitle=sprintf('%s, acqn%d',rawdat_dir, acqn);
annotation(figure2,'textbox',...
    [0.282608695652174 0.94695127409373 0.386287614404159 0.0299003316648511],...
    'String',OverallTitle,...
    'LineStyle','none',...
    'FontWeight','bold');
% Insert title for the whole figure;

%save everything
figurefile=sprintf('%s/FLiPEEGEMGStatefig_trunc_acqn%d', extracted_dir, acqn);
savefig(figure2, figurefile);
saveas(figure2,strcat(figurefile, '.png'))

close all
% Summaryfile=sprintf('%s/FLiPEEGEMGState_acqn%d_summary', extracted_dir, acqn);
% save(Summaryfile, 'ScaledEEGEMG', 'ScoringData', 'ScoringHeatMap');
