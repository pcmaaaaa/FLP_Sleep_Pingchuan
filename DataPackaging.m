function DataPackaging(rawdat_dir, acq_start, acq_end, animal_name, output_dir, HxInterval, tracked_state, epoch)

acq_dirs = dir([output_dir, 'Acq*']);
delta = [];
theta = [];
SleepStates = [];

for n = 1:size(acq_dirs,1)
    delta_file =  dir([output_dir,acq_dirs(n).name,'/*_delta.npy']);
    theta_file =  dir([output_dir,acq_dirs(n).name,'/*_theta.npy']);
    states_file = dir([output_dir,acq_dirs(n).name,'/StatesAcq*_hr0.npy']);
    temp_delta = readNPY(fullfile(delta_file.folder, delta_file.name));
    temp_theta = readNPY(fullfile(theta_file.folder, theta_file.name));
    temp_states = readNPY(fullfile(states_file.folder, states_file.name));
    delta = [delta temp_delta];
    theta = [theta temp_theta];
    SleepStates = [SleepStates temp_states];
end

delta = reshape(delta, 1, []);
theta = reshape(theta, 1, []);
SleepStates =  reshape(SleepStates, 1, []);

file_string = strcat(string(acq_start),'_', string(acq_end));
filename = strcat(rawdat_dir,sprintf('concat_Acq%s.mat',file_string));
load(filename);
load(strcat(rawdat_dir, 'autonotes.mat'));
global notebook

SS_time = 1:4:max(time_all);
if length(SS_time) > length(SleepStates)
    len_diff = length(SS_time)-length(SleepStates);
end

light_on  = '06:00:00';
light_off  = '18:00:00';
light_on_sec = seconds(duration(light_on));
light_off_sec = seconds(duration(light_off));
t1_date = extractBefore(string(notebook(1))," (");
t1 = seconds(duration(t1_date))-3600;
light_array = zeros(size(time_all));


lighton_dur = light_off_sec-t1;
lighton_idx = find(time_all <= lighton_dur);
light_array(lighton_idx) = 1;
dark_dur = light_off_sec-light_on_sec;
dark_idx = find(time_all > lighton_dur+dark_dur);
light_array(dark_idx)= 1;
% 
% if pe.Status == "NotLoaded"
%     exepath = '/usr/bin/python3';
%     pe = pyenv('Version',exepath);
% end

SleepData.SleepStates = SleepStates;
SleepData.BioData = tau_fit_G_all;
SleepData.BioDataTime = time_all;
SleepData.AnimalName = animal_name;
SleepData.DeltaPower = delta;
SleepData.ThetaPower = theta;
SleepData.PhotonCount = photoncount_all;
SleepData.SleepStateTime = SS_time(1:end-len_diff);
SleepData.Lights = light_array;
SleepData.SleepHistory = SleepHx(SleepStates, HxInterval, tracked_state, epoch);
output_file = strcat(output_dir, 'CombinedSleepData.mat');
save(output_file, 'SleepData');
display('Saved');
end

