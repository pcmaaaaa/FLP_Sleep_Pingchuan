import numpy as np
import glob
import os
from scipy import io, signal
from datetime import datetime
from statsmodels.nonparametric.kernel_regression import KernelReg
import FLP_Sleep_Pingchuan as PKA
import pandas as pd
import math
from neuroscience_sleep_scoring import SWS_utils
from matplotlib import mlab
from scipy.integrate import simps


#This is a library of common analysis tooks needed for most FLiP data analysis

def load_data(rawdatdir, data_filename):
	data_dict = {}
	io.loadmat(data_filename, mdict=data_dict)
	del data_dict['__header__']
	del data_dict['__version__']
	del data_dict['__globals__']
	these_keys = list(data_dict.keys())
	try:
		these_keys.remove('histograms')
	except ValueError:
		pass
	try:
		these_keys.remove('fits')
	except ValueError:
		pass

	for k in  these_keys:
		data_dict[k] = data_dict[k].flatten()
	delete_these = []
	try:
		for i,h in enumerate(data_dict['histograms']):
			if all(h == 0):
				delete_these.append(i)
	except KeyError:
		pass
	try:
		notebook_filename = os.path.join(rawdatdir, 'autonotes.mat')
		notebook = {}
		io.loadmat(notebook_filename, mdict=notebook)
		ts_format = '%H:%M:%S.%f'
		if len(notebook['notebook']) == 0:
			data_dict['timestamps'] = [[]]
		else:
			data_dict['timestamps'] = [notebook['notebook'][0][x][0][0:8] for x in np.arange(np.size(notebook['notebook'][0]))]
	except FileNotFoundError:
		data_dict['timestamps'] = [[]]

	return data_dict

def do_npreg(data_dict, bw = 20):
	i = list(filter(lambda x: 'tau_fit_G' in x, list(data_dict.keys())))
	data = data_dict[i[0]]
	n = np.size(data)
	kde = KernelReg(endog=data, exog=np.arange(n), var_type='c', bw=[bw])
	estimator = kde.fit(np.arange(n))
	estimator = np.reshape(estimator[0],n)
	data_dict['npreg'] = estimator
	return data_dict

def get_all_states(rawdatdir, filename, one_acq):
	if one_acq:
		acq_str = filename[filename.find('q')+1:filename.find('_analysis')]
		statefile = glob.glob(os.path.join(rawdatdir, '*extracted_data*', 'StatesAcq'+acq_str+'_hr0.npy'))[0]
		return np.load(statefile)
	state_files = glob.glob(os.path.join(rawdatdir, '*extracted_data*', 'StatesAcq*_hr0.npy'))
	acqs = [int(state_files[i][state_files[i].find('q')+1:state_files[i].find('_hr0')]) for i in range(len(state_files))]
	acqs.sort()
	all_states = np.array([0])
	for a in acqs:
		this_file = glob.glob(os.path.join(rawdatdir,'*extracted_data*', 'StatesAcq' + str(a) + '_hr0.npy' ))[0]
		these_states = np.load(this_file)
		all_states = np.concatenate((all_states, these_states))
	all_states = np.delete(all_states, 0)
	return all_states

def get_all_EEG(extract_dir):
	all_EEG = []
	EEG_files = glob.glob(os.path.join(extract_dir,'downsampEEG_*hr0.npy'))
	EEG_files.sort(key=lambda f: os.path.getctime(os.path.join(extract_dir, f)))
	for f in EEG_files:
		all_EEG.append(np.load(f))
	EEG = np.concatenate(all_EEG)
	return EEG

def get_zeit_time(data_dict):
	ts_format = '%H:%M:%S'
	lights_off = "18:00:00"
	lights_on = "06:00:00"
	if data_dict['timestamps'][0] == []:
		print('No Zeitgeiber Time calculated')
		data_dict['Zeit Time'] = []
		return data_dict
	t2 = datetime.strptime(data_dict['timestamps'][0], ts_format)
	lon_seconds = datetime.strptime(lights_on, ts_format)
	zeit_offset = (t2-lon_seconds).total_seconds()-3600
	data_dict['Zeit Time'] = (data_dict['time_all']+zeit_offset)/3600
	over24_idx, = np.where(data_dict['Zeit Time']>24)
	data_dict['Zeit Time'][over24_idx] = data_dict['Zeit Time'][over24_idx]-24
	return data_dict


def find_continuous(arr, this_state):
	if len(this_state) == 1:
		these_bins, = np.where(np.logical_or(arr == this_state, np.isnan(arr)))
	elif len(this_state) == 2:
		these_bins, = np.where(np.logical_or(arr == this_state[0], arr == this_state[1], np.isnan(arr)))
	else:
		print('I cannot handle looking for more than 2 states at the moment')
	cont_idx = []
	if np.size(these_bins)>0:
		temp = [these_bins[0]]
	else:
		# cont_idx.append([])
		return cont_idx
	for b in these_bins[1:]:
		if (np.size(temp) == 0) or (b == temp[-1]+1):
			temp.append(b)
		else:
			cont_idx.append(temp)
			temp = [b]
	cont_idx.append(temp)
	return cont_idx

def load_full_ex(rawdatdir, epoch_len = 4, binwidth = 2, filter_bounds = [0.003, 0.1], binned = False):
	ex_name = os.path.split(rawdatdir)[1]
	if binned:
		acq_files = glob.glob(os.path.join(rawdatdir, 'Acq*_analysis_binned.mat'))
	else:
		acq_files = glob.glob(os.path.join(rawdatdir, 'Acq*_analysis.mat'))
	acqs = [int(acq_files[i][acq_files[i].find('q')+1:acq_files[i].find('_analysis')]) for i in range(len(acq_files))]
	acqs.sort()
	if binned:
		experiment = [PKA.FLiPExperiment(os.path.join(rawdatdir, 'Acq'+str(i)+'_analysis_binned.mat'), one_acq=True, 
			epoch_len = epoch_len, binwidth = binwidth, fs =0.25, filter_bounds = filter_bounds) for i in acqs]
	else:
		experiment = [PKA.FLiPExperiment(os.path.join(rawdatdir, 'Acq'+str(i)+'_analysis.mat'), 
			one_acq=True, epoch_len = epoch_len, binwidth = binwidth, fs =1, filter_bounds = filter_bounds) for i in acqs]		
	return experiment, ex_name, acqs


def pull_experiment_names(filename = 
	'/Users/lizzie/Library/CloudStorage/Box-Box/ChenLab/Lizzie/FLiP_Experiment_Summary.xlsx'):
	
	#Filter by experiment type
	experiment_df = pd.read_excel(filename)
	exp_type = input('Do you want experiments without EEG/EMG? (y/n)') == 'n'
	if exp_type:
		focused_df = experiment_df.loc[experiment_df['Experiment Name'].str.contains("EEGEMG", case=False)]
	else:
		focused_df = experiment_df

	#Filter by genotype
	genotypes = list(focused_df['Genotype'].unique())
	print('Here are the available genotypes:')
	print(genotypes)
	genotype_str = input('Please list the genotypes you want, seperating each genotype with a comma. If you want all, type "all"')
	genotype_str = genotype_str.replace('+', '\+')
	genotype_str = genotype_str.replace(',','|')
	if genotype_str != 'all':
		focused_df = focused_df.loc[focused_df['Genotype'].astype(str).str.contains(genotype_str, case=False)]

	#Filter by implant location
	implant_loc = list(focused_df['Implant Location'].unique())
	print('Here are the available implant locations:')
	print(implant_loc)
	loc_str = input('Please list the locations you want, seperating each location with a comma. If you want all, type "all"')
	loc_str = loc_str.replace(',','|')
	if loc_str != 'all':
		focused_df = focused_df.loc[focused_df['Implant Location'].astype(str).str.contains(loc_str, case=False)]

	#Filter by sex
	sex = input('What sex do you want? (M/F/both)') != 'both'
	if sex:
		focused_df = focused_df.loc[focused_df['Sex'].astype(str).str.contains(sex, case=False)]

	#Filter by the length of recording
	recording_len_min = int(input('What is the minimum number of acquisitions you want?'))
	recording_len_max = int(input('What is the maximum number of acquisitions you want?'))
	focused_df = focused_df.loc[focused_df['# Acquisitions']>recording_len_min]
	focused_df = focused_df.loc[focused_df['# Acquisitions']<recording_len_max]

	#Filter by animal name
	animals = list(focused_df['Mouse ID'].unique())
	print('Here are the available animals:')
	print(animals)
	animal_ID = input('Please list the animals you want to EXCLUDE, seperating each animal with a comma. If you want all, type "all"')
	animal_ID = animal_ID.replace(',','|')
	if animal_ID != 'all':
		focused_df = focused_df.drop(focused_df.loc[focused_df['Mouse ID'].astype(str).str.contains(animal_ID, case=False)].index)

	#Filter by sleep dep
	sleep_dep = input('Do you want experiments that included sleep deprivation? (y/n)') == 'n'
	if sleep_dep:
		focused_df = focused_df.loc[~focused_df['Comments'].astype(str).str.contains('sleep dep', case=False)]
	#Filter out any where the photometry is not done
	focused_df = focused_df.loc[focused_df['Photometry Analysis'] == 'Yes']

	#Filter out any where the sleep scoring isn't done
	focused_df = focused_df.loc[focused_df['# Acquisitions'] == focused_df['# Acq Sleep Scored']]

	print(focused_df[['Experiment Name', 'Comments']])

	#Filter out any experiments that you don't want
	exclude_experiment = input('Please list the experiments you want to EXCLUDE, seperating each experiments with a comma. If you want all, type "all"')
	exclude_experiment = exclude_experiment.replace(',','|')
	if exclude_experiment != 'all':
		focused_df = focused_df.drop(focused_df.loc[focused_df['Experiment Name'].astype(str).str.contains(exclude_experiment, case=False)].index)

	print(focused_df[['Experiment Name', 'Comments']])

	return focused_df

def get_sleep_stats(sleep_states, epoch_len):
	sleep_stats = {}
	tot_epochs = np.size(sleep_states)
	tot_time = tot_epochs*epoch_len
	state_names = ['Active Wake', 'NREM', 'REM', 'Quiet Wake']
	sleep_stats['Total Time'] = tot_time
	sleep_stats['Total Num Epochs'] = tot_epochs
	for i,n in enumerate(state_names):
		num_epochs = np.size(np.where(sleep_states == i+1))
		bouts = find_continuous(sleep_states, [i+1])
		sleep_stats['Number '+ n + ' Epochs'] = num_epochs
		sleep_stats['Percentage '+ n] = num_epochs/tot_epochs
		sleep_stats['Time in ' + n] = num_epochs*epoch_len
		sleep_stats['Number '+ n + ' Bouts'] = len(bouts)
	return sleep_stats

def ss_onset_offset(ss, epoch_len):
	state_time =  np.arange(0,np.size(ss)*epoch_len,epoch_len)
	state_id = []
	onset_times = []
	offset_times  = []
	ss[ss == 4] = 1
	for i, s in enumerate(ss):
		if i == 0:
			curr_state = s
			state_id.append(s)
			onset_times.append(state_time[i])
		elif i == np.size(ss)-1:
			offset_times.append((state_time[i]+epoch_len))
		elif s == curr_state:
			continue
		elif s != curr_state:
			onset_times.append(state_time[i])
			offset_times.append(state_time[i])
			curr_state = s
			state_id.append(s)
		else:
			print('Not sure what the other option would be?')
				
	df = pd.DataFrame(columns = ['State', 'Start Time', 'End Time'])
	df['State'] = state_id
	df['Start Time'] = onset_times
	df['End Time'] = offset_times
	return df

def filt_lifetime(lifetime, fs = 1, filt_low = None, filt_high = None, N = 3):
	nyq = fs*0.5
	if (filt_low is not None) & (filt_high is not None):
		print('Using a bandpass')
		Wn = [filt_low/nyq,filt_high/nyq] # Cutoff frequencies
		B, A = signal.butter(N, Wn, btype='bandpass',output='ba')
	elif (filt_low is not None) & (filt_high is None):
		print('Using a highpass')
		Wn = filt_low/nyq
		B, A = signal.butter(N, Wn, btype='highpass',output='ba')
	elif (filt_low is None) & (filt_high is not None):
		print('Using a lowpass')
		Wn = filt_high/nyq
		B, A = signal.butter(N, Wn, btype='lowpass',output='ba')
	else:
		print('You didnt put any filter values')
		lifetime_filt = lifetime
		return lifetime_filt

	lifetime_filt = signal.filtfilt(B,A, lifetime)
	return lifetime_filt

def transient_threshold(filtered_liftime, num_std, discrete_cutoff = False):
	s = np.std(filtered_liftime)
	m = np.mean(filtered_liftime)
	if num_std:
		thresh = s*num_std
	else:
		thresh = discrete_cutoff
	return m, thresh, s

def transient_amplitudes(lifetime, transient_idxs):
	amplitudes = []
	start_vals = []
	troughs = []
	for t in transient_idxs:
		this_transient = lifetime[t]
		start_val = (lifetime[t[0]]+lifetime[t[0]-1])/2
		trough = np.min(this_transient)
		start_vals.append(start_val)
		troughs.append(trough)
		amplitudes.append(start_val-trough)
	return amplitudes, start_vals, troughs

def transient_FWHM(lifetime, time, transient_idxs):
	FWHM = []
	time_points = []
	for i, t in enumerate(transient_idxs):
		this_transient = lifetime[t]
		half_max = (this_transient[0]+(np.min(this_transient)))/2
		ii = 0
		while this_transient[ii]>half_max:
			ii += 1
		
		ee = len(t)-1
		while this_transient[ee]>half_max:
			ee -= 1

		t1 = t[ii]
		t2 = t[ee]

		FWHM.append(time[t2]-time[t1])
		time_points.append([t1,t2])
	return FWHM,time_points

def transient_window(filtered_liftime, idx, m):
	#look backwards for crossing mean
	i = idx[0]
	while filtered_liftime[i]<m:
		idx.insert(0, i)
		i = i-1
		if i < 1:
			break
	idx.insert(0, i)
	#look forward for crossing mean
	i = idx[-1]
	while filtered_liftime[i]<m:
		idx.append(i)
		i = i+1
		if i >= np.size(filtered_liftime)-1:
			break
	idx.insert(0, i)
	return list(np.unique(idx))

def get_these_ss(ss_idx, sleep_states, time, epoch_len):
	'''This function takes a list of photometry indexes and returns the corresponding sleep states 
	and sleep state indices'''
	these_states = []
	these_states_idx = []
	state_time =  np.arange(0, np.size(sleep_states)*epoch_len, epoch_len)
	time_seg = time[ss_idx] #Indexes correspond to photometry data
	for tt in np.arange(np.size(time_seg)-1):
		this_state_idx, = np.where(np.logical_and(state_time >= time_seg[tt], 
			state_time < time_seg[tt+1])) #Indexes correspond to sleep states data
		this_state = sleep_states[this_state_idx] 
		these_states.append(this_state)
		these_states_idx.append(this_state_idx) #Indexes correspond to sleep states data
	these_states = np.concatenate(these_states)
	these_states_idx = np.concatenate(these_states_idx) #Indexes correspond to sleep states data
	return these_states, these_states_idx #Indexes correspond to sleep states data

def find_ss_transition(this_state, this_state_idx, sleep_states):
	'''This function starts at a given state and moves backwards in time until it finds the previous state'''
	sleep_states[sleep_states == 4] = 1
	s = this_state
	i = this_state_idx #Indexes correspond to sleep states data
	while s == this_state:
		i = i - 1 #Indexes correspond to sleep states data
		s = sleep_states[i]
	return i, s

def number_per_sstype(sleep_states):
	data_dict = {}
	for this_state in [[1,4],[2],[3]]:
		cont_idx = PKA.find_continuous(sleep_states, this_state)
		end_idxs = [c[-1] for c in cont_idx]
		if end_idxs[-1] == np.size(sleep_states)-1:
			end_idxs = end_idxs[:-1]
		next_ss = np.asarray([sleep_states[e+1] for e in end_idxs])
		if this_state == [1,4]:
			data_dict['Wake-NREM'] = np.size(np.where(next_ss == 2))
			data_dict['Wake-REM'] = np.size(np.where(next_ss == 3))
		if this_state == [2]:
			data_dict['NREM-REM'] = np.size(np.where(next_ss == 3))
			data_dict['NREM-Wake'] = np.size(np.where(next_ss == 1)) + np.size(np.where(next_ss == 4))
		if this_state == [3]:
			data_dict['REM-NREM'] = np.size(np.where(next_ss == 2))
			data_dict['REM-Wake'] = np.size(np.where(next_ss == 1)) + np.size(np.where(next_ss == 4))
	return data_dict

def transient_property_per_sstype(transition_types, values, labels):
	data_dict = {}
	for i in labels:
		data_dict[i] = []
	for ii in np.arange(0, np.shape(transition_types)[0]):
		if (transition_types[ii] == [2,1]).all():
			data_dict['NREM-Wake'].append(values[ii])
		elif (transition_types[ii] == [3,1]).all():
			data_dict['REM-Wake'].append(values[ii])
		elif (transition_types[ii] == [2,3]).all():
			data_dict['NREM-REM'].append(values[ii]) 
		elif (transition_types[ii] == [1,3]).all():
			data_dict['Wake-REM'].append(values[ii])
		elif (transition_types[ii] == [3,2]).all():
			data_dict['REM-NREM'].append(values[ii])
		elif (transition_types[ii] == [1,2]).all():
			data_dict['Wake-NREM'].append(values[ii])
		else:
			data_dict['Unknown'].append(values[ii])

	return data_dict

def transient_associated_ss(transient_starts, num_seconds, exp):
	'''This function retrieves the sleep states around every transient with a window of +/-num_seconds'''
	time_start = [exp.Time[i] for i in transient_starts]
	ss_bins = np.zeros([int(len(transient_starts)), int((num_seconds*2)/exp.EpochLength)])
	x = np.arange(0, np.shape(ss_bins)[1])
	for i,tt in enumerate(time_start):
		window_idx, = np.where(np.logical_and(exp.Time > tt-num_seconds, exp.Time <= tt+num_seconds))
		these_states, these_states_idx = get_these_ss(window_idx, exp.SleepStates, exp.Time, exp.EpochLength)
		if np.size(these_states) == 0:
			continue
		ss_bins[i,:np.size(these_states)] = these_states

	return ss_bins

def shuffle_photometry(lifetime, window = 400):
	reshape_dimension = math.floor(len(lifetime)/window)
	remainder = len(lifetime)%window
	lifetime_reshape = np.reshape(lifetime[:-remainder], [int(reshape_dimension), -1])
	shuffling_order = np.arange(0, reshape_dimension)
	np.random.shuffle(shuffling_order)
	shuffled_lifetime = []
	for i in shuffling_order:
		shuffled_lifetime.append(lifetime_reshape[int(i)])
	shuffled_lifetime.append(lifetime[-remainder:])
	shuffled_lifetime = np.concatenate(shuffled_lifetime)
	
	return shuffled_lifetime

def get_epochs(sleep_states, diff_wake = False):
	epoch_dict = { 'NREM': find_continuous(sleep_states, [2]), 
	'REM':find_continuous(sleep_states, [3]),
	'Microarousals':find_continuous(sleep_states, [5])}
	if diff_wake:
		epoch_dict['Active Wake'] = find_continuous(sleep_states, [1])
		epoch_dict['Quiet Wake'] = find_continuous(sleep_states, [4])
	else:
		sleep_states[np.where(sleep_states == 4)[0]] = 1
		epoch_dict['Wake'] = find_continuous(sleep_states, [1])
	return epoch_dict


def get_previous_sleep_epoch(wake_epochs, sleep_states, scoring_epoch):
	sleep_epochs = find_continuous(sleep_states, [2,3])
	wake_starts = [w[0] for w in wake_epochs]
	sleep_ends = [s[-1] for s in sleep_epochs]
	sleep_lengths = []
	for i in wake_starts:
		if i >0:
			idx = sleep_ends.index(i-1)
			sleep_lengths.append(len(sleep_epochs[idx])*scoring_epoch)
		else:
			sleep_lengths.append(np.nan)
	return sleep_lengths

def transient_frequency_by_transition(exp, transient_dict, indices, microarousals = True):
	ss_time = np.linspace(0, exp.Time[-1], np.size(exp.SleepStates))
	time = exp.Time[indices]
	ss_idx, = np.where(np.logical_and(ss_time>=time[0], ss_time<time[-1]))
	sleep_states = exp.SleepStates[ss_idx]
	num_transitions = number_of_transitions(sleep_states, exp.EpochLength, microarousals = microarousals)
	transition_types, distance_from_transition = exp.ss_transition_per_transient(transient_dict, buffer_epochs = 2,
		microarousals = microarousals)

	transient_starts = [t[0] for t in transient_dict['Transient Idx']]
	vals, comm1, these_transients = np.intersect1d(indices, transient_starts, return_indices = True)

	state_labels = {'Wake':1, 'NREM':2, 'REM':3, 'Microarousal':5}
	tt_indices = {}
	frequency_dict = {}

	# for k in ['NREM-Wake', 'REM-Wake', 'NREM-REM', 'Wake-NREM', 'Sleep-Wake', 'Wake-Sleep', 'Microarousal', 'REM-NREM']:
	# 	tt_indices[k] = []
	for t in ['NREM-Wake', 'REM-Wake', 'NREM-REM', 'Wake-NREM', 'Microarousal', 'REM-NREM']:
		if t == 'Microarousal':
			tt_indices['Microarousal'] = np.where([any(x == 5) for x in transition_types[these_transients]])[0]
		else:
			num_code = [state_labels[t[:int(t.find('-'))]], state_labels[t[int(t.find('-'))+1:]]]
			tt_indices[t] = np.where([np.array_equal(x, num_code) for x in transition_types[these_transients]])[0]
			if num_transitions[t]>0:
				frequency_dict[t] = np.size(tt_indices[t])/num_transitions[t]
			else:
				frequency_dict[t] = 0
	frequency_dict['Sleep-Wake'] = (np.size(tt_indices['NREM-Wake'])+np.size(tt_indices['REM-Wake']))/(num_transitions['NREM-Wake']+num_transitions['REM-Wake'])
	frequency_dict['Wake-Sleep'] = frequency_dict['Wake-NREM']
	frequency_dict['Microarousal'] =  np.size(tt_indices['Microarousal'])/num_transitions['Microarousals']
	tt_indices['Sleep-Wake'] = np.concatenate([tt_indices['NREM-Wake'], tt_indices['REM-Wake']])
	tt_indices['Wake-Sleep'] = tt_indices['Wake-NREM']

	return frequency_dict, num_transitions

def find_behavior_length(epoch_dict, state, starttime, sleep_states, sleep_state_time, epoch_len):
	state_starts = [sleep_state_time[s[0]] for s in epoch_dict[state]]
	state_ends = [sleep_state_time[s[-1]] for s in epoch_dict[state]]
	epoch_idx = state_starts.index(starttime)
	behavior_len = state_ends[epoch_idx]-state_starts[epoch_idx]

	return (behavior_len, state_ends[epoch_idx])

def determine_transient_notransient(epoch_dict, transient_dict, exp):
	outcome_list = []
	for i, e in enumerate(epoch_dict['Wake']):
		transient_tracker = []
		for ti,t in enumerate(transient_dict['Transient Idx']):
			t_time = exp.Time[t]
			temp = [x for x in t_time if exp.SSTime[e[0]] <= x <= exp.SSTime[e[-1]]]
			if len(temp) == 0:
				transient_tracker.append(False)
			else:
				transient_tracker.append(True)
		if not any(transient_tracker):
			outcome_list.append(None)
		else:
			outcome_list.append(int(np.where(transient_tracker)[0][0]))
	transient_idx = [outcome_list.index(x) for x in outcome_list if isinstance(x, int)]
	no_transient_idx, = np.where(np.array(outcome_list) == None)

	return no_transient_idx, transient_idx

def get_all_powers(epoch_dict, fsd, freq_dict, exp, b,  NFFT_sec = 10, scale_by_freq = True, norm = True):
	all_powers_dict = {'Wake':[], 'NREM':[], 'REM':[], 'Microarousals':[]}
	NFFT = int(fsd*NFFT_sec)
	noverlap = int((NFFT_sec-0.1)*fsd)
	for state in list(epoch_dict.keys()):
		for w in epoch_dict[state]:
			this_power_dict = {}
			time_window = [exp.SSTime[w[0]], exp.SSTime[w[-1]]]
			eeg_seg = SWS_utils.get_eeg_segment(b, time_window)
			Pxx, freqs = mlab.psd(eeg_seg, Fs=fsd, scale_by_freq = scale_by_freq, 
				NFFT = NFFT, noverlap = noverlap)
			freq_res = freqs[1] - freqs[0]
			this_power_dict['Total Power'] = simps(Pxx, dx=freq_res, axis = 0)
			for f in list(freq_dict.keys()):
				freqIdx, = np.where(np.logical_and(freqs>=freq_dict[f][0],freqs<=freq_dict[f][1]))
				if norm:
					this_power_dict[f] = simps(Pxx[freqIdx], dx=freq_res, axis = 0)/this_power_dict['Total Power']
				else:
					this_power_dict[f] = simps(Pxx[freqIdx], dx=freq_res, axis = 0)
			all_powers_dict[state].append(this_power_dict)

	return all_powers_dict








