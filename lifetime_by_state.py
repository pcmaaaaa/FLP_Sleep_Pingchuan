import numpy as np
import glob
import os
from scipy import io
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
import PKA_Sleep as PKA
from statsmodels.nonparametric.kernel_regression import KernelReg
def lifetime_2_states(rawdatdir, binned = True, photoncounts = True, transition_only = False, transition_state = None, 
	transition_window = None, epoch_len = 4, npreg = False):
	print('Calculating....')
	data_dict = PKA.load_data(rawdatdir, binned=binned)
	data_dict = do_npreg(data_dict)
	state_files = glob.glob(os.path.join(rawdatdir, '*_extracted_data*', 'StatesAcq*_hr0.npy'))
	acqs = [int(state_files[i][state_files[i].find('q')+1:state_files[i].find('_hr0')]) for i in range(len(state_files))]
	acqs.sort()
	all_states = np.array([0])
	for a in acqs:
		this_file = glob.glob(os.path.join(rawdatdir,'*_extracted_data*', 'StatesAcq' + str(a) + '_hr0.npy' ))[0]
		these_states = np.load(this_file)
		all_states = np.concatenate((all_states, these_states))
	all_states = np.delete(all_states, 0)

	lifetime_dict = {}
	photon_dict = {}

	s_idx_dict = {}
	s_idx_dict['Wake'] = find_continuous(all_states, [1,4])
	s_idx_dict['Sleep'] = find_continuous(all_states, [2,3])

	if transition_only:
		get_transitions(transition_state, s_idx_dict, transition_window, epoch_len = epoch_len)

	for s in list(s_idx_dict.keys()):
		lifetime_dict[s] = []
		get_lifetimes(data_dict, lifetime_dict, s_idx_dict, s, all_states = all_states, npreg = npreg)
		if photoncounts:
			photon_dict[s] = []
			get_lifetimes(data_dict, photon_dict, s_idx_dict, s, all_states = all_states, npreg = npreg)
	if photoncounts:
		return data_dict, lifetime_dict, photon_dict
	else:
		return data_dict, lifetime_dict


def lifetime_4_states(rawdatdir, binned = True, photoncounts = True, transition_only = False, transition_state = None, 
	transition_window = None, epoch_len = 4, npreg = False):
	print('Calculating....')
	data_dict = PKA.load_data(rawdatdir, binned=binned)
	state_files = glob.glob(os.path.join(rawdatdir, '*_extracted_data*', 'StatesAcq*_hr0.npy'))
	acqs = [int(state_files[i][state_files[i].find('q')+1:state_files[i].find('_hr0')]) for i in range(len(state_files))]
	acqs.sort()

	all_states = np.array([0])
	for a in acqs:
		this_file = glob.glob(os.path.join(rawdatdir,'*_extracted_data*', 'StatesAcq' + str(a) + '_hr0.npy' ))[0]
		these_states = np.load(this_file)
		all_states = np.concatenate((all_states, these_states))
	all_states = np.delete(all_states, 0)

	lifetime_dict = {}
	photon_dict = {}

	s_idx_dict = {}
	s_idx_dict['Active Wake'] = find_continuous(all_states, [1])
	s_idx_dict['NREM'] = find_continuous(all_states, [2])
	s_idx_dict['REM'] = find_continuous(all_states, [3])
	s_idx_dict['Quiet Wake'] = find_continuous(all_states, [4])

	if transition_only:
		get_transitions(transition_state, s_idx_dict, transition_window, epoch_len = epoch_len)
	
	for s in list(s_idx_dict.keys()):
		lifetime_dict[s] = []
		get_lifetimes(data_dict, lifetime_dict, s_idx_dict, s, all_states = all_states, npreg = npreg)
		if photoncounts:
			photon_dict[s] = []
			get_lifetimes(data_dict, photon_dict, s_idx_dict, s, all_states = all_states, npreg = npreg)
	if photoncounts:
		return data_dict, lifetime_dict, photon_dict
	else:
		return data_dict, lifetime_dict

def get_stats(this_dict):
	means = {}
	sem = {}
	stacked = {}
	for s in list(this_dict.keys()):
		msize = (np.max([len(i) for i in this_dict[s]]))
		adjust_arr_len(this_dict, stacked, s, msize)
		means[s] = np.nanmean(stacked[s], axis=0)
		sem[s] = np.nanstd(stacked[s],axis=0)
	
	return means, sem, data_dict, this_dict

def get_transitions(state, s_idx_dict, window, epoch_len = 4):
	key_str = state + ' Transition'
	s_idx_dict[key_str] = []
	for s in s_idx_dict[state]:
		if len(s) > 0:
			before = int(window[0]/epoch_len)
			after =int(window[1]/epoch_len)
			temp = np.arange(s[0]+before, s[0]+after)
			s_idx_dict[key_str].append(temp)
		else:
			pass

def start_end_comp(rawdatdir, bin_size, binned = True, epoch_len = 4, num_states = 4):
	data_dict = PKA.load_data(rawdatdir, binned=binned)
	state_files = glob.glob(os.path.join(rawdatdir, '*_extracted_data*', 'StatesAcq*_hr0.npy'))
	acqs = [int(state_files[i][state_files[i].find('q')+1:state_files[i].find('_hr0')]) for i in range(len(state_files))]
	acqs.sort()

	all_states = np.array([0])
	for a in acqs:
		this_file = glob.glob(os.path.join(rawdatdir,'*_extracted_data*', 'StatesAcq' + str(a) + '_hr0.npy' ))[0]
		these_states = np.load(this_file)
		all_states = np.concatenate((all_states, these_states))
	all_states = np.delete(all_states, 0)
	state_time =  np.arange(0,np.size(all_states)*epoch_len,epoch_len)

	lifetime_dict = {}
	photon_dict = {}
	s_idx_dict = {}
	if num_states == 4:
		s_idx_dict['Active Wake'] = find_continuous(all_states, [1])
		s_idx_dict['NREM'] = find_continuous(all_states, [2])
		s_idx_dict['REM'] = find_continuous(all_states, [3])
		s_idx_dict['Quiet Wake'] = find_continuous(all_states, [4])
	if num_states == 2:
		s_idx_dict['Wake'] = find_continuous(all_states, [1,4])
		s_idx_dict['Sleep'] = find_continuous(all_states, [2,3])

	for s in list(s_idx_dict.keys()):
		lifetime_dict[s] = []
		for ss in s_idx_dict[s]:
			if len(ss) > (bin_size/epoch_len)*2:
				window1 = [state_time[ss[0]]-bin_size, state_time[ss[0]]]
				window2 = [state_time[ss[-1]]-bin_size, state_time[ss[-1]]]
				lifetime_idx1, = np.where(np.logical_and(data_dict['time_all']>=window1[0], data_dict['time_all']<window1[1]))
				lifetime_idx2, = np.where(np.logical_and(data_dict['time_all']>=window2[0], data_dict['time_all']<window2[1]))
				lifetime_dict[s].append([np.mean(data_dict['tau_fit_G_all'][lifetime_idx1]), np.mean(data_dict['tau_fit_G_all'][lifetime_idx2])])
			else:
				pass
	return lifetime_dict


def adjust_arr_len(lifetime_dict, lifetime_stacked, state, msize):
	temp = []
	for t in lifetime_dict[state]:
		nansize = msize - np.size(t)
		nans = np.full(nansize, np.nan)
		corrected = np.append(t, nans)
		temp.append(corrected)
	lifetime_stacked[state] = np.vstack(temp)

def plot_lifetime_4_state(lifetime_means, lifetime_sem, all_time):
	time_NREM = all_time[0:np.size(lifetime_means['NREM'])]
	time_REM = all_time[0:np.size(lifetime_means['REM'])]
	time_AWake = all_time[0:np.size(lifetime_means['Active Wake'])]
	time_QWake = all_time[0:np.size(lifetime_means['Quiet Wake'])]

	plt.ion()
	fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = [15,5])

	ax.plot(time_NREM, lifetime_means['NREM'], color = '#0339f8')
	ax.fill_between(time_NREM, lifetime_means['NREM'], lifetime_means['NREM']+lifetime_sem['NREM'],
		color = '#0339f8',alpha=0.5, edgecolor = None)
	ax.fill_between(time_NREM, lifetime_means['NREM'], lifetime_means['NREM']-lifetime_sem['NREM'],
		color = '#0339f8',alpha=0.5, edgecolor = None)

	ax.plot(time_REM, lifetime_means['REM'], color = '#e50000')
	ax.fill_between(time_REM, lifetime_means['REM'], lifetime_means['REM']+lifetime_sem['REM'],
		color = '#e50000',alpha=0.5, edgecolor = None)
	ax.fill_between(time_REM, lifetime_means['REM'], lifetime_means['REM']-lifetime_sem['REM'],
		color = '#e50000',alpha=0.5, edgecolor = None)


	ax.plot(time_AWake, lifetime_means['Active Wake'], color = '#35ad6b')
	ax.fill_between(time_AWake, lifetime_means['Active Wake'], lifetime_means['Active Wake']+lifetime_sem['Active Wake'],
		color = '#35ad6b',alpha=0.5, edgecolor = None)
	ax.fill_between(time_AWake, lifetime_means['Active Wake'], lifetime_means['Active Wake']-lifetime_sem['Active Wake'],
		color = '#35ad6b',alpha=0.5, edgecolor = None)


	ax.plot(time_QWake, lifetime_means['Quiet Wake'], color = '#7e1e9c')
	ax.fill_between(time_QWake, lifetime_means['Quiet Wake'], lifetime_means['Quiet Wake']+lifetime_sem['Quiet Wake'],
		color = '#7e1e9c',alpha=0.5, edgecolor = None)
	ax.fill_between(time_QWake, lifetime_means['Quiet Wake'], lifetime_means['Quiet Wake']-lifetime_sem['Quiet Wake'],
		color = '#7e1e9c',alpha=0.5, edgecolor = None)

	ax.set_xlim([0,ax.get_xlim()[1]])
	ax.set_xlabel('Time (s)', fontsize=18)
	ax.set_ylabel('Lifetime', fontsize=18)
	sns.despine()
	plt.tight_layout()
	plt.ioff()
	plt.show()


def plot_lifetime_2_state(lifetime_means, lifetime_sem, all_time):
	time_sleep = all_time[0:np.size(lifetime_means['Sleep'])]
	time_wake =  all_time[0:np.size(lifetime_means['Wake'])]

	plt.ion()
	fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = [15,5])

	ax.plot(time_sleep, lifetime_means['Sleep'], color = '#0339f8')
	ax.fill_between(time_sleep, lifetime_means['Sleep'], lifetime_means['Sleep']+lifetime_sem['Sleep'],
		color = '#0339f8',alpha=0.5, edgecolor = None)
	ax.fill_between(time_sleep, lifetime_means['Sleep'], lifetime_means['Sleep']-lifetime_sem['Sleep'],
		color = '#0339f8',alpha=0.5, edgecolor = None)

	ax.plot(time_wake, lifetime_means['Wake'], color = '#35ad6b')
	ax.fill_between(time_wake, lifetime_means['Wake'], lifetime_means['Wake']+lifetime_sem['Wake'],
		color = '#35ad6b',alpha=0.5, edgecolor = None)
	ax.fill_between(time_wake, lifetime_means['Wake'], lifetime_means['Wake']-lifetime_sem['Wake'],
		color = '#35ad6b',alpha=0.5, edgecolor = None)

	ax.set_xlim([0,ax.get_xlim()[1]])
	ax.set_xlabel('Time (s)', fontsize=18)
	ax.set_ylabel('Lifetime', fontsize=18)
	sns.despine()
	plt.tight_layout()
	plt.ioff()
	plt.show()







