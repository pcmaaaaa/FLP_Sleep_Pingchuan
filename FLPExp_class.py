import numpy as np
import glob
import os
from scipy import io,signal
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
import FLP_Sleep_Pingchuan as PKA
import copy
import scipy
import itertools
from scipy.integrate import simps
import pandas as pd
import random
from neuroscience_sleep_scoring import SWS_utils

class FLiPExperiment():
	"""A class representing a single FLiP Experiment"""
	def __init__(self, filename, epoch_len = 4, binwidth = 2, one_acq = False, fs =1, filter_bounds = [0.003, 0.1], shuffle_window = 400, 
		experimental_sensor = 'FLIM-AKAR', sleep_states = True, microarousals = False):
		self.Sensor = experimental_sensor
		self.filename = filename
		self.rawdatdir = os.path.dirname(filename)
		data_dict = PKA.load_data(self.rawdatdir, self.filename)
		data_dict = PKA.do_npreg(data_dict, bw = binwidth)
		if 'time_all' in data_dict.keys():
			data_dict = PKA.get_zeit_time(data_dict)
			self.ZeitTime = data_dict['Zeit Time']
		i = list(filter(lambda x: 'GWidth' in x, list(data_dict.keys())))
		self.GaussianWidth = data_dict[i[0]]
		i = list(filter(lambda x: 'chi_sq_G' in x, list(data_dict.keys())))
		self.ChiSquare = data_dict[i[0]]
		i = list(filter(lambda x: 'dpeak' in x, list(data_dict.keys())))
		self.DeltaPeakTime = data_dict[i[0]]
		i = list(filter(lambda x: 'photoncount' in x, list(data_dict.keys())))
		self.PhotonCount = data_dict[i[0]]
		if max(self.PhotonCount*fs > 550000):
			print('WARNING: THIS EXPERIMENT COULD CONTAIN DEAD TIME ISSUES')
			self.DeadTime = True
		else:
			self.DeadTime = False
		i = list(filter(lambda x: 'tau_fit_G' in x, list(data_dict.keys())))
		self.Lifetime = data_dict[i[0]]
		i = list(filter(lambda x: 'time' in x, list(data_dict.keys())))
		i.remove('timestamps')
		self.Time = data_dict[i[0]]
		
		self.Timestamps = data_dict['timestamps']
		self.NonParaReg = data_dict['npreg']
		self.Filt = PKA.filt_lifetime(self.Lifetime, fs = fs, filt_low = filter_bounds[0], filt_high = filter_bounds[1], N = 3)
		self.EpochLength = epoch_len

		if sleep_states:
			self.SleepStates = PKA.get_all_states(self.rawdatdir, self.filename, one_acq)
			self.SSTime = np.arange(0, len(self.SleepStates)*self.EpochLength, self.EpochLength)
			if microarousals:
				self.SleepStates = SWS_utils.define_microarousals(self.SleepStates,  self.EpochLength)
		self.Shuff = PKA.shuffle_photometry(self.Filt, window = shuffle_window)


	def get_lifetimes(self, lifetime_dict, photon_dict, s_idx_dict, this_state, npreg = True):
		state_time =  np.arange(0,np.size(self.SleepStates)*self.EpochLength,self.EpochLength)
		for s in s_idx_dict[this_state]:
			if len(s) > 0:
				window = [state_time[s[0]], state_time[s[-1]]]
				lifetime_idx, = np.where(np.logical_and(self.Time>=window[0], self.Time<window[1]))
				if np.size(lifetime_idx) == 0:
					continue
				if npreg:
					lifetime_dict[this_state].append(self.NonParaReg[lifetime_idx])
				else:
					lifetime_dict[this_state].append(self.Lifetime[lifetime_idx])
				photon_dict[this_state].append(self.PhotonCount[lifetime_idx])
			else:
				pass

		self.LifetimeByState = lifetime_dict
		self.PhotonCountByState = photon_dict

	def remove_short_states(self, threshold = 20):
		grouped_states = []
		temp = [self.SleepStates[0]]
		for i in np.arange(1, np.size(self.SleepStates)):
			previous_state = self.SleepStates[i-1]
			this_state = self.SleepStates[i]
			if this_state == previous_state:
				temp.append(this_state)
			else:
				grouped_states.append(temp)
				temp = [this_state]
		grouped_states.append(temp)

		for s in np.arange(1, len(grouped_states)-1):
			this_state = grouped_states[s][0]
			previous_state = grouped_states[s-1][0]
			next_state = grouped_states[s+1][0]
			num_epochs = len(grouped_states[s])
			if (num_epochs <= int(threshold/self.EpochLength)) & (previous_state == next_state):
				grouped_states[s][:] = [np.nan for i in range(num_epochs)]
		mod_states = list(itertools.chain(*grouped_states))
		mod_states = np.asarray(mod_states)
		nan_idx, = np.where(np.isnan(mod_states))

		state_time =  np.arange(0,np.size(self.SleepStates)*self.EpochLength,self.EpochLength)
		mod_lifetime = copy.copy(self.Lifetime)

		for n in np.arange(0, np.size(nan_idx)):
			start = state_time[nan_idx[n]]
			end = state_time[nan_idx[n]+1]
			time_idx, = np.where(np.logical_and(self.Time>=start, self.Time<end))
			mod_lifetime[time_idx] = np.nan

		return mod_lifetime, mod_states

	def find_transients(self, num_std = 3, FWHM_thresh = False, shuffled = False, discrete_cutoff = False):
		transient_dict = {self.Sensor: {'Data': self.Filt, 'Transient Idx': [], 'FWHM': []}}
		if shuffled:
			transient_dict['Shuffled'] = {'Data': self.Shuff,'Transient Idx': [], 'FWHM': []}
		for k in list(transient_dict.keys()):
			mean_lifetime, thresh, s = PKA.transient_threshold(transient_dict[k]['Data'], num_std = num_std, 
				discrete_cutoff = discrete_cutoff)
			# thresholded_vals, = np.where(self.Filt < (-thresh))
			thresholded_vals, = np.where(transient_dict[k]['Data'] < (mean_lifetime-thresh))
			grouped_thresh_vals = []
			prev_val = thresholded_vals[0]
			temp = [prev_val]
			for v in thresholded_vals[1:]:
				if v == prev_val + 1:
					temp.append(v)
				else:
					grouped_thresh_vals.append(temp)
					temp = [v]
				prev_val = v
			grouped_thresh_vals.append(temp)
			corrected_groups = []
			this_list = grouped_thresh_vals[0]
			for i in range(0,len(grouped_thresh_vals)-1):
				next_list = grouped_thresh_vals[i+1]
				in_between_idx = np.arange(this_list[-1], next_list[0])
				in_between_lifetime = transient_dict[k]['Data'][in_between_idx]
				if any(in_between_lifetime > mean_lifetime):
					corrected_groups.append(this_list)
					this_list = grouped_thresh_vals[i+1]
					if i == len(grouped_thresh_vals)-2:
						corrected_groups.append(next_list)
				else:
					combo_list = this_list + list(in_between_idx)+next_list
					this_list = combo_list
			transient_idxs = []
			for idx in corrected_groups:
				# transient_idxs.append(PKA.transient_window(self.Filt, idx, 0))
				temp = PKA.transient_window(transient_dict[k]['Data'], idx, mean_lifetime)
				if any(np.asarray(temp) < 0):
					delete = np.asarray(temp)[np.where(np.asarray(temp) < 0)[0]]
					temp.remove(delete)
				transient_idxs.append(temp)

			FWHM,time_points = PKA.transient_FWHM(transient_dict[k]['Data'], self.Time, transient_idxs) 
			if FWHM_thresh:
				these_idxs, = np.where(np.array(FWHM) > FWHM_thresh)
				transient_idxs = [transient_idxs[i] for i in these_idxs]
				FWHM = [FWHM[i] for i in these_idxs]
			transient_dict[k]['FWHM'] = FWHM
			transient_dict[k]['Transient Idx'] = transient_idxs
		# return transient_idxs, FWHM
		return transient_dict

	def get_net_change(self, state, remove_ss = True, threshold = 20, binsize = 30, window = 300):
		if remove_ss:
			mod_lifetime, mod_states = self.remove_short_states(threshold = threshold)
			ss = mod_states
		else:
			ss = self.SleepStates
		cont_idx = PKA.find_continuous(ss, state)
		state_time =  np.arange(0,np.size(self.SleepStates)*self.EpochLength,self.EpochLength)
		lifetime_idx = [np.where(np.logical_and(self.Time>state_time[i[0]]-binsize,self.Time<state_time[i[-1]]))[0] for i in cont_idx]
		cont_lifetimes = [self.Lifetime[i] for i in lifetime_idx]
		times = [self.Time[i] for i in lifetime_idx]

		net_change = np.zeros([len(cont_lifetimes), 2])
		for ii in np.arange(0, len(cont_lifetimes)):
			this_lifetime = cont_lifetimes[ii]
			this_time = times[ii]-times[ii][0] 
			if this_time[-1] < 2* binsize:
				net_change[ii] = np.array([np.nan,np.nan])
				continue
			elif this_time[-1] < window:
				first_bin, = np.where(this_time < binsize)
				second_bin, = np.where(this_time > this_time[-1]-binsize)
			else:
				first_bin, = np.where(this_time < binsize)
				second_bin, = np.where(np.logical_and(this_time > (window-binsize), this_time < window))
			net_change[ii] = np.array([np.mean(this_lifetime[first_bin]), np.mean(this_lifetime[second_bin])])

		return net_change
	def psd_per_ss(self, fsd = 200, transient_window = 100):
		onoff_df = PKA.ss_onset_offset(self.SleepStates, self.EpochLength)
		transient_end = []
		for i in onoff_df.index:
			if (onoff_df['End Time'][i]-onoff_df['Start Time'][i]) > transient_window:
				transient_end.append(onoff_df['Start Time'][i]+transient_window)
			else:
				transient_end.append(onoff_df['End Time'][i])
		onoff_df['Transient End'] = transient_end
		EEG = PKA.get_all_EEG(os.path.join(self.rawdatdir, '*extracted_data*'))
		realtime = np.linspace(0, self.Time[-1], np.size(EEG))
		onoff_df['freqs'] = ''
		onoff_df['psd'] = ''
		onoff_df['psd_normval'] = ''

		onoff_df['Transient Freqs'] = ''
		onoff_df['Transient psd'] = ''
		onoff_df['Transient Normval'] = ''
		for i in np.arange(0, onoff_df.shape[0]):
			time_idx, = np.where(np.logical_and(realtime > onoff_df['Start Time'][i], realtime < onoff_df['End Time'][i]))
			freqs, psd = signal.welch(EEG[time_idx], fsd, scaling='density') # for each freq, have a power value
			onoff_df.at[i,'psd'] = psd
			onoff_df.at[i,'freqs'] = freqs
			onoff_df.at[i,'psd_normval'] = simps(psd, freqs)

			transient_idx, = np.where(np.logical_and(realtime > onoff_df['Start Time'][i], realtime < onoff_df['Transient End'][i]))
			freqs, psd = signal.welch(EEG[transient_idx], fsd, scaling='density') # for each freq, have a power value
			onoff_df.at[i,'Transient psd'] = psd
			onoff_df.at[i,'Transient Freqs'] = freqs
			onoff_df.at[i,'Transient Normval'] = simps(psd, freqs)

		return onoff_df

	def ss_transition_per_transient(self, transient_dict, buffer_epochs = 4,microarousals = False):
		'''This function takes every transient and finds what type of sleep-wake transition occured right before it.'''
		state_time = np.arange(0, np.size(self.SleepStates)*self.EpochLength, self.EpochLength)
		if microarousals:
			self.SleepStates = SWS_utils.define_microarousals(self.SleepStates, self.EpochLength)
		transient_idxs = transient_dict['Transient Idx']
		transient_starts = [t[0] for t in transient_idxs] #Indexes correspond to photometry data
		transition_types = np.zeros([len(transient_starts), 2])
		distance_from_transition = np.zeros(len(transient_starts))
		for ii, i in enumerate(transient_starts):
			t_start = self.Time[i]
			buffer_start = t_start-(buffer_epochs*self.EpochLength)
			buffer_end = t_start+(buffer_epochs*self.EpochLength)
			buffer_idx, = np.where(np.logical_and(self.Time>=buffer_start, self.Time<=buffer_end)) #Indexes correspond to photometry data
			these_states, these_states_idx = PKA.get_these_ss(buffer_idx, self.SleepStates, 
				self.Time, self.EpochLength) #Indexes correspond to sleep states data
			if len(these_states) == 0:
				transition_types[ii] = [np.nan, np.nan]
				distance_from_transition[ii] = np.nan
				continue
			these_states[these_states == 4] = 1
			state_types = pd.unique(these_states)
			# common_state = int(scipy.stats.mode(these_states)[0])
			if len(state_types) == 1:
				state2 = state_types[0]
				transition_idx, state1 = PKA.find_ss_transition(state2, 
					these_states_idx[0], self.SleepStates) #Indexes correspond to sleep states data
				if transition_idx < 0:
					transition_types[ii] = [np.nan, np.nan]
					distance_from_transition[ii] = np.nan
					continue				
			else:
				state1 = state_types[0]
				state1_idx, = np.where(these_states == state1) #Indexes correspond to sleep states data
				state2 = state_types[1]
				transition_idx = np.where(these_states == state2)[0][0] #Indexes correspond to sleep states data
			transition_types[ii] = [state1, state2]
			distance_from_transition[ii] = t_start - state_time[transition_idx]
		return transition_types, distance_from_transition

	def transition_timestamps(self, microarousals = False, diff_wake = False):
		if microarousals:
			self.SleepStates = SWS_utils.define_microarousals(self.SleepStates, self.EpochLength)
		transition_dict = {
		'NREM-REM': [2,3], 
		'REM-NREM': [3,2]}

		if diff_wake:
			transition_dict['NREM-Active Wake'] = [2,1]
			transition_dict['NREM-Quiet Wake'] = [2,4]
			transition_dict['REM-Active Wake'] = [3,1]
			transition_dict['REM-Quiet Wake'] = [3,4]
			transition_dict['Active Wake-NREM'] = [1,2]
			transition_dict['Quiet Wake-NREM'] = [4,2]
			transition_dict['Active Wake-REM'] = [1,3]
			transition_dict['Quiet Wake-REM'] = [4,3]
			transition_dict['Active Wake-Quiet Wake'] = [1,4]
			transition_dict['Quiet Wake-Active Wake'] = [4,1]

		else:
			self.SleepStates[np.where(self.SleepStates == 4)[0]] = 1
			transition_dict['NREM-Wake'] = [2,1]
			transition_dict['REM-Wake'] = [3,1]
			transition_dict['Wake-NREM'] = [1,2]
			transition_dict['Wake-REM'] = [1,3]

		these_keys = list(transition_dict.keys())
		output_dict = {'Timestamps':{}, 'Number':{}}

		for k in these_keys:
			output_dict['Timestamps'][k] = []
			output_dict['Number'][k] = []
		
		for t in list(transition_dict.keys()):
			state_nums = transition_dict[t]
			epochs = PKA.find_continuous(self.SleepStates, [state_nums[1]])
			epoch_starts = [x[0] for x in epochs]
			these_transitions = [x for x in epoch_starts if self.SleepStates[x-1] == state_nums[0]]
			output_dict['Timestamps'][t] = self.SSTime[these_transitions]
			output_dict['Number'][t] = len(these_transitions)

		micros = PKA.find_continuous(self.SleepStates, [5])
		these_transitions = [x[0] for x in micros]
		output_dict['Timestamps']['Microarousals'] = self.SSTime[these_transitions]
		output_dict['Number']['Microarousals'] = len(these_transitions)
		if diff_wake:
			output_dict['Number']['Sleep-Wake'] = output_dict['Number']['NREM-Quiet Wake'] + output_dict['Number']['NREM-Active Wake'] + output_dict['Number']['REM-Active Wake'] + output_dict['Number']['REM-Quiet Wake']
			output_dict['Number']['Sleep-Wake'] = output_dict['Number']['Quiet Wake-NREM'] + output_dict['Number']['Active Wake-NREM'] + output_dict['Number']['Active Wake-REM'] + output_dict['Number']['Quiet Wake-REM']
			output_dict['Timestamps']['Sleep-Wake'] = np.concatenate([output_dict['Timestamps']['NREM-Quiet Wake'], output_dict['Timestamps']['NREM-Active Wake'], 
				output_dict['Timestamps']['REM-Active Wake'], output_dict['Timestamps']['REM-Quiet Wake']])
			output_dict['Timestamps']['Sleep-Wake'] = np.concatenate([output_dict['Timestamps']['Quiet Wake-NREM'], output_dict['Timestamps']['Active Wake-NREM'],
				output_dict['Timestamps']['Active Wake-REM'], output_dict['Timestamps']['Quiet Wake-REM']])


		else:
			output_dict['Number']['Sleep-Wake'] = output_dict['Number']['NREM-Wake'] + output_dict['Number']['REM-Wake']
			output_dict['Number']['Wake-Sleep'] = output_dict['Number']['Wake-NREM'] + output_dict['Number']['Wake-REM']
			output_dict['Timestamps']['Sleep-Wake'] = np.concatenate([output_dict['Timestamps']['NREM-Wake'], output_dict['Timestamps']['REM-Wake']])
			output_dict['Timestamps']['Wake-Sleep'] = np.concatenate([output_dict['Timestamps']['Wake-NREM'], output_dict['Timestamps']['Wake-REM']])

		return output_dict



















