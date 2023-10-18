import numpy as np
import glob
import os
import FLP_Sleep_Pingchuan as PKA
import pandas as pd
import Graphing_Utils as graph
from matplotlib import cm,patches
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from copy import deepcopy
from scipy import stats
from neuroscience_sleep_scoring import SWS_utils
from scipy.integrate import simps
import math


def get_transient_properties(df, filter_bounds = [0.003, 0.1], num_std  = 3, discrete_cutoff = False, 
	binned = False, FWHM_thresh = False, experimental_sensor = 'FLIM-AKAR', microarousals = False):
	property_dict = {}
	property_list = ['Amplitude', 'FWHM', 'Transient Duration', 'Experiment Name', 'Transition Types', 
	'Time from Transition', 'Transient Idx', 'SS Frequency']

	for p in property_list:
		property_dict[p] = []
	basenames = list(df['Experiment Name'])
	for b in basenames:
		print('Working on '+b+'...')
		rawdatdir = os.path.join('/Volumes/yaochen/Active/Lizzie/FLP_data/', b)
		if binned:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'binned_concat*.mat'))[0]
			exp = PKA.FLiPExperiment(concat_filename, filter_bounds = filter_bounds, fs = 0.25, experimental_sensor = experimental_sensor, 
				microarousals = microarousals)
		else:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'concat*.mat'))[0]
			exp = PKA.FLiPExperiment(concat_filename, filter_bounds = filter_bounds, experimental_sensor = experimental_sensor, microarousals = microarousals)
		transient_dict = exp.find_transients(num_std = num_std, FWHM_thresh = FWHM_thresh, discrete_cutoff = discrete_cutoff)
		transient_len = [exp.Time[t[-1]]-exp.Time[t[0]] for t in transient_dict[exp.Sensor]['Transient Idx']]
		amplitudes, start_vals, troughs = PKA.transient_amplitudes(exp.Lifetime, transient_dict[exp.Sensor]['Transient Idx'])
		transition_types, distance_from_transition = exp.ss_transition_per_transient(transient_dict[exp.Sensor], buffer_epochs = 2, 
			microarousals = microarousals)
		# FWHM,time_points = PKA.transient_FWHM(exp.Filt, exp.Time, transient_idxs)
		property_dict['Experiment Name'].append(b)
		property_dict['Amplitude'].append(amplitudes)
		property_dict['FWHM'].append(transient_dict[exp.Sensor]['FWHM'])
		property_dict['Transient Duration'].append(transient_len)
		property_dict['Transition Types'].append(transition_types)
		property_dict['Time from Transition'].append(distance_from_transition)
		property_dict['SS Frequency'].append(exp.transition_timestamps(microarousals = microarousals)['Number'])
	return property_dict

def get_basic_analyses(df, filter_bounds, num_std  = 3, discrete_cutoff = False, binned = False, FWHM_thresh = False, shuffled = False, 
	transient_detection = True, overlay_window = 50, stateplot_window = 120, lifetime_w_ss = True, state_plot = 'both', state_density = True,
	timing_ditribution = True, transient_frequency = True, savedir_dict = False, experimental_sensor = 'FLIM-AKAR', microarousals = True,
	intensity = True):
	'''This script will give you all or a subset of the basic figures used for PKA transient detection. 
			- transient_detection: Filtering and PKA transient detection fig (previously Transient_detection.ipynb)
			- lifetime_w_ss: Hourly figures with both filtered and raw photometry lined up with sleep state background 
			(previously Sleep_states_photometry_lifetimeonly_binned.ipynb)
			- state_plot: transient-triggered stateplot; per animal and whole group; indicate 'individual', 'grouped', or 'both' (previously transient_triggered_stateplot.ipynb)
			- state_density: probability of animal being in a each state before and after transient; per animal and whole group 
			(previously state_density.ipynb)
			- timing_ditribution: cummulative distribution of transient timing aligned to transition start, and cummulative distribution of 
			transition timing aligned to transient; only whole group (previously Transient_timing_cummdist.ipynb)
			- transient_frequency: calculates the percent of transitions with a transient for each type
		Filter bounds are usuing high pass at 0.0003 and FWHM is usually 6.8'''
	graph.make_bigandbold(xticksize = 20, yticksize = 20, axeslabelsize = 25)
	color_dict = graph.SW_colordict('numbers')

	if filter_bounds[0] is None:
		filter_type = 'Low Pass Filter: '
		filt_bound_str = str(filter_bounds[1]) + 'Hz'
	elif filter_bounds[1] is None:
		filter_type = 'High Pass Filter: '
		filt_bound_str = str(filter_bounds[0]) + 'Hz'
	else:
		filter_type = 'Bandpass Filter: '
		filt_bound_str = str(filter_bounds[0]) + 'Hz' + '-' + str(filter_bounds[1]) + 'Hz'
	if state_plot or state_density:
			all_data_dict = {}
			all_data_dict[experimental_sensor] = {'SS Data': []}
			if shuffled:
				all_data_dict['Shuffled'] = {'SS Data': []}
	if state_density:
		labels = ['Wake', 'REM','NREM', 'Unknown', 'Number of Transients', 'Animal Name']
		for g in list(all_data_dict.keys()):
			for i in labels:
				all_data_dict[g][i] = [] 
	if timing_ditribution:
		timing_data_dict = {}
		timing_data_dict[experimental_sensor] = {}
		if shuffled:
			timing_data_dict['Shuffled'] = {}
		transition_labels = ['NREM-Wake', 'REM-Wake', 'NREM-REM', 'Wake-REM', 'REM-NREM', 'Wake-NREM', 'Unknown','Sleep-Wake','Wake-Sleep','Number of Transients', 'Animal Name']
		for g in list(timing_data_dict.keys()):
			for i in transition_labels:
				timing_data_dict[g][i] = []
	# if transient_frequency:
	# 	frequency_dict = {}
	# 	frequency_labels = ['NREM-Wake', 'REM-Wake', 'NREM-REM', 'Wake-REM', 'REM-NREM', 'Wake-NREM','Sleep-Wake','Wake-Sleep']
	# 	for ID in np.unique(df['Mouse ID']):
	# 		frequency_dict[ID] = {}
	# 		for i in frequency_labels:
	# 			frequency_dict[ID][i] = []
				

	basenames = list(df['Experiment Name'])
	for b in basenames:
		if state_plot or state_density:
			for g in list(all_data_dict.keys()):
				all_data_dict[g]['Animal Name'].append(df.loc[df['Experiment Name'] == b]['Mouse ID'].item())
		print('Working on '+b+'...')
		rawdatdir = os.path.join('/Volumes/yaochen/Active/Lizzie/FLP_data/', b)
		if binned:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'binned_concat*.mat'))[0]
			exp = PKA.FLiPExperiment(concat_filename, filter_bounds = filter_bounds, fs = 0.25,experimental_sensor = experimental_sensor, microarousals = microarousals)
		else:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'concat*.mat'))[0]
			exp = PKA.FLiPExperiment(concat_filename, filter_bounds = filter_bounds, experimental_sensor = experimental_sensor, microarousals = microarousals)
		# if exp.DeadTime:
		# 	use_flag = input(b+' might have deadtime issues. Do you want to still include it (y/n)?')
		# 	if use_flag == 'y':
		# 		print('Ok, keeping...')
		# 		pass
		# 	else:
		# 		print('Ok, skipping....')
		# 		continue
		if transient_detection:
			transient_detection_fig, transient_detection_ax = plt.subplots(nrows = 4, ncols = 1, figsize = (18,24))
			transient_detection_fig, transient_detection_ax = graph.thick_axes(transient_detection_fig, transient_detection_ax)
			transient_overlay_fig, transient_overlay_ax = plt.subplots(nrows = 1, ncols = 1, figsize = (9,7))
			transient_detection_ax[0].plot(exp.Time, -(exp.Lifetime-exp.Lifetime[0]), linewidth = 0.5, color = 'k')
			transient_detection_ax[1].plot(exp.Time, -exp.Filt, linewidth = 0.5, color = 'k')
			transient_detection_ax[2].plot(exp.Time, -exp.Filt, linewidth = 0.5, color = 'k')
			transient_detection_ax[3].plot(exp.Time, -exp.Filt, linewidth = 0.5, color = 'k')
			transient_dict = exp.find_transients(num_std = num_std, discrete_cutoff = discrete_cutoff, FWHM_thresh = False, shuffled = shuffled)
			cmap = cm.get_cmap('plasma',len(transient_dict[exp.Sensor]['Transient Idx']))
			win = overlay_window
			for i,t in enumerate(transient_dict[exp.Sensor]['Transient Idx']):
				if any(np.asarray(t) < 0):
					delete = np.asarray(t)[np.where(np.asarray(t) < 0)[0]]
					t.remove(delete)
				c = cmap.colors[i]
				x = exp.Time[t]
				win_start = x[0]-win
				win_end = x[-1]+win
				if win_end > exp.Time[-1]:
				    win_end = exp.Time[-1]
				if win_start < exp.Time[0]:
				    win_start = exp.Time[0]
				window_idx, = np.where(np.logical_and(exp.Time>win_start, exp.Time<=win_end))
				x = exp.Time[window_idx]-exp.Time[window_idx[0]]
				y = exp.Lifetime[window_idx]-exp.Lifetime[t[0]]
				transient_detection_ax[2].plot(exp.Time[t], -exp.Filt[t], linewidth = 0.5, color = 'r')

			transient_dict = exp.find_transients(num_std = num_std, discrete_cutoff = discrete_cutoff, FWHM_thresh = FWHM_thresh, shuffled = shuffled)
			for i,t in enumerate(transient_dict[exp.Sensor]['Transient Idx']):
				if any(np.asarray(t) < 0):
					delete = np.asarray(t)[np.where(np.asarray(t) < 0)[0]]
					t.remove(delete)
				c = cmap.colors[i]
				x = exp.Time[t]
				win_start = x[0]-win
				win_end = x[-1]+win
				if win_end > exp.Time[-1]:
					win_end = exp.Time[-1]
				if win_start < exp.Time[0]:
					win_start = exp.Time[0]
				window_idx, = np.where(np.logical_and(exp.Time>win_start, exp.Time<=win_end))
				x = exp.Time[window_idx]-exp.Time[window_idx[0]]
				y = exp.Lifetime[window_idx]-exp.Lifetime[t[0]]
				transient_overlay_ax.plot(x, -y, color = c, linewidth = 1) 
				transient_overlay_ax.axvline(win, linestyle = '--', color = 'k')
				transient_detection_ax[3].plot(exp.Time[t], -exp.Filt[t], linewidth = 0.5, color = 'r')
			transient_overlay_fig, transient_overlay_ax = graph.thick_axes(transient_overlay_fig, transient_overlay_ax)

			graph.label_axes(transient_detection_ax[0], x = 'Time (s)', y = '-'+r'$\Delta$'+ ' Lifetime (ns)', title = 'Experiment: ' + b, title_fontsize = 25)
			graph.label_axes(transient_detection_ax[1], x = 'Time (s)', y = '-'+r'$\Delta$'+ ' Lifetime (ns)', 
				title = 'Filtered ('+filter_type+filt_bound_str+')', title_fontsize = 20)
			if num_std:
				graph.label_axes(transient_detection_ax[2], y = '-'+r'$\Delta$'+ ' Lifetime (ns)', x = 'Time (s)', 
					title = 'Amplitude Threshold ('+str(num_std) + ' StDev)', title_fontsize = 20)
			else:
				graph.label_axes(transient_detection_ax[2], y = '-'+r'$\Delta$'+ ' Lifetime (ns)', x = 'Time (s)', 
					title = 'Amplitude Threshold ('+str(discrete_cutoff) + ' ns)', title_fontsize = 20)

			graph.label_axes(transient_detection_ax[3],x = 'Time (s)', y = '-'+r'$\Delta$'+ ' Lifetime (ns)', 
				title = 'FWHM Threshold ('+str(FWHM_thresh) + ' ns)', title_fontsize = 20)

			transient_detection_ax[0].set_xlim([0, exp.Time[-1]])
			transient_detection_ax[1].set_xlim([0, exp.Time[-1]])
			transient_detection_ax[2].set_xlim([0, exp.Time[-1]])
			transient_detection_ax[3].set_xlim([0, exp.Time[-1]])
			yticks = np.linspace(round(transient_detection_ax[1].get_ylim()[0], 3), round(transient_detection_ax[1].get_ylim()[1], 3), 4)
			yticks = [round(y, 3) for y in yticks]
			transient_detection_ax[1].set_yticks(yticks)
			transient_detection_ax[2].set_yticks(yticks)
			transient_detection_ax[3].set_yticks(yticks)
			transient_overlay_ax.set_xticks(np.arange(0, 250, 75))
			transient_overlay_ax.set_xlim([0, 250])
			transient_overlay_ax.set_xticklabels(np.arange(0, 250, 75)-win)
			graph.label_axes(transient_overlay_ax, x = 'Time from Transient (s)', y = '-'+r'$\Delta$'+ ' Lifetime (ns)', title = 'Experiment: ' + b, 
				title_fontsize = 20)  
			transient_overlay_fig.tight_layout()
			transient_detection_fig.tight_layout()
			if savedir_dict:
				try:
					os.mkdir(os.path.join(savedir_dict['Transient Detection'],b,'Transient_detection_figs'))
				except FileExistsError:
					pass
				if binned:
					savefilename_detection = os.path.join(savedir_dict['Transient Detection'],b,'Transient_detection_figs','transient_detection_binned.png')
					savefilename_overlay = os.path.join(savedir_dict['Transient Detection'],b,'Transient_detection_figs','transient_overlay_binned.png')
				else:
					savefilename_detection = os.path.join(savedir_dict['Transient Detection'],b,'Transient_detection_figs','transient_detection.png')
				transient_detection_fig.savefig(savefilename_detection)
				transient_overlay_fig.savefig(savefilename_overlay)
			plt.close('all')
		if lifetime_w_ss:
			num_rows = 1
			if any(filter_bounds):
				num_rows += 1
			if intensity:
				num_rows += 1
			acq_files = glob.glob(os.path.join(rawdatdir, 'Acq*_analysis_binned.mat'))
			acqs = np.sort([int(f[f.find('q')+1:f.find('_analysis')]) for f in acq_files])
			for i,a in enumerate(acqs):
				hs = []
				ys = []
				lifetime_ss_fig, ax = plt.subplots(ncols = 1, nrows = num_rows, figsize = (20,5*num_rows))
				ax_idx = 0
				time_window = [i*3600, (i+1)*3600]
				plot_idx, = np.where(np.logical_and(exp.Time>=time_window[0], exp.Time<time_window[1]))
				#plotting raw lifetime
				ax[ax_idx].plot(exp.Time[plot_idx], exp.Lifetime[plot_idx], color = 'k', linewidth = 2)
				ax[ax_idx].set_xlim(i*3600, (i+1)*3600)
				graph.label_axes(ax[ax_idx],y = 'Lifetime (ns)', x = 'Time (s)', title = 'Experiment: '+b+'\n'+'Acquisition: '+str(a))


				y_low, y_high = ax[ax_idx].get_ylim()
				ys.append(y_low)
				hs.append(y_high-y_low)

				if any(filter_bounds):
					#plotting Filtered Lifetime (if applicable)
					ax_idx += 1
					ax[ax_idx].plot(exp.Time[plot_idx], exp.Filt[plot_idx], color = 'k', linewidth = 2)
					ax[ax_idx].set_xlim(i*3600, (i+1)*3600)
					y_low, y_high = ax[ax_idx].get_ylim()
					ys.append(y_low)
					hs.append(y_high-y_low)
					graph.label_axes(ax[ax_idx],y = 'Lifetime (ns)', x = 'Time (s)', title = 'Experiment: '+b+' Filtered ('+filter_type+filt_bound_str+')'+'\n'+'Acquisition: '+str(a))

				if intensity:
					ax_idx += 1
					ax[ax_idx].plot(exp.Time[plot_idx], exp.PhotonCount[plot_idx], color = 'k', linewidth = 2)
					order = np.sort(exp.PhotonCount[plot_idx])
					ax[ax_idx].set_xlim(i*3600, (i+1)*3600)
					y_low, y_high = ax[ax_idx].get_ylim()
					y_low = order[1]-10000
					ax[ax_idx].set_ylim([y_low, y_high])
					ys.append(y_low)
					hs.append(y_high-y_low)
					graph.label_axes(ax[ax_idx],y = 'Photon Count', x = 'Time (s)', title = 'Experiment: '+b+'\n'+'Acquisition: '+str(a))

				for state in [1,2,3,4,5]:
					cont_state = PKA.find_continuous(exp.SleepStates[int(i*900):int((i+1)*900)], [state])
					if len(cont_state) > 0:
						if len(cont_state[0])>0:
							for s in cont_state:
								x = (s[0]*exp.EpochLength)+time_window[0]
								w = ((s[-1]-s[0])+1)*exp.EpochLength
								for ii in np.arange(0, num_rows):
									rect = patches.Rectangle((x,ys[ii]), w, hs[ii], facecolor = color_dict[str(int(state))], 
									                          alpha = 1, edgecolor = None, zorder = 0)
									ax[ii].add_patch(rect)
								# rect2 = patches.Rectangle((x,y2), w, h2, facecolor = color_dict[str(int(state))], 
								#                           alpha = 1, edgecolor = None, zorder = 0)
								# filt_ax.add_patch(rect2)
			#     ax1.set_ylim(y_low, y_high)
				lifetime_ss_fig, ax = graph.thick_axes(lifetime_ss_fig, ax)
				lifetime_ss_fig.tight_layout()
				if savedir_dict:
					try:
						os.mkdir(os.path.join(savedir_dict['Lifetime w SS'],b,'photometry_w_SS'))
					except FileExistsError:
						pass
					if binned:
						savefilename_lifetime = os.path.join(savedir_dict['Lifetime w SS'],b,'photometry_w_SS','binned_Acq_'+str(a)+'.png')
					else:
						savefilename_lifetime = os.path.join(savedir_dict['Lifetime w SS'],b,'photometry_w_SS','Acq_'+str(a)+'.png')
					lifetime_ss_fig.savefig(savefilename_lifetime)
				plt.close('all')
		if state_plot:
			if not transient_detection:
				transient_dict = exp.find_transients(num_std = num_std, discrete_cutoff = discrete_cutoff, FWHM_thresh = FWHM_thresh, shuffled = shuffled)
			if state_plot == 'individual' or state_plot == 'both':
				stateplot_ind_fig, stateplot_ind_ax = plt.subplots(ncols = len(list(all_data_dict.keys())), figsize = [len(list(all_data_dict.keys()))*11,13])
				if len(list(all_data_dict.keys())) == 1:
					stateplot_ind_ax = [stateplot_ind_ax]
			for di,d in enumerate(list(transient_dict.keys())):
				transient_starts = [t[0] for t in transient_dict[d]['Transient Idx']]
				ss_bins = PKA.transient_associated_ss(transient_starts, stateplot_window, exp)
				all_data_dict[d]['SS Data'].append(ss_bins)

				if (state_plot == 'individual' or state_plot == 'both'):
					x_stateplot = np.arange(0, np.shape(ss_bins)[1])
					x_stateplot = (x_stateplot-np.shape(ss_bins)[1]/2)*exp.EpochLength
					all_distance = np.zeros(np.shape(ss_bins)[0]-1)
					for aa in np.arange(0, np.size(all_distance)):
						counter = 0
						idx = -1
						before_idx, = np.where(x_stateplot<=0)
						zero_idx, = np.where(x_stateplot==0)
						wake_idx, = np.where(np.logical_or(ss_bins[aa,before_idx] == 1, ss_bins[aa,before_idx] == 4))
						distance = zero_idx-wake_idx
						try:
							while distance[idx] == counter:
								counter = counter+1
								idx = idx - 1
						except IndexError:
							pass
						all_distance[aa] = counter
					sorted_idx = np.argsort(all_distance)

					for i, ii in enumerate(sorted_idx):
						for state in [1,2,3,4,5]:
							cont_state = PKA.find_continuous(ss_bins[ii, :], [state])
							if len(cont_state)>0:
								for s in cont_state:
									x_pos = x_stateplot[s][0]
									w = ((x_stateplot[s[-1]]-x_stateplot[s][0]))+exp.EpochLength
									rect1 = patches.Rectangle((x_pos,i), w, 1, facecolor = color_dict[str(int(state))], 
									                          alpha = 1, edgecolor = None, zorder = 0)
									stateplot_ind_ax[di].add_patch(rect1)
					stateplot_ind_ax[di].axvline(0, linestyle = '--', color = 'k')
					stateplot_ind_ax[di].set_xlim([x_stateplot[0], x_stateplot[-1]])
					stateplot_ind_ax[di].set_ylim([0, np.shape(ss_bins)[0]-1])
					stateplot_ind_fig.suptitle(b, fontweight = 'bold', fontsize = 30)
					graph.label_axes(stateplot_ind_ax[di], x = 'Time (s)', y = 'Transition ID', title = d)
					for axis in ['bottom','left', 'right', 'top']:
						stateplot_ind_ax[di].spines[axis].set_linewidth(3)
				stateplot_ind_fig.tight_layout()
				if savedir_dict:
					try:
						os.mkdir(savedir_dict['State Plot Individual'])
					except FileExistsError:
						pass
					savefilename_stateplot_ind = os.path.join(savedir_dict['State Plot Individual'],b+'_stateplot.png')
					stateplot_ind_fig.savefig(savefilename_stateplot_ind)
				plt.close('all')
		if state_density:
			color_dict_density = graph.SW_colordict('single state')
			if (state_density == 'both') or (state_density == 'individual'):
				density_ind_fig, density_ind_ax = plt.subplots(ncols = len(list(all_data_dict.keys())), figsize = [len(list(all_data_dict.keys()))*9,8])
				if len(list(all_data_dict.keys())) == 1:
					density_ind_ax = [density_ind_ax]
				density_ind_fig, density_ind_ax = graph.thick_axes(density_ind_fig, density_ind_ax)
			if not transient_detection and not state_plot:
				transient_dict = exp.find_transients(num_std = num_std, discrete_cutoff = discrete_cutoff, FWHM_thresh = FWHM_thresh, shuffled = shuffled)

			for di,d in enumerate(list(transient_dict.keys())):		
				transient_starts = [t[0] for t in transient_dict[d]['Transient Idx']]
				ss_bins = PKA.transient_associated_ss(transient_starts, stateplot_window, exp)
				if not state_plot:
					all_data_dict[d]['SS Data'].append(ss_bins)
				x_statedensity = np.arange(0, np.shape(ss_bins)[1])
				x_vals = (x_statedensity-len(x_statedensity)/2)*exp.EpochLength
				all_data_dict[d]['Number of Transients'].append(len(transient_dict[d]['Transient Idx']))
				wake_bin_counts = [np.size(np.where(ss_bins[:,i] == 1))+np.size(np.where(ss_bins[:,i] == 4)) for i in x_statedensity]
				all_data_dict[d]['Wake'].append(wake_bin_counts)
				NREM_bin_counts = [np.size(np.where(ss_bins[:,i] == 2)) for i in x_statedensity]
				all_data_dict[d]['NREM'].append(NREM_bin_counts)
				REM_bin_counts = [np.size(np.where(ss_bins[:,i] == 3)) for i in x_statedensity]
				all_data_dict[d]['REM'].append(REM_bin_counts)
				if (state_density == 'both') or (state_density == 'individual'):
					density_ind_ax[di].plot(x_vals, np.asarray(wake_bin_counts)/len(transient_dict[d]['Transient Idx']), color = color_dict_density['Wake'], label = 'Wake', linewidth = 2)
					density_ind_ax[di].plot(x_vals, np.asarray(NREM_bin_counts)/len(transient_dict[d]['Transient Idx']), color = color_dict_density['NREM'], label = 'NREM', linewidth = 2)
					density_ind_ax[di].plot(x_vals, np.asarray(REM_bin_counts)/len(transient_dict[d]['Transient Idx']), color = color_dict_density['REM'], label = 'REM', linewidth = 2)
					density_ind_ax[di].axvline(0, color = 'k', linestyle = '--')
					density_ind_ax[di].set_xlim([-(stateplot_window-16), (stateplot_window-16)])
					density_ind_fig.suptitle(b, fontweight = 'bold', fontsize = 30)
					graph.label_axes(density_ind_ax[di], x = 'Time from Transient', y = 'Probability of Behavior State', 
					                 title = d)
					density_ind_fig.tight_layout()				
					if savedir_dict:
						try:
							os.mkdir(savedir_dict['State Density Individual'])
						except FileExistsError:
							pass

						savefilename_statedensity_ind = os.path.join(savedir_dict['State Density Individual'],b+'_statedensity.png')
						density_ind_fig.savefig(savefilename_statedensity_ind)
					plt.close('all')
		if timing_ditribution:
			for d in list(timing_data_dict.keys()):
				transition_types, distance_from_transition = exp.ss_transition_per_transient(transient_dict[d], buffer_epochs = 2)
				data_dict_ind = PKA.transient_property_per_sstype(transition_types, distance_from_transition, transition_labels)
				for k in ['NREM-Wake', 'REM-Wake', 'NREM-REM', 'Wake-REM', 'REM-NREM', 'Wake-NREM']:
					timing_data_dict[d][k].append(data_dict_ind[k])
				timing_data_dict[d]['Sleep-Wake'].append(np.concatenate([data_dict_ind['NREM-Wake'], data_dict_ind['REM-Wake']]))
				timing_data_dict[d]['Wake-Sleep'].append(np.concatenate([data_dict_ind['Wake-NREM'], data_dict_ind['Wake-REM']]))
		
		# if transient_frequency:
		# 	transition_labels = ['NREM-Wake', 'REM-Wake', 'NREM-REM', 'Wake-REM', 'REM-NREM', 'Wake-NREM', 'Unknown','Sleep-Wake','Wake-Sleep','Number of Transients', 'Animal Name']
		# 	transition_types, distance_from_transition = exp.ss_transition_per_transient(buffer_epochs = 2, num_std = num_std, 
		# 		FWHM_thresh = FWHM_thresh)
		# 	data_dict_ind = PKA.transient_property_per_sstype(transition_types, distance_from_transition, transition_labels)
		# 	animal = df.loc[df['Experiment Name'] == b]['Mouse ID'].item()
		# 	keys_list = [c for c in list(exp.SS_Transition_Frequency.keys()) if c in list(data_dict_ind.keys())]
		# 	for k in keys_list:
		# 		if exp.SS_Transition_Frequency[k] > 0:
		# 			val = len(data_dict_ind[k])/exp.SS_Transition_Frequency[k]
		# 		else:
		# 			val = 0
		# 		frequency_dict[animal][k].append(val)
		# 	val = (len(data_dict_ind['NREM-Wake'])+len(data_dict_ind['REM-Wake']))/(exp.SS_Transition_Frequency['NREM-Wake']+exp.SS_Transition_Frequency['REM-Wake'])
		# 	frequency_dict[animal]['Sleep-Wake'].append(val)

		# 	val = (len(data_dict_ind['Wake-NREM'])+len(data_dict_ind['Wake-REM']))/(exp.SS_Transition_Frequency['Wake-NREM']+exp.SS_Transition_Frequency['Wake-REM'])
		# 	frequency_dict[animal]['Wake-Sleep'].append(val)

	if (state_density == 'both') or (state_density == 'grouped'):
		color_dict_density = graph.SW_colordict('single state')
		x_vals = (x_statedensity-len(x_statedensity)/2)*exp.EpochLength
		density_group_fig, density_group_ax = plt.subplots(ncols = len(list(all_data_dict.keys())), figsize = [len(list(all_data_dict.keys()))*9,8])
		if len(list(all_data_dict.keys())) == 1:
			density_group_ax = [density_group_ax]
		density_group_fig, density_group_ax = graph.thick_axes(density_group_fig, density_group_ax)
		for i,g in enumerate(list(all_data_dict.keys())):
			wakebins_by_animal = []
			NREMbins_by_animal = []
			REMbins_by_animal = []
			all_wake_bins = np.vstack(all_data_dict[g]['Wake'])
			all_NREM_bins = np.vstack(all_data_dict[g]['NREM'])
			all_REM_bins = np.vstack(all_data_dict[g]['REM'])
			for a in np.unique(all_data_dict[g]['Animal Name']):
				these_exps, = np.where(np.asarray(all_data_dict[g]['Animal Name']) == a)
				total_transients = sum(np.asarray(all_data_dict[g]['Number of Transients'])[these_exps])
				wakebins_by_animal.append(np.sum(all_wake_bins[these_exps,:], axis = 0)/total_transients)
				NREMbins_by_animal.append(np.sum(all_NREM_bins[these_exps,:], axis = 0)/total_transients)
				REMbins_by_animal.append(np.sum(all_REM_bins[these_exps,:], axis = 0)/total_transients)
			avg_wake = np.mean(np.vstack(wakebins_by_animal), axis = 0)
			avg_wakesem = stats.sem(np.vstack(wakebins_by_animal), axis = 0)
			
			avg_NREM = np.mean(np.vstack(NREMbins_by_animal), axis = 0)
			avg_NREMsem = stats.sem(np.vstack(NREMbins_by_animal), axis = 0)
			
			avg_REM = np.mean(np.vstack(REMbins_by_animal), axis = 0)
			avg_REMsem = stats.sem(np.vstack(REMbins_by_animal), axis = 0)
			
			graph.linegraph_w_error(density_group_ax[i], x_vals, avg_wake, avg_wakesem, color = color_dict_density['Wake'], label = 'Wake')
			graph.linegraph_w_error(density_group_ax[i], x_vals, avg_NREM, avg_NREMsem, color = color_dict_density['NREM'], label = 'NREM')
			graph.linegraph_w_error(density_group_ax[i], x_vals, avg_REM, avg_REMsem, color = color_dict_density['REM'], label = 'REM')

			density_group_ax[i].axvline(0, color = 'k', linestyle = '--')
			density_group_ax[i].set_xlim([x_vals[0]-0.5, x_vals[-1]+0.5])
			graph.label_axes(density_group_ax[i], x = 'Time from Transient', y = 'Probability of Behavior State', 
			                 title = g + ' (n=' + str(len(np.unique(all_data_dict[g]['Animal Name'])))+' Animals)')
			density_group_ax[i].legend(bbox_to_anchor=(1.02, 1), fontsize = 15)
			density_group_ax[i].set_xlim([-(stateplot_window-16), (stateplot_window-16)])
			density_group_ax[i].set_ylim([0,1])
			density_group_fig.tight_layout()

		if savedir_dict:
			try:
				os.mkdir(savedir_dict['State Density Individual'])
			except FileExistsError:
				pass

			savefilename_statedensity_group = os.path.join(savedir_dict['State Density Individual'],savedir_dict['Grouped Filename'])
			density_group_fig.savefig(savefilename_statedensity_group)


	if (state_plot == 'both' or state_plot == 'grouped'):
		state_names = ['Wake', 'NREM', 'REM', 'Wake', 'Microarousal']
		graph.make_bigandbold(xticksize = 35, yticksize = 35, axeslabelsize = 40)
		stateplot_group_fig, stateplot_group_ax = plt.subplots(ncols = len(list(all_data_dict.keys())),figsize = (len(list(all_data_dict.keys()))*15,25))
		if len(list(all_data_dict.keys())) == 1:
			stateplot_group_ax  = [stateplot_group_ax]
		for gi, genotype in enumerate(list(all_data_dict.keys())):
			data = np.concatenate(all_data_dict[genotype]['SS Data'], axis = 0)
			all_distance = np.zeros(np.shape(data)[0]-1)
			for aa in np.arange(0, np.size(all_distance)):
				counter = 0
				idx = -1
				before_idx, = np.where(x_stateplot<=0)
				zero_idx, = np.where(x_stateplot==0)
				wake_idx, = np.where(np.logical_or(data[aa,before_idx] == 1, data[aa,before_idx] == 4))
				#         wake_idx, = np.where(data[aa,before_idx] == 3)
				distance = zero_idx-wake_idx
				try:
					while distance[idx] == counter:
						counter = counter+1
						idx = idx - 1
				except IndexError:
					pass
				all_distance[aa] = counter
			sorted_idx = np.argsort(all_distance)
			#     num_wake
			for i, ii in enumerate(sorted_idx):
				for state in [1,2,3,4,5]:
					cont_state = PKA.find_continuous(data[ii, :], [state])
					if len(cont_state)>0:
						for s in cont_state:
							x_pos = x_stateplot[s][0]
							w = ((x_stateplot[s[-1]]-x_stateplot[s][0]))+exp.EpochLength
							rect1 = patches.Rectangle((x_pos,i), w, 1, facecolor = color_dict[str(int(state))], 
							                          edgecolor = None, zorder = 0, label = state_names[state-1])
							stateplot_group_ax[gi].add_patch(rect1)
			stateplot_group_ax[gi].axvline(0, linestyle = '--', color = 'k', linewidth = 5)
			stateplot_group_ax[gi].set_xlim([x_stateplot[0], x_stateplot[-1]])
			stateplot_group_ax[gi].set_ylim([0, np.shape(data)[0]-1])
			stateplot_group_ax[gi].set_xlabel('Time from Transient Onset (s)')
			stateplot_group_ax[gi].set_xlim([-70, 100])
		    
			stateplot_group_ax[gi].set_ylabel('Transient ID')
			stateplot_group_ax[gi].tick_params(width=5)
			stateplot_group_ax[gi].set_xticks([-40,0,40,80])
			h,l = stateplot_group_ax[gi].get_legend_handles_labels()
			try:
				legend_idx = [l.index(label) for label in ['NREM', 'REM', 'Wake', 'Microarousal']]
			except ValueError:
				legend_idx = [l.index(label) for label in ['NREM', 'Wake']]
			these_labels = list(np.asarray(l)[legend_idx])
			these_handles = list(np.asarray(h)[legend_idx])
			plt.legend(labels = these_labels, handles = these_handles, bbox_to_anchor=(1, 0.8), fontsize = 35)
			for axis in ['bottom','left', 'right', 'top']:
				stateplot_group_ax[gi].spines[axis].set_linewidth(5)
			stateplot_group_ax[gi].set_title(genotype, fontsize = 50, fontweight= 'bold')
			stateplot_group_fig.tight_layout()
			if savedir_dict:
				try:
					os.mkdir(savedir_dict['State Plot Individual'])
				except FileExistsError:
					pass
				savefilename_stateplot_group = os.path.join(savedir_dict['State Plot Individual'],savedir_dict['Grouped Filename'])
				stateplot_group_fig.savefig(savefilename_stateplot_group)
	if timing_ditribution:
		color_dict_transitions = graph.SW_colordict('transitions')
		color_dict_transitions['Shuffled'] = '#929591'
		tt_labels = ['NREM-Wake', 'REM-Wake', 'NREM-REM', 'Wake-NREM']
		tt_labels_simple = ['Sleep-Wake','Wake-Sleep', 'NREM-REM']
		if shuffled:
			stats_dict_transition = {}
			stats_dict_transient = {}
			for l in np.concatenate([tt_labels, tt_labels_simple]):
				stats_dict_transition[l] = {list(timing_data_dict.keys())[0]:[], list(timing_data_dict.keys())[1]:[], 'p-val':[], 'ks-stat':[]}
				stats_dict_transient[l] = {list(timing_data_dict.keys())[0]:[], list(timing_data_dict.keys())[1]:[], 'p-val':[], 'ks-stat':[]}
		graph.make_bigandbold(xticksize = 20, yticksize = 20, axeslabelsize = 25)
		transition_timing_fig1, transition_timing_ax1 = plt.subplots(ncols = 4, figsize = (16, 6))
		transition_timing_fig2, transition_timing_ax2 = plt.subplots(ncols = 3, figsize = (12, 6))
		transient_timing_fig1, transient_timing_ax1 = plt.subplots(ncols = 4, figsize = (16, 6))
		transient_timing_fig2, transient_timing_ax2 = plt.subplots(ncols = 3, figsize = (12, 6))

		for g in list(timing_data_dict.keys()):
			print('Plotting '+g)
			for ii, l in enumerate(tt_labels):
				transition_timing_ax1[ii].set_xlabel('Time from Transition (s)')
				vals, bins = np.histogram(np.concatenate(timing_data_dict[g][l]), bins = np.linspace(-100, 5000, 2000))
				yvals = np.cumsum(vals)/np.sum(vals)
				yvals = np.insert(yvals, 0, 0)
				if g == 'Shuffled':
					c = color_dict_transitions[g]
				else:
					c = color_dict_transitions[l]
				if shuffled:
					stats_dict_transition[l][g] = yvals
				transition_timing_ax1[ii].plot(bins,yvals, color = c, label = g+'\n(n = '+ str(np.sum(vals))+')', linewidth = 4)
				graph.label_axes(transition_timing_ax1[ii], title = l, x = 'Time from\nTransition (s)')
				transition_timing_ax1[ii].set_xlim([-300, transition_timing_ax1[ii].get_xlim()[1]])
				transition_timing_ax1[ii].legend()
				transition_timing_ax1[ii].set_yticks([0,0.5,1])
			graph.label_axes(transition_timing_ax1[0], y = 'Fraction of Total')
			graph.remove_yticks(transition_timing_fig1, transition_timing_ax1)
			transition_timing_fig1, transition_timing_ax1 = graph.thick_axes(transition_timing_fig1, transition_timing_ax1)
			transition_timing_fig1.tight_layout()

			for ii, l in enumerate(tt_labels_simple):
				vals, bins = np.histogram(np.concatenate(timing_data_dict[g][l]), bins = np.linspace(-100, 5000, 2000))
				yvals = np.cumsum(vals)/np.sum(vals)
				yvals = np.insert(yvals, 0, 0)
				if g == 'Shuffled':
					c = color_dict_transitions[g]
				else:
					c = color_dict_transitions[l]
				if shuffled:
					stats_dict_transition[l][g] = yvals
				transition_timing_ax2[ii].plot(bins,yvals, color = c, label = g+'\n(n = '+ str(np.sum(vals))+')', linewidth = 4)
				graph.label_axes(transition_timing_ax2[ii], title = l, x = 'Time from\nTransition (s)')
				transition_timing_ax2[ii].set_xlim([-300, transition_timing_ax1[ii].get_xlim()[1]])
				transition_timing_ax2[ii].legend()
				transition_timing_ax2[ii].set_yticks([0,0.5,1])

				# print(stats.kstest(yvals_AKAR, yvals_shuff))
			graph.label_axes(transition_timing_ax2[0], y = 'Fraction of Total')
			graph.remove_yticks(transition_timing_fig2, transition_timing_ax2)
			transition_timing_fig2, transition_timing_ax2 = graph.thick_axes(transition_timing_fig2, transition_timing_ax2)
			graph.label_axes(transition_timing_ax2[0], y = 'Fraction of Total')
			transition_timing_fig2.tight_layout()

			for ii, l in enumerate(tt_labels):
				if g == 'Shuffled':
					c = color_dict_transitions[g]
				else:
					c = color_dict_transitions[l]
				data = -np.concatenate(timing_data_dict[g][l])
				vals, bins = np.histogram(data, bins = np.linspace(-5000, 600, 2000))
				yvals = np.cumsum(vals)/np.sum(vals)
				yvals = np.insert(yvals, 0, 0)
				if shuffled:
					stats_dict_transient[l][g] = yvals
				transient_timing_ax1[ii].plot(bins,yvals, color = c, label = g+'\n(n = '+ str(np.sum(vals))+')',linewidth = 4)
				transient_timing_ax1[ii].legend()
				graph.label_axes(transient_timing_ax1[ii], title = l, x = 'Time from\nTransient (s)')
			transient_timing_fig1, transient_timing_ax1 = graph.thick_axes(transient_timing_fig1, transient_timing_ax1)
			graph.label_axes(transient_timing_ax1[0], y = 'Fraction of Total')
			graph.remove_yticks(transition_timing_fig1, transition_timing_ax1)

			transient_timing_fig1.tight_layout()

			for ii, l in enumerate(tt_labels_simple):
				if g == 'Shuffled':
					c = color_dict_transitions[g]
				else:
					c = color_dict_transitions[l]
				data = -np.concatenate(timing_data_dict[g][l])
				vals, bins = np.histogram(data, bins = np.linspace(-5000, 600, 2000))
				yvals = np.cumsum(vals)/np.sum(vals)
				yvals = np.insert(yvals, 0, 0)
				if shuffled:
					stats_dict_transient[l][g] = yvals
				transient_timing_ax2[ii].plot(bins,yvals, color = c, label = g+'\n(n = '+ str(np.sum(vals))+')', linewidth = 4)
				transient_timing_ax2[ii].axvline(0, color = 'k', linestyle = '--', linewidth = 2)
				transient_timing_ax2[ii].legend()
				graph.label_axes(transient_timing_ax2[ii], title = l, x = 'Time from\nTransient (s)')
				# print(stats.kstest(yvals_AKAR, yvals_shuff))
			transient_timing_fig2, transient_timing_ax2 = graph.thick_axes(transient_timing_fig2, transient_timing_ax2)
			graph.label_axes(transient_timing_ax2[0], y = 'Fraction of Total')
			graph.remove_yticks(transition_timing_fig2, transition_timing_ax2)

			transient_timing_fig2.tight_layout()
		if shuffled:
			for l in np.concatenate([tt_labels, tt_labels_simple]):
				stats_dict_transition[l]['ks-stat'], stats_dict_transition[l]['p-val'] = stats.kstest(stats_dict_transition[l][list(timing_data_dict.keys())[0]],
					stats_dict_transition[l][list(timing_data_dict.keys())[1]])
				stats_dict_transient[l]['ks-stat'], stats_dict_transient[l]['p-val'] = stats.kstest(stats_dict_transient[l][list(timing_data_dict.keys())[0]],
					stats_dict_transient[l][list(timing_data_dict.keys())[1]])
			for i,l in enumerate(tt_labels):
				if stats_dict_transition[l]['p-val'] < 0.001:
					txt = '***'
				elif stats_dict_transition[l]['p-val'] < 0.01:
					txt = '**'
				elif stats_dict_transition[l]['p-val'] < 0.05:
					txt = '*'
				else:
					txt = ''
				x_text = transition_timing_ax1[i].get_xlim()[-1]/2
				y_text = transition_timing_ax1[i].get_ylim()[-1]-0.05
				transition_timing_ax1[i].text(x_text, y_text, txt, fontweight= 'bold', fontsize = 20)
			transition_timing_fig1.tight_layout()
			for i,l in enumerate(tt_labels_simple):
				if stats_dict_transition[l]['p-val'] < 0.001:
					txt = '***'
				elif stats_dict_transition[l]['p-val'] < 0.01:
					txt = '**'
				elif stats_dict_transition[l]['p-val'] < 0.05:
					txt = '*'
				else:
					txt = ''
				x_text = transition_timing_ax2[i].get_xlim()[-1]/2
				y_text = transition_timing_ax2[i].get_ylim()[-1]-0.05
				transition_timing_ax2[i].text(x_text, y_text, txt, fontweight= 'bold', fontsize = 20)

			for i,l in enumerate(tt_labels):
				if stats_dict_transient[l]['p-val'] < 0.001:
					txt = '***'
				elif stats_dict_transient[l]['p-val'] < 0.01:
					txt = '**'
				elif stats_dict_transient[l]['p-val'] < 0.05:
					txt = '*'
				else:
					txt = ''
				x_text = transient_timing_ax1[i].get_xlim()[0]*0.25
				y_text = transient_timing_ax1[i].get_ylim()[-1]-0.05
				transient_timing_ax1[i].text(x_text, y_text, txt, fontweight= 'bold', fontsize = 20)

			for i,l in enumerate(tt_labels_simple):
				if stats_dict_transient[l]['p-val'] < 0.001:
					txt = '***'
				elif stats_dict_transient[l]['p-val'] < 0.01:
					txt = '**'
				elif stats_dict_transient[l]['p-val'] < 0.05:
					txt = '*'
				else:
					txt = ''
				x_text = transient_timing_ax2[i].get_xlim()[0]*0.25
				y_text = transient_timing_ax2[i].get_ylim()[-1]-0.05
				transient_timing_ax2[i].text(x_text, y_text, txt, fontweight= 'bold', fontsize = 20)

		if savedir_dict:
			try:
				os.mkdir(savedir_dict['Timing Distribution'])
			except FileExistsError:
				pass
			savefilename_timing3 = os.path.join(savedir_dict['Timing Distribution'],'timing_from_transient_'+savedir_dict['Grouped Filename'])
			savefilename_timing4 = os.path.join(savedir_dict['Timing Distribution'],'timing_from_transient_simple_'+savedir_dict['Grouped Filename'])
			transient_timing_fig1.savefig(savefilename_timing3)
			transient_timing_fig2.savefig(savefilename_timing4)
			savefilename_timing1 = os.path.join(savedir_dict['Timing Distribution'],'timing_from_transition_'+savedir_dict['Grouped Filename'])
			savefilename_timing2 = os.path.join(savedir_dict['Timing Distribution'],'timing_from_transition_simple_'+savedir_dict['Grouped Filename'])
			transition_timing_fig1.savefig(savefilename_timing1)
			transition_timing_fig2.savefig(savefilename_timing2)
	# if transient_frequency:
	# 	color_dict_transitions = graph.SW_colordict('transitions')
	# 	freq_fig_specific, freq_ax_specific = plt.subplots(figsize = (10, 6))
	# 	freq_fig_simple, freq_ax_simple = plt.subplots(figsize = (10, 6))
	# 	specific_labels = ['NREM-Wake', 'REM-Wake', 'NREM-REM', 'REM-NREM', 'Wake-NREM']
	# 	simple_labels = ['Sleep-Wake', 'Wake-Sleep', 'NREM-REM']
	# 	colors_specific = [color_dict_transitions[k] for k in specific_labels]
	# 	colors_simple = [color_dict_transitions[k] for k in simple_labels]

	# 	yvals_specific = []
	# 	yvals_simple = []
	# 	animal_names = list(frequency_dict.keys())
	# 	for a in animal_names:
	# 		yvals_specific.append([np.median(frequency_dict[a][k]) for k in specific_labels])
	# 		yvals_simple.append([np.median(frequency_dict[a][k]) for k in simple_labels])

	# 	freq_fig_specific, freq_ax_specific = graph.grouped_bargraph(freq_fig_specific, freq_ax_specific, 
	# 		yvals_specific, colors_specific, x_labels = animal_names, legend_labels = specific_labels)
	# 	freq_fig_simple, freq_ax_simple = graph.grouped_bargraph(freq_fig_simple, freq_ax_simple, 
	# 		yvals_simple, colors_simple, x_labels = animal_names, legend_labels = simple_labels)

	# 	freq_fig_specific, freq_ax_specific = graph.thick_axes(freq_fig_specific, freq_ax_specific)
	# 	freq_fig_simple, freq_ax_simple = graph.thick_axes(freq_fig_simple, freq_ax_simple)
		
	# 	if savedir_dict:
	# 		try:
	# 			os.mkdir(savedir_dict['Transient Frequency'])
	# 		except FileExistsError:
	# 			pass
	# 		savefilename_frequency_simp = os.path.join(savedir_dict['Transient Frequency'],'simple_'+savedir_dict['Grouped Filename'])
	# 		savefilename_frequency_spec = os.path.join(savedir_dict['Transient Frequency'],'specific_'+savedir_dict['Grouped Filename'])
	# 		freq_fig_specific.savefig(savefilename_frequency_spec)
	# 		freq_fig_simple.savefig(savefilename_frequency_simp)

	try:
		return stats_dict_transient, stats_dict_transition
	except UnboundLocalError:
		pass

def power_correlations(df, filter_bounds, freq_dict, num_std  = 3, discrete_cutoff = False, binned = False, FWHM_thresh = False, 
	experimental_sensor = 'FLIM-AKAR', savedir = False, microarousals = True, fsd = 200, norm = True, 
	NFFT_sec = 10, scale_by_freq = True):
	graph.make_bigandbold()
	basenames = list(df['Experiment Name'])
	powers_dict = {'Mouse ID': [], 'Experiment Name': [], 'Powers': [], 'Transient Idx': [], 'No Transient Idx':[]}
	color_dict = graph.SW_colordict('single state')

	for b in basenames:
		print('Working on '+b+'...')
		powers_dict['Experiment Name'].append(b)
		powers_dict['Mouse ID'].append(df.loc[df['Experiment Name'] == b]['Mouse ID'].i(tem))
		rawdatdir = os.path.join('/Volumes/yaochen/Active/Lizzie/FLP_data/', b)
		if binned:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'binned_concat*.mat'))[0]
			fs = 0.25
		else:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'concat*.mat'))[0]
			fs = 1
		exp = PKA.FLiPExperiment(concat_filename, filter_bounds = filter_bounds, fs = fs, 
			experimental_sensor = experimental_sensor, microarousals = microarousals)
		print(concat_filename)
		# if exp.DeadTime:
		# 	use_flag = input(b+' might have deadtime issues. Do you want to still include it (y/n)?')
		# 	if use_flag == 'y':
		# 		print('Ok, keeping...')
		# 		pass
		# 	else:
		# 		print('Ok, skipping....')
		# 		continue
		epoch_dict = PKA.get_epochs(exp.SleepStates)
		transient_dict = exp.find_transients(num_std = num_std, discrete_cutoff = discrete_cutoff, FWHM_thresh = FWHM_thresh)
		temp_powers = PKA.get_all_powers(epoch_dict, fsd, freq_dict, exp, b, NFFT_sec = NFFT_sec, 
			scale_by_freq = scale_by_freq, norm = norm)
		powers_dict['Powers'].append(temp_powers)
		no_transient_idx, transient_idx = PKA.determine_transient_notransient(epoch_dict, 
			transient_dict[experimental_sensor], exp)
		powers_dict['No Transient Idx'].append(no_transient_idx)
		powers_dict['Transient Idx'].append(np.asarray(transient_idx))
	return powers_dict

def get_correlations(df, filter_bounds, num_std  = 3, discrete_cutoff = False, binned = False, FWHM_thresh = False, 
	experimental_sensor = 'FLIM-AKAR', savedir = False, microarousals = True, wake_len_corr = True, sleep_len_corr = True, 
	movevment_corr = True, movement_threshold = 2, power_plot = True, fsd = 200):
	graph.make_bigandbold()
	basenames = list(df['Experiment Name'])
	correlation_dict = {'Mouse ID': [], 'Experiment Name': [], 'Transient': {}, 'No Transient': {}}
	color_dict = graph.SW_colordict('single state')
	if wake_len_corr:
		correlation_dict['Transient']['Wake Length'] = []
		correlation_dict['No Transient']['Wake Length'] = []
	if sleep_len_corr:
		correlation_dict['Transient']['Sleep Length'] = []
		correlation_dict['No Transient']['Sleep Length'] = []
	if movevment_corr:
		correlation_dict['Transient']['Average Movement'] = []
		correlation_dict['No Transient']['Average Movement'] = []
		correlation_dict['Transient']['Fraction Movement'] = []
		correlation_dict['No Transient']['Fraction Movement'] = []

	for b in basenames:
		print('Working on '+b+'...')
		correlation_dict['Experiment Name'].append(b)
		correlation_dict['Mouse ID'].append(df.loc[df['Experiment Name'] == b]['Mouse ID'].item())
		rawdatdir = os.path.join('/Volumes/yaochen/Active/Lizzie/FLP_data/', b)
		if binned:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'binned_concat*.mat'))[0]
			exp = PKA.FLiPExperiment(concat_filename, filter_bounds = filter_bounds, fs = 0.25, 
				experimental_sensor = experimental_sensor, microarousals = microarousals)
		else:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'concat*.mat'))[0]
			exp = PKA.FLiPExperiment(concat_filename, filter_bounds = filter_bounds, 
				experimental_sensor = experimental_sensor, microarousals = microarousals)
		# if exp.DeadTime:
		# 	use_flag = input(b+' might have deadtime issues. Do you want to still include it (y/n)?')
		# 	if use_flag == 'y':
		# 		print('Ok, keeping...')
		# 		pass
		# 	else:
		# 		print('Ok, skipping....')
		# 		continue
		epoch_dict = PKA.get_epochs(exp.SleepStates)
		transient_dict = exp.find_transients(num_std = num_std, discrete_cutoff = discrete_cutoff, FWHM_thresh = FWHM_thresh)
		no_transient_idx, transient_idx = PKA.determine_transient_notransient(epoch_dict, transient_dict, exp)

		ss_nomicro = deepcopy(exp.SleepStates)
		ss_nomicro[np.where(ss_nomicro == 5)[0]] = 2

		if wake_len_corr:
			wake_epoch_lengths = [len(e)*exp.EpochLength for e in epoch_dict['Wake']]
			correlation_dict['Transient']['Wake Length'].append(np.asarray(wake_epoch_lengths)[transient_idx])
			correlation_dict['No Transient']['Wake Length'].append(np.asarray(wake_epoch_lengths)[no_transient_idx])

		if sleep_len_corr:
			sleep_epoch_lengths = PKA.get_previous_sleep_epoch(epoch_dict['Wake'], ss_nomicro, exp.EpochLength)
			correlation_dict['Transient']['Sleep Length'].append(np.asarray(sleep_epoch_lengths)[transient_idx])
			correlation_dict['No Transient']['Sleep Length'].append(np.asarray(sleep_epoch_lengths)[no_transient_idx])

		if movevment_corr:
			try:
				v = np.load(os.path.join(rawdatdir,b+'_extracted_data/','velocity_vector.npy'))
			except FileNotFoundError:
				movement_df = pd.read_pickle(os.path.join(rawdatdir,b+'_extracted_data/', 'All_movement.pkl'))
				v = movement_processing(movement_df)
			avg_movement = []
			frac_movement = []
			for w in epoch_dict['Wake']:
				time_seg = exp.SSTime[w]
				movement_idx, = np.where(np.logical_and(v[1]>time_seg[0], v[1]<time_seg[-1]))
				movement_seg = v[0][movement_idx]
				avg_movement.append(np.mean(movement_seg))
				thresholded_movement, = np.where(movement_seg>movement_threshold)
				frac_movement.append(len(thresholded_movement)/len(movement_seg))
			correlation_dict['Transient']['Average Movement'].append(np.asarray(avg_movement)[transient_idx])
			correlation_dict['No Transient']['Average Movement'].append(np.asarray(avg_movement)[no_transient_idx])
			correlation_dict['Transient']['Fraction Movement'].append(np.asarray(frac_movement)[transient_idx])
			correlation_dict['No Transient']['Fraction Movement'].append(np.asarray(frac_movement)[no_transient_idx])

	if wake_len_corr:
		data_wake = [np.concatenate(correlation_dict['Transient']['Wake Length']),np.concatenate(correlation_dict['No Transient']['Wake Length'])]
		wakelen_fig, wakelen_ax = plt.subplots(figsize = [7,5])
		wakelen_fig, wakelen_ax = graph.thick_axes(wakelen_fig, wakelen_ax)
		colors = [color_dict['Wake'], 'k']
		wakelen_fig, wakelen_ax = graph.violin_plot(wakelen_fig, wakelen_ax, data_wake, [1,2], colors, showmeans = False, showmedians = True, 
			xlabels = ['Transient', 'No Transient'])
		wakelen_ax.set_yscale('log')
		graph.label_axes(wakelen_ax, y = 'Wake Length (s)')
		wakelen_fig.tight_layout()
		if savedir:
			wakelen_fig.savefig(savedir+'_wakelen_corr.png')

	if sleep_len_corr:
		data_sleep = [np.concatenate(correlation_dict['Transient']['Sleep Length']), np.concatenate(correlation_dict['No Transient']['Sleep Length'])]
		sleeplen_fig, sleeplen_ax = plt.subplots(figsize = [7,5])
		sleeplen_fig, sleeplen_ax = graph.thick_axes(sleeplen_fig, sleeplen_ax)
		colors = [color_dict['NREM'], 'k']
		sleeplen_fig,sleeplen_ax = graph.violin_plot(sleeplen_fig, sleeplen_ax, data_sleep, [1,2], colors, showmeans = False, showmedians = True, 
			xlabels = ['Transient', 'No Transient'])
		sleeplen_ax.set_yscale('log')
		graph.label_axes(sleeplen_ax, y = 'Sleep Length (s)')
		sleeplen_fig.tight_layout()
		if savedir:
			sleeplen_fig.savefig(savedir+'_sleeplen_corr.png')

	if movevment_corr:
		data_avgmove = [np.concatenate(correlation_dict['Transient']['Average Movement']),np.concatenate(correlation_dict['No Transient']['Average Movement'])]
		data_fracmove = [np.concatenate(correlation_dict['Transient']['Fraction Movement']),np.concatenate(correlation_dict['No Transient']['Fraction Movement'])]
		move_fig, (avgmove_ax,fracmove_ax) = plt.subplots(ncols = 2, figsize = [14,5])
		move_fig, (avgmove_ax,fracmove_ax) = graph.thick_axes(move_fig, (avgmove_ax,fracmove_ax))
		colors = ['#ff9408', 'k']
		move_fig,avgmove_ax = graph.violin_plot(move_fig, avgmove_ax, data_avgmove, [1,2], colors, showmeans = False, showmedians = True, 
			xlabels = ['Transient', 'No Transient'])
		graph.label_axes(avgmove_ax, y = 'Average Velocity')
		move_fig,avgmove_ax = graph.violin_plot(move_fig, fracmove_ax, data_fracmove, [1,2], colors, showmeans = False, showmedians = True, 
			xlabels = ['Transient', 'No Transient'])
		graph.label_axes(fracmove_ax, y = 'Fraction of Wake Time\nwith Movement')
		move_fig.tight_layout()
		if savedir:
			move_fig.savefig(savedir+'_move_corr.png')
	return correlation_dict


def make_summary_table(df, filter_bounds, num_std  = 3, discrete_cutoff = False,
	binned = False, FWHM_thresh = False, experimental_sensor = 'FLIM-AKAR', 
	microarousals = False, savedir = False):

	property_dict = get_transient_properties(df, filter_bounds = filter_bounds, num_std = num_std, discrete_cutoff = discrete_cutoff, binned = binned, FWHM_thresh= FWHM_thresh,
		experimental_sensor = experimental_sensor, microarousals = microarousals)
	property_dict['Transient Frequency'] = []
	state_labels = {'Wake':1, 'NREM':2, 'REM':3, 'Microarousal':5}
	property_dict_concat = {}
	tt_indices = {}

	for k in ['NREM-Wake', 'REM-Wake', 'NREM-REM', 'Wake-NREM', 'Sleep-Wake', 'Wake-Sleep', 'Microarousal', 'REM-NREM']:
		tt_indices[k] = []
	for i in np.arange(0, len(property_dict['Experiment Name'])):
		# transition_options = np.unique(property_dict['Transition Types'][i], axis = 0)
		property_dict['Transient Frequency'].append({})
		for t in ['NREM-Wake', 'REM-Wake', 'NREM-REM', 'Wake-NREM', 'Microarousal', 'REM-NREM']:
			if t == 'Microarousal':
				tt_indices['Microarousal'].append(np.where([any(x == 5) for x in property_dict['Transition Types'][i]])[0])
			else:
				num_code = [state_labels[t[:int(t.find('-'))]], state_labels[t[int(t.find('-'))+1:]]]
				tt_indices[t].append(np.where([np.array_equal(x, num_code) for x in property_dict['Transition Types'][i]])[0])
				if property_dict['SS Frequency'][i][t]>0:
					property_dict['Transient Frequency'][i][t] = np.size(tt_indices[t][i])/property_dict['SS Frequency'][i][t]
				else:
					property_dict['Transient Frequency'][i][t] = 0
		property_dict['Transient Frequency'][i]['Sleep-Wake'] = (np.size(tt_indices['NREM-Wake'][i])+np.size(tt_indices['REM-Wake'][i]))/(property_dict['SS Frequency'][i]['NREM-Wake']+property_dict['SS Frequency'][i]['REM-Wake'])
		property_dict['Transient Frequency'][i]['Wake-Sleep'] = property_dict['Transient Frequency'][i]['Wake-NREM']
		property_dict['Transient Frequency'][i]['Microarousal'] =  np.size(tt_indices['Microarousal'][i])/property_dict['SS Frequency'][i]['Microarousals']
		tt_indices['Sleep-Wake'].append(np.concatenate([tt_indices['NREM-Wake'][i], tt_indices['REM-Wake'][i]]))
		tt_indices['Wake-Sleep'].append(tt_indices['Wake-NREM'][i])
		assert all(len(item) == i+1 for item in list(tt_indices.values()))
	
	for k in list(property_dict.keys()):
		try:
			property_dict_concat[k] = np.concatenate(property_dict[k])
		except ValueError:
			property_dict_concat[k] = property_dict[k]

	property_df = pd.DataFrame(columns = list(tt_indices.keys()), index = ['Amplitude', 'Time from Transition', 'Transient Duration'])
	for c in property_df.columns:
		for r in property_df.index:
			these_vals = property_dict_concat[r][np.concatenate(tt_indices[c])]
			property_df[c][r] = np.median(these_vals)

	property_df = pd.concat([property_df,pd.DataFrame(columns = list(tt_indices.keys()), index = ['Transient Frequency'])])
	for c in property_df.columns:
		these_vals = [property_dict['Transient Frequency'][i][c] for i in np.arange(len(property_dict['Transient Frequency']))]
		property_df[c]['Transient Frequency'] = np.median(these_vals)
	property_df = property_df.astype(float)

	simple_df = property_df[['Wake-Sleep', 'Sleep-Wake', 'Microarousal']]
	simple_df = simple_df.round(3)
	
	fig, ax = plt.subplots()
	fig.patch.set_visible(False)
	ax.axis('off')
	ax.axis('tight')
	table = ax.table(cellText = simple_df.values, rowLabels=simple_df.index, 
		colLabels=simple_df.columns, loc='center')
	table.auto_set_font_size(False)
	table.scale(1, 2)
	table.set_fontsize(12)
	ax.set_title('\n\n'+df['Genotype'].iloc[0], weight = 'bold', size = 16, loc = 'left')
	fig.tight_layout()

	if savedir:
		fig.savefig(savedir)

def circadian_frequency(df, circadian_bins, filter_bounds, num_std  = 3, discrete_cutoff = False, binned = False, 
	FWHM_thresh = False, experimental_sensor = 'FLIM-AKAR', savename = False, microarousals = True):
	basenames = list(df['Experiment Name'])
	color_dict = graph.SW_colordict('single state')
	data = {}
	for c in circadian_bins[:-1]:
		data['ZT'+str(c)] = []
	frequency_dict = {'Mouse ID':[], 'Experiment Name':[], 'Data':deepcopy(data)}
	num_transitions_dict = {'Mouse ID':[], 'Experiment Name':[], 'Data':deepcopy(data)}

	for b in basenames:
		frequency_dict['Mouse ID'].append(df.loc[df['Experiment Name'] == b]['Mouse ID'].item())
		frequency_dict['Experiment Name'].append(b)
		num_transitions_dict['Mouse ID'].append(df.loc[df['Experiment Name'] == b]['Mouse ID'].item())
		num_transitions_dict['Experiment Name'].append(b)

		print('Working on '+b+'...')
		rawdatdir = os.path.join('/Volumes/yaochen/Active/Lizzie/FLP_data/', b)
		if binned:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'binned_concat*.mat'))[0]
			exp = PKA.FLiPExperiment(concat_filename, filter_bounds = filter_bounds, fs = 0.25, 
				experimental_sensor = experimental_sensor, microarousals = microarousals)
		else:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'concat*.mat'))[0]
			exp = PKA.FLiPExperiment(concat_filename, filter_bounds = filter_bounds, 
				experimental_sensor = experimental_sensor, microarousals = microarousals)
		transient_dict = exp.find_transients(num_std = num_std, discrete_cutoff = discrete_cutoff, FWHM_thresh = FWHM_thresh)
		for i in np.arange(np.size(circadian_bins)-1):
			indices, = np.where(np.logical_and(exp.ZeitTime >= circadian_bins[i], exp.ZeitTime < circadian_bins[i+1]))
			freq, num_trans = PKA.transient_frequency_by_transition(exp, transient_dict[experimental_sensor], 
				indices, microarousals = True)
			frequency_dict['Data']['ZT'+str(circadian_bins[i])].append(freq)
			num_transitions_dict['Data']['ZT'+str(circadian_bins[i])].append(num_trans)
	graph.make_bigandbold()
	circadian_fig, circadian_ax = plt.subplots(figsize = [8, 4])
	circadian_fig, circadian_ax = graph.thick_axes(circadian_fig, circadian_ax)
	num_wake_fig, num_wake_ax = plt.subplots(figsize = [8, 4])
	num_wake_fig, num_wake_ax = graph.thick_axes(num_wake_fig, num_wake_ax)

	for x,k in enumerate(frequency_dict['Data'].keys()):
		circadian_bar_vals = [frequency_dict['Data'][k][i]['Sleep-Wake'] for i in range(len(basenames))]
		num_wake_bar_vals = [num_transitions_dict['Data'][k][i]['Sleep-Wake'] for i in range(len(basenames))]
		circadian_ax.bar(x, np.mean(circadian_bar_vals), color = '#665fd1', zorder = 0, align = 'edge', 
			width = 1, edgecolor = 'k')
		num_wake_ax.bar(x, np.mean(num_wake_bar_vals), color = color_dict['Wake'], zorder = 0, align = 'edge', 
			width = 1, edgecolor = 'k')

	animals = np.unique(frequency_dict['Mouse ID'])
	markers = graph.pick_scatter_markers(len(animals))
	x_vals = np.arange(0.5, len(list(frequency_dict['Data'].keys())))
	for i in range(len(frequency_dict['Experiment Name'])):
		circadian_data_points = [frequency_dict['Data'][ZT][i]['Sleep-Wake'] for ZT in list(frequency_dict['Data'].keys())]
		num_wake_data_points = [num_transitions_dict['Data'][ZT][i]['Sleep-Wake'] for ZT in list(num_transitions_dict['Data'].keys())]
		marker_idx, = np.where(animals == frequency_dict['Mouse ID'][i])
		circadian_ax.plot(x_vals, circadian_data_points, color = 'k', zorder = 10, markersize = 8, 
			marker = markers[int(marker_idx)], label = frequency_dict['Mouse ID'][i])
		num_wake_ax.plot(x_vals, num_wake_data_points, color = 'k', zorder = 10, markersize = 8, 
			marker = markers[int(marker_idx)], label = frequency_dict['Mouse ID'][i])

	h,l = circadian_ax.get_legend_handles_labels()
	these_labels,h_idx = np.unique(l, return_index = True)
	these_handles = np.asarray(h)[h_idx]
	circadian_ax.legend(labels = list(these_labels), handles = list(these_handles))
	num_wake_ax.legend(labels = list(these_labels), handles = list(these_handles))

	circadian_ax.set_xticks(np.arange(0, len(list(frequency_dict['Data'].keys()))))
	circadian_ax.set_xticklabels(list(frequency_dict['Data'].keys()))
	circadian_ax = graph.label_axes(circadian_ax, y = 'Frequency', x = 'Zeitgeiber Time')
	circadian_fig.tight_layout()

	num_wake_ax.set_xticks(np.arange(0, len(list(frequency_dict['Data'].keys()))))
	num_wake_ax.set_xticklabels(list(frequency_dict['Data'].keys()))
	num_wake_ax = graph.label_axes(num_wake_ax, y = 'Number of Sleep-Wake\nTransitions', x = 'Zeitgeiber Time')
	num_wake_fig.tight_layout()


	if savename:
		circadian_fig.savefig(savename+'_frequency.png')
		num_wake_fig.savefig(savename+'_num_sw.png')
	return frequency_dict, circadian_fig, circadian_ax

def transition_triggered_lifetime(df,filter_bounds, data_folder = '/Volumes/yaochen/Active/Lizzie/FLP_data/', window = [30,30], num_std  = 3, discrete_cutoff = False, binned = False, 
	FWHM_thresh = False, experimental_sensor = 'FLIM-AKAR', savename = False, microarousals = True, 
	these_transitions = ['NREM-Wake','REM-NREM','Wake-REM','REM-Wake','NREM-REM','Wake-NREM','Microarousals','Sleep-Wake','Wake-Sleep'],
	intensity = False, dynamic_window = 75, diff_wake = False):
	if isinstance(df, pd.DataFrame):
		basenames = list(df['Experiment Name'])
	else:
		basenames = df
	color_dict = graph.SW_colordict('transitions')
	if intensity:
		lifetime_dict = {'Mouse ID':[], 'Experiment Name':[], 'Lifetime':{}, 'Time':{}, 'Intensity':{}}
	else:
		lifetime_dict = {'Mouse ID':[], 'Experiment Name':[], 'Lifetime':{}, 'Time':{}}
	for k in these_transitions:
		lifetime_dict['Lifetime'][k] = []
		lifetime_dict['Time'][k] = []
		if intensity:
			lifetime_dict['Intensity'][k] = []
	graph.make_bigandbold(axeslabelsize = 22)
	for b in basenames:
		# lifetime_dict['Mouse ID'].append(df.loc[df['Experiment Name'] == b]['Mouse ID'].item())
		lifetime_dict['Experiment Name'].append(b)

		print('Working on '+b+'...')
		rawdatdir = os.path.join(data_folder, b)
		if binned:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'binned_concat*.mat'))[0]
			fs = 0.25
		else:
			concat_filename = glob.glob(os.path.join(rawdatdir, 'concat*.mat'))[0]
			fs = 1

		exp = PKA.FLiPExperiment(concat_filename, filter_bounds = filter_bounds, 
			experimental_sensor = experimental_sensor, microarousals = microarousals, fs = fs)

		print(concat_filename)
		exp = RemoveFirst3Hours(exp)

		# if exp.DeadTime:
		# 	use_flag = input(b+' might have deadtime issues. Do you want to still include it (y/n)?')
		# 	if use_flag == 'y':
		# 		print('Ok, keeping...')
		# 		pass
		# 	else:
		# 		print('Ok, skipping....')
		# 		continue

		transition_dict = exp.transition_timestamps(microarousals = microarousals, diff_wake = diff_wake)
		epoch_dict = PKA.get_epochs(exp.SleepStates, diff_wake = diff_wake)
		for k in these_transitions:
			if 'Microarousal' in k:
				window[1] = 100
			else:
				second_state = k[k.find('-')+1:]
				behavior_lens = np.asarray([PKA.find_behavior_length(epoch_dict, second_state, t, exp.SleepStates, exp.SSTime, exp.EpochLength) for t in transition_dict['Timestamps'][k]])
				if dynamic_window:
					window[1] = np.percentile(behavior_lens[:,0], dynamic_window)
			win_len = int(sum(window))
			stacked_lifetime = np.empty([len(transition_dict['Timestamps'][k]), int(win_len*fs)])
			stacked_lifetime[:] = np.nan
			if intensity:
				stacked_intensity = np.empty([len(transition_dict['Timestamps'][k]), int(win_len*fs)])
				stacked_intensity[:] = np.nan
			stacked_time = np.empty([len(transition_dict['Timestamps'][k]), int(win_len*fs)])
			stacked_time[:] = np.nan
			for i,t in enumerate(transition_dict['Timestamps'][k]):
				if behavior_lens[i][0] < win_len:
					idx, = np.where(np.logical_and(exp.Time>=t-window[0], exp.Time<behavior_lens[i][1]))
				else:
					idx, = np.where(np.logical_and(exp.Time>=t-window[0], exp.Time<t+window[1]))
				if len(idx) > win_len*fs:
					idx = idx[:int(win_len*fs)]
				normIdx = np.where(exp.Time>=t)[0][0]
				stacked_lifetime[i, 0:len(idx)] = exp.Filt[idx]-exp.Filt[normIdx]
				if intensity:
					stacked_intensity[i, 0:len(idx)] = exp.PhotonCount[idx]-exp.PhotonCount[normIdx]
				stacked_time[i, 0:len(idx)] = exp.Time[idx]-t
			lifetime_dict['Lifetime'][k].append(stacked_lifetime)
			lifetime_dict['Time'][k].append(stacked_time)
			if intensity:
				lifetime_dict['Intensity'][k].append(stacked_intensity)
		write_lifetime_dict_to_csv(b, data_folder, these_transitions, lifetime_dict)

	lifetime_fig, lifetime_ax = plt.subplots(nrows = 1, ncols = len(these_transitions), figsize = [4*len(these_transitions), 6])
	y_max_lft = np.zeros(len(these_transitions))
	y_min_lft = np.zeros(len(these_transitions))
	if intensity:
		intensity_fig, intensity_ax = plt.subplots(nrows = 1, ncols = len(these_transitions), figsize = [4*len(these_transitions), 6])
		y_max_int = np.zeros(len(these_transitions))
		y_min_int = np.zeros(len(these_transitions))
	# loc_lft = plticker.MultipleLocator(base=(y_max_lft-y_min_lft)/5)
	# loc_int = plticker.MultipleLocator(base=(y_max_int-y_min_int)/5)
	# loc_int = plticker.MultipleLocator(base=100000)
	for ii, s in enumerate(these_transitions):
		lengths = [np.shape(lifetime_dict['Time'][s][i])[1] for i in range(len(lifetime_dict['Time'][s]))]
		for aa in range(len(lifetime_dict['Time'][s])):
			overflow = lengths[aa]-min(lengths)
			delete_these = np.arange(lengths[aa]-overflow, lengths[aa])
			lifetime_dict['Time'][s][aa] = np.delete(lifetime_dict['Time'][s][aa],delete_these, axis = 1)
			lifetime_dict['Lifetime'][s][aa] = np.delete(lifetime_dict['Lifetime'][s][aa],delete_these, axis = 1)
			if intensity:
				lifetime_dict['Intensity'][s][aa] = np.delete(lifetime_dict['Intensity'][s][aa],delete_these, axis = 1)
		these_times = np.concatenate(lifetime_dict['Time'][s], axis = 0)
		x = np.nanmean(these_times, axis = 0)
		these_lifetimes = np.concatenate(lifetime_dict['Lifetime'][s], axis = 0)
		y_lft = np.nanmedian(these_lifetimes, axis = 0)
		sem_lft = stats.sem(these_lifetimes, axis = 0, nan_policy='omit')
		if intensity:
			these_intensities = np.concatenate(lifetime_dict['Intensity'][s], axis = 0)
			y_int = np.nanmedian(these_intensities, axis = 0)
			sem_int = stats.sem(these_intensities, axis = 0, nan_policy='omit')
			intensity_ax[ii] = graph.linegraph_w_error(intensity_ax[ii], x, y_int, sem_int, color = color_dict[s], label = s, linewidth = 3)
			intensity_ax[ii].axvline(0, linestyle = '--', linewidth = 2, color = 'k')
			intensity_ax[ii] = graph.label_axes(intensity_ax[ii], x = 'Time from\nTransition (s)', title = s)
			y_min_int[ii] = intensity_ax[ii].get_ylim()[0]
			y_max_int[ii] = intensity_ax[ii].get_ylim()[1]
			# intensity_ax[ii].yaxis.set_major_locator(loc_int)
		lifetime_ax[ii] = graph.linegraph_w_error(lifetime_ax[ii], x, y_lft, sem_lft, color = color_dict[s], label = s, linewidth = 3)
		lifetime_ax[ii].axvline(0, linestyle = '--', linewidth = 2, color = 'k')
		lifetime_ax[ii] = graph.label_axes(lifetime_ax[ii], x = 'Time from\nTransition (s)', title = s, title_fontsize = 20)
		y_min_lft[ii] = lifetime_ax[ii].get_ylim()[0]
		y_max_lft[ii] = lifetime_ax[ii].get_ylim()[1]

		if ii > 0:
			lifetime_ax[ii].set_yticklabels([])
			if intensity:
				intensity_ax[ii].set_yticklabels([])

	for ax in lifetime_ax:
		ax.set_ylim([min(y_min_lft)-0.01, max(y_max_lft)+0.01])
		base = np.round((max(y_max_lft)+0.01-min(y_min_lft)-0.01)/5, 2)
		loc_lft = plticker.MultipleLocator(base=base)
		ax.yaxis.set_major_locator(loc_lft)
		

	if intensity:
		for ax in intensity_ax:
			ax.set_ylim([min(y_min_int)-10000, max(y_max_int)+10000])

	lifetime_ax[0] = graph.label_axes(lifetime_ax[0], y = r'$\Delta$'+ ' Lifetime (ns)')
	lifetime_fig.suptitle(experimental_sensor + ' ' + basenames[0], fontsize = 30, fontweight = 'bold')
	lifetime_fig, lifetime_ax = graph.thick_axes(lifetime_fig, lifetime_ax)
	lifetime_fig.tight_layout()
	if intensity:
		intensity_ax[0] = graph.label_axes(intensity_ax[0], y = r'$\Delta$'+ 'Photon Count')
		intensity_fig.suptitle(experimental_sensor + ' ' + basenames[0], fontsize = 30, fontweight = 'bold')
		intensity_fig, intensity_ax = graph.thick_axes(intensity_fig, intensity_ax)
		intensity_fig.tight_layout()

	if savename:
		lft_savename = rawdatdir + '/lifetime_' + savename
		lifetime_fig.savefig(lft_savename)
		if intensity:
			# idx = savename.find(os.path.splitext(savename)[1])
			int_savename = rawdatdir + '/intensity_' + savename
			intensity_fig.savefig(int_savename)

def write_lifetime_dict_to_csv(basename, data_folder, these_transitions, lifetime_dict):
	for transition in these_transitions:
		csv_filename_lifetime = data_folder + basename + '/' + transition + '_lifetime.csv'
		csv_filename_time = data_folder + basename + '/' + transition + '_time.csv'
		csv_filename_intensity = data_folder + basename + '/' + transition + '_intensity.csv'

		Lifetime_list = pd.DataFrame(lifetime_dict['Lifetime'][transition][0])
		time_list = pd.DataFrame(lifetime_dict['Time'][transition][0])
		intensity_list = pd.DataFrame(lifetime_dict['Intensity'][transition][0])

		Lifetime_list.to_csv(csv_filename_lifetime, index = False)
		time_list.to_csv(csv_filename_time, index = False)
		intensity_list.to_csv(csv_filename_intensity, index = False)

def RemoveFirst3Hours(exp):
	RemainedIndex, = np.where(exp.ZeitTime > 15)
	exp.ZeitTime = exp.ZeitTime[RemainedIndex]
	exp.PhotonCount = exp.PhotonCount[RemainedIndex]
	exp.Lifetime = exp.Lifetime[RemainedIndex]
	exp.GaussianWidth = exp.GaussianWidth[RemainedIndex]
	exp.ChiSquare = exp.ChiSquare[RemainedIndex]
	exp.DeltaPeakTime = exp.DeltaPeakTime[RemainedIndex]
	exp.Filt = exp.Filt[RemainedIndex]
	exp.Shuff = exp.Shuff[RemainedIndex]

	SSTime_remainedIndex, = np.where((exp.SSTime >= exp.Time[RemainedIndex[0]]) & (exp.SSTime <= exp.Time[RemainedIndex[-1]]))
	exp.Time = exp.Time[RemainedIndex]
	exp.SSTime = exp.SSTime[SSTime_remainedIndex]
	exp.SleepStates = exp.SleepStates[SSTime_remainedIndex]

	assert np.size(exp.ZeitTime) == np.size(exp.PhotonCount) == np.size(exp.Lifetime) == np.size(exp.GaussianWidth) == np.size(exp.ChiSquare) == np.size(exp.DeltaPeakTime) == np.size(exp.Shuff) == np.size(exp.Filt) == np.size(exp.Time)

	assert np.size(exp.SSTime) == np.size(exp.SleepStates)

	return exp
	

	
	




