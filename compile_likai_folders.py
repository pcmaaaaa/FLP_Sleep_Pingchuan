import numpy as np
import os
from shutil import copy2
from neuroscience_sleep_scoring import SWS_utils
import sys
from scipy import signal


def compile_data(basename, acq_range):
	acq_dir = os.path.join('/Volumes/yaochen/Active/Lizzie/FLP_data/Data_for_likai/', 
		basename)
	data_dir = os.path.join('/Volumes/yaochen/Active/Lizzie/FLP_data', basename)
	extracted_dir = os.path.join('/Volumes/yaochen/Active/Lizzie/FLP_data', basename, basename+'_extracted_data')
	try:
		os.mkdir(acq_dir)
	except FileExistsError:
		print(acq_dir + ' already exists!')
	for acq in acq_range:
		#Make acquisition directory
		this_dir = os.path.join(acq_dir,'Acq'+str(acq))
		try:
			os.mkdir(this_dir)
		except FileExistsError:
			print(this_dir + ' already exists!')
		#copy analysis file
		copy_analysis_files(this_dir, data_dir, acq)
		#copy AD files
		copy_EEG_files(this_dir, data_dir, acq)
		#copy sleep data
		copy_state_array(this_dir, extracted_dir, acq)
		#get theta and delta bands
		get_bands(this_dir, extracted_dir, acq)
		#copy figure
		copy_fig(this_dir, extracted_dir, acq)
	f = os.path.join(data_dir, 'autonotes.mat')
	dst = os.path.join(acq_dir,'autonotes.mat')
	copy2(f,dst)


def copy_analysis_files(this_dir, data_dir, acq):
	f = os.path.join(data_dir, 'Acq'+str(acq)+'_analysis_binned.mat')
	dst = os.path.join(this_dir,'Acq'+str(acq)+'_analysis_binned.mat')
	copy2(f,dst)
	f = os.path.join(data_dir, 'Acq'+str(acq)+'_analysis.mat')
	dst = os.path.join(this_dir,'Acq'+str(acq)+'_analysis.mat')
	copy2(f,dst)

def copy_EEG_files(this_dir, data_dir, acq):
	for ch in [0,2,3]:
		f = os.path.join(data_dir, 'AD' + str(ch) + '_' + str(acq) + '.mat')
		dst = os.path.join(this_dir, 'AD' + str(ch) + '_' + str(acq) + '.mat')
		copy2(f,dst)
def copy_state_array(this_dir, extracted_dir, acq):
	f = os.path.join(extracted_dir, 'StatesAcq'+str(acq)+'_hr0.npy')
	dst = os.path.join(this_dir, 'StatesAcq'+str(acq)+'_hr0.npy')
	copy2(f,dst)
def copy_fig(this_dir, extracted_dir, acq):
	f = os.path.join(extracted_dir, 'FLiPEEGEMGStatefig_acqn' + str(acq) + '.png')
	dst = os.path.join(this_dir, 'FLiPEEGEMGStatefig_acqn' + str(acq) + '.png')
	copy2(f,dst)

def get_bands(this_dir, extracted_dir, acq):
	this_eeg = np.load(os.path.join(extracted_dir, 
		'downsampEEG_Acq' + str(acq) + '_hr0.npy'))
	this_eeg_reshape = np.reshape(this_eeg, (-1,800))
	minfreq = 0.5 # min freq in Hz
	maxfreq = 35 # max freq in Hz
	f,Pxx = signal.welch(this_eeg_reshape, fs = 200, nperseg = 800, noverlap = 400)
	delta_band = np.sum(Pxx[:,np.where(np.logical_and(f>=1,f<=4))[0]], axis = 1)
	theta_band = np.sum(Pxx[:,np.where(np.logical_and(f>=5,f<=8))[0]], axis = 1)
	np.save(os.path.join(this_dir, 'Acq' + str(acq) + '_delta'), delta_band/np.mean(delta_band))
	np.save(os.path.join(this_dir, 'Acq' + str(acq) + '_theta'), theta_band/np.mean(theta_band))

if __name__ == "__main__":
	args = sys.argv
	basename = str(args[1])
	acq_start = int(args[2])
	acq_end = int(args[3])
	acq_range = np.arange(acq_start, acq_end+1)
	compile_data(basename, acq_range)
