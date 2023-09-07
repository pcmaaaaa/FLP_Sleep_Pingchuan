import PKA_Sleep as PKA
import pandas as pd
import numpy as np
import glob
import os
import time
from neuroscience_sleep_scoring import SWS_utils
from datetime import datetime, timedelta
import Graphing_Utils as graph
import matplotlib.pyplot as plt
basename = 'ltEEGEMG0006'
annotations_fn = '/Volumes/yaochen/Active/Lizzie/FLP_data/ltEEGEMG0006/Default_2023-04-20_16_45_51_annotations.tsv'
extracted_dir = '/Volumes/yaochen/Active/Lizzie/FLP_data/ltEEGEMG0006/ltEEGEMG0006_extracted_data/'
rawdatdir = '/Volumes/yaochen/Active/Lizzie/FLP_data/ltEEGEMG0006/'

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

SS_df = pd.DataFrame(columns = ['Sleep State', 'Timestamp'])
ss_files = glob.glob(os.path.join(extracted_dir, 'StatesAcq*.npy'))
acqs = np.sort([int(s[s.find('StatesAcq')+9:s.find('_hr0')]) for s in ss_files])
for a in acqs:
	temp_df = pd.DataFrame(columns = ['Sleep State', 'Timestamp'])
	SS_fn = os.path.join(extracted_dir, 'StatesAcq'+str(a)+'_hr0.npy')
	eeg_fn = os.path.join(rawdatdir, 'AD0_'+str(a)+'.mat')
	temp_df['Sleep State'] = np.load(SS_fn)
	ctime = time.ctime(os.path.getmtime(eeg_fn))
	start_datetime = datetime.strptime(ctime, '%a %b %d %H:%M:%S %Y')-timedelta(0,3600)
	temp_df['Timestamp'] = [start_datetime + timedelta(0,int(4*i)) for i in np.arange(1, np.size(list(temp_df['Sleep State']))+1)]
	SS_df = pd.concat([SS_df,temp_df], ignore_index=True)
annotations = pd.read_csv(annotations_fn, sep='\t', skiprows = [0,1,2,3,4,5])
t0 = datetime.strptime(annotations['Start Time'][0][:-4], '%m/%d/%y %H:%M:%S')
SS_df['Offset Time'] = [(t-t0).seconds for t in SS_df['Timestamp']]
SS_df['Sleep State'] = SWS_utils.define_microarousals(SS_df['Sleep State'], 4)
motor_on = annotations.loc[annotations['Annotation'].str.contains("8229 Motor On", case=False)]
wake_idx = PKA.find_continuous(np.asarray(SS_df['Sleep State']), [1])
wake_starts = [w[0] for w in wake_idx[1:]]
end_time = 52928
distance_from_transition = []
for w in wake_starts:
	if SS_df['Offset Time'][w]<end_time:
		x =find_nearest(motor_on['Time From Start'], SS_df['Offset Time'][w])
		distance_from_transition.append(x-SS_df['Offset Time'][w])

vals, bins = np.histogram(distance_from_transition, bins = 50)
yvals = np.cumsum(vals)/np.sum(vals)
yvals = np.insert(yvals, 0, 0)

color_dict = graph.SW_colordict('single state')
graph.make_bigandbold()
fig, ax = plt.subplots()
ax.plot(bins, yvals, color = 'k', linewidth = 3)
ax.axvline(0, color = color_dict['Wake'], linewidth = 3, linestyle = '--')
fig, ax = graph.thick_axes(fig, ax)
ax = graph.label_axes(ax, x = 'Time from Wake (s)', y = 'Fraction of Total')
fig.tight_layout()
fig.savefig('/Users/lizzie/Documents/Chen_Lab/F30/Figures/Feedback_timing.png')
