#This is a package of graphing things I use often
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from copy import deepcopy
import random
import math
def thick_axes(fig, ax, width = 3):
	num_ax = len(fig.get_axes())
	if num_ax > 1:
		if len(np.shape(ax)) > 1:
			rows = np.shape(ax)[0]
			cols = np.shape(ax)[1]
			for r in np.arange(0, rows):
				for c in np.arange(0, cols):
					ax[r,c].tick_params(width=3)
					for axis in ['bottom','left']:
						ax[r,c].spines[axis].set_linewidth(3)
					for axis in ['right','top']:
						ax[r,c].spines[axis].set_visible(False)
		else:
			for ii in np.arange(0, num_ax):
				ax[ii].tick_params(width=3)
				for axis in ['bottom','left']:
					ax[ii].spines[axis].set_linewidth(3)
				for axis in ['right','top']:
					ax[ii].spines[axis].set_visible(False)
	else:
		ax.tick_params(width=3)
		for axis in ['bottom','left']:
			ax.spines[axis].set_linewidth(3)
		for axis in ['right','top']:
			ax.spines[axis].set_visible(False)

	return fig, ax

def linegraph_w_error(ax, x,y, error, color = 'k', label = None, linewidth = 1):
	ax.plot(x,y, color = color, label = label, linewidth =linewidth)
	upper_bound = y+error
	lower_bound = y-error
	ax.fill_between(x, upper_bound, lower_bound, color = color, alpha = 0.5, edgecolor = None)
	return ax


def make_bigandbold(xticksize = 15, yticksize = 15, axeslabelsize = 17):
	plt.rc('xtick', labelsize = xticksize)
	plt.rc('ytick', labelsize = yticksize)
	plt.rc('axes', labelsize = axeslabelsize)
	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"

def label_axes(ax, x = False, y= False, title= False, title_fontsize = 25, fontweight = 'bold'):
	if x:
		ax.set_xlabel(x)
	if y:
		ax.set_ylabel(y)
	if title:
		ax.set_title(title, fontsize = title_fontsize, fontweight = fontweight)

	return ax

def SW_colordict(keys):
	color_dict = {}
	if keys == 'numbers':
		color_dict['1'] = '#54ac68'
		color_dict['2'] = '#a2bffe'
		color_dict['3'] = '#ff6163'
		color_dict['4'] = '#54ac68'
		color_dict['5'] = '#fac205'
	if keys == 'single state':
		color_dict['Wake'] = '#54ac68'
		color_dict['NREM'] = '#a2bffe'
		color_dict['REM'] = '#ff6163'
		color_dict['Wake'] = '#54ac68'
		color_dict['Microarousals'] = '#fac205'
	if keys == 'transitions':
		color_dict['NREM-Wake'] = '#54ac68'
		color_dict['NREM-Active Wake'] = '#54ac68'
		color_dict['NREM-Quiet Wake'] = '#a55af4'

		color_dict['Sleep-Wake'] = '#54ac68'
		color_dict['REM-Wake'] = '#9be5aa'
		color_dict['REM-Active Wake'] = '#9be5aa'
		color_dict['REM-Quiet Wake'] = '#c48efd'

		color_dict['NREM-REM'] = '#ff6163'
		color_dict['Wake-REM'] = '#ff000d'

		color_dict['REM-NREM'] = '#488ee4'
		color_dict['Wake-NREM'] = '#a2bffe'
		color_dict['Wake-Sleep'] = '#a2bffe'
		color_dict['Active Wake-NREM'] = '#a2bffe'
		color_dict['Quiet Wake-NREM'] = '#a2bffe'

		color_dict['Active Wake-Quiet Wake'] = '#c48efd'
		color_dict['Quiet Wake-Active Wake'] = '#9be5aa'

		color_dict['Unknown'] = '#d8dcd6'

		color_dict['Microarousal'] = '#fac205'
		color_dict['Microarousals'] = '#fac205'

	return color_dict

def remove_yticks(fig, ax):
	num_ax = len(fig.get_axes())
	for ii in np.arange(1, num_ax):
		ax[ii].set_yticklabels([])
	return fig, ax

def grouped_bargraph(fig, ax, yvals, colors, x_labels = [], legend_labels = [], edgecolor = None, 
	linewidth = None):
	assert len(yvals) == len(x_labels)
	assert len(yvals[0]) == len(colors) == len(legend_labels)
	tot_group_width = len(colors)
	xvals = []
	count = 0
	for i in np.arange(0, len(x_labels)):
		xvals.append(np.arange(count, count+len(colors)))
		count += len(colors)+1
	assert len(yvals) == len(xvals) == len(x_labels)
	for ii in np.arange(0, len(yvals)):
		bars = ax.bar(xvals[ii], yvals[ii], color = colors, width = 1, align = 'edge', 
			edgecolor = edgecolor, linewidth = linewidth)
	x_ticks = [x[0]+(len(x)/2) for x in xvals]
	ax.set_xticks(x_ticks)
	ax.set_xticklabels(x_labels)

	color_map = dict(zip(yvals[0], colors))
	patches = [Patch(color=v, label=k) for k, v in color_map.items()]
	ax.legend(labels = legend_labels, handles = patches, fontsize = 15, ncols = math.ceil(len(legend_labels)/3))

	return fig, ax

def get_jittered_x(x, size, width = 0.25):
	x_vals = np.random.uniform(low=x-0.25, high=x+0.25, size=size)
	return x_vals

def violin_plot(fig, ax, data, x_positions, colors, showmeans = False, showmedians = True, xlabels = []):
	for i,d in enumerate(data):
		d = d[~np.isnan(d)]
		violin_parts = ax.violinplot(d, showmeans = showmeans, showmedians = showmedians, positions = [x_positions[i]])
		violin_parts['bodies'][0].set_color(colors[i])
		violin_parts['cmins'].set_color(colors[i])
		violin_parts['cmaxes'].set_color(colors[i])
		violin_parts['cbars'].set_color(colors[i])
		if showmedians:
			violin_parts['cmedians'].set_color(colors[i])
		if showmeans:
			violin_parts['cmeans'].set_color(colors[i])
	ax.set_xticks(x_positions)
	ax.set_xticklabels(xlabels)
	return fig, ax
def pick_scatter_markers(num_markers):
	marker_dict = deepcopy(Line2D.markers)
	for m in ['None', None, ' ', '', '_', '.',',','|', 0,1]:
		marker_dict.pop(m)
	all_markers = len(marker_dict.keys())
	these_markers = random.sample(list(marker_dict.keys()), num_markers)
	return these_markers





