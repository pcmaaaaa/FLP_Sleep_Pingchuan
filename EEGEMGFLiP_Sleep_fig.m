rawdat_dir = '/Volumes/yaochen/Active/Lizzie/FLP_data/ltFLiPAKAREEGEMG0033/';
extracted_dir = '/Volumes/yaochen/Active/Lizzie/FLP_data/ltFLiPAKAREEGEMG0033/ltFLiPAKAREEGEMG0033_extracted_data/';
frequency = 400;
ylims = [1.85, 1.91];
for acqn = 4:26
    PlotEEGEMGFLiP(rawdat_dir, acqn, extracted_dir, frequency, ylims);
end