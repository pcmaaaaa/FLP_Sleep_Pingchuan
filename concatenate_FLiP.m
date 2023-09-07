% rawdat_dir = '/Volumes/PcSSDA/1610/20220219_1610_SW_002/';
% % FLipAnalysisInRow(rawdat_dir, 1,1);
% acq_range = 11:22;
% FirstAcq=10;
% SavedFilePostFix = '20220315';

tau_emp_all = [];
photoncount_all = [];
p1_all = [];
time_all = [];
tau_fit_G_all = [];
chi_sq_G_all = [];
dpeak_all = [];
GWidth_all = [];
AnalyzedFileRange = acq_range-FirstAcq+1;
filename = strcat(rawdat_dir, 'concat_Acq',num2str(acq_range(1)), '_', num2str(acq_range(end)));
for i = 1:length(acq_range)

    iAcq=AnalyzedFileRange(i);
%     file = strcat(rawdat_dir,sprintf('20220219_1610_SW_002_analysis_%dh_20220315.mat',iAcq));
    file = [SavedFilePreFix, num2str(iAcq), 'h_', SavedFilePostFix, '.mat']
    load(file);
%     plotFLiPdata(file)
    photoncount_all = [photoncount_all, photoncount'];
    p1_all = [p1_all, p1'];
    tau_fit_G_all = [tau_fit_G_all, tau_empTrunc'];
    chi_sq_G_all = [chi_sq_G_all, chi2'];
    dpeak_all = [dpeak_all, beta5'];
    GWidth_all = [GWidth_all, beta6'];
    if iAcq == acq_range(1)-FirstAcq+1
        time_all = time';
    else
        time_all = [time_all, time'+time_all(end)];
    end
    tau_emp_all = tau_emp_all(~isnan(tau_emp_all));
    photoncount_all = photoncount_all(~isnan(photoncount_all));
    p1_all = p1_all(~isnan(p1_all));
    tau_fit_G_all = tau_fit_G_all(~isnan(tau_fit_G_all));
    chi_sq_G_all = chi_sq_G_all(~isnan(chi_sq_G_all));
    time_all = time_all(~isnan(time_all));
    dpeak_all = dpeak_all(~isnan(dpeak_all));
    GWidth_all = GWidth_all(~isnan(GWidth_all));
end
save(filename, 'time_all', 'photoncount_all', 'tau_fit_G_all','p1_all', 'chi_sq_G_all', 'GWidth_all', 'dpeak_all');
time_hrs = time_all/3600;

concat_figure = figure('Position', [10 10 1400 400]);
subplot(4, 1, 1);
plot(time_hrs,tau_fit_G_all,'.', 'Color', 'k');
ylabel('lifetime (ns)');
title('Fitted Lifetime Gaussian');
xlim([0 max(time_hrs)]);


subplot(4, 1, 2);
plot(time_hrs,chi_sq_G_all,'.', 'Color','k');
xlabel('time (s)');
ylabel('Chi Square Value');
title('Chi Square Gaussian');
xlim([0 max(time_hrs)]);

subplot(4, 1, 3);
plot(time_hrs,photoncount_all,'.', 'Color','k');
ylabel('photoncount');
title('Photoncount');
xlim([0 max(time_hrs)]);

subplot(4, 1, 4);
plot(time_hrs,dpeak_all,'.', 'Color','k');
ylabel('Delta Peak Time');
title('Delta Peak Time');
xlim([0 max(time_hrs)]);


saveas(concat_figure, strcat(filename,'_figure.png'))
saveas(concat_figure, strcat(filename,'_figure.fig'))