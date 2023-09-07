function plotFLiPdata(filename)
load(filename);
time = time(~isnan(time));
photoncount = photoncount(~isnan(photoncount));
tau_fit_G = tau_fit_G(~isnan(tau_fit_G));
chi_sq_G = chi_sq_G(~isnan(chi_sq_G));
figure1 = figure(1); subplot(3,1,1);
plot(time,tau_fit_G, '.', 'Color', 'k');
title (filename);
ylabel ('fluorescence lfietime (ns)');
xlim([1,max(time)])
subplot(3,1,2);
plot(time,photoncount, '.', 'Color', 'k');
ylabel('photon counts');
xlim([1,max(time)])
subplot(3,1,3);
plot(time,chi_sq_G, '.', 'Color', 'k');
xlabel('time (sec)'); ylabel('chi square value');
xlim([1,max(time)])
figurefile = strcat(filename(1:end-4), '_FLiPfigure');
savefig(figure1, figurefile);
saveas(figure1,strcat(figurefile, '.png'))
end

