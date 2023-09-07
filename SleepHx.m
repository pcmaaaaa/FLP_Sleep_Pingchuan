function [sleephx] = SleepHx(SleepStates, HxInterval, tracked_state, epoch)
%SleepHx tracks history of certain states over a certain past interval

sleephx = zeros(size(SleepStates));
num_bins = ceil((HxInterval*60)/epoch);
for h = 1:length(sleephx)
    if h <= num_bins
        window = 1:h;
    else
        window = h-num_bins:h;
    end
    hx = SleepStates(window);
    temp = ismember(hx, tracked_state);
    num_state = length(find(temp));
    sleephx(h) = num_state/num_bins;
end
end