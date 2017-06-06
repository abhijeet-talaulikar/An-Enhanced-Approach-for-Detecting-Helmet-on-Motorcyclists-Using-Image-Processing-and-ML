function [avg] = average_hue(I)
    I = rgb2hsv(I);
    I = I(:,:,1);
    avg = sum(sum(I)) / numel(I);
end