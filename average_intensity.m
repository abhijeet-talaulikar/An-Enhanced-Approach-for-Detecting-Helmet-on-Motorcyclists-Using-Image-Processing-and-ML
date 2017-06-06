function [avg] = average_intensity(I)
    I = rgb2gray(I);
    avg = sum(sum(I)) / numel(I);
end