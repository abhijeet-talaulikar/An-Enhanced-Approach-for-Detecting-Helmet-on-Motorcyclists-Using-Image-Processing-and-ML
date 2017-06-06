function [I1] = get_quadrant(I, Quadrant)
    x = size(I,1);
    y = size(I,2);
    if Quadrant == 1
        I1 = I(1:floor(x/2), floor(y/2):end, :);
    elseif Quadrant == 2
        I1 = I(1:floor(x/2), 1:floor(y/2), :);
    elseif Quadrant == 3
        I1 = I(floor(x/2):end, 1:floor(y/2), :);
    elseif Quadrant == 4
        I1 = I(floor(x/2):end, floor(y/2):end, :);
    else return;
    end
end