function [ac] = arc_circularity(I, Quadrant)
    if ismember(Quadrant,[1 2]) == 0
        return;
    elseif Quadrant == 2
        I = I(:,end:-1:1,:);
    end
    y = size(I,1);
    x = size(I,2);
    ox = 1; %origin x coord
    oy = size(I,1); %origin y coord
    maxBlobWidth = 0;
    for i = 1:x
        if I(y,i)==1
            break;
        end
        maxBlobWidth = maxBlobWidth + 1;
    end
    radii = zeros(1,maxBlobWidth);
    for i = 1:maxBlobWidth
        %if first pixel is black
        if I(size(I,1),i)==0
            for j = size(I,1):-1:1
                if I(j,i)==1
                    break;
                end
            end
            radii(1,i) = sqrt(( (i-ox)^2)+( (j-oy)^2));
        else
            %if first pixel is white then loop till lowermost black pixel and
            %then run same algo as above
            for k = size(I,1):-1:1
                if I(k,i)==0
                    break;
                end
            end
            for j = k:-1:1
                if I(j,i)==1
                    break;
                end
            end
            radii(1,i) = sqrt(( (i-ox)^2)+( (j-oy)^2));
        end
    end
    ac = mean(radii) / std(radii);
end