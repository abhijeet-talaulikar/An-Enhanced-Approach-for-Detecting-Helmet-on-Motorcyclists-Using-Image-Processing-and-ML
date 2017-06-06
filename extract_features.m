function extract_features(Directory, Class)
    if Directory(end) ~= '/'
        Base = strcat(Directory,'\');
        Directory = strcat(Directory,'\*.jpg');
    else
        Base = Directory;
        Directory = strcat(Directory,'*.jpg');
    end
    contents = dir(Directory);
    m = zeros(numel(contents), 8 + 1980 + 1);
    for i = 1:numel(contents)
        I = imread(strcat(Base,contents(i).name));
        avg_int1 = average_intensity(get_quadrant(I,1));
        avg_int2 = average_intensity(get_quadrant(I,2));
        avg_int3 = average_intensity(get_quadrant(I,3));
        avg_int4 = average_intensity(get_quadrant(I,4));
        avg_hue3 = average_hue(get_quadrant(I,3));
        avg_hue4 = average_hue(get_quadrant(I,4));
        I = preprocess(I);
        arc_circ1 = arc_circularity(get_quadrant(I,1), 1);
        if isnan(arc_circ1) || isinf(arc_circ1)
            arc_circ1 = 0;
        end
        arc_circ2 = arc_circularity(get_quadrant(I,2), 2);
        if isnan(arc_circ2)  || isinf(arc_circ1)
            arc_circ2 = 0;
        end
        m(i, 1:8) = [arc_circ1, arc_circ2, avg_int1, avg_int2, avg_int3, avg_int4, avg_hue3, avg_hue4];
        m(i, 9:end-1) = extractHOGFeatures(I(1:floor(size(I,1)/2),:));
        m(i, end) = Class;
    end
    
    %Feature scaling
    for i = 1:size(m,2)-1-1980
        m(:,i) = (m(:,i) - min(m(:,i))) / (max(m(:,i)) - min(m(:,i)));
    end
    
    if Class == 0
        csvwrite('face.csv',m);
    else
        csvwrite('helmet.csv',m);
end