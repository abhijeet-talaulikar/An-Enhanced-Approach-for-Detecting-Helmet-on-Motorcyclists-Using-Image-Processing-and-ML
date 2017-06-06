function preprocess_batch(Directory)
    if Directory(end) ~= '/'
        Base = strcat(Directory,'\');
        Directory = strcat(Directory,'\*.jpg');
    else
        Base = Directory;
        Directory = strcat(Directory,'*.jpg');
    end
    contents = dir(Directory);
    for i = 1:numel(contents)
        I = imread(strcat(Base,contents(i).name));
        I = preprocess(I);
        imwrite(I, contents(i).name);
    end
end