function [I1] = preprocess(I)
    I = ~im2bw(I,0.98);
    I = medfilt2(I);
    I = imfill(I,8,'holes');
    I = bwareaopen(I,50);
    SE = strel('square',5);
    I = imerode(I,SE);
    I1 = ~imdilate(I,SE);
end