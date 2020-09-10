% clc; clear;
% load('incremental_image.mat');
% cimg = imread('./input_images/testimage.jpg');
% img = rgb2gray(cimg);
% sig = 0.07;
% nimg = imnoise(img,'salt & pepper',sig);
% noise_img = nimg;
% load('input_image.mat');

input_image = uint8(denoised_image);
fid = fopen('test_image.txt','w');
% fid = fopen('input_image.txt','w');
[col, row] = size(input_image);
% imshow(input_image);
for i=1:col
    if mod(i,100)==0
        fprintf("%d row is on process\n",i);
    end
    for j=1:row
        inpixel = input_image(i,j,1);
        if(inpixel<=15)
            fprintf(fid,"0%x",inpixel);
        else
            fprintf(fid,"%x",inpixel);
        end
        fprintf(fid,"\n");
    end
end

fclose(fid);