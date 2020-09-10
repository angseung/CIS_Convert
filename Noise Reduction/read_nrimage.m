% load('incremental_image.mat');
% clc; clear;
output_image = zeros(1080, 1920);

M = textread('./output_image_lena.txt','%8c');

for i=1:1080
    if mod(i,100)==0
        fprintf("%d row is on process\n",i);
    end
    for j=1:1920
        idx= 1920*(i-1)+j;
        outpixel  = bin2dec(char(M(idx,1:8)));
        output_image(i,j) = outpixel;
    end
end

figure(2);
imshow(uint8(output_image));