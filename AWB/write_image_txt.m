clc; clear;
% load('incremental_image.mat');
A = imread("../dataset/colorchecker/1D/1_8D5U5524.tiff");
input_image = imresize(A,[1080 1920]);

%fid = fopen('test_image.txt','w');
fid = fopen('tt.txt','w');
[col, row, ~] = size(input_image);

for i=1:1
    for j=1:row
        inrgb = input_image(i,j,1);
        if(inrgb<=15)
            fprintf(fid,"0%x",inrgb);
        else
            fprintf(fid,"%x",inrgb);
        end
        inrgb = input_image(i,j,2);
        if(inrgb<=15)
            fprintf(fid,"0%x",inrgb);
        else
            fprintf(fid,"%x",inrgb);
        end
        inrgb = input_image(i,j,3);
        if(inrgb<=15)
            fprintf(fid,"0%x\n",inrgb);
        else
            fprintf(fid,"%x\n",inrgb);
        end
    end
end

fclose(fid);
