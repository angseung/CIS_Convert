% clear variables
%% Main Script...
clc;
clear; 

%% Set parameters
enable_plot = 1; % 0 = OFF, 1 = ON

%% Read the testing image
% testImages = ["barbara","baboon","goldhill","peppers","lena","1920test1","1920test2"];
% testImages = ["barbara"];
testImages = ["1920test2"];

% plotImagename = "goldhill"; % one image to plot
plotImagename = "1920test2"; % one image to plot
numImages = size(testImages,2);

% noiseRatio = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25];
% noiseRatio = [0.005, 0.01, 0.015, 0.02];
% noiseRatio = [0.03];
noiseRatio = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
numNoise = size(noiseRatio,2);

%filters = ["median","SEPD","RSEPD","fuzzy","HTF","HTF_Fuzzy"];
% filters = ["median","HW-SEPD", "HW-DTBDM", "HTF"];
filters = ["HW-RSEPD"];
    % "HTF","HW-DTBDM","HW-SEPD","HW-RSEPD","SEPD"];
numFilter = size(filters,2);
psnrs = zeros(numFilter, numNoise);

if numFilter >= 7
    fprintf("Too many filters are considered!!!\n");
    plotshape = 0;
else
    plotshape = [13,23,23,23,33,33,33];
end

for nr = 1:numNoise
    for imnum = 1:numImages
        % load image
        ori_image = imgload(testImages(imnum));
      
        % applying noise
        noise_img = imnoise(ori_image,'salt & pepper',noiseRatio(nr));
        if enable_plot == 1 && strcmp(plotImagename,testImages(imnum))
            figure((nr-1)*numImages+imnum); 
            subplot(plotshape(numFilter)*10+1)
            imshow(uint8(ori_image));
            title(strcat('Original'));
            subplot(plotshape(numFilter)*10+2)
            imshow(uint8(noise_img));
            title(strcat('Noise image, Noise Ratio=',num2str(noiseRatio(nr))));
        end
        
        % filter processing
        for f = 1:numFilter
            [filtered_img, psnr] = imgfilter(filters(f),ori_image,noise_img);
            psnrs(f,nr) = psnrs(f,nr) + psnr;
            if enable_plot == 1 && strcmp(plotImagename,testImages(imnum))
                subplot(plotshape(numFilter)*10+f+2)
                imshow(uint8(filtered_img));
                title(strcat(filters(f)," PSNR=",num2str(psnr)));
            end
        end
    end
end
psnrs = psnrs / numImages;
xlswrite('psnr_result.xls',psnrs);

%% Funcion - load image

function img = imgload(name)

if(strcmp(name,'barbara'))
    img=imread('./input_images/barbara512.bmp');
elseif(strcmp(name,'baboon'))
    img=imread('./input_images/baboon512.bmp');
elseif(strcmp(name,'goldhill'))
    img=imread('./input_images/goldhill512.bmp');
elseif(strcmp(name,'peppers'))
    img=imread('./input_images/peppers512.bmp');
elseif(strcmp(name,'lena'))
    img=imread('./input_images/lena512.bmp');
elseif(strcmp(name,'1920test1'))
    cimg = imread('./input_images/1920image2.jfif');
    img = rgb2gray(cimg);
elseif(strcmp(name,'1920test2'))
    cimg = imread('./input_images/testimage.jpg');
    img = rgb2gray(cimg);
else
    fprintf("Incorrect image name!!\n");
    img=0;
end

end

%% Funcion - Applying filter

function [filtered_img, psnr] = imgfilter(name, ori_img, noise_img)

if(strcmp(name,'median'))
    filtered_img = medfilt2(noise_img,[3 3]);
elseif(strcmp(name,'mean'))
    kernel = ones(3, 3) / 9;
    filtered_img = conv2(noise_img, kernel, 'same');
elseif(strcmp(name,'HTF_Fuzzy'))
    filtered_img = hw_HTF(noise_img, 20, 1);
elseif(strcmp(name,'HTF'))
    filtered_img = hw_HTF(noise_img, 20, 0);
elseif(strcmp(name,'HW-DTBDM'))
    filtered_img = hw_DTBDM(noise_img);
elseif(strcmp(name,'HW-SEPD'))
    filtered_img = hw_SEPD(noise_img, 20);
elseif(strcmp(name,'HW-RSEPD'))
    filtered_img = hw_RSEPD(noise_img, 20);
elseif(strcmp(name,'SEPD'))
    filtered_img = SEPD(noise_img, 20);
elseif(strcmp(name,'RSEPD'))
    filtered_img = RSEPD(noise_img, 20);
elseif(strcmp(name,'DTBDM'))
    filtered_img = DTBDM(noise_img);
elseif(strcmp(name,'fuzzy'))
    filtered_img = NAFSM(noise_img);
elseif(strcmp(name,'adaptive median'))
    filtered_img = Adaptive_Median_filter(noise_img,3);    
else
    fprintf("Incorrect filter name!!\n");
    filtered_img=0;
end
psnr = PSNR(ori_img, filtered_img);
end