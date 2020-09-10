clc; clear;
% incremental_image = zeros(1080,1920,3);
% 
% [col, row, ~] = size(incremental_image);
% x=0;
% for j=1:col
%     for i=1:row
%         in = bitand(x,255);
%         incremental_image(j,i,1)=in;
%         incremental_image(j,i,2)=in;
%         incremental_image(j,i,3)=in;
%         x=x+1;
%     end
% end
% save('incremental_image','incremental_image');

% load('incremental_image.mat');
% load('incremental_image.mat');
% input_image = incremental_image;

X = imread("../dataset/colorchecker/1D/1_8D5U5524.tiff");
input_image = imresize(X,[1080 1920]);

N = 4;
rowsize = floor(size(input_image,2)/N);
%% Blurring & Edge detection
now = 0;
dbg_absdiff = zeros(rowsize,1080);
dbg_acc = zeros(1080,rowsize);
for i = 1:rowsize
    % horizontal mean + down-sampling (sampled by N-size)
    before = now;
    now    = squeeze(sum(input_image(:,N*(i-1)+1:N*i,:),2)); % 4 bits for each now
    
    % horizontal [1 -1] convolution filter
    if (i==1)
        gain = now;
    else
        t = abs(before-now);
        dbg_absdiff(i,:) = t(:,1);
        gain = gain + t; % 29 bit long
    end
    dbg_acc(:,i)=gain(:,1);
end
dbe_acc1 = dbg_acc;
for i=2:1080
    dbg_acc(i,:) = dbg_acc(i,:)+dbg_acc(i-1,480);
end

dbg_gain = zeros(1080,1);

for i=1:1080
    dbg_gain(i) = sum(gain(1:i));
end

finalGain = sum(gain,1); % 29 bits

% Find maximum value
maxGain = max(finalGain);

% find MSB bit location
maxLoc = floor(log2(maxGain)); 
truncBW = 4; % valid bit-width from MSB
mask = 0;
for i=1:truncBW
    mask = mask + 2^(maxLoc-i+1);
end

% Quantize and truncate to truncBW (accumulated gain value)
truncRed =  bitshift(bitand(finalGain(1), mask),-(maxLoc-truncBW+1));
truncGreen= bitshift(bitand(finalGain(2), mask),-(maxLoc-truncBW+1));
truncBlue = bitshift(bitand(finalGain(3), mask),-(maxLoc-truncBW+1));

bw2 = 6;
sumRGB = truncRed^2+truncGreen^2+truncBlue^2; % (2*bw1)+2 bit = 10bit
q_sumRGB = floor(sumRGB/2^(10-bw2))*2^(10-bw2);

som=round(sqrt(q_sumRGB)); % 5 bit (maximum = 26)

gain_R=som/truncRed; % 5 bit/{bw1}
gain_G=som/truncGreen;
gain_B=som/truncBlue;

gain_R=floor(gain_R*2^5)/2^5;
gain_G=floor(gain_G*2^5)/2^5;
gain_B=floor(gain_B*2^5)/2^5;

%% Correction of input image
% output_image(:,:,1)=input_image(:,:,1)/(white_R*sqrt(3));
% output_image(:,:,2)=input_image(:,:,2)/(white_G*sqrt(3));
% output_image(:,:,3)=input_image(:,:,3)/(white_B*sqrt(3));

sw_output_image(:,:,1)=input_image(:,:,1)*gain_R;
sw_output_image(:,:,2)=input_image(:,:,2)*gain_G;
sw_output_image(:,:,3)=input_image(:,:,3)*gain_B;

subplot(2,1,1);
imshow(lin2rgb(input_image));
subplot(2,1,2);
imshow(lin2rgb(sw_output_image));