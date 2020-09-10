function [truncRed ,truncGreen ,truncBlue,output_image] = paper_jrt(input_image, N)

truncRed=0;
truncGreen=0;
truncBlue=0;

rowsize = floor(size(input_image,1)/N);

%% Blurring & Edge detection
now = 0;
for i = 1:rowsize
    % horizontal mean + down-sampling (sampled by N-size)
    before = now;
    now    = squeeze(sum(input_image(:,N*(i-1)+1:N*i,:),2)); % 4 bits for each now
    
    % horizontal [1 -1] convolution filter
    if (i==1)
        gain = now;
    else
        gain = gain + abs(before-now); % 29 bit long
    end
end
finalGain = sum(gain,1); % 29 bits

som=sqrt(finalGain(1)^2+finalGain(2)^2+finalGain(3)^2);
    
gain_R=1/(finalGain(1)/som);
gain_G=1/(finalGain(2)/som);
gain_B=1/(finalGain(3)/som);

% % Find maximum value
% maxGain = max(finalGain);
% 
% % find MSB bit location
% maxLoc = floor(log2(maxGain)); 
% truncBW = 4; % valid bit-width from MSB
% mask = 0;
% for i=1:truncBW
%     mask = mask + 2^(maxLoc-i+1);
% end
% 
% % Quantize and truncate to truncBW (accumulated gain value)
% truncRed=bitshift(bitand(finalGain(1), mask),-(maxLoc-truncBW+1));
% truncGreen=bitshift(bitand(finalGain(2), mask),-(maxLoc-truncBW+1));
% truncBlue=bitshift(bitand(finalGain(3), mask),-(maxLoc-truncBW+1));
% 
% bw2 = 6;
% sumRGB = truncRed^2+truncGreen^2+truncBlue^2; % (2*bw1)+2 bit = 10bit
% q_sumRGB = floor(sumRGB/2^(10-bw2))*2^(10-bw2);
% 
% som=round(sqrt(q_sumRGB)); % 5 bit (maximum = 26)
% 
% gain_R=som/truncRed; % 5 bit/{bw1}
% gain_G=som/truncGreen;
% gain_B=som/truncBlue;
% 
% gain_R=floor(gain_R*2^5)/2^5;
% gain_G=floor(gain_G*2^5)/2^5;
% gain_B=floor(gain_B*2^5)/2^5;
% %% Correction of input image
% % output_image(:,:,1)=input_image(:,:,1)/(white_R*sqrt(3));
% % output_image(:,:,2)=input_image(:,:,2)/(white_G*sqrt(3));
% % output_image(:,:,3)=input_image(:,:,3)/(white_B*sqrt(3));

output_image(:,:,1)=input_image(:,:,1)*gain_R;
output_image(:,:,2)=input_image(:,:,2)*gain_G;
output_image(:,:,3)=input_image(:,:,3)*gain_B;
end