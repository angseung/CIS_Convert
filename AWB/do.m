clear;
clc;

% Images received from Clair-Pixel
%input_image = double(imread('./CP_images/2M_IMG_0000.ppm')); % 8 bit
%input_image = double(imread('./CP_images/2M_IMG_BNR.ppm')); % 8 bit
%input_image = double(imread('./CP_images/CP8120_dark_0000.pgm')); % 16 bit (1 channel)
%input_image = double(imread('./CP_images/CP8120_dark_BNR.ppm'));  % 8 bit (3 channel)
%input_image = double(imread('./CP_images/CP8220_bayer_10bit_women_LP_0000.pgm')); % 16 bit (1 channel)
%input_image = double(imread('./CP_images/CP8220_bayer_women_BNR.ppm')); % 8 bit (3 channel)

% Images received from open-sources
%input_image=uint8(imread('./input_images/building1.jpg'));
%input_image=uint8(imread('./input_images/cow2.jpg'));
%input_image=uint8(imread('./input_images/dog3.jpg'));
%input_image=uint8(imread('./input_images/cat.png'));
%input_image=uint8(imread('./input_images/tungsten.png'));

enable_plot = 1; % 0 = 'OFF', 1 = 'ON'
enable_saveplot = 1; % 0 = 'OFF', 1 = 'ON'
load('../dataset/colorchecker/groundtruthcoordinates/ColorCheckerData');
fid = fopen('../dataset/colorchecker/file_list.txt','r');
f_ae = fopen('./ae_results.txt','w');

x=textscan(fid,'%s');
fclose(fid);
x=x{1}; % File name list
angularError = zeros(568,5);

for iter=1:10
    filename = x{iter};
    fprintf('%d iter, filename %s \n',iter,x{iter});
    input_image = uint8(imread(filename));
    C = strsplit(filename,'_');
    CC = strsplit(C{1},'/');
    filenum = str2num(CC{5});
    %mask_chart = uint8(imread('../dataset/colorchecker/1D/masks/mask1_8D5U5525.tiff'));

    illuminant_groundtruth = [.5, .5, .5];
%     illuminant_groundtruth = REC_groundtruth(filenum,:);
    illuminant_groundtruth = illuminant_groundtruth ./ norm(illuminant_groundtruth);
    
    % AWB <-- (input_data, njet, mink_norm, on_gauss)
    % Gray-World    
%     tic;
    on_gauss = 0;
    [~,~,~,out1]=AWB(input_image, 0, 1, on_gauss);
    [extracted_chart, mask_chart] = get_chart(out1,final_coord(:,:,filenum));

    illuminant = get_illuminant(extracted_chart,mask_chart);
    illuminant = illuminant ./ norm(illuminant);
    ae1 = AngularError(illuminant_groundtruth,illuminant);
    angularError(iter,1) = angularError(iter,1) + ae1;    
%     fprintf("GrayWorld time : %f", toc);
    
    % White-Patch
%     tic;
    on_gauss = 0;
    [~,~,~,out2]=AWB(input_image, 0, -1, on_gauss);
    [extracted_chart, mask_chart] = get_chart(out2,final_coord(:,:,filenum));

    illuminant = get_illuminant(extracted_chart,mask_chart);
    illuminant = illuminant ./ norm(illuminant);
    ae2 = AngularError(illuminant_groundtruth,illuminant);
    angularError(iter,2) = angularError(iter,2) + ae2;
%     fprintf("WhitePatch time : %f", toc);
    
    % Shades of Grey
%     tic;
    on_gauss = 0;
    mink_norm=5;    % any number between 1 and infinity
    [~,~,~,out3]=AWB(input_image, 0, mink_norm, on_gauss);
    [extracted_chart, mask_chart] = get_chart(out3,final_coord(:,:,filenum));

    illuminant = get_illuminant(extracted_chart,mask_chart);
    illuminant = illuminant ./ norm(illuminant);
    ae3 = AngularError(illuminant_groundtruth,illuminant);
    angularError(iter,3) = angularError(iter,3) + ae3;    
%     fprintf("Shade-of-Grey time : %f", toc);

    % Grey-Edge
%     tic;
    mink_norm=5;    % any number between 1 and infinity
    on_gauss=0;        % sigma 
    diff_order=1;   % differentiation order (1 or 2)

    [~,~,~,out4]=AWB(input_image, diff_order, mink_norm, on_gauss);
    [extracted_chart, mask_chart] = get_chart(out4,final_coord(:,:,filenum));
    illuminant = get_illuminant(extracted_chart,mask_chart);
    illuminant = illuminant ./ norm(illuminant);
    ae4 = AngularError(illuminant_groundtruth,illuminant);
    angularError(iter,4) = angularError(iter,4) + ae4;
%     fprintf("Grey-edge time : %f", toc);

%     tic;
    [~,~,~,out5]=paper_jrt(input_image, 4);
    [extracted_chart, mask_chart] = get_chart(out5,final_coord(:,:,filenum));
    illuminant = get_illuminant(extracted_chart,mask_chart);
    illuminant = illuminant ./ norm(illuminant);
    ae5 = AngularError(illuminant_groundtruth,illuminant);
    angularError(iter,5) = angularError(iter,5) + ae5;    
%     fprintf("paper_jrt time : %f", toc);

    if(enable_plot==1)
        h=figure(1);
        subplot(2,4,1); imshow(lin2rgb(input_image));
        title('Original image');
        subplot(2,4,2); imshow(lin2rgb(out1));
        title(strcat('Gray-World, Error : ',num2str(ae1)));
        subplot(2,4,3);imshow(lin2rgb(out2));
        title(strcat('White-Patch, Error : ',num2str(ae2)));
        subplot(2,4,4);imshow(lin2rgb(out3));
        title(strcat('Shade-of-Gray, Error : ',num2str(ae3)));
        subplot(2,4,5);imshow(lin2rgb(out4));
        title(strcat('Gray-Edge, Error : ',num2str(ae4)));
        subplot(2,4,6);imshow(lin2rgb(out5));
        title(strcat('JRT paper, Error : ',num2str(ae5)));        
        subplot(2,4,7);imshow(lin2rgb(imoverlay(extracted_chart,mask_chart)));
        title('Region-of-Interest');
        fprintf(f_ae,"%f %f %f %f %f\n",ae1,ae2,ae3,ae4,ae5);
        fprintf('Angular error Gray-World = %f\n',ae1);
        fprintf('Angular error White-Patch = %f\n',ae2);
        fprintf('Angular error Shade-of-Gray = %f\n',ae3);
        fprintf('Angular error Gray-Edge = %f\n',ae4);
        fprintf('Angular error Paper_JRT = %f\n',ae5);
        if(enable_saveplot==1)
            saveas(h,strcat('./AWB_images/',int2str(filenum),'.bmp'));
        end
    end
end
fclose(f_ae);
AE = mean(angularError,1);

