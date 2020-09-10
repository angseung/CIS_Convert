function [white_R ,white_G ,white_B,output_image] = AWB(input_image,njet,mink_norm,on_gauss)
%% Variable initialization
if(nargin<2), njet=0; end
if(nargin<3), mink_norm=1; end
if(nargin<4), on_gauss=1; end

gauss_image = zeros(size(input_image,1),size(input_image,2),3);
deriv_image = zeros(size(input_image,1),size(input_image,2),3);

tmp_input_image=input_image; % Replica

%% Gaussian-filter (pre-processing)
if(on_gauss~=0)
    for channel = 1:3
        %gauss_image(:,:,channel) = gDer(input_image(:,:,channel),sigma,0,0);
        %filter = fspecial('gaussian',5,0.8);
        filter = fspecial('gaussian',13,2); % imrotate due to conv2 function
        %filter = fspecial('gaussian',9,1.4);
        gauss_image(:,:,channel) = conv2(input_image(:,:,channel),filter,'same');
    end
else
    gauss_image = input_image;
end

%% Edge detection with (Sobel filter or Laplace filter)
if (njet>0)
    % In conv2, filter is rotated 90 degree
    % In fspecial, filter is generated with 90 degree rotated
    % So, imrotate(x,90) is not required
    sobel_x = fspecial('sobel')/8;
    sobel_y = sobel_x'; %  transposed
    R_x = conv2(gauss_image(:,:,1),sobel_x,'same');
    R_y = conv2(gauss_image(:,:,1),sobel_y,'same');
    R = sqrt(R_x.^2 + R_y.^2);
    
    G_x = conv2(gauss_image(:,:,2),sobel_x,'same');
    G_y = conv2(gauss_image(:,:,2),sobel_y,'same');
    G = sqrt(G_x.^2 + G_y.^2);
    
    B_x = conv2(gauss_image(:,:,3),sobel_x,'same');
    B_y = conv2(gauss_image(:,:,3),sobel_y,'same');
    B = sqrt(B_x.^2 + B_y.^2);    
    
    deriv_image(:,:,1)=R;
    deriv_image(:,:,2)=G;
    deriv_image(:,:,3)=B;
else
    deriv_image = gauss_image;
end
% 
% x = gammma;
% ma = max(max(max(x)));
% mi = min(min(min(x)));
% x = (x-mi)/(ma-mi);

%% Remove saturated pixels in derivated image
deriv_image=abs(deriv_image);
deriv_image(deriv_image>= 255) = 0;

%% Minkowski norm
if(mink_norm == -1) % White-Patch (mink_norm = infinity)
    white_R=double(max(max(deriv_image(:,:,1))));
    white_G=double(max(max(deriv_image(:,:,2))));
    white_B=double(max(max(deriv_image(:,:,3))));
    
    som=sqrt(white_R^2+white_G^2+white_B^2);
    
    white_R=white_R/som;
    white_G=white_G/som;
    white_B=white_B/som;
    
elseif(mink_norm == -2) % Gray-World + White-Patch
    
    
else % minkowski norm
    kleur=power(deriv_image,mink_norm);
 
    white_R = power(sum(sum(kleur(:,:,1))),1/mink_norm);
    white_G = power(sum(sum(kleur(:,:,2))),1/mink_norm);
    white_B = power(sum(sum(kleur(:,:,3))),1/mink_norm);

    som=sqrt(white_R^2+white_G^2+white_B^2);

    white_R=white_R/som;
    white_G=white_G/som;
    white_B=white_B/som;
end

%% Correction of input image
output_image(:,:,1)=tmp_input_image(:,:,1)/white_R;
output_image(:,:,2)=tmp_input_image(:,:,2)/white_G;
output_image(:,:,3)=tmp_input_image(:,:,3)/white_B;

