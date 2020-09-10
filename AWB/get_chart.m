function [extracted_chart, mask_chart] = get_chart(input_image,coord)
    % Im = lin2rgb(input_image);
    Im = input_image;
    
    % Crop input image
    roiy = round(coord(:,1));
    roix = round(coord(:,2));
    crop_xmin = min(roix);
    crop_xmax = max(roix);
    crop_ymin = min(roiy);
    crop_ymax = max(roiy);
    roix = roix - crop_xmin + 1;
    roiy = roiy - crop_ymin + 1;
    cropped_Im = imcrop(Im,[crop_ymin, crop_xmin, crop_ymax-crop_ymin, crop_xmax-crop_xmin]);
    
    % Image warping
    U = [roiy roix];
    X = [0 0; 300 0; 300 200; 0 200]; % The size of chart
    T = maketform('projective',U,X); % Transformation matrix
%     A = T.tdata.T;
%     tform = projective2d(A);
%     J = imwarp(Im,tform,'XYScale',1);
    J = imtransform(cropped_Im,T,'XYScale',1); % Image warping
    
    mask_chart = roipoly(cropped_Im,roiy,roix);  % ROI region (polygon)
    rotated_mask = imtransform(mask_chart,T,'XYScale',1); % Image warping
    
    [Xmin, Xmax, Ymin, Ymax] = get_vertex(rotated_mask);
    extracted_chart = J(Xmin:Xmax,Ymin:Ymax,:);

    % Draw rectangular regions
    xspace = round(size(extracted_chart,1)/4);
    yspace = round(size(extracted_chart,2)/6);
    r = 7;
    xdefault=-10;
    ydefault=3;
    mask_chart = false(size(extracted_chart,1), size(extracted_chart,2));
    xpoint = round(xspace*7/2)+xdefault;
    ypoint = round(yspace/2)+ydefault;
    for k = 1:6
        mask_chart(xpoint-r:xpoint+r,ypoint+yspace*(k-1)-r:ypoint+yspace*(k-1)+r) = true;
    end
    
%     figure(1);
%     subplot(2,2,1);
%     imshow(lin2rgb(Im));
%     subplot(2,2,2);
%     imshow(lin2rgb(J));
%     subplot(2,2,3);
%     imshow(rotated_mask);
%     subplot(2,2,4);
%     imshow(lin2rgb(imoverlay(extracted_chart,mask_chart)));
end