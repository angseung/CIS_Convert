function mask_chart = get_mask_chart(input_image,coord)
    % Im = lin2rgb(input_image);
    Im = input_image;
    
    roiy = round(coord(:,1));
    roix = round(coord(:,2));
    U = [roiy roix];
    X = [0 0; 300 0; 300 200; 0 200]; % The size of chart
    T = maketform('projective',U,X); % Transformation matrix
    
    mask_roi = roipoly(Im,roiy,roix);  % ROI region (polygon)
    mask_rotated = imtransform(mask_roi,T,'XYScale',1); % Image warping
    mask_chart = false(size(mask_rotated,1), size(mask_rotated,2));
    
    [Xmin, Xmax, Ymin, Ymax] = get_vertex(mask_chart); 
    
    % Draw rectangular regions
    xspace = round(200/4);
    yspace = round(300/6);
    r = 7;
    xdefault=-10+Xmin;
    ydefault=3+Ymin;
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