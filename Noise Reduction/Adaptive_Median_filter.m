function final_image = Adaptive_Median_filter(input_image,MaxSizeFilter)
%         AdaptiveFilter(image,MaxSizeFilter)
%                 remove noise by changing the size of filter
%                 image : tasvir noisy
%                 MaxSizeFilter : maximum size of filter 
%
%                   AdaptiveFilter start from 3*3 filter
%                   and if noise don't remove,greats size of filter
%                   until MaxSizeFilter and repeat removing noise
imag=double(input_image);
MaxSizeFilterSide = floor((MaxSizeFilter-1)/2);
padsize=[MaxSizeFilterSide, MaxSizeFilterSide];
pad_imag = padarray(imag,padsize,'symmetric');

[x,y]=size(pad_imag);
filtered_imag=zeros(x,y);

[xb,yb]=size(input_image);
StartPoint=MaxSizeFilter-floor(MaxSizeFilter/2);
for i=StartPoint:StartPoint+(xb-1)
    for j=StartPoint:StartPoint+(yb-1)
        filtered_imag = compute_AMF(pad_imag, filtered_imag, 3, MaxSizeFilter,i,j);
    end
end

% Cut out side values
final_image = filtered_imag(StartPoint:StartPoint+xb-1,StartPoint:StartPoint+yb-1);
fprintf("adaptive median end\n");
end

 
function output_imag = compute_AMF(imag, output_imag, FilterSize,MaxFilterSize,i,j)

roi_size=ceil((FilterSize-1)/2);
AreaNeighberhood=imag(i-roi_size:i+roi_size,j-roi_size:j+roi_size);
sortedArea=sort(AreaNeighberhood(:));

% obtain variables for compute_AMF
Zmin=sortedArea(1);
Zmax=sortedArea(end);
Zmed=median(sortedArea);
Zxy=imag(i,j);

B1=Zxy-Zmin;
B2=Zxy-Zmax;

if(B1>0 && B2<0) % Zmin < Zxy < Zmax
    output_imag(i,j)=Zxy;
    return;
else % Zxy == (Zmax or Zmin) : +++ Noise detected +++
    A1=Zmed-Zmin;
    A2=Zmed-Zmax;
    if(A1>0 && A2<0) % Zmin < Zmed < Zmax
        output_imag(i,j)=Zmed;
        return;
    else % Zmed == (Zmax or Zmin) : +++ Small filter size +++ ?????????
        if(FilterSize<MaxFilterSize)
            FilterSize=FilterSize+2; % Try the bigger filter size
            compute_AMF(imag, output_imag, FilterSize,MaxFilterSize,i,j);
            return;
        else
            output_imag(i,j)=Zmed;
        end
    end
end

end
