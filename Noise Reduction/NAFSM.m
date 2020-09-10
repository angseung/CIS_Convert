% Noise adaptive fuzzy switching median filter for salt-and-pepper noise
% reduction
% Kenny Kal Vin Toh, etc
function denoised_image = NAFSM(input_image)


x = padarray(input_image,[3 3],'symmetric');
padIm = double(x);

M = input_image;
denoised_image = zeros(size(padIm));

N = zeros(size(padIm));
N(padIm~=0 & padIm~=255) = 1;

[xlen, ylen] =size(padIm);
T1 = 10;
T2 = 30;

%% Noise detected
for i=4:1:xlen-3
    for j=4:1:ylen-3
        
        if N(i,j) == 0
            s = 1;
            g = N(i-s:i+s,j-s:j+s);
            
            if sum(g(:)>0)
                
               %% clear tmp;
               tmp = padIm(i-s:i+s,j-s:j+s);
               tmp = tmp(g(:)==1);
               
               M(i,j) = median(tmp);
               
            else
                
               M(i,j) = median([padIm(i-1,j-1),padIm(i,j-1),padIm(i+1,j-1),padIm(i-1,j)]);
               
            end % if s>smax
            
        end % if N(i,j) == 0
    end
end % for i=4:1:xlen-3

 %% Noise detected
for i=4:1:xlen-3
    for j=4:1:ylen-3
        
        if N(i,j) == 0
            %% clear tmp;
            tmp = abs(padIm(i-1:i+1,j-1:j+1)-padIm(i,j)*ones(3,3));
            
            d = max(tmp(:));
            
            if d<T1
                f = 0;
            else
                if d>=T1 && d<T2
                    f = (d-T1)/(T2-T1);
                else
                    f =1;
                end
            end % d<T1
            
            denoised_image(i,j) = (1-f)*padIm(i,j)+f*M(i,j);
        else
            denoised_image(i,j) = padIm(i,j);
        end %  if N(i,j) == 0
        
    end
end % for i=4:1:xlen-3

denoised_image = uint8(denoised_image(4:xlen-3,4:ylen-3));



% rowsize = size(input_image,1);
% colsize = size(input_image,2);
% denoised_image = zeros(rowsize, colsize);
% 
% padIm = double(padarray(input_image,[3 3],'symmetric'));
% 
% M = double(input_image);
% 
% N = zeros(size(padIm));
% N(padIm~=0 & padIm~=255) = 1;
% 
% T1 = 10;
% T2 = 30;
% 
% %% Noise detected
% for i=4:rowsize-3
%     for j=4:colsize-3
%         
%         if N(i,j) == 0 % If MIN or MAX value
%             s = 1;
%             g = N(i-s:i+s,j-s:j+s);
%                         
%             if sum(g(:)>0)
%                 
%                %% clear tmp;
%                tmp = padIm(i-s:i+s,j-s:j+s);
%                tmp = tmp(g(:)==1);
%                
%                M(i-1,j-1) = median(tmp);
%                
%             else
%                 
%                M(i-1,j-1) = median([padIm(i-1,j-1),padIm(i,j-1),padIm(i+1,j-1),padIm(i-1,j)]);
%                
%             end % if s>smax
%             
%         end % if N(i,j) == 0
%     end
% end % for i=4:1:xlen-3
% 
% 
% %% Noise detected
% for i=4:rowsize-3
%     for j=4:colsize-3
%         
%         if N(i,j) == 0  % If MIN or MAX value
%             %% clear tmp;
%             tmp = abs(padIm(i-1:i+1,j-1:j+1)-padIm(i,j)*ones(3,3));
% 
%             d = max(tmp(:));
% 
%             if d<T1
%                 f = 0;
%             else
%                 if d>=T1 && d<T2
%                     f = (d-T1)/(T2-T1);
%                 else
%                     f = 1;
%                 end
%             end % d<T1
% 
%             denoised_image(i-1,j-1) = (1-f)*padIm(i,j)+f*M(i-1,j-1);
%         else
%             denoised_image(i-1,j-1) = padIm(i,j);
%         end %  if N(i,j) == 0
%         
%     end
% end
% denoised_image = uint8(denoised_image);

fprintf("Fuzzy end\n");

end