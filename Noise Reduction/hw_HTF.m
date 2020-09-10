function denoised_image = hw_HTF(input_image, Th_diff, fuzzy_on)

rowsize = size(input_image,1);
colsize = size(input_image,2);
denoised_image = zeros(rowsize, colsize);

padIm = double(padarray(input_image,[1 1],'symmetric'));

row_buffer = zeros(colsize,1);
MINinW = 1;
MAXinW = 255;

Th1_Fuzzy = 10;
Th2_Fuzzy = 30;

for i=2:rowsize+1
    for j=2:colsize+1
        % impulse checking
        if(padIm(i,j) <= MINinW || padIm(i,j) >= MAXinW)
            impulse = 1;
        else
            impulse = 0;
        end
        
        if (impulse == 1)
            % If an impulse pixel
            med_sorted = sort([padIm(i-1,j-1),padIm(i-1,j),padIm(i-1,j+1),...
                padIm(i,j-1),padIm(i,j+1),padIm(i+1,j-1),padIm(i+1,j),...
                padIm(i+1,j+1)]);
            
            diff = abs(med_sorted(6)-med_sorted(3));
            
            if(diff <= Th_diff)
                mode = 0; % Plane : similar surrounding pixels
            else
                mode = 1; % Edge : fluctuating surrounding pixels
            end
            
            if (mode == 0) % Plane
                dist_plane1 = abs(padIm(i-1,j-1)-padIm(i-1,j)) + ...
                              abs(padIm(i-1,j-1)-padIm(i-1,j+1)) + ...
                              abs(padIm(i-1,j)-padIm(i-1,j+1));
                          
                dist_plane2 = abs(padIm(i-1,j+1)-padIm(i,j+1)) + ...
                              abs(padIm(i-1,j+1)-padIm(i+1,j+1)) + ...
                              abs(padIm(i,j+1)-padIm(i+1,j+1));
                          
                dist_plane3 = abs(padIm(i+1,j-1)-padIm(i+1,j)) + ...
                              abs(padIm(i+1,j-1)-padIm(i+1,j+1)) + ...
                              abs(padIm(i+1,j)-padIm(i+1,j+1));
                          
                dist_plane4 = abs(padIm(i-1,j-1)-padIm(i,j-1)) + ...
                              abs(padIm(i-1,j-1)-padIm(i+1,j-1)) + ...
                              abs(padIm(i,j-1)-padIm(i+1,j-1));
                
%                 dist_plane5 = abs(padIm(i-1,j)-padIm(i-1,j-1)) + ...
%                               abs(padIm(i-1,j)-padIm(i,j)) + ...
%                               abs(padIm(i-1,j-1)-padIm(i,j));
%                           
%                 dist_plane6 = abs(padIm(i,j-1)-padIm(i+1,j-1)) + ...
%                               abs(padIm(i,j-1)-padIm(i+1,j)) + ...
%                               abs(padIm(i+1,j-1)-padIm(i+1,j));
%                           
%                 dist_plane7 = abs(padIm(i+1,j)-padIm(i+1,j+1)) + ...
%                               abs(padIm(i+1,j)-padIm(i,j+1)) + ...
%                               abs(padIm(i+1,j+1)-padIm(i,j+1));
%                           
%                 dist_plane8 = abs(padIm(i,j+1)-padIm(i-1,j+1)) + ...
%                               abs(padIm(i,j+1)-padIm(i-1,j)) + ...
%                               abs(padIm(i-1,j+1)-padIm(i-1,j));
                
                plane1 = (padIm(i-1,j-1)+padIm(i-1,j)+padIm(i-1,j+1))/3;
                plane2 = (padIm(i-1,j+1)+padIm(i,j+1)+padIm(i+1,j+1))/3;
                plane3 = (padIm(i+1,j-1)+padIm(i+1,j)+padIm(i+1,j+1))/3;
                plane4 = (padIm(i-1,j-1)+padIm(i,j-1)+padIm(i+1,j-1))/3;
%                 plane5 = (padIm(i-1,j)+padIm(i-1,j-1)+padIm(i,j))/3;
%                 plane6 = (padIm(i,j-1)+padIm(i+1,j-1)+padIm(i+1,j))/3;
%                 plane7 = (padIm(i+1,j)+padIm(i+1,j+1)+padIm(i,j+1))/3;
%                 plane8 = (padIm(i,j+1)+padIm(i-1,j+1)+padIm(i-1,j))/3;
                
                [~, minplaneidx] = min([dist_plane1, dist_plane2, dist_plane3, dist_plane4]);
                                     %   dist_plane5, dist_plane6, dist_plane7, dist_plane8]);
                
                if (minplaneidx==1)
                    f_bar = plane1;
                elseif(minplaneidx==2)
                    f_bar = plane2;
                elseif(minplaneidx==3)
                    f_bar = plane3;
                elseif(minplaneidx==4)
                    f_bar = plane4;
%                 elseif(minplaneidx==5)
%                     f_bar = plane5; 
%                 elseif(minplaneidx==6)
%                     f_bar = plane6;
%                 elseif(minplaneidx==7)
%                     f_bar = plane7; 
%                 elseif(minplaneidx==8)
%                     f_bar = plane8;  
                else
                    f_bar = 0;
                end
                
            else % Edge
                dist_edge1 = abs(padIm(i-1,j-1)-padIm(i+1,j+1));
                dist_edge2 = abs(padIm(i-1,j)  -padIm(i+1,j));
                dist_edge3 = abs(padIm(i-1,j+1)-padIm(i+1,j-1));
                dist_edge4 = abs(padIm(i,j-1)  -padIm(i,j+1));

                edge1 = (padIm(i-1,j-1)+padIm(i+1,j+1))/2;
                edge2 = (padIm(i-1,j)  +padIm(i+1,j))/2;
                edge3 = (padIm(i-1,j+1)+padIm(i+1,j-1))/2;
                edge4 = (padIm(i,j-1)  +padIm(i,j+1))/2;
                
                [~, minedgeidx] = min([dist_edge1, dist_edge2, dist_edge3, dist_edge4]);
                
                if (minedgeidx==1)
                    f_bar = edge1;
                elseif(minedgeidx==2)
                    f_bar = edge2;
                elseif(minedgeidx==3)
                    f_bar = edge3;
                elseif(minedgeidx==4)
                    f_bar = edge4;
                else
                    f_bar = 0; % Default
                end
            end
        else
            % Not an impulse pixel
            f_bar = padIm(i,j);
        end
        
        if(fuzzy_on==1)
            % Fuzzy concept
            tmp = abs(padIm(i-1:i+1,j-1:j+1)-padIm(i,j)*ones(3,3));
            d = max(tmp(:));

            if d<Th1_Fuzzy
                rate = 0;
            else
                if d>=Th1_Fuzzy && d<Th2_Fuzzy
                    rate = (d-Th1_Fuzzy)/(Th2_Fuzzy-Th1_Fuzzy);
                else
                    rate =1;
                end
            end % d<T1

            f_hat = (1-rate)*padIm(i,j)+rate*f_bar;
            if (f_hat >= 255)
                f_hat = 255;
            end
        else
            f_hat = f_bar;
        end
        
        row_buffer(j-1) = f_hat;
    end
    row_buffer = round(row_buffer);
    denoised_image(i-1,:) = row_buffer;
    padIm(i,:) = padarray(row_buffer,[1],'symmetric');
end

% denoised_image = uint8(denoised_image);
fprintf("hw_HTF end\n");
end