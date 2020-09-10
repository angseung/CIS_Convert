function denoised_image = RSEPD(input_image, Ts)

rowsize = size(input_image,1);
colsize = size(input_image,2);
denoised_image = zeros(rowsize, colsize);

padIm = double(padarray(input_image,[1 1],'symmetric'));
%dbg_restored = zeros(rowsize, colsize);
% padIm = double(padarray(input_image,[1 1],0,'both'));
% MINinW = 255;
% MAXinW = 0;

% rowsize = rowsize-2;
% colsize = colsize-2;
% denoised_image = zeros(rowsize-2, colsize-2);
% padIm = input_image;

for i=2:rowsize+1
	for j=2:colsize+1
		% ***---- Extreme Data Detector ----***
% 		W = padIm(i-1:i+1,j-1:j+1);
%         tmpMINinW = min(min(W));
% 		tmpMAXinW = max(max(W));
%         if tmpMINinW < MINinW
%             MINinW = tmpMINinW;
%         end
%         if tmpMAXinW > MAXinW
%             MAXinW = tmpMAXinW;
%         end
        MINinW = 0;
        MAXinW = 255;
		pi = 0;
		
		if((padIm(i,j) == MINinW) || (padIm(i,j) == MAXinW))
            % if((padIm(i-1,j-1) == MINinW) && (padIm(i-1,j) == MINinW) && (padIm(i-1,j+1) == MINinW))
                
            % if((padIm(i-1,j-1) == MAXinW) && (padIm(i-1,j) == MAXinW) && (padIm(i-1,j+1) == MAXinW))
			pi = 1; % Noisy pixel
		end
		
		if (pi == 0)  % If not noisy pixel
			denoised_image(i-1,j-1) = padIm(i,j);
        else % If noisy pixel
			b = 0;
			if(padIm(i+1,j) == MINinW || padIm(i+1,j) == MAXinW)
				b = 1; % Check surrounding pixel
			end
			
			% ***---- Edge-Oriented Noise filter ----***
			if(b==1) % If surrounding pixel is noisy
                f_hat = (padIm(i-1,j-1)+2*padIm(i-1,j)+padIm(i-1,j+1))/4;
            else % If surrounding pixel is not noisy
             	Da = abs(padIm(i-1,j-1) - padIm(i+1,j));
				Db = abs(padIm(i-1,j)   - padIm(i+1,j));
				Dc = abs(padIm(i-1,j+1) - padIm(i+1,j));
				
				f_hat_Da = (padIm(i-1,j-1) + padIm(i+1,j))/2;
				f_hat_Db = (padIm(i-1,j)   + padIm(i+1,j))/2;
				f_hat_Dc = (padIm(i-1,j+1) + padIm(i+1,j))/2;
				
                D = [Da Db Dc];
                Dmin = min(D);
                
                if(Dmin == Da)
                    f_hat = f_hat_Da;
                elseif(Dmin == Db)
                    f_hat = f_hat_Db;
                else
                    f_hat = f_hat_Dc;
                end
			end
			
			% ***---- Impulse Arbiter ----***
			if (abs(padIm(i,j)-f_hat) > Ts)
				denoised_image(i-1,j-1) = f_hat;
                padIm(i,j) = f_hat;
                %dbg_restored(i,j) = 255;
			else
				denoised_image(i-1,j-1) = padIm(i,j);
			end
		end
	end
end

fprintf("RSEPD end\n");
    end
