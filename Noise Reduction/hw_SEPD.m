function denoised_image = hw_SEPD(input_image, Ts)

rowsize = size(input_image,1);
colsize = size(input_image,2);
denoised_image = zeros(rowsize, colsize);

padIm = double(padarray(input_image,[1 1],'symmetric'));
% MINinW = 255;
% MAXinW = 0;

b_D = zeros(31,4);
b_D(1  ,:) = [2, 5, 8,  10];
b_D(2  ,:) = [3, 5, 8,  10];
b_D(3  ,:) = [2, 8, 10, 12];
b_D(4  ,:) = [1, 6, 8,  10];
b_D(5  ,:) = [2, 5, 7,  10];
b_D(6  ,:) = [3, 5, 7,  10];
b_D(7  ,:) = [2, 4, 9,  10];
b_D(8  ,:) = [1, 9, 10, 0 ];
b_D(9  ,:) = [2, 5, 8,  11];
b_D(10 ,:) = [3, 5, 7,  9 ];
b_D(11 ,:) = [2, 6, 8,  11];
b_D(12 ,:) = [6, 8, 9,  0 ];
b_D(13 ,:) = [2, 5, 9,  11];
b_D(14 ,:) = [3, 5, 9,  0 ];
b_D(15 ,:) = [2, 4, 9,  11];
b_D(16 ,:) = [9, 0, 0,  0 ];
b_D(17 ,:) = [2, 5, 8,  12];
b_D(18 ,:) = [1, 5, 8,  12];
b_D(19 ,:) = [2, 4, 8,  12];
b_D(20 ,:) = [1, 6, 8,  12];
b_D(21 ,:) = [1, 2, 5,  7 ];
b_D(22 ,:) = [1, 5, 7,  0 ];
b_D(23 ,:) = [1, 2, 4,  0 ];
b_D(24 ,:) = [1, 0, 0,  0 ];
b_D(25 ,:) = [2, 5, 6,  8 ];
b_D(26 ,:) = [3, 5, 6,  8 ];
b_D(27 ,:) = [2, 4, 6,  8 ];
b_D(28 ,:) = [6, 8, 0,  0 ];
b_D(29 ,:) = [2, 4, 5,  7 ];
b_D(30 ,:) = [3, 5, 7,  0 ];
b_D(31 ,:) = [2, 4, 0,  0 ];

idx_D = zeros(12,4);
idx_D(1 ,:) = [-1, -1, 0, 1];   % [8], [3]
idx_D(2 ,:) = [-1, -1, 1, 1];   % [8], [0]
idx_D(3 ,:) = [-1, -1, 1, 0];   % [8], [1]
idx_D(4 ,:) = [-1,  0, 1, 1];   % [7], [0]
idx_D(5 ,:) = [-1,  0, 1, 0];   % [7], [1]
idx_D(6 ,:) = [-1,  0, 1, -1];  % [7], [2]
idx_D(7 ,:) = [-1,  1, 1, 0];   % [6], [1]
idx_D(8 ,:) = [-1,  1, 1, -1];  % [6], [2]
idx_D(9 ,:) = [-1,  1, 0, -1];  % [6], [5]
idx_D(10,:) = [0,  -1, 0, 1];   % [5], [3]
idx_D(11,:) = [0 , -1, 1, 1];   % [5], [0]
idx_D(12,:) = [0 ,  1, 1, -1];  % [3], [2]

dbg_b = zeros(colsize,1);
dbg_Dmean = zeros(colsize,4);
dbg_Dsub  = zeros(colsize,12);
dbg_ia    = zeros(colsize,5);
row_buffer = zeros(colsize,1);
dgb_readData = zeros(colsize,3);

MINinW = 1;
MAXinW = 255;
for i=2:rowsize+1
	for j=2:colsize+1
		% ---- Extreme Data Detector ----
% 		W = padIm(i-1:i+1,j-1:j+1);
%         tmpMINinW = min(min(W));
% 		tmpMAXinW = max(max(W));
%         if tmpMINinW < MINinW
%             MINinW = tmpMINinW;
%         end
%         if tmpMAXinW > MAXinW
%             MAXinW = tmpMAXinW;
%         end
        
        
		pi = 0;
		
		if((padIm(i,j) <= MINinW) || (padIm(i,j) >= MAXinW))
			pi = 1; % Noisy pixel
		end
		
		if (pi == 0)  % If not noisy pixel
			f_bar = padIm(i,j);
        else % If noisy pixel
			b = 0;
            if(padIm(i,j-1) <= MINinW || padIm(i,j-1) >= MAXinW)
                b = b + 2^4; % Check surrounding pixel [5]
            end
            if(padIm(i,j+1) <= MINinW || padIm(i,j+1) >= MAXinW)
				b = b + 2^3; % Check surrounding pixel [3]
            end
            if(padIm(i+1,j-1) <= MINinW || padIm(i+1,j-1) >= MAXinW)
				b = b + 2^2; % Check surrounding pixel [2]
            end
            if(padIm(i+1,j) <= MINinW || padIm(i+1,j) >= MAXinW)
				b = b + 2^1; % Check surrounding pixel [1]
            end
			if(padIm(i+1,j+1) <= MINinW || padIm(i+1,j+1) >= MAXinW)
				b = b + 2^0; % Check surrounding pixel [0]
			end
            if i==2
                dbg_b(j-1) = b;
            end
			% ---- Edge-Oriented Noise filter ----
			if(b==31) % If surrounding pixel is noisy
%                 if(padIm(i-1,j-1) <= MINinW && padIm(i-1,j) <= MINinW && padIm(i-1,j+1) <= MINinW)
%                     % f_hat = MINinW;
%                     f_hat = padIm(i,j);
%                 elseif(padIm(i-1,j-1) >= MAXinW && padIm(i-1,j) >= MAXinW && padIm(i-1,j+1) >= MAXinW)
%                     % f_hat = MAXinW;
%                     f_hat = padIm(i,j);
%                 else
%                     ss = sort(padIm(i-1:i+1,j-1:j+1));
%                     f_hat = ss(5);
%                     % f_hat = (padIm(i-1,j-1)+2*padIm(i-1,j)+padIm(i-1,j+1))/4;
%                 end
                ss = sort(padIm(i-1:i+1,j-1:j+1));
                f_hat = ss(5);
                
%                 if((padIm(i-1,j-1) == MINinW || padIm(i-1,j-1) == MAXinW) && ...
%                    (padIm(i-1,j)   == MINinW || padIm(i-1,j)   == MAXinW) && ...
%                    (padIm(i-1,j+1) == MINinW || padIm(i-1,j+1) == MAXinW))
%                     if(nummax > nummin)
%                         f_hat = MAXinW;
%                     else
%                         f_hat = MINinW;
%                     end
%                 else
%                     f_hat = (padIm(i-1,j-1)+2*padIm(i-1,j)+padIm(i-1,j+1))/4;
%                 end
            else % If surrounding pixel is not noisy
				Dlist = b_D(b+1,:); % list of possible D
                Dmin = 100000;
                meanidx = 1; % To find out index of Dmin in Dlist
                for z=1:nnz(Dlist)
                    idx = idx_D(Dlist(z),:);
                    % Get D value
                    x = padIm(i+idx(1),j+idx(2));
                    y = padIm(i+idx(3),j+idx(4));
                    D = abs(x - y);
                    if i==2
                        dbg_Dsub(j-1,3*(z-1)+1:3*z) = [x, y, D];
                    end
                    % Find Dmin
                    if D < Dmin
                        Dmin = D;
                        meanidx = z; % get index of Dmin
                    end
                end
                Dminidx = Dlist(meanidx); % Dminidx : index of Dmin in Dlist
                idx = idx_D(Dminidx,:);
                f_hat = (padIm(i+idx(1),j+idx(2)) + padIm(i+idx(3),j+idx(4)))/2;
			end
			
			% ---- Impulse Arbiter ----
            if i==2
                dbg_ia(j-1,1) = 1;          % pi
                dbg_ia(j-1,2) = padIm(i,j); % f
                dbg_ia(j-1,3) = f_hat;      % f_hat
            end
			if (abs(padIm(i,j)-f_hat) > Ts)
				f_bar = f_hat;
                if i==2
                    dbg_ia(j-1,4) = 1;      % comp
                end
			else
				f_bar = padIm(i,j);
            end
        end
        if i==2
            dbg_ia(j-1,5) = f_bar; 
        end
        row_buffer(j-1) = f_bar;
		%denoised_image(i-1,j-1) = f_bar;
        %padIm(i,j) = f_bar;
    end
    row_buffer = round(row_buffer);
    denoised_image(i-1,:) = row_buffer;
    padIm(i,:) = padarray(row_buffer,[1],'symmetric');
%     dgb_readData(:,1) = row_buffer;
%     dgb_readData(:,2) = input_image(i,:)';
%     dgb_readData(:,3) = input_image(i+1,:)';
end

fprintf("hw_SEPD end\n");

end