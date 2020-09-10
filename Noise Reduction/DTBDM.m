function denoised_image = DTBDM(input_image)

rowsize = size(input_image,1);
colsize = size(input_image,2);
denoised_image = zeros(rowsize, colsize);

padIm = double(padarray(input_image,[1 1],'symmetric'));
MINinW = 255;
MAXinW = 0;

Th_IMa = 20;
Th_IMb = 25;
Th_FMa = 40;
Th_FMb = 80;
Th_SMa = 15;
Th_SMb = 60;

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

for i=2:rowsize+1
	for j=2:colsize+1
        % -------- Isolation module -------- %
		TopHalf_max = max([padIm(i-1,j-1), padIm(i,j-1), padIm(i+1,j-1), padIm(i-1,j)]);
        TopHalf_min = min([padIm(i-1,j-1), padIm(i,j-1), padIm(i+1,j-1), padIm(i-1,j)]);
        BottomHalf_max = max([padIm(i,j+1), padIm(i-1,j+1), padIm(i,j+1), padIm(i+1,j+1)]);
        BottomHalf_min = min([padIm(i,j+1), padIm(i-1,j+1), padIm(i,j+1), padIm(i+1,j+1)]);
        
        TopHalf_diff = TopHalf_max - TopHalf_min;
        BottomHalf_diff = BottomHalf_max - BottomHalf_min;
        
        if((TopHalf_diff >= Th_IMa) || (BottomHalf_diff >= Th_IMa))
            decision1 = 1; % Noise suspected
        else
            decision1 = 0;
        end
        
        if((abs(padIm(i,j)-TopHalf_max)>=Th_IMb) || (abs(padIm(i,j)-TopHalf_min)>=Th_IMb))
            decision2 = 1;  % Noise suspected
        else
            decision2 = 0;
        end
        
        if((abs(padIm(i,j)-BottomHalf_max)>=Th_IMb) || (abs(padIm(i,j)-BottomHalf_min)>=Th_IMb))
            decision2 = 1;  % Noise suspected
        else
            decision2 = decision2;
        end
        
        % -------- Fringe module -------- %
        % a, h = padIm(i-1,j-1), padIm(i+1,j+1)
        if((abs(padIm(i-1,j-1)-padIm(i,j))>=Th_FMa) || ...
           (abs(padIm(i+1,j+1)-padIm(i,j))>=Th_FMa) || ...
           (abs(padIm(i-1,j-1)-padIm(i+1,j+1))>=Th_FMb))
            decision3 = 1;  % Noise suspected
        else
            decision3 = 0;
        end
        
        % c, f = padIm(i+1,j-1), padIm(i-1,j+1)
        if((abs(padIm(i+1,j-1)-padIm(i,j))>=Th_FMa) || ...
           (abs(padIm(i-1,j+1)-padIm(i,j))>=Th_FMa) || ...
           (abs(padIm(i+1,j-1)-padIm(i-1,j+1))>=Th_FMb))
            decision3 = decision3;  % Noise suspected
        else
            decision3 = 0;
        end
        
        % b, g = padIm(i,j-1), padIm(i,j+1)
        if((abs(padIm(i,j-1)-padIm(i,j))>=Th_FMa) || ...
           (abs(padIm(i,j+1)-padIm(i,j))>=Th_FMa) || ...
           (abs(padIm(i,j-1)-padIm(i,j+1))>=Th_FMb))
            decision3 = decision3;  % Noise suspected
        else
            decision3 = 0;
        end
        
        % d, e = padIm(i-1,j), padIm(i+1,j)
        if((abs(padIm(i-1,j)-padIm(i,j))>=Th_FMa) || ...
           (abs(padIm(i+1,j)-padIm(i,j))>=Th_FMa) || ...
           (abs(padIm(i-1,j)-padIm(i+1,j))>=Th_FMb))
            decision3 = decision3;  % Noise suspected
        else
            decision3 = 0;
        end
        
        % -------- Similarity module -------- %
        sorted = sort(padIm(i-1:i+1,j-1:j+1));
        
        maxij = sorted(6) + Th_SMa;
        minij = sorted(4) - Th_SMa;
        tmax = sorted(5) + Th_SMb;
        tmin = sorted(5) - Th_SMb;
        
        if (maxij <= tmax)
            Nmax = maxij;
        else
            Nmax = tmax;
        end
        
        if (minij >= tmin)
            Nmin = minij;
        else
            Nmin = tmin;
        end
        
        if (padIm(i,j) >= Nmax || padIm(i,j) <= Nmin)
            decision4 = 1;
        else
            decision4 = 0;
        end
        
        % -------- Edge-preserving image filter -------- %
        if (decision1 == 1 || decision2 == 1 || decision3 == 1 || decision4 == 1)
            noise_detected = 1;
        else
            noise_detected = 0;
        end
        
        if (noise_detected == 0)
            f_bar = padIm(i,j);
        else
            b = 0;
            if(padIm(i,j-1) == MINinW || padIm(i,j-1) == MAXinW)
                b = b + 2^4; % Check surrounding pixel
            end
            if(padIm(i,j+1) == MINinW || padIm(i,j+1) == MAXinW)
				b = b + 2^3; % Check surrounding pixel
            end
            if(padIm(i+1,j-1) == MINinW || padIm(i+1,j-1) == MAXinW)
				b = b + 2^2; % Check surrounding pixel
            end
            if(padIm(i+1,j) == MINinW || padIm(i+1,j) == MAXinW)
				b = b + 2^1; % Check surrounding pixel
            end
			if(padIm(i+1,j+1) == MINinW || padIm(i+1,j+1) == MAXinW)
				b = b + 2^0; % Check surrounding pixel
			end
            
			% ---- Edge-Oriented Noise filter ----
			if(b==31) % If surrounding pixel is noisy
                if(padIm(i-1,j-1) == MINinW && padIm(i-1,j) == MINinW && padIm(i-1,j+1) == MINinW)
                    f_hat = MINinW;
                elseif(padIm(i-1,j-1) == MAXinW && padIm(i-1,j) == MAXinW && padIm(i-1,j+1) == MAXinW)
                    f_hat = MAXinW;
                else
                    f_hat = (padIm(i-1,j-1)+2*padIm(i-1,j)+padIm(i-1,j+1))/4;
                end
            else % If surrounding pixel is not noisy
				Dlist = b_D(b+1,:); % list of possible D
                Dmin = 100000;
                meanidx = 1; % To find out index of Dmin in Dlist
                for z=1:nnz(Dlist)
                    idx = idx_D(Dlist(z),:);
                    % Get D value
                    D = abs(padIm(i+idx(1),j+idx(2)) - padIm(i+idx(3),j+idx(4)));
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
			sorted = sort([f_hat, padIm(i-1,j), padIm(i,j-1), padIm(i+1,j), padIm(i,j+1)]);
            f_bar = sorted(3);
		end
		denoised_image(i-1,j-1) = f_bar;
        padIm(i,j) = f_bar;
	end
end

fprintf("DTBDM end\n");
end

