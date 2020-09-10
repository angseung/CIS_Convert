% load('incremental_image.mat');
output_image = zeros(1080, 1920,3);

M = textread('./output_image.txt','%24c');

for i=1:1080
    if mod(i,100)==0
        fprintf("%d row is processing\n",i);
    end
    for j=1:1920
        idx= 1920*(i-1)+j;
        red   = bin2dec(char(M(idx,17:24)));
        green = bin2dec(char(M(idx,9:16)));
        blue  = bin2dec(char(M(idx,1:8)));
        output_image(i,j,1) = uint8(red);
        output_image(i,j,2) = uint8(green);
        output_image(i,j,3) = uint8(blue);
    end
end

imshow(lin2rgb(uint8(output_image)));

