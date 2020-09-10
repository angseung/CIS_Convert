clc; clear;


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

idx_pixel = zeros(12,2);
idx_pixel(1 ,:) = [8, 3] ;
idx_pixel(2 ,:) = [8, 0] ;
idx_pixel(3 ,:) = [8, 1] ;
idx_pixel(4 ,:) = [7, 0] ;
idx_pixel(5 ,:) = [7, 1] ;
idx_pixel(6 ,:) = [7, 2] ;
idx_pixel(7 ,:) = [6, 1] ;
idx_pixel(8 ,:) = [6, 2] ;
idx_pixel(9 ,:) = [6, 5] ;
idx_pixel(10,:) = [5, 3] ;
idx_pixel(11,:) = [5, 0] ;
idx_pixel(12,:) = [3, 2] ;

fid = fopen('./write_mapping_D.txt','w');

for i=1:31
    fprintf(fid,"5'd%d : begin\n",i-1);
    Dlist = b_D(i,:);
    for j=1:nnz(Dlist)
        if j==1
            fprintf(fid,"\t// ");
            for z=1:nnz(Dlist)
                fprintf(fid,"D%d ",Dlist(z));
            end
            fprintf(fid,"\n");
        end
        D = Dlist(j);
        pixelidx = idx_pixel(D,:);
        fprintf(fid,"\t// D%d = [%d], [%d]\n",D,pixelidx(1),pixelidx(2));
        fprintf(fid,'\to_D%da <= pixelinW[%d];\n',j,pixelidx(1));
        fprintf(fid,'\to_D%db <= pixelinW[%d];\n',j,pixelidx(2));
    end
    fprintf(fid,"end\n");
end

fclose(fid);