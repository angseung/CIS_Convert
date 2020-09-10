function [Xmin, Xmax, Ymin, Ymax] = get_vertex(mask_chart)
Xmin = 0;
Ymin = 0;

imXmax = size(mask_chart,1);
imYmax = size(mask_chart,2);

a = -1*imYmax/imXmax;
pass = 0;

if(imXmax > imYmax)
    for x_intercept=1:imXmax
        for x=1:x_intercept
            y = round(a*(x - x_intercept));
            if(x>0 && y>0 && x<=imXmax && y<=imYmax)
                if(mask_chart(x,y) == true)
                    Xmin = x;
                    Ymin = y;
                    break;
                end
            end
        end
        if(x>0 && y>0 && x<=imXmax && y<=imYmax)
            if(mask_chart(x,y) == true)
                pass = 1;
                break;
            end
        end
    end
    
    if(pass==0)
        for x_intercept=imXmax:2*imXmax
            for x=(x_intercept-imXmax):imXmax
                y = round(a*(x - x_intercept));
                if(x>0 && y>0 && x<=imXmax && y<=imYmax)
                    if(mask_chart(x,y) == true)
                        Xmin = x;
                        Ymin = y;
                        break;
                    end
                end
            end
            if(x>0 && y>0 && x<=imXmax && y<=imYmax)
                if(mask_chart(x,y) == true)
                    break;
                end
            end
        end
    end
else
    for y_intercept=1:imYmax
        for y=1:y_intercept
            x = round(1/a*(y-y_intercept));
            if(x>0 && y>0 && x<=imXmax && y<=imYmax)
                if(mask_chart(x,y) == true)
                    Xmin = x;
                    Ymin = y;
                    break;
                end
            end
        end
        if(x>0 && y>0 && x<=imXmax && y<=imYmax)
            if(mask_chart(x,y) == true)
                pass = 1;
                break;
            end
        end
    end
    
    if(pass==0)
        for y_intercept=imYmax:2*imYmax
            for y=(y_intercept-imYmax):imYmax
                x = round(1/a*(y - y_intercept));
                if(x>0 && y>0 && x<=imXmax && y<=imYmax)
                    if(mask_chart(x,y) == true)
                        Xmin = x;
                        Ymin = y;
                        break;
                    end
                end
            end
            if(x>0 && y>0 && x<=imXmax && y<=imYmax)
                if(mask_chart(x,y) == true)
                    break;
                end
            end
        end   
    end
end

Xmax = Xmin + 198;
Ymax = Ymin + 298;

% truepoint = find(mask_chart==true,1);
% Xmin = truepoint - imXmax*floor(truepoint/imXmax);
% Ymin = floor(truepoint/imXmax);
% Xmax = Xmin + 200;
% Ymax = Ymin + 300;


% vertexs = zeros(4,2);

% for x=1:imXmax
%     for y=1:imYmax
%         if(mask_chart(x,y) == true)
%             vertexs(1,1) = x;
%             vertexs(1,2) = y;
%             break;
%         end
%     end
%     if(mask_chart(x,y) == true)
%        break; 
%     end
% end
% 
% for y=imYmax:-1:1
%     for x=1:imXmax
%         if(mask_chart(x,y) == true)
%             vertexs(2,1) = x;
%             vertexs(2,2) = y;
%             break;
%         end       
%     end
%     if(mask_chart(x,y) == true)
%        break; 
%     end
% end
% 
% for x=imXmax:-1:1
%     for y=imYmax:-1:1
%         if(mask_chart(x,y) == true)
%             vertexs(3,1) = x;
%             vertexs(3,2) = y;
%             break;
%         end       
%     end
%     if(mask_chart(x,y) == true)
%        break; 
%     end
% end
% 
% for y=1:imYmax
%     for x=imXmax:-1:1
%         if(mask_chart(x,y) == true)
%             vertexs(4,1) = x;
%             vertexs(4,2) = y;
%             break;
%         end       
%     end
%     if(mask_chart(x,y) == true)
%        break; 
%     end
% end
% 
% Length1 = (vertexs(1,1)-vertexs(2,1))^2 + (vertexs(1,2)-vertexs(2,2))^2;
% Length2 = (vertexs(2,1)-vertexs(3,1))^2 + (vertexs(2,2)-vertexs(3,2))^2;
% if(Length1 > Length2)
%     Xmin = vertexs(1,1);
%     Ymin = vertexs(1,2);
%     Xmax = vertexs(3,1);
%     Ymax = vertexs(3,2);    
% else
%     Xmin = vertexs(1,1);
%     Ymin = vertexs(4,2);
%     Xmax = vertexs(3,1);
%     Ymax = vertexs(2,2);
% end

end