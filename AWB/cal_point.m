function [Xf, Yf] = cal_point(Xs, Ys, angle, dist)
    % Calculate point (Xf,Yf) at specific distance from start point (Xs,Ys)
    % Using angle (angle) and distance (dist)
    % Yf = angle*(Xf-Xs)+Ys
    % (Xf-Xs)^2 + (Yf-Ys)^2 = dist^2
    
    t = angle*Xs - Ys;
    R0 = angle*angle + 1;
    R1 = -2*(Xs + angle*Ys + angle*t);
    R2 = Xs*Xs + Ys*Ys + t*t + 2*Ys*t - dist^2;
    
    p = [R0, R1, R2];
    r = roots(p);
    Xf = r(r>0);
    Yf = angle*Xf - t;
end