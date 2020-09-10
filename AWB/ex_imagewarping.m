% 영상 워핑 (Image Warping) 테스트 코드
%
% 2012.06.13 김지한 paeton@yonsei.ac.kr
% https://paeton.tistory.com/entry/%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%9B%8C%ED%95%91-image-Warping
clear all;
close all;
clc

% control point의 개수 지정
ctrNx = 5;
ctrNy = 5;

% 워핑할 이미지 준비
I = checkerboard(16);
[nrow, ncol] = size(I);    % 이미지의 사이즈 얻는다

% control point grid 설정
[x1, y1] = meshgrid( linspace(1,ncol,ctrNx), linspace(1,nrow,ctrNy) );

% 무작위 값을 가지도록 motion vector 생성
amp  = 5;  % motion vector의 크기를 키워주는 factor
%Vx = padarray(randn([ctrNy-2 ctrNx-2])*amp,[1, 1]);   % 외곽의 motion vector는 움직이지 않도록 설정
%Vy = padarray(randn([ctrNy-2 ctrNx-2])*amp,[1, 1]);
%Vx = randn([ctrNy ctrNx]);   % 외곽의 motion vector는 움직이지 않도록 설정
%Vy = randn([ctrNy ctrNx]);
xx = [100 120 140 160 180];
Vx = [xx; xx; xx; xx; xx;];   % 외곽의 motion vector는 움직이지 않도록 설정
Vy = [xx; xx; xx; xx; xx;]; 

% motion vector를 이용해서 deformation map Ux, Uy 생성
[x2, y2] = meshgrid( 1:ncol, 1:nrow);
Ux = interp2( x1, y1, Vx, x2, y2 );
Uy = interp2( x1, y1, Vy, x2, y2 );

% pixel을 이동해서 warping 수행
Iwarped = interp2(x2, y2, I, x2 - Ux, y2 - Uy );

% 표시
figure;
subplot(1,2,1);
imagesc(I, [ 0 1]); colormap gray; axis image;
hold on;
scatter( x1(:), y1(:), 25, 'g');

subplot(1,2,2);
imagesc(Iwarped,[0 1]); colormap gray; axis image;
hold on;
scatter( x1(:)+Vx(:), y1(:)+Vy(:), 25, 'g');