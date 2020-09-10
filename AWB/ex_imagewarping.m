% ���� ���� (Image Warping) �׽�Ʈ �ڵ�
%
% 2012.06.13 ������ paeton@yonsei.ac.kr
% https://paeton.tistory.com/entry/%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%9B%8C%ED%95%91-image-Warping
clear all;
close all;
clc

% control point�� ���� ����
ctrNx = 5;
ctrNy = 5;

% ������ �̹��� �غ�
I = checkerboard(16);
[nrow, ncol] = size(I);    % �̹����� ������ ��´�

% control point grid ����
[x1, y1] = meshgrid( linspace(1,ncol,ctrNx), linspace(1,nrow,ctrNy) );

% ������ ���� �������� motion vector ����
amp  = 5;  % motion vector�� ũ�⸦ Ű���ִ� factor
%Vx = padarray(randn([ctrNy-2 ctrNx-2])*amp,[1, 1]);   % �ܰ��� motion vector�� �������� �ʵ��� ����
%Vy = padarray(randn([ctrNy-2 ctrNx-2])*amp,[1, 1]);
%Vx = randn([ctrNy ctrNx]);   % �ܰ��� motion vector�� �������� �ʵ��� ����
%Vy = randn([ctrNy ctrNx]);
xx = [100 120 140 160 180];
Vx = [xx; xx; xx; xx; xx;];   % �ܰ��� motion vector�� �������� �ʵ��� ����
Vy = [xx; xx; xx; xx; xx;]; 

% motion vector�� �̿��ؼ� deformation map Ux, Uy ����
[x2, y2] = meshgrid( 1:ncol, 1:nrow);
Ux = interp2( x1, y1, Vx, x2, y2 );
Uy = interp2( x1, y1, Vy, x2, y2 );

% pixel�� �̵��ؼ� warping ����
Iwarped = interp2(x2, y2, I, x2 - Ux, y2 - Uy );

% ǥ��
figure;
subplot(1,2,1);
imagesc(I, [ 0 1]); colormap gray; axis image;
hold on;
scatter( x1(:), y1(:), 25, 'g');

subplot(1,2,2);
imagesc(Iwarped,[0 1]); colormap gray; axis image;
hold on;
scatter( x1(:)+Vx(:), y1(:)+Vy(:), 25, 'g');