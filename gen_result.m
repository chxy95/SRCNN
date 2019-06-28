%close all;
%clear all;

%% read ground truth image
%filepaths = dir(fullfile('images','*.bmp'));
%for i = 1 : length(filepaths)
%im = imread(fullfile('images',filepaths(i).name));

im  = imread('images\butterfly_GT.bmp');
img_ori = im;

%% set parameters
up_scale = 3;
model = 'model\weights.mat';
%model = 'model\9-5-5(ImageNet)\x3.mat';
%model = 'model\9-1-5(91 images)\x3.mat';

% up_scale = 3;
% model = 'model\9-3-5(ImageNet)\x3.mat';
% up_scale = 3;
% model = 'model\9-1-5(91 images)\x3.mat';
% up_scale = 2;
% model = 'model\9-5-5(ImageNet)\x2.mat'; 
% up_scale = 4;
% model = 'model\9-5-5(ImageNet)\x4.mat';

%% work on illuminance only
if size(im,3)>1
    im = rgb2ycbcr(im);
    im_cbcr = im(:, :, 2:3);
    im = im(:, :, 1);
end

im_gnd = modcrop(im, up_scale);
im_gnd = single(im_gnd)/255;

%% bicubic interpolation
im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
im_b = imresize(im_l, up_scale, 'bicubic');

%% SRCNN
im_h = SRCNN(model, im_b);

%% remove border
im_h = shave(uint8(im_h * 255), [up_scale, up_scale]);
im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);

%% compute PSNR
psnr_bic = compute_psnr(im_gnd,im_b);
psnr_srcnn = compute_psnr(im_gnd,im_h);

%% show results
fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
fprintf('PSNR for SRCNN Reconstruction: %f dB\n', psnr_srcnn);

im_cbcr =  modcrop(im_cbcr, up_scale);
im_cbcr = shave(im_cbcr, [up_scale, up_scale]);

img_ori = shave(modcrop(img_ori, up_scale), [up_scale, up_scale]);
%figure, imshow(img_ori); title('img_ori');

img1 = cat(3, im_b, im_cbcr);
img1 = ycbcr2rgb(img1);
%figure, imshow(img1); title('Bicubic Interpolation');

img2 = cat(3, im_h, im_cbcr);
img2 = ycbcr2rgb(img2);
%figure, imshow(img2); title('SRCNN Reconstruction');

img3 = cat(3, im_gnd, im_cbcr);
img3 = ycbcr2rgb(img3);
%figure, imshow(img3); title('gnd');

%imwrite(img1, [filepaths(i).name 'Bicubic_Interpolation' '.bmp']);
imwrite(img2, [filepaths(i).name 'SRCNN_Reconstruction' '.bmp']);
%imwrite(img3, [filepaths(i).name 'Image_Groundtruth' '.bmp']);

%end
