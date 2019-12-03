%This script does the graduated RPCA with motion coherency constraint on a video

%paths and parameters
name = 'test';
filename = ['input/' name '.avi'];
outdir = ['output/' name '/'];
if ~exist(outdir,'dir')
    mkdir(outdir);
end

%read data
obj = VideoReader(filename);
numFrames = obj.NumberOfFrames;
height=obj.Height;
width=obj.Width;
imgSize = [height, width];
HxW = height*width;
D = zeros(HxW, numFrames);
for k = 1 : numFrames
    frame = read(obj,k);
    if ndims(frame)==3
        frame = rgb2gray(frame);
    end
    D(:, k) = reshape(frame, HxW, 1);
end
D = mat2gray(D);

%run algorithm
%mog-RPCA
r = 1;
param.mog_k = 3;
param.lr_init = 'SVD';
param.maxiter = 100;
param.initial_rank = 2*r;
param.tol = 1e-3;
lr_prior.a0 = 1e-6;
lr_prior.b0 = 1e-6;
mog_prior.mu0 = 0;
mog_prior.c0 = 1e-3;
mog_prior.d0 = 1e-3;
mog_prior.alpha0 = 1e-3;
mog_prior.beta0 = 1e-3;
[lr_model, mog_model, r] = mog_rpca(D, param, lr_prior, mog_prior);
L1 = lr_model.U*lr_model.V';
S1 = D - L1;
%S1 = mat2gray(S1);L1 = mat2gray(L1);
%mcr_rpca
lambda1 = 0.5/sqrt(HxW);
lambda2 = 0.2/sqrt(HxW);
tol3 = 1e-4;
[L,S,T] = mcr_rpca(S1,height,width,lambda1,lambda2,tol3);
L2 = D-S;

%save output
for k = 1 : size(D,2)
    im = S(:,k);
    im = reshape(im, imgSize);
    im = mat2gray(im);
    imwrite(im,[outdir 'S_',num2str(k),'.jpg']);
end
