clear all
load TRAIN_IM_UCM.mat


for i = 1:270
for k = 1:5
[imout] = CCSIM_2D_CB_test_RGBN(TRAIN_IM(:,:,:,i), [256 256 3], [90 90], [4 4], 3, 0);
FULL_OUTPUT(:,:,:,(i*5-5+k))=imout;
end
end
