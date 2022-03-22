clear all
load TRAIN_IM_SAT.mat


for i = 1:60
for k = 1:5
[imout] = CCSIM_2D_CB_test_RGBN(TRAIN_IM(:,:,:,i), [28 28 4], [12 12], [4 4], 3, 0);
FULL_OUTPUT(:,:,:,(i*5-5+k))=imout;
end
end
