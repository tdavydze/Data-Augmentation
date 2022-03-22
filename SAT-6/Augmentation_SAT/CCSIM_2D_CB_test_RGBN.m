function [imout] = CCSIM_2D_CB_test_RGBN(imin, sizeout, T, OL, cand, seed)

if seed ~=0
    rand('seed',seed)
end

imin = double(imin);

imout = zeros(sizeout);
% imoutwo = zeros(size(imout));
sizein = [size(imin,1) size(imin,2)];

temp = ones([OL(1) T(2)]);
errtop = xcorr2(imin(:,:,1).^2, temp);
temp = ones([T(1) OL(2)]);
errside = xcorr2(imin(:,:,1).^2, temp);
temp = ones([T(1)-OL(1) OL(2)]);
errsidesmall = xcorr2(imin(:,:,1).^2, temp);


tic;
for i=[1:T(1)-OL(1):sizeout(1)-T(1), sizeout(1)-T(1)+1]
    for j=[1:T(2)-OL(2):sizeout(2)-T(2), sizeout(2)-T(2)+1]
        
        
        if (i > 1) && (j > 1)
            shared = imout(i:i+OL(1)-1,j:j+T(2)-1);
            err = errtop - 2 * xcorr2(imin(:,:,1), shared) + sum(shared(:).^2);
            
            err = err(OL(1):end-T(1)+1,T(2):end-T(2)+1);
            
            shared = imout(i+OL(1):i+T(1)-1,j:j+OL(2)-1);
            err2 = errsidesmall - 2 * xcorr2(imin(:,:,1), shared) + sum(shared(:).^2);
            
            err = err + err2(T(1):end-T(1)+OL(1)+1, OL(2):end-T(2)+1);
            
            [~,loc] = sort(err(:));
            
            [ibest, jbest] = ind2sub(size(err),loc(1:cand,1));
            c = ceil(rand * length(ibest));
            pos = [ibest(c) jbest(c)];
            
            
        elseif i > 1
            shared = imout(i:i+OL(1)-1,j:j+T(2)-1);
            err = errtop - 2 * xcorr2(imin(:,:,1), shared) + sum(shared(:).^2);
            
            err = err(OL(1):end-T(1)+1,T(2):end-T(2)+1);
            
            [~,loc] = sort(err(:));
            
            [ibest, jbest] = ind2sub(size(err),loc(1:cand,1));
            c = ceil(rand * length(ibest));
            pos = [ibest(c) jbest(c)];
                       
        elseif j > 1
            shared = imout(i:i+T(1)-1,j:j+OL(2)-1);
            err = errside - 2 * xcorr2(imin(:,:,1), shared) + sum(shared(:).^2);
            
            err = err(T(1):end-T(1)+1,OL(2):end-T(2)+1);
            
            [~,loc] = sort(err(:));
            
            [ibest, jbest] = ind2sub(size(err),loc(1:cand,1));
            c = ceil(rand * length(ibest));
            pos = [ibest(c) jbest(c)];
            
        else
            pos = ceil(rand([1 2]) .* (sizein-T+1));
        end
        
        Target1 = imin(pos(1):pos(1)+T(1)-1, pos(2):pos(2)+T(2)-1,1);
        M1 = mincut_func(Target1, imout(i:i+T(1)-1, j:j+T(2)-1,1), T, OL, i, j);
        imout(i:i+T(1)-1,j:j+T(2)-1,1) = combine_2D(imout(i:i+T(1)-1, j:j+T(2)-1,1),Target1, M1);
        
        Target2 = imin(pos(1):pos(1)+T(1)-1, pos(2):pos(2)+T(2)-1,2);
        M2 = mincut_func(Target2, imout(i:i+T(1)-1, j:j+T(2)-1,2), T, OL, i, j);
        imout(i:i+T(1)-1,j:j+T(2)-1,2) = combine_2D(imout(i:i+T(1)-1, j:j+T(2)-1,2),Target2, M2);
        
        Target3 = imin(pos(1):pos(1)+T(1)-1, pos(2):pos(2)+T(2)-1,3);
        M3 = mincut_func(Target3, imout(i:i+T(1)-1, j:j+T(2)-1,3), T, OL, i, j);
        imout(i:i+T(1)-1,j:j+T(2)-1,3) = combine_2D(imout(i:i+T(1)-1, j:j+T(2)-1,3),Target3, M3);
        
        Target4 = imin(pos(1):pos(1)+T(1)-1, pos(2):pos(2)+T(2)-1,4);
        M4 = mincut_func(Target4, imout(i:i+T(1)-1, j:j+T(2)-1,4), T, OL, i, j);
        imout(i:i+T(1)-1,j:j+T(2)-1,4) = combine_2D(imout(i:i+T(1)-1, j:j+T(2)-1,4),Target4, M4);
        
        %     keyboard
%         imagesc(imout); axis equal tight xy; axis off
        
    end
end

toc;
% subplot(1,2,1); imagesc(imin); axis equal tight xy; title('Training Image');
% % colormap parula
% colormap gray
% subplot(1,2,2); imagesc(imout); axis equal tight xy; title('Boundary Corrected');
% subplot(1,3,3); imagesc(imoutwo); axis equal tight xy; title('Original');


