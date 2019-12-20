%% Region segmentation

load('data.mat');

N = size(ori_imgs, 1);
result = cell(N, 1);

for i = N:-1:1
    disp(i);
    
    M = size(ori_imgs{i}, 1);
    result{i} = cell(M, 1);
    se = strel('disk', 3);         

    for j = ceil(M/2):M
        img = ori_imgs{i}{j};
        mask = imerode(cams{i}{j}, se);
        
        if sum(sum(mask)) < 5
            break
        end
        
        result{i}{j} = chan_vese(img, mask, 200);
    end
    
    for j = ceil(M/2)-1:-1:1
        img = ori_imgs{i}{j};
        mask = imerode(cams{i}{j}, se);
        
        if sum(sum(mask)) < 5
            break
        end
        
        result{i}{j} = chan_vese(img, mask, 200);
    end
end
