binMs = 10;
timeStart = -100;
timeEnd = 199;
daysToKeep = 1;
chunkSize = 100;

selectedElectrodes = 1:1024;

p = "E:\Database\THINGS\TVSD\monkeyF\THINGS_MUA_trials_2.mat";
mapFile = "D:\Research\Project\THINGS\dataset\TVSD\monkeyF\_logs\1024chns_mapping_20220105.mat";
thingsFile = "D:\Research\Project\THINGS\dataset\TVSD\monkeyF\_logs\things_imgs.mat";
imgRoot = "E:\Database\THINGS\concepts and images\images_THINGS";
outFile = "D:\Research\Project\THINGS\data\data.mat";
metadataFile = replace(outFile, ".mat", "_metadata.csv");
targetSize = [100 100];

m = matfile(p);
Smap = load(mapFile);
Simg = load(thingsFile);

mapping = Smap.mapping(:);
ALLMAT = m.ALLMAT;
tb = m.tb;

timeIdx = find(tb >= timeStart & tb <= timeEnd);
nBins = numel(timeIdx) / binMs;

keep = ismember(ALLMAT(:, 6), daysToKeep);
ALLMAT_keep = ALLMAT(keep, :);
nTrials = size(ALLMAT_keep, 1);

trial_idx = ALLMAT_keep(:, 1);
train_idx = ALLMAT_keep(:, 2);
test_idx = ALLMAT_keep(:, 3);
rep = ALLMAT_keep(:, 4);
count = ALLMAT_keep(:, 5);
day = ALLMAT_keep(:, 6);

electrode_roi = strings(1024, 1);
electrode_roi(1:512) = "V1";
electrode_roi(513:832) = "IT";
electrode_roi(833:1024) = "V4";
selectedElectrodeRoi = electrode_roi(selectedElectrodes);

if isfile(outFile)
    delete(outFile)
end

data = struct;
data.ALLMAT = ALLMAT_keep;
data.tb = mean(reshape(tb(timeIdx), binMs, nBins), 1);
data.binMs = binMs;
data.timeStart = timeStart;
data.timeEnd = timeEnd;
data.days = daysToKeep;
data.mapping = mapping;
data.selectedElectrodes = selectedElectrodes(:);
data.electrode_roi = selectedElectrodeRoi;
data.trial_idx = trial_idx;
data.train_idx = train_idx;
data.test_idx = test_idx;
data.rep = rep;
data.count = count;
data.day = day;

nSelected = numel(selectedElectrodes);

data.ALLMUA = zeros(nSelected, nTrials, nBins, "single");
data.IMAGES = zeros(3, targetSize(1), targetSize(2), nTrials, "uint8");
data.image_rel_path = strings(nTrials, 1);
data.image_source = strings(nTrials, 1);
data.image_name = strings(nTrials, 1);
data.concept_name = strings(nTrials, 1);
data.image_label = strings(nTrials, 1);

dstStart = 1;

for srcStart = 1:chunkSize:size(ALLMAT, 1)
    srcEnd = min(srcStart + chunkSize - 1, size(ALLMAT, 1));
    src = srcStart:srcEnd;
    keepBlock = keep(src);

    if any(keepBlock)
        block = m.ALLMUA(:, src, timeIdx);
        block = block(mapping, :, :);
        block = block(selectedElectrodes, :, :);
        block = block(:, keepBlock, :);

        nBlock = sum(keepBlock);
        block = reshape(block, nSelected, nBlock, binMs, nBins);
        block = mean(block, 3);
        block = reshape(block, nSelected, nBlock, nBins);

        dst = dstStart:(dstStart + nBlock - 1);
        data.ALLMUA(:, dst, :) = single(block);
        dstStart = dstStart + nBlock;

        clear block
    end
end

for i = 1:nTrials
    trIdx = train_idx(i);
    teIdx = test_idx(i);

    if trIdx > 0
        stim = Simg.train_imgs(trIdx);
    elseif teIdx > 0
        stim = Simg.test_imgs(teIdx);
    else
        error("Trial %d has both train_idx and test_idx equal to 0.", i)
    end

    relPath = string(stim.things_path);
    relPath = replace(relPath, "/", filesep);
    relPath = replace(relPath, "\", filesep);

    [imageFolder, imageName, imageExt] = fileparts(relPath);
    [~, conceptName] = fileparts(imageFolder);
    imageLabel = conceptName + "/" + imageName + imageExt;

    imgPath = fullfile(imgRoot, "object_images", relPath);

    if ~isfile(imgPath)
        error("Image not found: %s", imgPath)
    end

    img = imread(imgPath);

    if ismatrix(img)
        img = repmat(img, 1, 1, 3);
    elseif size(img, 3) == 1
        img = repmat(img, 1, 1, 3);
    elseif size(img, 3) > 3
        img = img(:, :, 1:3);
    end

    img = imresize(img, targetSize, "lanczos3");

    data.IMAGES(:, :, :, i) = permute(img, [3 1 2]);
    data.image_rel_path(i) = relPath;

    data.image_name(i) = imageName + imageExt;
    data.concept_name(i) = conceptName;
    data.image_label(i) = imageLabel;

    if trIdx > 0
        data.image_source(i) = "train";
    else
        data.image_source(i) = "test";
    end
end

image_rel_path = data.image_rel_path;
image_name = data.image_name;
concept_name = data.concept_name;
image_label = data.image_label;

metadata = table(trial_idx, train_idx, test_idx, rep, count, day, image_rel_path, image_name, concept_name, image_label);
writetable(metadata, metadataFile);

save(outFile, "data", "-v7.3")