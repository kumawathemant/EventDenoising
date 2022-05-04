clear, clc, close all

addpath('utils')
addpath(['edncnn' filesep 'code'])
addpath(['edncnn' filesep 'camera'])
addpath('aedatLoaders')

edncnnDir = '/home/cpslab/HemantWorkspace/CSE8803Project/6_features'
epmDir = '/home/cpslab/HemantWorkspace/CSE8803Project/edncnn_output/'; 
output_dir = '/home/cpslab/HemantWorkspace/CSE8803Project/tore_randomSample/';

if ~exist(output_dir,'dir')
    mkdir(output_dir)
end

%For each dataset from EDnCNN - read in representations/labels and make Tore to match
files = dir([epmDir '*epm.mat'])

numSamplesPerFile = 500;
k = 7;

for fLoop = 1:numel(files)
    
    if exist([output_dir files(fLoop).name(1:end-4) '_tore.mat'],'file')
        disp('file already processed')
        continue
    end

    load([epmDir files(fLoop).name])
    
    %convert to doubles
    aedat.data.polarity.timeStamp = double(aedat.data.polarity.timeStamp);
    
    %Ensure events are sorted by time
    if ~issorted(aedat.data.polarity.timeStamp)
        [aedat.data.polarity.timeStamp,idx] = sort(aedat.data.polarity.timeStamp);
        aedat.data.polarity.y = aedat.data.polarity.y(idx);
        aedat.data.polarity.x = aedat.data.polarity.x(idx);
        aedat.data.polarity.polarity = aedat.data.polarity.polarity(idx);
        aedat.data.polarity.closestFrame = aedat.data.polarity.closestFrame(idx);
        aedat.data.polarity.frameTimeDelta = aedat.data.polarity.frameTimeDelta(idx);
        aedat.data.polarity.duringAPS = aedat.data.polarity.duringAPS(idx);
        aedat.data.polarity.apsIntensity = aedat.data.polarity.apsIntensity(idx);
        aedat.data.polarity.apsIntGood = aedat.data.polarity.apsIntGood(idx);
        aedat.data.polarity.Jt = aedat.data.polarity.Jt(idx);
        aedat.data.polarity.Prob = aedat.data.polarity.Prob(idx);
    end
    
    numRows = double(aedat.data.frame.size(1));
    numCols = double(aedat.data.frame.size(2));
    numEvents = aedat.data.polarity.numEvents;

    inputVar.neighborhood = 4; 
    
    timeQuantiles = quantile(aedat.data.polarity.timeStamp,[0.15 0.85]);
    qFilter = aedat.data.polarity.timeStamp >= timeQuantiles(1) & ...
        aedat.data.polarity.timeStamp <= timeQuantiles(2);
    
   
    nearEdgeIdx = ((aedat.data.polarity.y-inputVar.neighborhood) < 1) | ...
        ((aedat.data.polarity.x-inputVar.neighborhood) < 1) | ...
        ((aedat.data.polarity.y+inputVar.neighborhood) > numRows) | ...
        ((aedat.data.polarity.x+inputVar.neighborhood) > numCols);
    
    
    sampleIdx = ~nearEdgeIdx & qFilter & (aedat.data.polarity.duringAPS>0) & aedat.data.polarity.apsIntGood;
    
    samples = find(sampleIdx);
    sampleList = randsample(numel(samples), numSamplesPerFile);
    sampleList = samples(sampleList);
    samples = false(numEvents,1);
    samples(sampleList) = true;
    
    Xtore = events2ToreChip(...
        aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, ...
        aedat.data.polarity.y(samples), aedat.data.polarity.x(samples), aedat.data.polarity.timeStamp(samples), aedat.data.polarity.polarity(samples), ...
        k, inputVar.neighborhood);
    
    Ytore = categorical(aedat.data.polarity.Prob(samples)>0.5,[true false],{'valid' 'noise'});
    samples_tore = samples;

    save([output_dir files(fLoop).name(1:end-4) '_tore.mat'], 'Xtore', 'Ytore', 'samples_tore')
    
end
    

buildTrainTestData_tore(output_dir)

% load X/Y from the all_label.mat file that contains 10k samples per dataset
Y = load([output_dir 'all_labels.mat'],'Y');
Y = Y.Y;
grpLabel = load([output_dir 'all_labels.mat'],'grpLabel');
grpLabel = grpLabel.grpLabel;

testIdx = ismember(grpLabel,[2 3 10]); %test on bench, bigChecker, labFast
testY = Y(testIdx);
trainY = Y(~testIdx);
clear Y

X = load([output_dir 'all_labels.mat'],'Xtore');
X = X.Xtore;
testX = X(:,:,:,testIdx);
X(:,:,:,testIdx) = [];
trainX = X;
clear X

% results = train_network(output_dir, trainX, trainY, testX, testY); 


%% Use network to predict data labels (real/noise)

testSet = [1:3]; %test on bench, bigChecker, labFast

for fLoop = 1:numel(testSet)
     
    file = [epmDir files(testSet(fLoop)).name]
    [fp,fn,fe] = fileparts(file);
    
    if exist([output_dir fn '_pred_MPF.mat'],'file')
        disp('file already processed')
        continue
    end
    
    load(file, 'aedat', 'inputVar')
    
    %Ensure events are sorted by time
    aedat.data.polarity.timeStamp = double(aedat.data.polarity.timeStamp);
    if ~issorted(aedat.data.polarity.timeStamp)
        [aedat.data.polarity.timeStamp,idx] = sort(aedat.data.polarity.timeStamp);
        aedat.data.polarity.y = aedat.data.polarity.y(idx);
        aedat.data.polarity.x = aedat.data.polarity.x(idx);
        aedat.data.polarity.polarity = aedat.data.polarity.polarity(idx);
        aedat.data.polarity.closestFrame = aedat.data.polarity.closestFrame(idx);
        aedat.data.polarity.frameTimeDelta = aedat.data.polarity.frameTimeDelta(idx);
        aedat.data.polarity.duringAPS = aedat.data.polarity.duringAPS(idx);
        aedat.data.polarity.apsIntensity = aedat.data.polarity.apsIntensity(idx);
        aedat.data.polarity.apsIntGood = aedat.data.polarity.apsIntGood(idx);
        aedat.data.polarity.Jt = aedat.data.polarity.Jt(idx);
        aedat.data.polarity.Prob = aedat.data.polarity.Prob(idx);
    end
    
    %Smaller spatial / deeper temporal
    inputVar.neighborhood = 4; %9x9 chip centered on event of interest
    inputVar.depth = k;
    
    load('pretrainedNetworks/denoise/denoise_acc_0p77134_size_9__9_14_tore.mat', 'net')
    YPred = makeLabeledAnimations_4Tore(aedat, inputVar, net);
    
    save([output_dir fn '_pred_Tore_9x9x14_0p77134.mat'],'YPred','-v7.3')
    
    YPred_multiColumn = YPred;
    YPred = YPred(:,1);

    YPred_s = nan(numel(YPred),1);
    YPred_s(YPred>0.5) = 1;
    YPred_s(YPred<=0.5) = 0;
    [noisyScore(fLoop), denoiseScoreTore(fLoop)] = scoreDenoise(aedat, YPred_s);
    
    YPred = categorical(YPred_s,[1 0],{'valid' 'noise'});

        writeOutGifExamples
    
end

%Average results for each scene and plot
figure
bar(cat(1,mean(reshape(noisyScore,3,[]),1),mean(reshape(denoiseScoreTore,3,[]),1))')
legend('Noisy','Denoised')
xlabel('Scene')
ylabel('RPMD')
grid on


    