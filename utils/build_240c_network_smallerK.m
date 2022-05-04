%% Create Deep Learning Network Architecture
% Script for creating the layers for a deep learning network with the following 
% properties:
%%
% 
%  Number of layers: 70
%  Number of connections: 74
%
%% 
% Run the script to create the layers in the workspace variable |lgraph|.
% 
% To learn more, see <matlab:helpview('deeplearning','generate_matlab_code') 
% Generate MATLAB Code From Deep Network Designer>.
% 
% Auto-generated by MATLAB on 12-Jan-2021 09:37:17
%% Create Layer Graph
% Create the layer graph variable to contain the network layers.

lgraph = layerGraph();
%% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.

tempLayers = [
    imageInputLayer([180 240 8],"Name","ImageInputLayer","Normalization",)
    convolution2dLayer([3 3],16,"Name","Encoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-1-ReLU-1")
    convolution2dLayer([3 3],16,"Name","Encoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-1-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-1-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","Encoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],32,"Name","Encoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-2-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-2-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","Encoder-Stage-3-Conv-1","Padding",[1 0 1 1],"WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-3-ReLU-1")
    convolution2dLayer([3 3],64,"Name","Encoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-3-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-3-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","Encoder-Stage-4-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-4-ReLU-1")
    convolution2dLayer([3 3],128,"Name","Encoder-Stage-4-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-4-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-4-MaxPool_2","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","Bridge-Conv-1_2","Padding",[1 0 1 0],"WeightsInitializer","he")
    reluLayer("Name","Bridge-ReLU-1_2")
    convolution2dLayer([3 3],128,"Name","Bridge-Conv-2_2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Bridge-ReLU-2_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    dropoutLayer(0.5,"Name","Encoder-Stage-4-DropOut")
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-4-MaxPool_1","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","Bridge-Conv-1_1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Bridge-ReLU-1_1")
    convolution2dLayer([3 3],256,"Name","Bridge-Conv-2_1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Bridge-ReLU-2_1")
    dropoutLayer(0.5,"Name","Bridge-DropOut")
    transposedConv2dLayer([2 2],256,"Name","Decoder-Stage-1-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-1-DepthConcatenation_2")
    convolution2dLayer([3 3],128,"Name","Decoder-Stage-1-Conv-1_2","Padding",[1 2 1 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-1_2")
    convolution2dLayer([3 3],128,"Name","Decoder-Stage-1-Conv-2_2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-2_2")
    transposedConv2dLayer([2 2],128,"Name","Decoder-Stage-2-UpConv_2","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-UpReLU_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-1-DepthConcatenation_1")
    convolution2dLayer([3 3],128,"Name","Decoder-Stage-1-Conv-1_1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-1_1")
    convolution2dLayer([3 3],128,"Name","Decoder-Stage-1-Conv-2_1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-2_1")
    transposedConv2dLayer([2 2],128,"Name","Decoder-Stage-2-UpConv_1","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-UpReLU_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-2-DepthConcatenation")
    convolution2dLayer([3 3],64,"Name","Decoder-Stage-2-Conv-1","Padding",[1 2 1 1],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],64,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-2")
    transposedConv2dLayer([2 2],64,"Name","Decoder-Stage-3-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-3-DepthConcatenation")
    convolution2dLayer([3 3],32,"Name","Decoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-ReLU-1")
    convolution2dLayer([3 3],32,"Name","Decoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-ReLU-2")
    transposedConv2dLayer([2 2],32,"Name","Decoder-Stage-4-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-4-DepthConcatenation")
    convolution2dLayer([3 3],16,"Name","Decoder-Stage-4-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-ReLU-1")
    convolution2dLayer([3 3],16,"Name","Decoder-Stage-4-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-ReLU-2")
    convolution2dLayer([1 1],1,"Name","Final-ConvolutionLayer","Padding","same","WeightsInitializer","he")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Encoder-Stage-1-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Decoder-Stage-4-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Encoder-Stage-2-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Decoder-Stage-3-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Encoder-Stage-3-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Decoder-Stage-2-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-4-ReLU-2","Encoder-Stage-4-MaxPool_2");
lgraph = connectLayers(lgraph,"Encoder-Stage-4-ReLU-2","Decoder-Stage-1-DepthConcatenation_1/in2");
lgraph = connectLayers(lgraph,"Bridge-ReLU-2_2","Encoder-Stage-4-DropOut");
lgraph = connectLayers(lgraph,"Bridge-ReLU-2_2","Decoder-Stage-1-DepthConcatenation_2/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpReLU","Decoder-Stage-1-DepthConcatenation_2/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU_2","Decoder-Stage-1-DepthConcatenation_1/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU_1","Decoder-Stage-2-DepthConcatenation/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpReLU","Decoder-Stage-3-DepthConcatenation/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-4-UpReLU","Decoder-Stage-4-DepthConcatenation/in1");
%% Plot Layers

plot(lgraph);