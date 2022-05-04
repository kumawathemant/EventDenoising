function aedat = readUAVdata(fn)

%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: /home/wescomp/Dropbox/WesDocs/UD/Research/eventCameraFeatures/uavData/events.txt
%
% Auto-generated by MATLAB on 04-Jan-2021 11:57:37

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 4);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = " ";

% Specify column names and types
opts.VariableNames = ["timestamp", "x", "y", "polarity"];
opts.VariableTypes = ["double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";

% Import the data
events = readtable(fn, opts);


%% Clear temporary variables
clear opts

aedat.data.polarity.timeStamp = events.timestamp.*1e6;
aedat.data.polarity.x = events.x + 1;
aedat.data.polarity.y = events.y + 1;
aedat.data.polarity.polarity = events.polarity;

