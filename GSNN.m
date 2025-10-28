%% ====================================================================
% Author: Khalid Youssef, PhD (2023)
% Email: khyous@iu.edu
% ====================================================================
% Supplemental code for demonstrating how to implement the SNN optimization 
% pipeline for landslide susceptibility modelling
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% PLEASE ACKNOWLEDGE THE EFFORT THAT WENT INTO DEVELOPMENT 
% BY REFERENCING THE PAPER:
%
% K. Youssef, K. Shao, S. Moon & L.-S. Bouchard Landslide susceptibility 
% modeling by interpretable neural network. 
% Communications Earth & Environment 
% https://doi.org/10.1038/s43247-023-00806-5
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% ====================================================================
% ====================================================================
% ====================================================================
% Dependencies:
%
% This code was implemented using MATLA 2021b
% It requires statistics & machine learning toolbox
% 
% *Earlier versions of MATLAB might require the deep learning toolbox
% ====================================================================
% Hardware requirements:
%
% This has been successfully tested with a minimum of 64GB RAM using 
% 4 workers, and took approximately 90 minutes to run under these settings
% with an i7-11800H Intel CPU.
% If more RAM is available, the number of workers can be increased for
% faster processing. 
% If you have less than 64GB RAM, you can try reducing the number of
% workers.
% To set the number of workers, use the command parpool(n) before you run 
% this script, where n is the number of workers.
% ====================================================================
% Optimization instructions:
%
% Download the dataset "GSNN_Demo_Data.mat" from:
% https://dataverse.ucla.edu/dataset.xhtml?persistentId=doi:10.25346/S6/D5QPUA.
%
% Run this file for the full pipeline, or call the funcions described
% herein individually.
%
% % Data preparation GSNN_Data_Prep. Loads the dataset and prepares the
% data for optimization. Select the included dataset "GSNN_Demo_Data.mat" 
% for this demonstration. See documentation for how to prepare a dataset 
% for your own data. 
%
% % Tournament ranking is performed in two steps: 
% The function GSNN_TR1 for backwards elimination, and the function 
% GSNN_TR2 for forward selection
%
% % Teacher MST network training is performed by calling the function:
% GSNN_MST  
%
% % SNN network training is performed by caslling the function:
% GSNN_SNN
%
% % Save the SNN model when the optimization is complete.
%
% --------------------------------------------------------------------
% The optimization is to be performed multiple times with different 
% initial conditions, where the model with the highest AUC is selected.
% --------------------------------------------------------------------
% ====================================================================
% Inference instructions:
%
% The second part of this file demonstrates how to use the SNN model for 
% inference, and how to extract and plot the additive feature functions.
%
% The first popup window is for selecting the saved SNN model
% 
% The second popup window is for select the dataset 
%
% GSNN_Inference_Data_Prep is used for preparing the data for inference.
% This data preparation function does not normalize the data, and keeps the
% features in their original range. 
%
% ====================================================================
% For questions and comments, please email: land.slide.snn@gmail.com
% ====================================================================
%% ====================================================================

close all
clear

% parpool('local','IdleTimeout',Inf);

% Main parameter values  
composite_level = 2; % composite level options: 1, 2
SNN_iterations = 50; % number of SNN training iterations

% Functions paramater values
NN1 = 5;% number of nurons per layer for Tournament ranking - backwards elimination
NE1 = 50;% number of training epochs for Tournament ranking - backwards elimination
reps1 = 4000;% number of models for Tournament ranking - backwards elimination
thr1 = 0.005;% elimination threshold for Tournament ranking - backwards elimination
NN2 = 8;% number of nurons per layer for Tournament ranking - forward selection
NE2 = 50;% number of training epochs for Tournament ranking - forward selection
reps2 = 3;% number of models per step for Tournament ranking - forward selection

%##########################################################################
%##########################################################################

% Data preparation function
display('Data preparation ...')
[TR,TAR,VL,TARV,TST,TART,names,nameID,MN,MX]...
    = GSNN_Data_Prep(composite_level);
display('Data preparation complete')

% Tournament ranking - backwards elimination
display('Tournament ranking step 1/2 ...')
TR1indx = GSNN_TR1(TR,TAR,VL,TARV,NN1,NE1,reps1,thr1);

% select winning Features
names = names(TR1indx);% feature names
nameID = nameID(TR1indx);% feature name IDs
TR = TR(TR1indx,:);% Training partition
VL = VL(TR1indx,:);% Validation partition
TST = TST(TR1indx,:);% Testing partition
MX = MX(TR1indx);% feature minimums
MN = MN(TR1indx);% feature maximums
display('Tournament ranking step 1/2 complete')

% Tournament ranking - forward selection
display('Tournament ranking step 2/2 ...')
TR2indx = GSNN_TR2(TR,TAR,VL,TARV,NN2,NE2,reps2,composite_level);

% select winning Features
names = names(TR2indx);
nameID = nameID(TR2indx);
TR = TR(TR2indx,:);
VL = VL(TR2indx,:);
TST = TST(TR2indx,:);
MX = MX(TR2indx);
MN = MN(TR2indx);
display('Tournament ranking step 2/2 complete')

% Teacher MST network training
display('Teacher MST training ...')
[res,resV,resT] ...
    = GSNN_MST(TR,TAR,VL,TARV,TST,TART);
display('Teacher MST training complete')

% SNN network training
display('SNN training ...')
[SNN,ranks] = GSNN_SNN(TR,TAR,VL,TARV,TST,TART,res,resV,MN,MX,SNN_iterations);
display('SNN training complete')
breake;

[file,path] = uiputfile({'*.mat'},...
               'Save SNN model','C:\');
namesR = names(ranks);
save([path file],'SNN','namesR')


display('Optimization complete. Click any key to proceed to inference and visualization examples.')
pause
%##########################################################################

%##########################################################################
%######################Inference & Visualization Demo######################
%##########################################################################
clear
[file,path] = uigetfile({'*.mat'},...
               'Select SNN model','C:\');
load([path file])

% Inference data preparation function
composite_level = 2;
Features = GSNN_Inference_Data_Prep(composite_level,namesR);

% Inference:
% SNN model parameters
a = SNN.a; b = SNN.b; w = SNN.w; c = SNN.c;

% SNN single-sample inference, example:
display('SNN single-sample inference example:')
sample = Features(:,1);
% individual Features contribution
functions = sum(w'.*exp(-(a.*repmat(sample,[1,size(a,2)])+b).^2)')'+c;
display(['------------------'])
display(['------------------'])
for j = 1:numel(namesR)
    display([namesR{j} ': ' mat2str(sample(j))]);
    display(['f(' namesR{j} '): ' mat2str(functions(j))]);
    display(['------------------'])
end
% total sample susceptibility
susceptibility = sum(sum(w'.*exp(-(a.*repmat(sample,[1,size(a,2)])+b).^2)')'+c);
display(['------------------'])
display(['susceptibility = ' mat2str(susceptibility)])

% SNN batch inference:
Susceptibility = sum(squeeze(sum(repmat(w',[1,1,size(Features,2)]).*...
    exp(-(repmat(a',[1,1,size(Features,2)]).*...
    permute(repmat(Features,[1,1,size(a,2)]),[3,1,2])+...
    repmat(b',[1,1,size(Features,2)])).^2)))+...
    repmat(c,[1,size(Features,2)]));

% SNN batch inference with individual functions:
Functions = squeeze(sum(repmat(w',[1,1,size(Features,2)]).*...
    exp(-(repmat(a',[1,1,size(Features,2)]).*...
    permute(repmat(Features,[1,1,size(a,2)]),[3,1,2])+...
    repmat(b',[1,1,size(Features,2)])).^2)))+...
    repmat(c,[1,size(Features,2)]);

% Plot feature functions:
figure('units','normalized','outerposition',[0 0 1 1])
d = ceil(sqrt(numel(namesR)));
%d=1;
for j = 1:numel(namesR)
%for j = 1:1
    subplot(d,d,j)
    fig = plot(Features(j,:),Functions(j,:),'.');
    tmp = namesR{j};
    ind = find(tmp == '&');
    clear('tmp2')
    if numel(ind) > 0
    tmp2{1} = tmp(1:ind);
    tmp2{2} = tmp(ind+2:end);
    else
        tmp2 = tmp;
    end
    title(tmp2)  
    mx = max(Features(j,:));
    mn = min(Features(j,:));
    fig.Parent.XLim = [mn mx];
    fig.Parent.XTick = [mn mx];
    fig.Parent.XTickLabel = round([mn mx],1);
    fig.Parent.XTickLabelRotation = 10; 
    fig.Parent.YLim = [0 max(Functions(:))];
    fig.Parent.FontSize = 10;
end
%relief=[Features(7,:)',Functions(7,:)'];

% % % % Dem = GRIDobj('demmin.tif');
% % % % w1=Dem.size(1);
% % % % w2=Dem.size(2);
% % % % dem=double(Dem.Z);
% % % % [m,n]=find((AA==1));
% % % % fsus=Functions(2,:)';
% % % % AA(m,:)=fsus;
% % % % Susce=reshape(AA,w1,w2);
% % % % Dem.Z=Susce;
% % % % GRIDobj2geotiff(Dem);
% % % % ww=wwwww(~isnan(wwwww(:,2)),:);