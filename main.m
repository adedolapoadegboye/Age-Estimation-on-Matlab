close all
clear
clc
addpath(genpath(cd))
%%
load LBPFeatures
load gaborFeatures

%% preprocessing of features
%1. for LBP features
for ii=1:size(LBPFeatures,2)
    features_LBP(ii,:)=[LBPFeatures(ii).feature];
    labels(ii,1)=gaborFeatures(ii).age;
end
% features_LBP=normalize(features_LBP);

[NewData,~,mm] = preprocssing (features_LBP,labels);
Newfeatures_LBP=NewData(:,1:end-1);

%% BPNN peroformance
[acc_NN,y,net,NNinput,NNlabels]=NNscript_new(Newfeatures_LBP,mm);
labels_NN=vec2ind(y);
%% ask for the input test image and use the model with highest accuracy
[filename,path]=uigetfile('*');
testimg=imread([path,filename]);

% extract gabor features
if size(testimg,3)==3
    graytestimg=rgb2gray(testimg);
else
    graytestimg=testimg;
end
features_test = extractLBPFeatures(graytestimg);

testimgop=net(features_test');
[~,position]=max(testimgop);

if position==1
    testimgopstrng='0-9';
elseif position==2
    testimgopstrng='10-18';
elseif position==3
    testimgopstrng='19-25';
elseif position==4
    testimgopstrng='26-30';
elseif position==5
    testimgopstrng='31-40';
elseif position==6
    testimgopstrng='41-50';
elseif position==7
    testimgopstrng='51-60';
elseif position==8
    testimgopstrng='above 60';
end
    
textposition= [77 107]; 
box_color = {'yellow'};
testimg = insertText(testimg,textposition,(testimgopstrng),'FontSize',18,'BoxColor',...
    box_color,'BoxOpacity',0.4,'TextColor','white');
figure
imshow(testimg)
