function [newdata,class,multiclass] = preprocssing (features,labels)

%% remove any empty feature row from the matrix
emptyIndex = find(arrayfun(@(features) isempty(features),features));
features(emptyIndex,:)=[];
featuresize=length(labels);
%% divide the age into age groups 
% 1 = 0-9
% 2 = 10-18
% 3 = 19-25
% 4 = 26-30
% 5= 31 - 40
% 6 = 41-50
% 7 = 51 -60
% 8= above 60
trainIDX = ones(featuresize,1);
for ii=1:numel(labels)
    if 0<=labels(ii) && labels(ii)<=9
        trainIDX (ii)=1;
    elseif 10<=labels(ii) && labels(ii)<=18
        trainIDX (ii)=2;
    elseif 19<=labels(ii) && labels(ii)<=25
        trainIDX (ii)=3;
    elseif 26<=labels(ii) && labels(ii)<=30
        trainIDX (ii)=4;
    elseif 31<=labels(ii) && labels(ii)<=40
        trainIDX (ii)=5;
    elseif 41<=labels(ii) && labels(ii)<=50
        trainIDX (ii)=6;
    elseif 51<=labels(ii) && labels(ii)<=60
        trainIDX (ii)=7;
    elseif labels(ii)>60
        trainIDX (ii)=8;
    end
end
%%
newf=[features, trainIDX];
[sortedtrainIDX,idx]=sort(trainIDX);
sortedata=newf(idx,:);
newdata=mySMOTE(sortedata,100,10,sortedtrainIDX);
newlabels=newdata(:,end);
% converting into multiclass
[class]=unique(trainIDX);
multiclass=zeros([size(newlabels,1),numel(class)]);
for ii=1:numel(class)
    multiclass(newlabels==class(ii),ii)=1;
end
end