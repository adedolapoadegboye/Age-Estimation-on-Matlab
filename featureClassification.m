function [acc_NN, acc_GANN, acc_GWONN,labels_NN,labels_GANN,labels_GWONN,...
    Convergence_curve_GWO,GWOnet,newGAnet,net] = ...
                                               featureClassification(attr,multiclass)
%% BPNN peroformance
[acc_NN,y,net,NNinput,NNlabels]=NNscript_new(attr,multiclass);
labels_NN=vec2ind(y);
%% GA optimisation
NNweights_biases=getwb(net);             % get teh input weights and biases
                                         %of pretrained neural network
dim=numel(NNweights_biases);             % dimesnion of searching space
fitness=  @(x)Objective_function(x,net,NNinput,NNlabels);
% global valueFitt 
options = gaoptimset('MutationFcn',@mutationadaptfeasible,'PopulationSize',...
                     100,'Generations',100);
% options = gaoptimset('MutationFcn',@mutationadaptfeasible,'Generations',100);                 
options = gaoptimset(options,'PlotFcns',{@gaplotbestf}, ...
    'Display','iter','OutputFcn',@myoutputfcn);
%  options = gaoptimset(options,'OutputFcn',@myoutputfcn);
[x,fval] = ga(fitness,dim,[],[],[],[],-1.*ones(1,dim),1.*ones(1,dim),[],[],options);
GAnet=net;
newGAnet=setwb(GAnet,x);
% Test the Network
y = newGAnet(NNinput);
y=abs(round(y));
cp=classperf(NNlabels,y);
acc_GANN=cp.CorrectRate;
labels_GANN=vec2ind(y);
%% GWO
SearchAgents_no=50;
Max_iteration=100;
[Alpha_score,w_GWO,Convergence_curve_GWO]=GWO(SearchAgents_no,Max_iteration,...
                                NNinput,NNlabels,net);
GWOnet=net;
GWOnet=setwb(GWOnet,w_GWO);
y=sim(GWOnet, NNinput);
predictedLables_GWO=abs(round(y));
cp=classperf(NNlabels,predictedLables_GWO);
acc_GWONN=cp.CorrectRate;
labels_GWONN=vec2ind(y);