% NLDA: Face Recognition based on Synthesized Sketches
% Written by Nannan Wang
% 2016.10.26
% Xidian University
% nnwang@xidian.edu.cn


clear;
clc;
close all;

addpath('Codes/Utilities');
addpath('Codes/NLDA');

Database = 'CUFSF';
Methods = {'DPGM'};

Path = ['/data/shengqingjie/outputs/Fss/Results/',Database,'/'];
if strcmp(Database,'CUFS')
    nTrain = 150;
    nTotal = 338;
else
    nTrain = 300;
    nTotal = 944;
end

ACC_Ours = 0;

ntest = 20;

for counter = 1:ntest
    
    fprintf('Random test %d/%d\n',counter,ntest);
    
    index = randperm(nTotal);
    trainindex = index(1:nTrain);
    testindex  = index(nTrain+1:end);
    Data = LoadAllData(trainindex,testindex,Path,Methods);
    
    index_set(counter).trainindex = trainindex;
    index_set(counter).testindex  = testindex;
    
    NLDA_Result = NLDA_Classification(Methods,Data,nTrain,Database);

    ACC_Ours = NLDA_Result{1}.accuracy + ACC_Ours;
       
end

ACC_Ours = ACC_Ours/ntest;

fprintf('NLDA Ours:%f\n',ACC_Ours);
%save([Path,'Result_NLDA.mat'],'Result_NLDA');

%colorset = [255  0  255;
%             0   255  0;
%             0    0  255;
%             138  43 226;
%             0   199 140;
%             255 215  0;
%             0 0 0;
%           ];
%
%colorset = colorset./255;
%
%figure;
%linewidth = 1;
%if strcmp(Database,'CUFSF')
%    numindex = 299;
%else
%    numindex = 149;
%end

%
%plot(1:numindex,100*NLDA_Ours(1:numindex),'-','Color',colorset(6,:),'LineWidth',linewidth);
%hold on;
%grid on;
%xlabel('The reduced number of dimensions');
%ylabel('Recognition rate (%)');
%if strcmp(Database,'CUFSF')
%    axis([1 numindex 0 90]);
%else
%    axis([1 numindex 0 100]);
%end
%%legend('LLE','SSD','MRF','MWF','Fast-RSLCR','RSLCR','Location','SouthEast');
%legend(Ours','Location','SouthEast');
%title(['NLDA on ',Database]);
%%saveas(gcf,[Path,Database,'_NLDA.jpg']);