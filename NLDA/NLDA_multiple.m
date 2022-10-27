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

Database = 'CUFS';
Methods = {'MRF', 'SSD', 'RSLCR', 'DGFL', 'BP-GAN', 'pix2pix', 'SCA-GAN', 'SCA-GAN-GR', 'Ours_byy'};

Path = ['/data/shengqingjie/outputs/Fss/Results/',Database,'/'];
if strcmp(Database,'CUFS')
    nTrain = 150;
    nTotal = 338;
else
    nTrain = 300;
    nTotal = 944;
end

ACC_MRF = 0;
ACC_SSD = 0;
ACC_RSLCR = 0;
ACC_DGFL = 0;
ACC_BPGAN = 0;
ACC_pix2pix = 0;
ACC_SCA_GAN = 0;
ACC_Ours = 0;
NLDA_MRF = 0;
NLDA_SSD = 0;
NLDA_RSLCR = 0;
NLDA_DGFL = 0;
NLDA_BPGAN = 0;
NLDA_pix2pix = 0;
NLDA_SCA_GAN = 0;
NLDA_Ours = 0;

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

    ACC_MRF = NLDA_Result{1}.accuracy + ACC_MRF;
    ACC_SSD = NLDA_Result{2}.accuracy + ACC_SSD;
    ACC_RSLCR = NLDA_Result{3}.accuracy + ACC_RSLCR;
    ACC_DGFL = NLDA_Result{4}.accuracy + ACC_DGFL;
    ACC_BPGAN = NLDA_Result{5}.accuracy + ACC_BPGAN;
    ACC_pix2pix = NLDA_Result{6}.accuracy + ACC_pix2pix;
    ACC_SCA_GAN = NLDA_Result{7}.accuracy + ACC_SCA_GAN;
    ACC_Ours = NLDA_Result{8}.accuracy + ACC_Ours;

    NLDA_MRF = NLDA_Result{1}.RecRate + NLDA_MRF;
    NLDA_SSD = NLDA_Result{2}.RecRate + NLDA_SSD;
    NLDA_RSLCR = NLDA_Result{3}.RecRate + NLDA_RSLCR;
    NLDA_DGFL = NLDA_Result{4}.RecRate + NLDA_DGFL;
    NLDA_BPGAN = NLDA_Result{5}.RecRate + NLDA_BPGAN;
    NLDA_pix2pix = NLDA_Result{6}.RecRate + NLDA_pix2pix;
    NLDA_SCA_GAN = NLDA_Result{7}.RecRate + NLDA_SCA_GAN;
    NLDA_Ours = NLDA_Result{8}.RecRate + NLDA_Ours;
       
end

ACC_MRF = ACC_MRF/ntest;
ACC_SSD = ACC_SSD/ntest;
ACC_RSLCR = ACC_RSLCR/ntest;
ACC_DGFL = ACC_DGFL/ntest;
ACC_BPGAN = ACC_BPGAN/ntest;
ACC_pix2pix = ACC_pix2pix/ntest;
ACC_SCA_GAN = ACC_SCA_GAN/ntest;
ACC_Ours = ACC_Ours/ntest;

NLDA_MRF = NLDA_MRF/ntest;
NLDA_SSD = NLDA_SSD/ntest;
NLDA_RSLCR = NLDA_RSLCR/ntest;
NLDA_DGFL = NLDA_DGFL/ntest;
NLDA_BPGAN = NLDA_BPGAN/ntest;
NLDA_pix2pix = NLDA_pix2pix/ntest;
NLDA_SCA_GAN = NLDA_SCA_GAN/ntest;
NLDA_Ours = NLDA_Ours/ntest;

Result_NLDA.indexset = index_set;
Result_NLDA.NLDA_MRF = NLDA_MRF;
Result_NLDA.NLDA_SSD = NLDA_SSD;
Result_NLDA.NLDA_RSLCR = NLDA_RSLCR;
Result_NLDA.NLDA_DGFL = NLDA_DGFL;
Result_NLDA.NLDA_BPGAN = NLDA_BPGAN;
Result_NLDA.NLDA_pix2pix = NLDA_pix2pix;
Result_NLDA.NLDA_SCA_GAN = NLDA_SCA_GAN;
Result_NLDA.NLDA_Ours = NLDA_Ours;

save([Path,'Result_NLDA.mat'],'Result_NLDA');

colorset = [255  0  255;
             0   255  0;
             0    0  255;
             138  43 226;
             0   199 140;
             255 215  0;
             220 0 130;
             0 0 0;
           ];
       
colorset = colorset./255;

figure;
linewidth = 1;
if strcmp(Database,'CUFSF')
    numindex = 299;
else
    numindex = 149;
end
fprintf('NLDA MRF:%f\n',ACC_MRF);
fprintf('NLDA SSD:%f\n',ACC_SSD);
fprintf('NLDA RSLCR:%f\n',ACC_RSLCR);
fprintf('NLDA DGFL:%f\n',ACC_DGFL);
fprintf('NLDA BP-GAN:%f\n',ACC_BPGAN);
fprintf('NLDA pix2pix:%f\n',ACC_pix2pix);
fprintf('NLDA SCA-GAN:%f\n',ACC_SCA_GAN);
fprintf('NLDA Ours:%f\n',ACC_Ours);

plot(1:numindex,100*NLDA_MRF(1:numindex),'-','Color',colorset(8,:),'LineWidth',linewidth);
hold on;
grid on;
plot(1:numindex,100*NLDA_SSD(1:numindex),'-','Color',colorset(7,:),'LineWidth',linewidth);
plot(1:numindex,100*NLDA_RSLCR(1:numindex),'-','Color',colorset(6,:),'LineWidth',linewidth);
plot(1:numindex,100*NLDA_DGFL(1:numindex),'-','Color',colorset(5,:),'LineWidth',linewidth);
plot(1:numindex,100*NLDA_BPGAN(1:numindex),'-','Color',colorset(4,:),'LineWidth',linewidth);
plot(1:numindex,100*NLDA_pix2pix(1:numindex),'-','Color',colorset(3,:),'LineWidth',linewidth);
plot(1:numindex,100*NLDA_SCA_GAN(1:numindex),'-','Color',colorset(2,:),'LineWidth',linewidth);
plot(1:numindex,100*NLDA_Ours(1:numindex),'-','Color',colorset(1,:),'LineWidth',linewidth);
xlabel('The reduced number of dimensions');
ylabel('Recognition rate (%)');
if strcmp(Database,'CUFSF')
    axis([1 numindex 0 90]);
else
    axis([1 numindex 0 100]);
end
%legend('LLE','SSD','MRF','MWF','Fast-RSLCR','RSLCR','Location','SouthEast');
legend('MRF','SSD','RSLCR','DGFL','BP-GAN','pix2pix','SCA-GAN','Ours','Location','SouthEast');
title(['NLDA on ',Database]);
saveas(gcf,[Path,Database,'_NLDA.jpg']);