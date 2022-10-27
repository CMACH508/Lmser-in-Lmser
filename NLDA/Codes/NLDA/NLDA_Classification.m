function NLDA_Result = NLDA_Classification(Methods,Data,nTrain,Database)

indics = 1;
for i = 1:length(Methods)
    
    %Synthesized Sketch based Recognition
    if strcmp(Methods{i},'SCA-GAN')
        TrainData  = [Data{i+2}.TrSketch Data{i+1}.TrSketch];
        TestData = Data{i+2}.TeSketch;
    elseif strcmp(Methods{i},'SCA-GAN-GR')
        continue;
    else
        TrainData  = [Data{1}.TrSketch Data{i+1}.TrSketch];
        TestData   = Data{1}.TeSketch;
    end

    if strcmp(Database,'CUFS')
        nDim = 300;
    else
        nDim = 600;
    end
    
    GalleryData  = Data{i+1}.TeSketch;
    TrainLabel   = [1:nTrain 1:nTrain];
    TestLabel    = [nTrain+1:nTrain+size(TestData,2)];
    GalleryLabel = [nTrain+1:nTrain+size(GalleryData,2)];
    NLDA_Result{indics} = NLDA(TrainData,TestData,GalleryData,...
        TrainLabel,TestLabel,GalleryLabel,nDim,Methods{i});
    indics = indics + 1;
    
end