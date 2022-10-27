function Data = LoadAllData(trainindex,testindex,Path,Methods)

Data{1}.trainindex = trainindex;
Data{1}.testindex  = testindex;

%Methods2(2:7) = Methods;
Methods2(2:10) = Methods;
Methods2{1}   = 'GR';

for i = 1:length(Methods2)
    
    [TrSketch TeSketch] = LoadData(trainindex,testindex,Path,Methods2{i});
    Data{i}.TrSketch = TrSketch;
    Data{i}.TeSketch = TeSketch;
    
end


