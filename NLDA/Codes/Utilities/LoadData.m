function [TrSketch TeSketch] = LoadData(trainindex,testindex,Path,Method)

TrSketch = [];
TeSketch = [];

imlist = readImageNames([Path,Method,'/Sketch/']);
imlist = sort(imlist.name)

for i = 1:length(trainindex)
    
    sketch      = imread([Path,Method,'/Sketch/',imlist(trainindex(i)).name]);
    if size(sketch,3) == 3
        sketch = rgb2gray(sketch);
    end
    TrSketch = [TrSketch sketch(:)];
    
end
TrSketch = double(TrSketch);

for i = 1:length(testindex)
    
    sketch      = imread([Path,Method,'/Sketch/',imlist(testindex(i)).name]);
    if size(sketch,3) == 3
        sketch = rgb2gray(sketch);
    end
    TeSketch = [TeSketch sketch(:)];
    
end
TeSketch = double(TeSketch);