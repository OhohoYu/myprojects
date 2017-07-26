%% Task2
%% Loading and storing the images to "Images"

close all

addpath(genpath('data'))
addpath(genpath('for_students'))

Images = dir(['data' filesep 'images' filesep 'IM*.nii']);
regs = dir(['data' filesep 'reg_results' filesep 'Reg*.nii']);

niis = [];

for i = 1 : length(Images)
    s = load_untouch_nii(Images(i).name);
    niis = cat(3, niis, s.img);
end

%% Area of focus 
%for each image focus on only at y=375-as specified, and choose an x range
%of 50 to 150 - adequate to cover the cover all the pixel invovled in the
%deformation during motion - making interpolation easier/faster

%Image_Slice is a 101 by 300 matrix of which each column corresponds to the
%line of pixes of our interest(at y=375), from each image (we have 300
%images)
Image_Slice = [];
for i=1:size(niis,3)
    Image_Slice = [Image_Slice niis(50:150, 375, i)];
end

%% Linearly interpolate each line 
%x-range - matching the one used in previous section
Xrange = 50:150; % Original x range

%create an even finner range for the range of x - steop of 0.025 (sub-divide pixel)
Xrange_Finner = 50:0.025:150; 

%carry out the interp, using interp1 - 1D interpolation
Interp_Image_Slice = interp1(Xrange, single(Image_Slice), Xrange_Finner, 'linear');
%% FIdx the x coordinate corresponding to an intensity value of 70 (surrogate signal) and plot it

Idx = zeros(1, size(Interp_Image_Slice, 2));

for i=1:length(Idx)
    %any value >= 69 round up to 70 and anything <=71 round down to 70
    %then the correspondinf idices are sotred to Idx and used for the plot.
    Idx(i) = find(ceil(Interp_Image_Slice(:,i))>=69 & floor(Interp_Image_Slice(:,i)<=71), 1);
end
                                        %shifting all by 1 - for plot
SurrogateSignal = Xrange_Finner(Idx) + repmat(1, 1, length(Idx)); 
%create figure - image with star and the corresponding plot
figure(1)
    set(gcf,'color','w'); % set figure background to white
    subplot(1,2,1)
        imshow(niis(:,:,6)',[]);
        axis on
        hold on
        %add red star
        plot(SurrogateSignal(6), 375,'r*','MarkerSize',16);
        xlabel('Anterior-Posterior','FontSize',14); 
        ylabel('Superior-Inferior','FontSize',14);
        hold off
    subplot(1,2,2)
        plot(SurrogateSignal);
        xlabel('Image','FontSize',14); 
        ylabel('Surrogate Signal','FontSize',14);
        title('Surrogate Signal - AP displacement', 'FontSize',16);
%% Store workspace 
% need data for later 
save('Task2.mat');