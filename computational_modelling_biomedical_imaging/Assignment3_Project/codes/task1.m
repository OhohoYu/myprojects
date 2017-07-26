%% Display the images - constant looping - ctrl c to stop

close all

%% Add functions and data paths
addpath(genpath('data'))
addpath(genpath('for_students'))

Images = dir(['data' filesep 'images' filesep 'IM*.nii']);
regs = dir(['data' filesep 'reg_results' filesep 'Reg*.nii']);

Mask = load_untouch_nii('mask.nii');

niis = [];
CPs = [];

RefNii = load_untouch_nii(Images(6).name);

for i = 1 : length(Images)
    niis = [niis ;  load_untouch_nii(Images(i).name)];
    CPs = [CPs ; load_untouch_nii(regs(i).name)];
end

deformations = [];
differences = [];

while (true)
    for i = 1 : length(niis)
        plot(2,2);
        set(gcf,'color','w'); % set figure background to white
        deformations = [deformations; deformNiiWithCPG(CPs(i), niis(i), RefNii, false)];
        difference = RefNii;
        difference.img = (deformations(i).img - double(RefNii.img)) .* double(Mask.img);
        differences = [differences; difference]; 
        %Floating image 
        subplot(2,2,1); 
        dispNiiSlice(niis(i), 'z', 1, [0, 100]);
        xlabel('Anterior-Posterior')
        ylabel('Superior-Inferior')
        %Warped
        subplot(2,2,2);
        dispNiiSlice(deformations(i), 'z', 1, [0, 100]);
        xlabel('Anterior-Posterior')
        ylabel('Superior-Inferior')
        %difference between ref and warped
        subplot(2,2,3);%diff rev and warped
        dispNiiSlice(differences(i), 'z', 1, [-10, 10]);
        xlabel('Anterior-Posterior')
        ylabel('Superior-Inferior')        
        %Warped - only mask area
        subplot(2,2,4);  
        deformations(i).img = deformations(i).img .* double(Mask.img);
        %redifines deformations after a while only mask area is printed -ok
        dispNiiSlice(deformations(i), 'z', 1, [0, 100]);
        xlabel('Anterior-Posterior')
        ylabel('Superior-Inferior')
        pause(0.01);
    end
end