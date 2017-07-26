function points_trans = transPointsWithCPG(cpg_nii, points_orig, cpg_is_disp)
%function to transform one or more points using the provided control point
%grid which defines a b-spline transformation
%
%INPUTS:
%   cpg_nii - the control point grid as a nifti structure
%   points_orig - an n x 3 vector (or n x 2 if 2D) containing the point
%   coordinates before applying the transformtion
%   cpg_is_disp - if true the cpg contains a displacement field, if false
%       the cpg contains a deformation field. [false]
%OUTPUTS:
%   points_trans - an n x 3 vector containing the point coordinates after
%   applying the transformtion
%
%Note: deformation field = target coords + displacement field

%check for optional inputs
if ~exist('cpg_is_disp','var') || isempty(cpg_is_disp)
    cpg_is_disp = false;
end

%get coords for cpg
[cxs,cys,czs] = coords_from_nii(cpg_nii);

%loop over points and apply transformation
points_trans = zeros(size(points_orig));
if size(points_orig,2) == 2
    points_orig = [points_orig ones(size(points_orig,1),1)];
end
for pt = 1:size(points_orig,1)
    %get deformation/displacement for this point
    points_trans(pt,:) = squeeze(calcDefField(permute(cpg_nii.img,[1 2 3 5 4]),cxs,cys,czs,points_orig(pt,1),points_orig(pt,2),points_orig(pt,3)))';
end

%if cpg contains displacements need to add original point coords
if cpg_is_disp
    points_trans = points_trans + points_orig;
end