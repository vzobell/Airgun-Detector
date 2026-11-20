function d = find_xwav(site, file)
% FIND_XWAV  Locate a df100 xwav of a GOM site on the frosty decimated shares.
%   d = find_xwav('GC_18','GOM_GC_18_250705_145800_df100.x.wav') returns the dir() entry.
roots = {'//frosty.ucsd.edu/GofMX_Decimated_8','//frosty.ucsd.edu/GofMX_Decimated_10', ...
         '//frosty.ucsd.edu/GofMX_Decimated_12'};
for k = 1:numel(roots)
    d = dir(fullfile(roots{k},"GOM_" + site,'*_df100',file));
    if ~isempty(d), return, end
end
