function sweep_pr(only)
% SWEEP_PR  Precision and recall of the original detector and the prototype
% over a range of settings, from strict to lenient, on sets 1, 2 and 3.
%
%   recall    = shots found / shots marked
%   precision = detections that hold a shot / all detections
%               (detections in windows without airguns all count against it)
%
% Original detector: the fixed floor on the squared correlation
% (minCorrThresh) is swept; everything else stays as written. With the
% floor at 0 only the "15% above the segment median" rule is left, so the
% last points lower that offset as well.
%
% Prototype: it has no single threshold, so its four acceptance settings
% are moved together, from strict to lenient (shapeWeak, shapeThresh,
% loudProm, clearPP_dB). The row marked current is the file as written.
%
% Results go to pr_curve.csv (one row per set) and pr_curve_sites.csv (one row
% per set and site); plot_pr.py draws the curves together with the network's.
%
%   sweep_pr              all three sets
%   sweep_pr({'set3'})    only these sets; writes pr_curve_sites.csv only

here = fileparts(mfilename('fullpath')); proto = fullfile(fileparts(here),'xcorr_shape','xcorr_airgun_shape.m');
%        floor     offset   current
origS = {'1e17',   '0.15',  0
         '5e16',   '0.15',  0
         '2e16',   '0.15',  0
         '7e15',   '0.15',  1
         '3e15',   '0.15',  0
         '1e15',   '0.15',  0
         '3e14',   '0.15',  0
         '1e14',   '0.15',  0
         '0',      '0.15',  0
         '0',      '0.05',  0
         '0',      '0',     0};
%         weak    thresh   loudProm  clearPP  current
protoS = {'0.8',  '0.95',  'Inf',    'Inf',   0
          '0.7',  '0.9',   'Inf',    'Inf',   0
          '0.6',  '0.85',  '4',      '16',    0
          '0.5',  '0.75',  '2',      '12',    0
          '0.4',  '0.65',  '1',      '8',     1
          '0.3',  '0.55',  '0.75',   '6',     0
          '0.2',  '0.45',  '0.5',    '4',     0
          '0.1',  '0.35',  '0.25',   '2',     0};
sets = {'set1','set2','set3'}; rows = {}; srows = {};
if nargin > 0, sets = only; end
for s = 1:numel(sets)
    for i = 1:size(origS,1)
        T = score_detector('pr_tmp',{'minCorrThresh',origS{i,1}; 'parm.c2_offset',origS{i,2}},'',sets{s});
        lab = sprintf('floor %s offset %s',origS{i,1:2});
        rows(end+1,:) = add(T,'original',lab,i,origS{i,3},sets{s}); %#ok<AGROW>
        srows = [srows; bysite(T,'original',lab,i,origS{i,3},sets{s})]; %#ok<AGROW>
    end
    for i = 1:size(protoS,1)
        T = score_detector('pr_tmp',{'parm.shapeWeak',protoS{i,1}; 'parm.shapeThresh',protoS{i,2}; ...
            'parm.loudProm',protoS{i,3}; 'parm.clearPP_dB',protoS{i,4}},proto,sets{s});
        lab = sprintf('weak %s shape %s loud %s clear %s',protoS{i,1:4});
        rows(end+1,:) = add(T,'prototype',lab,i,protoS{i,5},sets{s}); %#ok<AGROW>
        srows = [srows; bysite(T,'prototype',lab,i,protoS{i,5},sets{s})]; %#ok<AGROW>
    end
end
vars = {'detector','setting','step','current','set','shots','detections','found','spare'};
if nargin == 0, writetable(cell2table(rows,'VariableNames',vars),fullfile(here,'pr_curve.csv')); end
S = cell2table(srows,'VariableNames',[vars,{'site'}]);
f = fullfile(here,'pr_curve_sites.csv');
if exist(f,'file')                           % keep the rows of sets not rerun this time
    old = readtable(f,'Delimiter',',','TextType','string');
    old = old(~ismember(old.set,string(sets)),:);
    S.detector = string(S.detector); S.setting = string(S.setting); S.set = string(S.set); S.site = string(S.site);
    S = [old; S];
end
writetable(S,f);
for s = 1:numel(sets)                        % remove the temporary score files
    delete(fullfile(here,sets{s},'*_pr_tmp.csv'));
end
end

function r = add(T,det,label,step,cur,setName)
r = {det,label,step,cur,setName,sum(T.shots),sum(T.detections),sum(T.found),sum(T.false)};
fprintf('PR %-9s %s %-42s found %d of %d, detections %d, without a shot %d\n',det,setName,label,r{8},r{6},r{7},r{9});
end

function r = bysite(T,det,label,step,cur,setName)
% the same totals, one row per site
u = unique(T.site); r = cell(numel(u),10);
for k = 1:numel(u)
    b = T.site == u(k);
    r(k,:) = {det,label,step,cur,setName,sum(T.shots(b)),sum(T.detections(b)),sum(T.found(b)),sum(T.false(b)),char(u(k))};
end
end
