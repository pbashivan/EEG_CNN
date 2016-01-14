locs = readlocs('Z:\CVPIA\Pouya\SDrive\SkyDrive\Documents\IBM\Codes\Older\data\Neuroscan_ch_locations.ced', ...
    'elecind', [1:59 61:63]);
% Compute distances between all consequtive electrodes
for i = 1 : 59
    dist(i) = norm([(locs(i+1).X - locs(i).X) ...
        (locs(i+1).Y - locs(i).Y) ...
        (locs(i+1).Z - locs(i).Z)], 2);
end

% Project into 2D
for i = 1 : length(locs)
    Xs(i) = locs(i).X;
    Ys(i) = locs(i).Y;
    Zs(i) = locs(i).Z;
    labels{i} = locs(i).labels;
end

%% Azimuth projection
for i = 1 : length(A)
    proj(i, :) = elproj(A(i, :), 'polar');
end

figure;scatter(proj(:,1), proj(:,2))

%% Save to spherical coordinates
for i = 1 : length(A)
    [proj(i, 1), proj(i, 2), proj(i, 3)] = cart2sph(A(i,1), A(i,2), A(i,3));
end