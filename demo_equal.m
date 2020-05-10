clear;
addpath(genpath('./'))
testData = imageDatastore('real_data/image/');
I = readimage(testData, 1);
f_gt = 3442; 
grayImage = rgb2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage, points);
numImages = numel(testData.Files);
tforms(numImages) = projective2d(eye(3));

gravitypath = strcat('real_data/gravity/'); 
gravityname = dir(fullfile(gravitypath,'*.mat'));
gravity = importdata([gravitypath, gravityname(1).name]);

% Iterate over remaining image pairs
for n = 2:numImages
    
    % Store points, features and gravith for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
    gravityPrevious = gravity;
    
    % Read I(n), gravity(n)
    I = readimage(testData, n);
    gravity = importdata([gravitypath, gravityname(n).name]);
    
    % Convert image to grayscale.
    grayImage = rgb2gray(I);
    
    % Detect and extract SURF features for I(n).
    points = detectSURFFeatures(grayImage);
    [features, points] = extractFeatures(grayImage, points);
    
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
    
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);
    
    
    [tforms(n), inlierpoints1, inlierpoints2]= estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    
    pt1 = inlierpoints1.Location - [size(grayImage,2) size(grayImage,1)]/2; % shift the orign to the image center
    pt2 = inlierpoints2.Location - [size(grayImage,2) size(grayImage,1)]/2;
    
    scale = normalize2d(pt1);
    
    x1 = scale*pt1; x1(:,3) = 1;
    x2 = scale*pt2; x2(:,3) = 1;
    x1 = x1.'; x2 = x2.';
    % least squares (can be replaced with RANSAC)
    Npts = size(x1,2);
    A = zeros(2*Npts,9);
    Z = [0 0 0];
    for m = 1:Npts
        X = x1(:,m).';
        x = x2(1,m); y = x2(2,m);
        A(2*m,:) = [  Z  -X  y*X];
        A(2*m-1,:) = [ X   Z  -x*X];
    end
    [~,~,V] = svd(A);
    V8 = V(:,8)';
    V9 = V(:,9)';
    
    R1 = gravityPrevious/gravityPrevious(1,1); % make the first entry become 1
    R2 = gravity/gravity(1,1);
    a1 = R1(1,2); a2 = R1(2,1); a3 = R1(2,2); a4 = R1(2,3); a5 = R1(3,1); a6 = R1(3,2); a7 = R1(3,3);
    b1 = R2(1,2); b2 = R2(2,1); b3 = R2(2,2); b4 = R2(2,3); b5 = R2(3,1); b6 = R2(3,2); b7 = R2(3,3);
    
    sols = solver_fHf([V8 V9 a1 a2 a3 a4 a5 a6 a7 b1 b2 b3 b4 b5 b6 b7]); % 1st row - normalized focal length, 2nd row - scalar of V8, 3rd row - scalar of Hy.
    good = []; G = zeros(3,3,1); H = zeros(3,3,1); Hy = zeros(3,3,1); 
    for k = 1:size(sols,2)
        G(:,:,k) = sols(2,k)*reshape(V8,3,3).'+reshape(V9,3,3).';
        H(:,:,k) = [1/sols(1,k) 0 0; 0 1/sols(1,k) 0; 0 0 1]*G(:,:,k)*[sols(1,k) 0 0; 0 sols(1,k) 0; 0 0 1]/sols(3,k);
        Hy(:,:,k) = R2*H(:,:,k)*R1.';
        
        motions = homodec(Hy(:,:,k));
        Ra = motions(1).R; Rb = motions(2).R;
        if ( (abs(Ra(1,2))+abs(Ra(2,1))+abs(Ra(2,3))+abs(Ra(3,2)))<0.0005 || (abs(Rb(1,2))+abs(Rb(2,1))+abs(Rb(2,3))+abs(Rb(3,2)))<0.0005) % The rotation matrix should satisfy Ry
            good = [good k];
        end
        
    end
    inliers = zeros(1,1);
    if length(good)>1 % if there are multiple candidates
        for k = 1:length(good)
            
            Hx1    = G(:,:,good(k))*x1;
            invHx2 = G(:,:,good(k))\x2;
            
            x1     = hnormalise(x1);
            x2     = hnormalise(x2);
            Hx1    = hnormalise(Hx1);
            invHx2 = hnormalise(invHx2);
            
            d2 = sum((x1-invHx2).^2)  + sum((x2-Hx1).^2);
            inliers(k) = length(find(abs(d2) < 0.0001));
        end
        [~,I] = max(inliers);
        best = good(I);
    else
        best = good;
    end
    
    
    f_est = sols(1,best)/scale;
    error_f = abs(f_est-f_gt)/f_gt;
    
end


