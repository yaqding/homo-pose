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
    
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
    gravityPrevious = gravity;
    
    % Read I(n).
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
    
    pt1 = inlierpoints1.Location - [size(grayImage,2) size(grayImage,1)]/2;
    pt2 = inlierpoints2.Location - [size(grayImage,2) size(grayImage,1)]/2;
    
    scale = normalize2d(pt1);
    
    x1o = 1/f_gt*pt1; x1o(:,3) = 1; % known focal length
    
    x1 = gravityPrevious*x1o.';
    x1(1,:) = x1(1,:)./x1(3,:); x1(2,:) = x1(2,:)./x1(3,:); x1(3,:) = 1;
    %
    x2 = scale*pt2; x2(:,3) = 1;
    x2 = x2.';
    % ransac
    currbest = 0;
    for s = 1:100
        Npts = 4;
        ind = randsample(length(inlierpoints1),Npts);
        
        A = zeros(2*Npts,9);
        Z = [0 0 0];
        for m = 1:Npts
            X = x1(:,ind(m)).';
            x = x2(1,ind(m)); y = x2(2,ind(m));
            A(2*m,:) = [  Z  -X  y*X];
            A(2*m-1,:) = [ X   Z  -x*X];
        end
        [~,~,V] = svd(A);
        V8 = V(:,8)';
        V9 = V(:,9)';
        
        
        R2 = gravity/gravity(1,1);
        b1 = R2(1,2); b2 = R2(2,1); b3 = R2(2,2); b4 = R2(2,3); b5 = R2(3,1); b6 = R2(3,2); b7 = R2(3,3);
        
        sols = solver_Hf([V8 V9 b1 b2 b3 b4 b5 b6 b7]); % 1st row - normalized focal length, 2nd row - scalar of V8, 3rd row - scalar of Hy.
        good = []; G = zeros(3,3,1); H = zeros(3,3,1); Hy = zeros(3,3,1);
        for k = 1:size(sols,2)
            G(:,:,k) = sols(2,k)*reshape(V8,3,3).'+reshape(V9,3,3).';
            H(:,:,k) = [1 0 0; 0 1 0; 0 0 sols(1,k)]*G(:,:,k)/sols(3,k);
            Hy(:,:,k) = R2*H(:,:,k);
            
            motions = homodec(Hy(:,:,k));
            Ra = motions(1).R; Rb = motions(2).R;
            if ( (abs(Ra(1,2))+abs(Ra(2,1))+abs(Ra(2,3))+abs(Ra(3,2)))<0.001 || (abs(Rb(1,2))+abs(Rb(2,1))+abs(Rb(2,3))+abs(Rb(3,2)))<0.001) % The rotation matrix should satisfy Ry
                good = [good k];
            end
            
        end
        inliers = zeros(1,1);
        for k = 1:length(good)
            
            Hx1    = G(:,:,good(k))*x1;
            invHx2 = G(:,:,good(k))\x2;
            
            x1     = hnormalise(x1);
            x2     = hnormalise(x2);
            Hx1    = hnormalise(Hx1);
            invHx2 = hnormalise(invHx2);
            
            d2 = sum((x1-invHx2).^2)  + sum((x2-Hx1).^2);
            inliers(k) = length(find(abs(d2) < 0.00001));
        end
        [bestIn, I] = max(inliers);
        
        if bestIn > currbest
            currbest = bestIn;
            best_f = sols(1,I)/scale;
        end
        
    end
    
   
    error_f = abs(best_f-f_gt)/f_gt;
    
end


