function scale = normalize2d(pt)

dist = sqrt(pt(:,1).^2 + pt(:,2).^2);
meandist = mean(dist(:));
scale = sqrt(2)/meandist;

