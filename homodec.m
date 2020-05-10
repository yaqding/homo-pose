% Input - Euclidean homography matrix H
% Output - motion parameters R, T, N
% Reference:
% An invitation to 3D vision, section 5.3.3 Page 136

function motions = homodec(H)


s = svd(H);
H = H/s(2);

[~,S,V] = svd(H'*H);

if det(V) < 0
    V = -V;
end

s1 = S(1,1);
s3 = S(3,3);

v1 = V(:,1); v2 = V(:,2); v3 = V(:,3);

u1 = (sqrt(1-s3)*v1 + sqrt(s1-1)*v3) / sqrt(s1-s3);
u2 = (sqrt(1-s3)*v1 - sqrt(s1-1)*v3) / sqrt(s1-s3);

U1 = [v2 u1 cross(v2,u1)];
W1 = [H*v2 H*u1 skew(H*v2)*H*u1];

U2 = [v2 u2 cross(v2,u2)];
W2 = [H*v2 H*u2 skew(H*v2)*H*u2];

R1 = W1*U1';
R2 = W2*U2';

n = cross(v2, u1);
if n(3) > 0
    motions(1).n = n;
    motions(1).t = (H-R1)*n;
    
else
    motions(1).n = -n;
    motions(1).t = -(H-R1)*n;
end

motions(1).R = R1;

n = cross(v2, u2);
if n(3) > 0
    motions(2).n = n;
    motions(2).t = (H-R2)*n;
else
    motions(2).n = -n;
    motions(2).t = -(H-R2)*n;
end
motions(2).R = R2;


end


function V = skew(vec)

V = [0 -vec(3)  vec(2); vec(3) 0 -vec(1); -vec(2) vec(1) 0];

end


