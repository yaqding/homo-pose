clear;
num = 100000;
fHf = zeros(1,num); Hf = zeros(1,num); Hvar = zeros(1,num);
for i = 1:num
    data = rand(1,32);
    tic
    solver_fHf(data);
    tt = toc;
    fHf(i) = tt;
    
    data2 = data(1:25);
    tic
    solver_Hf(data2);
    tt = toc;
    Hf(i) = tt;
    
    data3 = data(1:23);
    tic
    solver_var(data3);
    tt = toc;
    Hvar(i) = tt;
end

T_mean = [mean(fHf) mean(Hf) mean(Hvar)]; % evaluate with C++ is more efficient