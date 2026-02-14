function [R] = BALM(B, numClust, Max_iter, lambda, gamma)
% BALM - main function for Auto-weighted and Intrinsic Multi-view Spectral Clustering via Anchor Graphs
%
% Syntax: [R] = ELMSC(B, numClust, Max_iter, lambda, gamma)
%
% shuangxun.ma@chd.edu.cn
% 2025/09/1

num_views = size(B, 2);
[nSmp, num_anchor] = size(B{1});
 
for v = 1:num_views
    wv(v) = 1/num_views;
end
P = zeros(nSmp, num_anchor);

quadprog_options = optimset( 'Algorithm','interior-point-convex','Display','off');

for i = 1:Max_iter
    
    % Update P
    for iii = 1:nSmp
        qu_tmp = wv(1);
        tmp_2 = wv(1)*B{1}(iii,:);
        for v = 2:num_views
            qu_tmp = qu_tmp + wv(v);
            tmp_2 = tmp_2+wv(v)*B{v}(iii,:);
        end
        tmp_2 = -2*tmp_2;
        tmp_1 = qu_tmp + lambda;
        tmp_1 = tmp_1*eye(num_anchor);
        
        tmp_1 = 2*tmp_1;
        tmp_1 = (tmp_1'+tmp_1)/2;
        
        P(iii,:) = quadprog(tmp_1,tmp_2',[],[],ones(1,num_anchor),1,zeros(num_anchor,1),ones(num_anchor,1),P(iii,:),quadprog_options);
    end
    
    % Update wv
    for v = 1:num_views
        wv(v)=(-((+norm((P-B{v}),'fro')^2))/gamma)^(1/(gamma-1));
    end
    
    if i >= Max_iter
        fprintf('Reach the max iterations!\n');
        break;
    end
    
end

feaSum = full(sqrt(sum(P,1)));
feaSum = max(feaSum, 1e-12);
P = P./feaSum(ones(size(P,1),1),:);
R = mySVD(P,numClust+1);
R(:,1) = [];

R=R./repmat(sqrt(sum(R.^2,2)),1,numClust);

end