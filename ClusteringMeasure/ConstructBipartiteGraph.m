function [B] = ConstructBipartiteGraph_v2(data, nSmp, num_views, num_anchor, k)
% Input: 
% 
conData = [];
for v = 1:num_views
    conData = [conData data{v}'];
end

H = cell(num_views,1);
 
[~,ind,~] = graphgen_anchor(conData,num_anchor);

for v = 1:num_views
    H{v} = data{v}(:,ind);
end

for v = 1:num_views
    D = L2_distance_1(data{v}, H{v});
    [~, idx] = sort(D, 2); % sort each row
    B{v} = zeros(nSmp,num_anchor);
    for ii = 1:nSmp
        id = idx(ii,1:k+1);
        di = D(ii, id);
        B{v}(ii,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end
%     B{v}= B{v}';
end%构造的B是n*m的

end

