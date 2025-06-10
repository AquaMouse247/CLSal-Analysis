function kl_div_value=KLDiv(P,Q)
%  kl_div_value = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  P and Q  are automatically normalised to have the sum of one on rows
% have the length of one at each
% P =  n x nbins
% Q =  1 x nbins or n x nbins(one to one)
% kl_div_value = n x 1
if size(P,2)~=size(Q,2)
    error('the number of columns in P and Q should be the same');
end
if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
   error('the inputs contain non-finite values!')
end
% normalizing the P and Q
if size(Q,1)==1
    Q = Q ./sum(Q);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    kl_div_value = sum(P.*log(P./repmat(Q,[size(P,1) 1])),2);

elseif size(Q,1)==size(P,1)

    Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    kl_div_value =  sum(P.*log(P./Q),2);
end
% resolving the case when P(i)==0
kl_div_value(isnan(kl_div_value))=0;