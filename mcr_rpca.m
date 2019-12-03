function [L,S,T] = mcr_rpca(D,l,m,lambda1,lambda2,tol)
% Motion coherency regularized RPCA
%     [L,S,T] = arg min ||L||_* + lambda1*||S||_1 + lambda2*||T||_TV,
%         s.t. D = L + S, T = S
%     D: input data
%     [l,m]: image size
%     tol: stop criterion (default: 1e-4)
% Refs:
%     Extracting contrast-filled vessels in X-ray angiography by graduated RPCA with motion coherency constraint
% Contact:
%   Mingxin Jin
%   Shanghai Jiao Tong University
%   jmx106@sjtu.edu.cn


n = size(D,2);

if nargin < 2
    l = sqrt(size(D,1));
end

if nargin < 3
    m = size(D,1)/l;
end

if nargin < 4
    lambda1 = 0.5/sqrt(max(l*m,n));
end

if nargin < 5
    lambda2 = 0.2/sqrt(max(l*m,n));
end

if nargin < 6
    tol = 1e-4;
end

addpath PROPACK;

%Initialize
L = zeros(l*m,n);
S = L;
T = L;
X = L;
Y = L;

%parameters
norm_two = lansvd(D, 1, 'L');
mu = 1.25/norm_two;
mu_bar = mu * 1e7;
rho = 1.5;
iter = 0;
d_norm = norm (D, 'fro');
maxIter = 1000;
converged = false;
sv = 10;

while ~converged
    iter = iter+1;
    % update L
    if choosvd(n, sv) == 1
        [U,S0,V] = lansvd(D-S+X/mu, sv, 'L');
    else
        [U,S0,V] = svd(D-S+X/mu, 'econ');
    end
    diagS = diag(S0);
    svp = length(find(diagS > 1/mu));
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    L = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';
    
    % update S
    Q = (D-L+T+(X+Y)/mu)/2;
    S = sign(Q) .* max(abs(Q)-lambda1/2/mu,0);
    
    % update T
    K = S-Y/mu;
    if any(K(:))
        T = SplitBregman3DROF(K,mu/lambda2,0.001,m,l);
    else
        T = K;
    end
    
    % Update multipliers
    X = X + mu * ( D - L - S );
    Y = Y + mu * ( T - S );
    
    mu = min(mu*rho, mu_bar);
    stopCriteria = norm(D-L-S,'fro')/d_norm;
    
    if mod(iter,10)==0
        disp(['iter ' num2str(iter) ' stopCriterion ' num2str(stopCriteria)]);
    end
    if stopCriteria < tol;
       converged = true;
    end
    if iter == maxIter
       converged = true;
    end
end
end
