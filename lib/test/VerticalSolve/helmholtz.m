
% TODO:
% Regular stencil
% Fix the ends for dphi/dz = 0
% Clean up so number of bands is minimized

for res=1:3
    
N=5*2^res;
h = 1/N;
coef = .01;

% Build the regular matrix
sG=[1 -15 15 -1]/(12*h);
sG1 = [-145 159 -15 1]/(120*h);
sL=[-1 16 -30 16 -1]/(12*h^2);
L = zeros(N);
for i=-2:2
    ix = i+3;
    len = N - abs(i);
    L = L + diag(ones(len,1)*sL(ix),i);
end

% Build the rows of L with bc's at ends
L(1,1:4) = sG1 / h;
L(2,1:4) = (sG-sG1) / h;
L(N,N-3:N) = -(-fliplr(sG1)) / h;
L(N-1,N-3:N) = (-fliplr(sG1)-sG) / h;

% 2nd order version
% L = diag((2/h^2 + 1)*ones(N,1),0)
% L = L + diag(-1/h^2*ones(N-1,1),1)
% L = L + diag(-1/h^2*ones(N-1,1),-1)
% L(1,1) = 1/h^2+1
% L(N,N) = 1/h^2+1

% This is to calculate the stencil
% P = 4;
% M = zeros(P+1,P+1);
% % Fill in all p rows for the first P columns
% for p=1:P+1
%     M(p,1:P) = (([1:P]-0).^p - ([0:P-1]-0).^p )/p;
% end
% % Add the Neumann bc for the last column
% % M(:,P+1) = (((-1).^[0:P]) .* )';
% M(:,P+1) = zeros(P+1,1);
% M(2,P+1) = 1;
% M = M';
% 
% % Stencil we want is for grad phi at x=1
% f = [0:P]';
% % f = zeros(P+1,1);
% % f(2) = 1;
% sG1 = pinv(M') * f;
% sG1 = sG1(1:P)/h;

% Initialize the solution with cell averages
x = [0:N]'*h;
% phi = ((x(2:N+1) - 1).^3 - (x(1:N)-1).^3)/(3*h);
kx = 2*pi;
phi = (sin(kx * x(2:N+1)) - sin(kx * x(1:N)))/(kx * h);
% phi0 = sin(kx*
rhs = - kx^2 * phi;

err(res) = max(abs(rhs - L*phi))
end





