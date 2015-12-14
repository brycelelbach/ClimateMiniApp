
% Build the moment matrix
x = [-3:2]';
N = size(x,1) - 1;
P = 5;
for p=1:P
    M(:,p) = (x(2:N+1).^p - x(1:N).^p) / p;
end

s = pinv(M') * eye(N,1)

% check exact solutions
nu = .1;
k_x = 1;
k_z = 1;
omega_z = nu*k_z^2;
N = 10;
dx = 1/N;
xc = (.5:N-.5)'*dx;

% Check the implicit operator
k_z = 3;
phi = cos(pi*k_z*xc);
L = spdiags(ones(N,1)*[1 -2 1],-1:1,N,N)
L(1,1) = -1;
L(N,N) = -1;
L = L / dx^2;
% amp = L*phi ./ phi;
% amp = amp(1);
gamma = nu * (2 - 2*cos(pi*k_z*dx))/dx^2
dt = .5;
H = speye(N,N) - dt * nu * L;
amp = (H \ phi) ./ phi;
amp = amp(1);
amp - (1 + dt*gamma)^(-1)

% Check the explicit operator
k_x = 2;
u_x = 1;
phi = sin(2*pi*k_x*xc);
sop = ([0 s'] - [s' 0])/dx;
L = spdiags(ones(N,1)*s',-3:1,N,N)
