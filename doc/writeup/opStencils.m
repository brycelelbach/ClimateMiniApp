
% Build the moment matrix
x = [-3:2]';
N = size(x,1) - 1;
P = 5;
for p=1:P
    M(:,p) = (x(2:N+1).^p - x(1:N).^p) / p;
end

s = pinv(M') * eye(N,1)

% check exact solutions
N = 20;
dx = 1/N;
xc = (.5:N-.5)'*dx;

% Check the implicit operator
k_z = 3;
nu = .1;
phi = cos(pi*k_z*xc);
Lz = spdiags(ones(N,1)*[1 -2 1],-1:1,N,N);
Lz(1,1) = -1;
Lz(N,N) = -1;
Lz = Lz / dx^2;
% amp = L*phi ./ phi;
% amp = amp(1);
gamma = nu * (2*cos(pi*k_z*dx) - 2)/dx^2
dt = 1.5/N;
H = speye(N,N) - dt * nu * Lz;
amp = (H \ phi) ./ phi;
amp = amp(1);
amp - (1 - dt*gamma)^(-1)

% Check the explicit operator
k_x = 1;
u_x = 1;
phi = sin(2*pi*k_x*xc);
% s = [0 0 1 0 0]';
sop = ([0 s'] - [s' 0]);
Lxy = spdiags(ones(N,1)*sop,-3:2,N,N);
Lxy(1,N-2:N) = sop(1:3);
Lxy(2,N-1:N) = sop(1:2);
Lxy(3,N:N) = sop(1:1);
Lxy(N-1,1:1) = sop(6:6);
Lxy(N,1:2) = sop(5:6);
% sop = ([-1 1]);
% Lxy = spdiags(ones(N,1)*sop,-1:0,N,N);
% Lxy(1,N) = sop(1);

% Forward Euler
cfl = dt * u_x/dx;
% H = speye(N,N) - cfl * Lxy
% Third-order Taylor
% H = speye(N,N) - cfl * Lxy + (1/2) * (cfl * Lxy)^2 - (1/6) * (cfl * Lxy)^3;
% Fourth-order Taylor
H = speye(N,N) - cfl * Lxy + (1/2) * (cfl * Lxy)^2 ...
    - (1/6) * (cfl * Lxy)^3 + (1/24) * (cfl * Lxy)^4;

Lphi = zeros(size(phi));
for is=1:6
    shift = is - 4;
    x = xc + shift*dx;
    Lphi = Lphi + sop(is)*sin(2*pi*k_x*x);
end

norm(Lxy*phi - Lphi)

clf
hold
plot(phi)
plot(H^30*phi,'red')
grid on

