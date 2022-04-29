clear all
format compact
pref = 1                 % if 1, split out the exp(-alpha*x.^2/2) term and fit the rest
ddx = 2e-2               % finite difference step
xmax = 5                 % max value of the coordinate
maxpts = 2000            % number of grid points is 2*maxpts+1
Nfit = 500               % how many points to use in GPR
x = -maxpts:maxpts;      
x = x/maxpts*xmax;       % x grid
dx = x(2)-x(1);          % dx for rectangular quadrature
Vharm = 0.5*x.^2;        % harmonic potential
plot(x,Vharm)  
title('The potential curve')
xlabel('x') 
ylabel('V')
disp('Press a key !')  
pause;

% initialize NN with the harmonic wavefunction
n = 1                    % quantum number
Eharm = 0.5+n            % exact energy
E = Eharm + 0.1          % initial energy with an error vs exact
alpha = 1.00             % distortion of initial guess via the exp term. ND: exponential sensitivity to this parameter
beta = 0.03              % distortion of initial guess via the Hermite polynomial term
t = pi^(-0.25)/sqrt(2^n*factorial(n))*exp(-alpha*x.^2/2).*hermiteH(n,x+beta*x.^2); % initial wavefunction
  if pref, t = t./exp(-alpha*x.^2/2); end;                               % if removing the exp factor
  
% build a GPR model
x = x';
t = t';
% inital hyperparameters
Sigma0 = 0.01;
SigmaF0 = Sigma0;
kparams0 = [50, 0.2]
if n == 0,
    FM = 'none';         % don't fit hyperpamaters for the ground state as the constant GPR target at n=0 (with pref = 1) does not allow to determine them well 
else
    FM = 'exact';        % fit hyperparameters
end;    
order = randperm(max(size(x)));
gprMdl = fitrgp(x(order(1:Nfit)),t(order(1:Nfit)),'FitMethod',FM,'ConstantSigma',false,'KernelParameters',kparams0,'Sigma',Sigma0);
% keep GPR hyparameters for the self-consistency cycle
kparams0 = gprMdl.KernelInformation.KernelParameters
Sigma0 = gprMdl.Sigma

% Test the model
[y,~,yint1] = predict(gprMdl,x);
plot(x,t,'b',x,y,'r')            % NB: if the exp factor is split out, t follows the Hermite polynomial. The wavefunction will be plotted in the end
title('The ML target (psi or t): initial (blue) and GPR fit (red)')
xlabel('x') 
ylabel('target')
disp('Press a key !')  
pause;

% 7-point finite different stencil for the kinetic energy operator
Ty = -0.5*( 2*predict(gprMdl,x-3*ddx)-27*predict(gprMdl,x-2*ddx)+270*predict(gprMdl,x-ddx)-490*predict(gprMdl,x)+270*predict(gprMdl,x+ddx)-27*predict(gprMdl,x+2*ddx)+2*predict(gprMdl,x+3*ddx) )/(180*ddx^2); % compute the second derivative (KEO) with a 7-point stencil
      if pref, Ty = -0.5*( 2*predict(gprMdl,x-3*ddx).*exp(-alpha*(x-3*ddx).^2/2)-27*predict(gprMdl,x-2*ddx).*exp(-alpha*(x-2*ddx).^2/2)+270*predict(gprMdl,x-ddx).*exp(-alpha*(x-ddx).^2/2)-490*y.*exp(-alpha*x.^2/2)+270*predict(gprMdl,x+ddx).*exp(-alpha*(x+ddx).^2/2)-27*predict(gprMdl,x+2*ddx).*exp(-alpha*(x+2*ddx).^2/2)+2*predict(gprMdl,x+3*ddx).*exp(-alpha*(x+3*ddx).^2/2) )/(180*ddx^2); end; % if removing the bell

% norm of the wavefunction. Strictly speaking we do not need to maintain
% <psi|psi>=1 but we want to prevent the wavefunction from collapsing to
% zero during self-consistency cycles
norm = dot(y,y)*dx
  if pref, 
      norm = dot(y.*exp(-alpha*x.^2/2),y.*exp(-alpha*x.^2/2))*dx, 
  end;  % if removing the bell
gprMdlold = gprMdl;
tharm = t;
V = Vharm';
delta = 0.05;            % vicinity of the point where E-V(x)
Nsteps = 200;            % number of iterations
mix = 0.2;               % mixing parameter to stabilize the cycles
for i=1:Nsteps,          % self-consistency cycles
    cycle = i
    told = t;
    Eold = E;
    yold = y;    
    Tyold = Ty;
    Ty = -0.5*( 2*predict(gprMdl,x-3*ddx)-27*predict(gprMdl,x-2*ddx)+270*predict(gprMdl,x-ddx)-490*predict(gprMdl,x)+270*predict(gprMdl,x+ddx)-27*predict(gprMdl,x+2*ddx)+2*predict(gprMdl,x+3*ddx) )/(180*ddx^2); % compute the second derivative (KEO) with a 7-point stencil
      if pref, Ty = -0.5*( 2*predict(gprMdl,x-3*ddx).*exp(-alpha*(x-3*ddx).^2/2)-27*predict(gprMdl,x-2*ddx).*exp(-alpha*(x-2*ddx).^2/2)+270*predict(gprMdl,x-ddx).*exp(-alpha*(x-ddx).^2/2)-490*predict(gprMdl,x).*exp(-alpha*x.^2/2)+270*predict(gprMdl,x+ddx).*exp(-alpha*(x+ddx).^2/2)-27*predict(gprMdl,x+2*ddx).*exp(-alpha*(x+2*ddx).^2/2)+2*predict(gprMdl,x+3*ddx).*exp(-alpha*(x+3*ddx).^2/2) )/(180*ddx^2); end; % if removing the bell
    Ty = Ty/sqrt(norm);
    Ty = mix*Ty+(1-mix)*Tyold;       % mixer to stabilize the cycles
      % E = <psi|H|psi>/<psi|psi>
      if pref, 
          E = dot(y.*exp(-alpha*x.^2/2),(Ty+V.*y.*exp(-alpha*x.^2/2)))*dx/norm, 
      else
          E = dot(y,(Ty+V.*y))*dx/norm, 
      end;
      E = mix*E+(1-mix)*Eold;
    t = Ty./(E - V);  
      if pref, t = t./exp(-alpha*x.^2/2); end; % if the exp term is split out
    t = mix*t+(1-mix)*told;
    
    % GPR
    xgpr = x(abs(E-V)>delta);
    tgpr = t(abs(E-V)>delta);
    order = randperm(max(size(xgpr)));
    gprMdl = fitrgp(xgpr(order(1:Nfit)),tgpr(order(1:Nfit)),'FitMethod','none','KernelParameters',kparams0,'Sigma',Sigma0); 
    [y,~,yint1] = predict(gprMdl,x); 
    
    norm = dot(y,y)*dx;
      if pref, norm = dot(y.*exp(-alpha*x.^2/2),y.*exp(-alpha*x.^2/2))*dx; end;  % if the exp term is split out
    % Plotting intermediate results in each cycle: initial guess in blue,
    % final result in r and GPR target in green
    plot(x(abs(E-V)>delta),t(abs(E-V)>delta),'g',x,tharm,'b',x,y,'r')
    title('The ML target (psi or t): initial (blue), GPR fit (red), Eq. 3.2.5 (green)')
    xlabel('x') 
    ylabel('target')
    pause(0.01) 
end;
disp('Press a key !')  
pause;
% Plot the final wavefunction: GPR target in green, initial guess in blue,
% and the final fitted wavefunction in red
if pref, plot(x,t.*exp(-alpha*x.^2/2),'g',x,tharm.*exp(-alpha*x.^2/2),'b',x,y.*exp(-alpha*x.^2/2),'r'); end; % if removing the bell
title('The final wavefunction: initial (blue), GPR fit (red), Eq. 3.2.5 (green)')
xlabel('x') 
ylabel('wavefunction')
Final_energy = E
norm
