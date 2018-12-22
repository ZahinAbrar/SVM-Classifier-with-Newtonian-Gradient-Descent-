clc
clear all
close all

alpha = 0.01;
beta = .8;
q = 20;
m = 100;
n =100;
t =1;
gamma = 10;
train = importdata('linearly_separable_2.txt');
test = importdata('linearly_separable_2.txt');
y = train(:,1); % seperating class
one = find(y==1);
x_train = train(:,[2,3]);
K = Kernel(x_train);
M = diag(y)* K * diag(y);
lambda = 0.1*ones(200,1);
mu = 9.9*ones(200,1);
epsilon = 1e-5;

m_11 = (t*M) + diag(1./ (lambda.^2));
m_12 = zeros(200);
m_13 = eye(200);
m_14 = y;
m_21 = zeros(200);
m_22 = diag(1./ (mu.^2));
m_23 = eye(200);
m_24 = zeros(200,1);
m_31 = eye(200);
m_32 = eye(200);
m_33 = zeros(200);
m_34 = zeros(200,1);
m_41 = y';
m_42 = zeros(1,200);
m_43 = zeros(1,200);
m_44 = 0;

 L = [m_11 m_12 m_13 m_14; m_21 m_22 m_23 m_24 ;m_31 m_32 m_33 m_34; m_41 m_42 m_43 m_44];
% L = eye(601);

d_1 = (t*M*lambda) - (t * ones(200,1)) - (1./lambda);
d_2 = -1./mu;
d_3 = zeros(200,1);
d_4 = 0;

D = -[d_1 ; d_2 ; d_3 ; d_4];

S = L\D;

del_lambda = S(1:200);
del_mu = S(201:400);
nu = S(401:600)
w = S(600)


n_a = (t*M*lambda) - (t*ones(200,1) - (1./lambda)+ nu + (w*y));
n_b = -1./mu + nu;
N = [n_a ; n_b];


t =1 ;
cnt = 1;
while t < 10^9;
    
   D = -[d_1 ; d_2 ; d_3 ; d_4];
   S = L\D;
   %s_wrong = L\D
   del_lambda = S(1:200);
   del_mu = S(201:400);
   nu = S(401:600)
   w = S(600)
   stack = [del_lambda ; del_mu];
   int_cnt = 1;
   while abs((N' * stack)^2)/2 > epsilon;
       s = 1;
       while any((lambda + (s* del_lambda)) < 0) || any((mu + (s* del_mu)) < 0 );
           s = s*beta;
       end
       
       fun = @(lambda) ((t/2)*lambda'*M*lambda - t*ones(1,200)*lambda);
       phi = @(lambda,mu)( - sum(log(lambda)) - sum(log(mu)));
       while fun(lambda + (s*del_lambda)) + phi((lambda + (s*del_lambda)),(mu + (s*del_mu))) > fun(lambda) + phi(lambda,mu) - abs(s*alpha*(N'*stack));
           s = s*beta ;
       end
       
       lambda = lambda + (s*del_lambda); %updating lambda
       mu = mu + (s*del_mu);             %updating mu
       
       m_11 = (t*M) + diag(1./ (lambda.^2));
       m_12 = zeros(200);
       m_13 = eye(200);
       m_14 = y;
       m_21 = zeros(200);
       m_22 = diag(1./ (mu.^2));
       m_23 = eye(200);
       m_24 = zeros(200,1);
       m_31 = eye(200);
       m_32 = eye(200);
       m_33 = zeros(200);
       m_34 = zeros(200,1);
       m_41 = y';
       m_42 = zeros(1,200);
       m_43 = zeros(1,200);
       m_44 = 0;

       L = [m_11 m_12 m_13 m_14; m_21 m_22 m_23 m_24 ;m_31 m_32 m_33 m_34; m_41 m_42 m_43 m_44]; % calculating L for KKT

       d_1 = (t*M*lambda) - (t * ones(200,1)) - (1./lambda);
       d_2 = -1./mu;
       d_3 = zeros(200,1);
       d_4 = 0;
       
       D = -[d_1 ; d_2 ; d_3 ; d_4]; %calculating D for KKT

       S = inv(L) *D; %calculating S with the updated value
       del_lambda = S(1:200);  %extracting new del_lambda
       del_mu = S(201:400);    %extracting new del_mu
       stack = [del_lambda ; del_mu];
       n_a = (t*M*lambda) - (t*ones(200,1) - (1./lambda)+ nu + (w*y));
       n_b = -1./mu + nu;
       N = [n_a ; n_b];
       fprintf('int_count = %f, cnt = %f, s =%f\n', int_cnt, cnt,s);
       fprintf('condition= %f',abs((N' * stack)^2)/2 );
       int_cnt = int_cnt+1
       
    end

    t= t*q;
    cnt = cnt+1
end  

plot(lambda,'-o')