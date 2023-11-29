clc
clear
close all

%% Acquisição dos dados
arquivo = fopen('Evolucao_fbest_1000.txt', 'r');
str = fscanf(arquivo, '%c');
fclose(arquivo);

% Remove os caracteres de quebra de linha da string
str = strrep(str, newline, '');

% Transforma a string em uma matriz de números
data = str2num(str);

% Reorganiza a matriz para ser 1x1000
y = reshape(data', [1, 1000]);

%% DLS
alfa = 0.01;
lambda = 0.001;
Dmax = 0.5;
t = 1:length(y);
pontos = 100;
t = t(1:pontos)';
y = y(1:pontos)';
th = 0.1*rand(3,1);
ym = th(1)*exp(-th(2)*t) + th(3);

figure()
scatter(t,y,'b','filled')
hold on
grid on
p1 = plot(0,0,'r','linewidth',3);
legend("Medido","Modelado (DLS)")
err = 10^(-5);
dth = 1000;

while((max(abs(dth))>err))
   
   dy =  y - ym;

   dh_th1 = exp(-th(2)*t);
   dh_th2 = th(1)*(exp(-th(2)*t)).*(-t); 
   dh_th3 = t.^0; 

   J = [dh_th1 dh_th2 dh_th3];
   dth = inv(J'*J + lambda*eye(size(J'*J)))*J'*dy;
   
   dth = alfa.*dth;
   
   for(k = 1:length(dth))
     if(dth(k) > Dmax) dth(k) = Dmax*(dth(k)/abs(dth(k))); end  
   end
   th = th + dth;

   ym = th(1)*exp(-th(2)*t) + th(3);
    
   %plota modelo atual
   set(p1,'XData',t,'YData',ym)
   drawnow


   %Cirterios de parada
    if(max(abs(dth))<err || sum(isnan(dth))) break; end

end
%% Parâmetros
tau = 1/th(2);
c = exp(-1/(tau));
c = 1.05*c;
t = 1:1:6*tau;
A = th(1);

%% Plot da exponencial
figure()
y = A*exp(-t/tau);
plot(t,y,'linewidth',3);
grid on
hold on
%% Plot da curva y = c*y;
y2 = A;
for k = 2:length(t)
  y2(k) = c*y2(k-1);
end
plot(t,y2,'*');

legenda = legend("Modelo da formulação original","Pior caso na nova reformulação");
set(legenda,"FontSize",14)