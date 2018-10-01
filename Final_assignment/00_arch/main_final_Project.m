%% Users input

clearvars; close all; clc;

roadWidth = 10;
roadPoints_qual = 0.5;
roadPoints = [0 0;
              50.00 00.00;
              53.00 00.41;
              55.92 01.88;
              58.47 04.59;
              59.88 08.42;
              59.91 11.42;
              58.57 15.18;
              56.02 18.00;
              52.82 19.57;
              49.97 20.00;
              47.72 20.27;
              45.26 21.20;
              43.15 22.69;
              41.45 24.80;
              40.37 27.33;
              40.01 30.34;
              40.61 33.43;
              42.45 36.55;
              44.71 38.48;
              47.21 39.60;
              50.00 40.00;
              80.00 40.00;
              81.85 40.36;
              83.97 41.94;
              85.00 44.79;
              84.30 47.49;
              64.32 82.15;
              59.67 88.00;
              54.52 91.68;
              49.68 93.60;
              44.44 94.55;
              39.54 94.41;
              34.61 93.28;
              29.44 90.84;
              25.53 87.86;
              22.21 84.00;
              19.38 78.69;
              18.07 74.17;
              17.69 70.91;
              17.82 66.80;
              18.59 63.00;
              19.41 60.52;
              21.03 57.13;
              23.22 52.45;
              24.15 49.31;
              24.83 45.54;
              25.03 40.84;
              24.44 36.25;
              22.75 30.67;
              19.56 24.86;
              13.96 18.87;
              09.60 15.91;
              05.30 13.99];
roadPoints(:,2) = roadPoints(:,2) + 5;



%% Start and compute Road Map

roadMap = RoadMap(roadPoints, roadPoints_qual, roadWidth);
roadMap.plotGrid2D();
roadMap.plotGrid3D();

save(fullfile(pwd, 'ROADMAP'), 'roadMap')

%% Load Roadmap

varLoad = load(fullfile(pwd, 'ROADMAP'));
roadMap = varLoad.roadMap;



%% Test (executar somente para teste e visualizacao da animacao)

% m=1500;
% Izz=3000;
% lf=1.3;
% lr=1.7;
% cf=25000;
% cr=40000;
% sensors = [-60 -30 0 30 60]/180*pi;
% 
% vehicle = Vehicle(m, Izz, lf, lr, cf, cr, sensors, roadMap.roadGridX, roadMap.roadGridY, roadMap.roadGridZ);
% 
% timeStep = 0.05;
% timeLimit = 20;
% s0=[10,5]';
% v0=[2,0]';
% psi0=0;
% psip0=0;
% beta0=0;
% 
% tic
% netStructure = [6,3];
% genes = rand(1,1000);
% numGens = vehicle.create_Autonomous_driver(netStructure, genes);
% vehicle.sim(timeStep, timeLimit, s0, v0, psi0, psip0, beta0);
% toc
% 
% vehicle.reward.dist
% 
% roadMap.plotGrid2D();
% hold on;
% vehicle.animate();



%% Optimize

use_parallel = true;

if use_parallel && isempty(gcp('nocreate'))
    parpool(8);
end
     

% hidden layer structure (number of neurons in each hidden layer)
netStructure = [3];

aux = [7,netStructure,2];
numberOfVariables = 0;
for i=1:numel(aux)-1
    numberOfVariables = numberOfVariables + (aux(i)+1)*aux(i+1);
end


LB = [];
UB = [];

% https://www.mathworks.com/help/gads/how-the-genetic-algorithm-works.html
% "When Elite count is at least 1, the best fitness value can only decrease from one generation to the next. "

opts = gaoptimset('PlotFcns',{@gaplotbestf,@gaplotscores,@gaplotstopping},...
                  'PopulationSize',30,...
                  'Generations',500,...
                  'StallGenLimit', 20,...
                  'UseParallel',true,...
                  'Vectorized','off',...
                  ...% 'Display','iter',...
                  'SelectionFcn',@selectionroulette,... %{@selectiontournament,4}
                  'Elitecount',2);   
fitnessfcn = @(genes) fun_ga_vehicle(genes, roadMap, netStructure);
[besFitness,Fval,exitFlag,Output, Population] = ga(fitnessfcn, numberOfVariables,[],[],[],[],LB,UB,[],opts);
besFitness

%%

% vehicle design parameters
m=1500;
Izz=3000;
lf=1.3;
lr=1.7;
cf=25000;
cr=40000;
sensors = [-60 -30 0 30 60]/180*pi;

vehicle = Vehicle(m, Izz, lf, lr, cf, cr, sensors, roadMap.roadGridX, roadMap.roadGridY, roadMap.roadGridZ);

% simulation time step and limit
timeStep = 0.05;
timeLimit = 100;

% initial conditions
s0=[10,5]';
v0=[2,0]';
psi0=0;             
psip0=0;
beta0=0;

vehicle.create_Autonomous_driver(netStructure, besFitness);
vehicle.sim(timeStep, timeLimit, s0, v0, psi0, psip0, beta0);

roadMap.plotGrid2D();
hold on;
vehicle.animate();