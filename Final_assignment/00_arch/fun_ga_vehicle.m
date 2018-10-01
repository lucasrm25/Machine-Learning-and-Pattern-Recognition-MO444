function fitness = fun_ga_vehicle(genes, roadMap, netStructure)

    % vehicle design parameters
    m=1500;
    Izz=3000;
    lf=1.3;
    lr=1.7;
    cf=25000;
    cr=40000;
    sensors = [-60 -30 0 30 60]/180*pi;

    vehicle = Vehicle(m, Izz, lf, lr, cf, cr, sensors, roadMap);

    % simulation time step and limit
    timeStep = 0.05;
    timeLimit = 100;
    
    % initial conditions
    s0=[10,5]';
    v0=[2,0]';
    psi0=0;             
    psip0=0;
    beta0=0;
    
    vehicle.create_Autonomous_driver(netStructure, genes);
    vehicle.sim(timeStep, timeLimit, s0, v0, psi0, psip0, beta0);
    fitness = -(vehicle.reward.dist^1);

end