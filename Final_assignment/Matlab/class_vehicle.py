import numpy as np

class class_vehicle:
    
#% m=1500;
#% J_ZZ=3000;
#% l_v=1.3;
#% l_h=1.7;
#% l=l_h+l_v;
#% c_v=25000;
#% c_h=40000;
#% v=30;
#% w0_fak=1;
#% Tz_fak=1;
#% D_fak=1.25;
#% i_Ges=-0.5;
    
    def __init__(self, m, Izz, cf, cr, lf, lr, roadGridX, roadGridY, roadGridZ):
        self.m = m
        self.Izz = Izz
        self.cf = cf
        self.cr = cr
        self.lf = lf
        self.lr = lr
        self.roadGridX = roadGridX
        self.roadGridY = roadGridY
        self.roadGridZ = roadGridZ
        
    def sim(self, s0, v0, psi0, psip0, beta0, timeStep=0.1, timeLimit=50):
        for idx, i in enumerate(np.arange(0, timeLimit, timeStep)):
            print(1)
        return 0