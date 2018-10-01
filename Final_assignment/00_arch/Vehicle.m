classdef Vehicle < handle
    % m=1500;
    % J_ZZ=3000;
    % l_v=1.3;
    % l_h=1.7;
    % l=l_h+l_v;
    % c_v=25000;
    % c_h=40000;
    % i_Ges=-0.5;
    
    properties
        m = 1500
        Izz = 3000
        cf = 25000
        cr = 40000
        lf = 1.3
        lr = 1.7
        tyre_r = 16*2.54
        sensors = [-60 -30 0 30 60]/180*pi
        
        tyre_trq_Acc_Max = 100;
        tyre_trq_Brk_Max = 150;
        tyre_ang_Max = 40/180*pi
        
%         roadGridX
%         roadGridY
%         roadGridZ        
        roadMap
        
        ANN_autonomous_driver
        
        time
        space
        
        reward = struct('dist',0, 'time',inf)
    end
    properties (Constant)
        T_v = @(ang) [cos(ang), -sin(ang), 0;
                      sin(ang),  cos(ang), 0;
                      0,         0,        1];
    end
    
    methods
        function self = Vehicle(m, Izz, lf, lr, cf, cr, sensors, roadMap)
            self.m = m;
            self.Izz = Izz;
            self.cf = cf;
            self.cr = cr;
            self.lf = lf;
            self.lr = lr;
            self.sensors = sensors;
            self.roadMap = roadMap;
        end
        
        function dist = get_sensor_dist(self, psi, s)
            ds = 1;  % sensor precision
            dist = zeros(1,numel(self.sensors));
            for i=1:numel(self.sensors)
                mapdist = 0;
                s_wall = s;
                direction = psi(3) + self.sensors(i);
                while ~isnan(mapdist)
                    s_wall = s_wall + ds * Vehicle.T_v(direction) * [1;0;0];
                    mapdist = interp2(self.roadMap.roadGridX, self.roadMap.roadGridY, self.roadMap.roadGridZ, s_wall(1), s_wall(2));
                end
                dist(i) = norm(s_wall - s);
            end
            dist = dist';
            dist = zeros(1,numel(self.sensors));
            for i=1:numel(self.sensors)
                direction = psi(3) + self.sensors(i);
                dist(i) = interp3( self.roadMap.roadGridX_sensor, self.roadMap.roadGridY_sensor, self.roadMap.roadGridPSI_sensor, self.roadMap.roadGridDist_sensor,...
                                   s(1), s(2), direction); 
            
            end
            dist = dist';
        end
        
        function numGens = create_Autonomous_driver(self, net_structure, genes)
            numInputs = numel(self.sensors) + 2; % sensors, linear vel. and angular vel.
            numOutputs = 2;

            net = MLP_ga([numInputs, net_structure, numOutputs], @tansig);
            net.setwb(genes);
            
            numGens = net.numWeightElements;
            self.ANN_autonomous_driver = net;
        end
        
        function sim(self, timeStep, timeLimit, s0, v0, psi0, psip0, beta0)
            
            % STATES
            a = nan(3,timeLimit/timeStep+1);
            v = nan(3,timeLimit/timeStep+1);
            s = nan(3,timeLimit/timeStep+1);
            psipp = nan(3,timeLimit/timeStep+1);
            psip = nan(3,timeLimit/timeStep+1);
            psi = nan(3,timeLimit/timeStep+1);          
            % beta = nan(3,timeLimit/timeStep+1);
            v(:,1) = [v0; 0];
            s(:,1) = [s0; 0];
            psi(:,1) = [0;0;psi0];
            psip(:,1) = [0;0;psip0];
            % beta(:,1) = [0;0;beta0];
            
            % ACTIONS
            tyre_trq = nan(1,timeLimit/timeStep+1);
            st_wh_ang = nan(1,timeLimit/timeStep+1);
            tyre_trq(1) = 0;
            st_wh_ang(1) = 0;
            
            % Begin calculations                      
            V_s_wf = [self.lf; 0; 0];
            V_s_wr = [-self.lr; 0; 0];
            
            i=1;
            for t=0:timeStep:timeLimit
                % Calculate actions based on states
                dist = self.get_sensor_dist(psi(:,i), s(:,i));
                states = [dist; norm(v(:,i)); norm(psip(:,i))]';
                actions = self.ANN_autonomous_driver.sim(states);
%                 if t==0
%                     states;
%                     actions;
%                 end
                
                if actions(1) >= 0
                    tyre_trq(i)  = self.tyre_trq_Acc_Max * actions(1);
                else
                    tyre_trq(i)  = self.tyre_trq_Brk_Max * actions(1);
                end
                st_wh_ang(i) = self.tyre_ang_Max * actions(2);

                % front/rear wheel velocity - Inertial CS
                I_v_wf(:,i) = v(:,i) + cross( psip(:,i), Vehicle.T_v(psi(3,i))*V_s_wf );
                I_v_wr(:,i) = v(:,i) + cross( psip(:,i), Vehicle.T_v(psi(3,i))*V_s_wr );
                
                % front/rear wheel velocity - wheel CS
                wf_v_wf(:,i) = Vehicle.T_v(psi(3,i) + st_wh_ang(i))' * I_v_wf(:,i);
                wr_v_wr(:,i) = Vehicle.T_v(psi(3,i)               )' * I_v_wr(:,i);
                
                % coornering stiffness
                if wf_v_wf(1,i)==0, alpha_f = 0;
                else alpha_f = - atan( wf_v_wf(2,i) / wf_v_wf(1,i) ); end                                
                if wr_v_wr(1,i)==0, alpha_r = 0;
                else alpha_r = - atan( wr_v_wr(2,i) / wr_v_wr(1,i) ); end
                
                % front/rear wheel force - wheel CS
                wf_F_wf = [0;                 alpha_f * self.cf;  0];
                wr_F_wr = [tyre_trq(i)/self.tyre_r; alpha_r * self.cr;  0];
                
                % F = m.a - Inertial CS
                a(:,i) = (2 * Vehicle.T_v(psi(3,i) + st_wh_ang(i)) * wf_F_wf + ...
                          2 * Vehicle.T_v(psi(3,i)               ) * wr_F_wr ) / self.m;
                v(:,i+1)= v(:,i) + a(:,i) * timeStep;
                s(:,i+1)= s(:,i) + v(:,i) * timeStep;
                
                % M = I.wp - Vehicle CS
                psipp(:,i) = (cross(V_s_wf , 2 * Vehicle.T_v(st_wh_ang(i)) * wf_F_wf) + ...
                              cross(V_s_wr , 2                                * wr_F_wr)) / self.Izz;  
                psip(:,i+1)= psip(:,i) + psipp(:,i) * timeStep;
                psi(:,i+1) = psi(:,i)  + psip(:,i)  * timeStep;

                % Calculate rewards
                dist = interp2(self.roadMap.roadGridX, self.roadMap.roadGridY, self.roadMap.roadGridZ, s(1,i), s(2,i));
                if ~isnan(dist)
                    self.reward.dist = dist;
                    self.reward.time = t;
                else break;
                end
                
                i = i+1;
            end
            self.time  = 0:timeStep:t;
            self.space = s(:,1:numel(self.time));
        end

        function animate(self)
            sct = scatter(self.space(1,1), self.space(2,1), 80, 'MarkerEdgeColor', 'k' , 'MarkerFaceColor', 'r', 'Marker','s');
            dt = 1/40;
            for t=min(self.time):dt:max(self.time)
                [~, idx] = min(abs(self.time-t));
                sct.XData = self.space(1,idx);
                sct.YData = self.space(2,idx);
                pause(dt/2);
            end
        end
    end
    
end

