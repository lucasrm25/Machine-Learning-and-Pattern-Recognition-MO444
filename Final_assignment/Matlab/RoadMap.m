classdef RoadMap < handle
    
    properties
        roadWidth
        roadPoints_qual
        roadPoints
        
        roadPoints_fine
        roadPoints_dist
        
        roadGridX
        roadGridY
        roadGridZ
        
        
        roadGridX_sensor
        roadGridY_sensor
        roadGridPSI_sensor
        roadGridDist_sensor      
    end
    properties (Constant)
        T_v = @(ang) [cos(ang), -sin(ang);
                      sin(ang),  cos(ang)];
    end
    
    
    methods
        function self = RoadMap(roadPoints, roadPoints_qual, roadWidth)
            self.roadPoints = roadPoints;
            self.roadPoints_qual = roadPoints_qual;
            self.roadWidth = roadWidth;
            
            self.createMap;
            
            self.createDistSensors;
        end
        
        function dist = get_sensor_dist(self, psi, s)
            ds = 1;  % sensor precision
            mapdist = 0;
            s_wall = s;
            while ~isnan(mapdist)
                s_wall = s_wall + ds * RoadMap.T_v(psi) * [1;0];
                mapdist = interp2(self.roadGridX, self.roadGridY, self.roadGridZ, s_wall(1), s_wall(2));
            end
            dist = norm(s_wall - s);
        end
        
        function createDistSensors(self)
            [self.roadGridX_sensor, self.roadGridY_sensor, self.roadGridPSI_sensor] = meshgrid( self.roadGridX(1,:), self.roadGridY(:,1), (0:1:360)*pi/180 );
            self.roadGridDist_sensor = NaN(size(self.roadGridX_sensor));
            
            
            
            dispstat('init')
            for ix = 1:size(self.roadGridX_sensor,2)
                for iy = 1:size(self.roadGridY_sensor,1)
                    if isnan(self.roadGridZ(iy,ix))
                        continue;
                    end
                    for iz = 1:size(self.roadGridPSI_sensor,3)
                        self.roadGridDist_sensor(iy,ix,iz) = self.get_sensor_dist(self.roadGridPSI_sensor(iy,ix,iz), [self.roadGridX_sensor(iy,ix,iz), self.roadGridY_sensor(iy,ix,iz)]');
                    end
                end
                dispstat(sprintf('\n--- %3.1f%%',ix/size(self.roadGridX_sensor,2)*100))
            end
        end
        
        function createMap(self)
            roadPoints_fine(1,:) = self.roadPoints(1,:);
            roadPoints_dist(1,1) = 0;
            for i=1:size(self.roadPoints,1)-1
                vec = self.roadPoints(i+1,:) - self.roadPoints(i,:);
                vecNorm = norm(vec);
                vecUnit = vec/vecNorm;
                for j=self.roadPoints_qual:self.roadPoints_qual:vecNorm
                    roadPoints_fine(end+1,:) = self.roadPoints(i,:) + j*vecUnit;
                    roadPoints_dist(end+1,1) = roadPoints_dist(end,1) + self.roadPoints_qual;
                end
                roadPoints_fine = [roadPoints_fine; self.roadPoints(i+1,:)];
                roadPoints_dist(end+1,1) = roadPoints_dist(end,1) + self.roadPoints_qual;
            end


            s = {0: self.roadPoints_qual*2: max(self.roadPoints(:,1))+self.roadWidth;
                 0: self.roadPoints_qual*2: max(self.roadPoints(:,2))+self.roadWidth};

            [roadGridX, roadGridY] = meshgrid( s{1}, s{2} );
            roadGridZ = NaN(size(roadGridX));

            n_sx = numel(s{1});
            n_sy = numel(s{2});

            % dispstat('Calculating Road Map','keepprev','keepthis')
            hbar = parfor_progressbar(n_sx,'Computing...');

%             if isempty(gcp('nocreate')), parpool(4); end

            for i=1:n_sx
                for j=1:n_sy
                    distMin = inf;
                    for k=1:size(roadPoints_fine,1)
                        dist = norm( [roadGridX(j,i) roadGridY(j,i)] - roadPoints_fine(k,:) );
                        if dist < distMin && dist <= self.roadWidth/2
                            distMin = dist;
                            roadGridZ(j,i) = roadPoints_dist(k);
                        end
                    end
                end
                hbar.iterate(1);
            %     dispstat(sprintf('--- %3.1f%%',i/n_sx*100))
            end
            close(hbar);
            
            self.roadPoints_fine = roadPoints_fine;
            self.roadGridX = roadGridX;
            self.roadGridY = roadGridY;
            self.roadGridZ = roadGridZ;
            self.roadPoints_dist = roadPoints_dist;
        end
        
        function plotGrid3D(self)
            figure('Color','white')
            surf(self.roadGridX, self.roadGridY, self.roadGridZ, self.roadGridZ, 'edgeColor', 'none'); hold on;
            plot(self.roadPoints(:,1),self.roadPoints(:,2))
            scatter(self.roadPoints(:,1),self.roadPoints(:,2), 20, 'filled')
            colorbar
            xlim([min(min(self.roadGridX)) max(max(self.roadGridX))])
            ylim([min(min(self.roadGridY)) max(max(self.roadGridY))])
            grid on
            set(gca,'DataAspectRatio',[1 1 10])
            xlabel('x position [m]')
            ylabel('y position [m]')
            zlabel('Travelled distance [m]')
        end
        
        function plotGrid2D(self)
            figure('Color','white')
            imagesc([min(self.roadGridX) max(self.roadGridX)], [min(self.roadGridY) max(self.roadGridY)], self.roadGridZ, 'AlphaData',0.5*(~isnan(self.roadGridZ)))
            set(gca,'YDir','normal')
            xlabel('x position [m]')
            ylabel('y position [m]')
        end
    end
    
end

