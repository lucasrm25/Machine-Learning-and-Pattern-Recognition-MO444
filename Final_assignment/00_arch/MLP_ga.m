classdef MLP_ga < handle
    
    properties
        transferfcn
        netStructure
        wb
    end
    
    methods
        function obj = MLP_ga(netStructure, transferfcn)
            obj.netStructure = netStructure;
            obj.transferfcn = transferfcn;
        end
        
        function setwb(obj, genes)
            obj.wb = cell(1,numel(obj.netStructure)-1);
            idx = 0;
            for i=2:numel(obj.netStructure)
                ninp = obj.netStructure(i-1)+1;
                nout = obj.netStructure(i);
                idx = (1:ninp*nout) + max(idx);
                obj.wb{i-1} = reshape( genes(idx) ,ninp,nout  );
            end
        end
        
        function Y = sim(obj, X)
            for i=2:numel(obj.netStructure)
                X(:,end+1) = 1;
                X = X * obj.wb{i-1};
                X = arrayfun( obj.transferfcn , X );
            end
            Y = X;
        end
        
        function sum = numWeightElements(obj)
            sum = 0;
            for i=1:numel(obj.wb)
                sum = sum + numel(obj.wb{i});
            end
        end
    end
    
end

% netStructure = [7,6,3,2];
% genes = rand(1,1000);
% mlp = MLP_ga(netStructure, @tansig)
% mlp.setwb(genes)
% mlp.sim(rand(100,7))