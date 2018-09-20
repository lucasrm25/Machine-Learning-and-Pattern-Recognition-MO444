classdef ANN_lin < handle

    properties
        type        % 'linear' or 'quadratic'
        m           % number of hidden neurons
        w           % neuron weights
        act_fun     % activation function
        k           % number of outputs   
        n           % number of inputs
        C           % regularization coefficients
        settingsX
        settingsY
        
        
        regularization = 'optimized'
        normalization = false
        algorithm = 'NE'
        options
                
        MSE = @(y, f) norm( y - f )^2 / numel(y);
        RMSE = @(y, f) ( norm( y - f )^2 / numel(y) )^0.5;
        R2  = @(y, f) 1 - (norm( y - f )^2)/(norm( y - mean(y) )^2);
    end
    
    methods
        function obj = ANN_lin(type)
            if strcmp(type, 'linear')
                obj.act_fun = @obj.act_fun_linear;
            else
                obj.act_fun = @obj.act_fun_quadratic;
            end
            obj.type = type;
        end
        
        function act_val = act_fun_linear(obj,X,m)
            act_val = X(:,mod(m-1,size(X,2))+1).^ceil(m/size(X,2));
        end
        
        function act_val = act_fun_quadratic(obj,X, m)
            if m <= sum(1:size(X,2))
                [idx, idy]= find(tril(ones(size(X,2))));
                act_val = X(:,idx(m)).*X(:,idy(m));
            else
                act_val = X(:,m-sum(1:size(X,2)));
            end
        end
        
        function f_t = sim(obj,X, varargin)         
            if ~isempty(varargin)
                argin_norm = varargin{1};
            else
                argin_norm = true;
            end
            if obj.normalization && argin_norm
                aux = mapstd('apply',X',obj.settingsX); %mapminmax
                X = aux';
            end   
            N = size(X,1);
            H = ones(N,obj.m+1);
            %dispstat('','init');
            for im=1:obj.m
                %dispstat(sprintf('Calculating H matrix %.1f%%',im/obj.m*100));
                H(1:N,im+1) = obj.act_fun(X,im);
            end
            f_t = H*obj.w;
            
            if obj.normalization && argin_norm
                aux = mapstd('reverse',f_t',obj.settingsY); %mapminmax
                f_t = aux';
            end
        end
        
        function read_varargin(obj, inputs)
            for i = 1:2:length(inputs)
                obj.(inputs{i}) = inputs{i+1}; 
            end
        end
        
        function info = reduce_features(obj, X, S, numel2remove)
            idx = 1:size(X,2);
            idx_removed  = [];
            MSE_removed = [];
            
            while numel(idx_removed) < numel2remove
                MSE_worst = 0;
                idx_removed(end+1)  = NaN;
                MSE_removed(end+1) = NaN;
                for i=1:numel(idx)
                    info = obj.train( X(:,idx([1:i-1,i+1:end])), S, 'algorithm','NE', 'regularization', 0, 'normalization', true );
                    if info.MSE_tr_va > MSE_worst
                        idx_removed(end) = idx(i);
                        MSE_worst = info.MSE_tr_va;
                    end
                end
                MSE_removed(end) = MSE_worst;
                idx(idx==idx_removed(end)) = [];
            end
            info = struct('MSE_removed',MSE_removed, 'idx',idx, 'idx_removed',idx_removed);
        end
        
        function info = train(obj, X, S, varargin)          
            
            obj.read_varargin(varargin);
            
            if strcmp(obj.type,'linear')
                obj.m = size(X,2);
            else
                obj.m = sum(1:size(X,2)) + size(X,2);
            end
            obj.k = size(S,2);
            obj.n = size(X,2);
            
            if obj.normalization
                [aux, obj.settingsX] = mapstd(X',0,1); %mapminmax
                X = aux';
                [aux, obj.settingsY] = mapstd(S',0,1); %mapminmax
                S = aux';
            end
         
            % Define training and validation data for the regularization coefficient training
            porc_tr_RC = 0.8;
            idx_tr  = randperm(size(X,1));
            X_tr = X(idx_tr(1:floor(porc_tr_RC*numel(idx_tr))),:);
            S_tr = S(idx_tr(1:floor(porc_tr_RC*numel(idx_tr))),:);
            X_va = X(idx_tr(floor(porc_tr_RC*numel(idx_tr))+1:end),:);
            S_va = S(idx_tr(floor(porc_tr_RC*numel(idx_tr))+1:end),:);

            N_tr = size(X_tr,1);        % Number of training data
            N_va = size(X_va,1);        % Number of training data

            % Calculate H matricess
            dispstat('','init');
            H_tr = ones(N_tr,obj.m+1);
            H_va = ones(N_va,obj.m+1);
            for im=1:obj.m
                if (mod(im,10)==0 || im==obj.m), dispstat(sprintf('Calculating H matrix %.1f%%',im/obj.m*100)); end
                H_tr(1:N_tr,im+1) = obj.act_fun(X_tr,im);        % H(iN,im) = h_im(X_iN)
                H_va(1:N_va,im+1) = obj.act_fun(X_va,im);
            end

            if strcmp(obj.regularization,  'optimized')
                fun_wk = @(Ck,k) pinv(H_tr'*H_tr + abs(Ck)*eye(obj.m+1)) * H_tr' * S_tr(:,k);
                MSE_va = @(Ck,k) norm(H_va*fun_wk(Ck,k)-S_va(:,k))^2/numel(S_va(:,k));

                dispstat(sprintf('Calculating Regularization Coefficients %.1f%%...',0),'keepprev');
                obj.C = rand(obj.k,1);
                optsoptimset = optimset('Display','off', 'TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter',500,'PlotFcns',{@optimplotfval,@optimplotx });
                for ik=1:obj.k
%                     obj.C(ik) = fminsearch( @(Ck) MSE_va(Ck,ik), obj.C(ik),optsoptimset);
                    obj.C(ik) = lsqnonlin( @(Ck) H_va*fun_wk(Ck,ik)-S_va(:,ik), obj.C(ik),[],[],optsoptimset);
    %                 obj.C(ik) = fminunc( @(Ck)MSE_va(Ck,ik), obj.C(ik),optsoptimset);
                    dispstat(sprintf('Calculating Regularization Coefficients %.1f%%...',ik/obj.k*100));
                end
                obj.C = abs(obj.C);
            else % no regularization
                obj.C = obj.regularization * eye(obj.k,1);
            end
                        
            H  = [H_tr ; H_va];
            Sh = [S_tr; S_va];

            if strcmp(obj.algorithm, 'NE')
                fun_wk = @(Ck,k) pinv(H'*H + abs(Ck)*eye(obj.m+1)) * H' * Sh(:,k);
                dispstat('','init');
                obj.w = zeros(obj.m+1,obj.k);
                for ik=1:obj.k
                    dispstat(sprintf('Calculating ANN weight vector %.1f%%\n',ik/obj.k*100));
                    obj.w(:,ik) = fun_wk(obj.C(ik),ik);
                end
                f_X_tr_va = obj.sim(X, false);
                if obj.normalization
                    MSE_tr_va = obj.MSE(mapstd('reverse',f_X_tr_va',obj.settingsY)' , mapstd('reverse',S',obj.settingsY)');
                else
                    MSE_tr_va = obj.MSE(f_X_tr_va, S);
                end
                info = struct('iter',inf, 'MSE_tr_va',MSE_tr_va);
            else    % Gradient Descent
                obj.w = rand(obj.m+1,obj.k) - 0.5;
                for ik=1:obj.k
                    dispstat('','init');
                    gradient = ones(obj.options.maxsteps, obj.m+1);
                    for iter=1:obj.options.maxsteps
                        f_X_tr = obj.sim(X_tr, false);
                        f_X_va = obj.sim(X_va, false);
                        for im=1:obj.m+1
                            gradient(iter,im) = (1/N_tr)* ( sum((f_X_tr-S_tr).*H_tr(:,im)) + obj.C*obj.w(im,ik) );
                            if strcmp(obj.options.algorithm,'ADAGRAD')
                                obj.w(im,ik) = obj.w(im,ik) - (obj.options.alpha/norm(gradient(:,im)))*gradient(iter,im);
                            else
                                obj.w(im,ik) = obj.w(im,ik) - obj.options.alpha*gradient(iter,im);
                            end
                        end
                        if iter >= 2 && norm(gradient(iter,:)) > norm(gradient(iter-1,:))
                            obj.options.alpha = obj.options.alpha / 10;
                        end
%                         if sum(abs(gradient)) <= 1e-6, break; end     
                        if obj.normalization
                            MSE_tr(iter) = obj.MSE(mapstd('reverse',f_X_tr',obj.settingsY)' , mapstd('reverse',S_tr',obj.settingsY)');
                            MSE_va(iter) = obj.MSE(mapstd('reverse',f_X_va',obj.settingsY)' , mapstd('reverse',S_va',obj.settingsY)');
                        else
                            MSE_tr(iter) = obj.MSE(f_X_tr , S_tr);
                            MSE_va(iter) = obj.MSE(f_X_va , S_va);
                        end
                        if mod(iter,10)==0, dispstat(sprintf('Calculating ANN weight vector %.1f%%\n',iter/obj.options.maxsteps*100)); end
                    end
                end
                info = struct('iter',1:obj.options.maxsteps, 'MSE_tr',MSE_tr, 'MSE_va',MSE_va);
            end
        end
        
    end
    
end
