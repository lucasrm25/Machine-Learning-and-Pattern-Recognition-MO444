clearvars; close all; clc

cd 'C:\Users\Renata\Desktop\MO444\First_assignment'
addpath(fullfile(pwd,'data'));


MSE = @(f, y) norm( y - f )^2 / numel(y);
RMSE = @(f, y) ( norm( y - f )^2 / numel(y) )^0.5;
R2  = @(f, y) 1 - (norm( y - f )^2)/(norm( y - mean(y) )^2);

%% Read Data

aux =  csvread('train.csv',1,2);
Xtr = aux(:,1:end-1);
Ytr = aux(:,end);

Xtest = csvread('test.csv',1,2);
% Xtest = Xtest';
Ytest = csvread('test_target.csv',1,0);
% Ytest = Ytest';


%% Part 1 - Linear Neural Network
clc
linear = ANN_lin('linear');

linear.train(Xtr,Ytr, 'algorithm','NE', 'regularization', 0, 'normalization', false)
f_Xtr = linear.sim(Xtr);
f_Xtest = linear.sim(Xtest);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

linear_quad = ANN_lin('quadratic');

linear_quad.train(Xtr,Ytr, 'algorithm','NE', 'regularization', 0, 'normalization', false)
f_Xtr = linear_quad.sim(Xtr);
f_Xtest = linear_quad.sim(Xtest);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

linear_quad.train(Xtr,Ytr, 'algorithm','NE', 'regularization','optimized', 'normalization',false);
f_Xtr = linear_quad.sim(Xtr);
f_Xtest = linear_quad.sim(Xtest);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

%% FIG - Comparison NE, GD
clc

linear = ANN_lin('quadratic');

ne_info = linear.train(Xtr,Ytr, 'algorithm','NE', 'regularization', 0, 'normalization', false);
f_Xtr = linear.sim(Xtr);
f_Xtest = linear.sim(Xtest);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

gd_opts = struct('maxsteps', 300, 'alpha', 1e-25, 'algorithm', 'none');
gd_info = linear.train(Xtr,Ytr, 'algorithm','GD', 'regularization',0, 'normalization',false, 'options',gd_opts);
f_Xtr = linear.sim(Xtr);
f_Xtest = linear.sim(Xtest);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

figure('Color','white')
plot(gd_info.iter, gd_info.MSE_tr, 'k-', 'LineWidth',1); hold on;
plot(gd_info.iter, gd_info.MSE_va, 'k--', 'LineWidth',1);
grid on
xlabel('Epochs')
ylabel('Mean Square Error')
xlim([min(gd_info.iter) max(gd_info.iter)])
legend({'Training data','Validation data'})


%% GD NORMALIZED
clc

linear = ANN_lin('quadratic');

ne_info = linear.train(Xtr,Ytr, 'algorithm','NE', 'regularization', 0, 'normalization', true);
f_Xtr = linear.sim(Xtr);
f_Xtest = linear.sim(Xtest);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

gd_opts = struct('maxsteps', 100, 'alpha', 1e-5, 'algorithm', 'none');
gd_info = linear.train(Xtr,Ytr, 'algorithm','GD', 'regularization',0, 'normalization',true, 'options',gd_opts);
f_Xtr = linear.sim(Xtr);
f_Xtest = linear.sim(Xtest);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

figure('Color','white')
plot(gd_info.iter, gd_info.MSE_tr, 'k-', 'LineWidth',1); hold on;
plot(gd_info.iter, gd_info.MSE_va, 'k--', 'LineWidth',1);
% plot(gd_info_adagrad.iter, gd_info_adagrad.MSE_tr, '-', 'LineWidth',2);
% plot(gd_info_adagrad.iter, gd_info_adagrad.MSE_va, '-', 'LineWidth',2);
% plot([-100 1000], [ne_info.MSE_tr_va ne_info.MSE_tr_va], '--k', 'LineWidth',2);
grid on
xlabel('Epochs')
ylabel('Mean Square Error')
xlim([min(gd_info.iter) max(gd_info.iter)])
% ylim([0 inf])
legend({'Training data','Validation data'})
% legend({'Const-GD Training data','Const-GD Validation data','ADAGRAD-GD Training data','ADAGRAD-GD Validation data','NE Training + Validation data'})
% title('Root Mean Square Error')

%% ADAGRAD
clc

linear = ANN_lin('quadratic');

ne_info = linear.train(Xtr,Ytr, 'algorithm','NE', 'regularization', 0, 'normalization', true);
f_Xtr = linear.sim(Xtr);
f_Xtest = linear.sim(Xtest);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

gd_opts = struct('maxsteps', 100, 'alpha', 1, 'algorithm', 'ADAGRAD');
gd_info = linear.train(Xtr,Ytr, 'algorithm','GD', 'regularization',0, 'normalization',true, 'options',gd_opts);
f_Xtr = linear.sim(Xtr);
f_Xtest = linear.sim(Xtest);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

figure('Color','white')
plot(gd_info.iter, gd_info.MSE_tr, 'k-', 'LineWidth',1); hold on;
plot(gd_info.iter, gd_info.MSE_va, 'k--', 'LineWidth',1);
% plot(gd_info_adagrad.iter, gd_info_adagrad.MSE_tr, '-', 'LineWidth',2);
% plot(gd_info_adagrad.iter, gd_info_adagrad.MSE_va, '-', 'LineWidth',2);
% plot([-100 1000], [ne_info.MSE_tr_va ne_info.MSE_tr_va], '--k', 'LineWidth',2);
grid on
xlabel('Epochs')
ylabel('Mean Square Error')
xlim([min(gd_info.iter) max(gd_info.iter)])
% ylim([0 inf])
legend({'Training data','Validation data'})
% legend({'Const-GD Training data','Const-GD Validation data','ADAGRAD-GD Training data','ADAGRAD-GD Validation data','NE Training + Validation data'})
% title('Root Mean Square Error')


%% Wraper
clc
numel2remove = 15;
fontsize = 10;

linear = ANN_lin('linear');
info = linear.reduce_features(Xtr, Ytr, 15);

fig = figure('Color','white','Position',[390.6000  189.0000  557.6000  267.2000]);
hold on
plot(1:numel2remove, info.MSE_removed, 'k-o',  'LineWidth',1.5,...
                                                'MarkerSize',5,...
                                                'MarkerEdgeColor','k',...
                                                'MarkerFaceColor','k')
grid on
xlim([0 numel2remove+1])
ylabel('MSE')

axs = gca;
for i=1:numel2remove
    xpos = i - (axs.XLim(2)-axs.XLim(1))*0.01;
    ypos = info.MSE_removed(i) + (axs.YLim(2)-axs.YLim(1))*0.07;
    text(xpos, ypos, num2str(info.idx_removed(i)+1), 'FontSize',fontsize)
end
axs = gca;
axs.LabelFontSizeMultiplier = 1;
axs.FontSize = fontsize;


%% WRAPER TEST
clc

linear = ANN_lin('quadratic');

Xtr_new = Xtr;
Xtr_new(:,[26 13 25 11 6 43 23 24 3 8 22 27 29 28 20]) = [];
Xtest_new = Xtest;
Xtest_new(:,[26 13 25 11 6 43 23 24 3 8 22 27 29 28 20]) = [];

ne_info = linear.train(Xtr_new,Ytr, 'algorithm','NE', 'regularization', 0, 'normalization', true);
f_Xtr = linear.sim(Xtr_new);
f_Xtest = linear.sim(Xtest_new);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

gd_opts = struct('maxsteps', 300, 'alpha', 1e-3, 'algorithm', 'ADAGRAD');
gd_info = linear.train(Xtr_new,Ytr, 'algorithm','GD', 'regularization',0, 'normalization',true, 'options',gd_opts);
f_Xtr = linear.sim(Xtr_new);
f_Xtest = linear.sim(Xtest_new);
fprintf('Training:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtr, Ytr), MSE(f_Xtr, Ytr), R2(f_Xtr, Ytr));
fprintf('Test:\n RMSE: %.3d, MSE: %.3d, R2: %.2f\n',RMSE(f_Xtest, Ytest), MSE(f_Xtest, Ytest), R2(f_Xtest, Ytest));

figure('Color','white')
plot(gd_info.iter, gd_info.MSE_tr, 'k-', 'LineWidth',1); hold on;
plot(gd_info.iter, gd_info.MSE_va, 'k--', 'LineWidth',1);
% plot(gd_info_adagrad.iter, gd_info_adagrad.MSE_tr, '-', 'LineWidth',2);
% plot(gd_info_adagrad.iter, gd_info_adagrad.MSE_va, '-', 'LineWidth',2);
% plot([-100 1000], [ne_info.MSE_tr_va ne_info.MSE_tr_va], '--k', 'LineWidth',2);
grid on
xlabel('Epochs')
ylabel('Mean Square Error')
xlim([min(gd_info.iter) max(gd_info.iter)])
legend({'Training data','Validation data'})