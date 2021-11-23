clear all, clc
%Christian Rafael Mora Parga, Proyecto Final, Proc. Señales Biomédicas

% Clasificación y predicción de "estados mentales" a partir de señales
% cerebrales electroencefalográficas (EEG) a 4 sujetos de prueba. Se 
% capturaron estas señales a 5 canales, para los estados mentales de
% "concentración", "neutral" y "relajado". 
% Ejecutar
%% Truncar los archivos a 60 segundos 200*60 muestras:
path = 'eeg-feature-generation-master/data/';
Fs = 200;

S = dir(fullfile(path,'subject*'));

coun = 1;
for i = 1:size(S,1)
    T = readtable(strcat(path,S(i).name));
    sis = size(T,1);
    if sis > Fs*60
        sizes(coun) = sis;
        T = cell2mat(table2cell(T));
        T = T(1:Fs*60,:);
        name = S(i).name; name = name(1:end-4);
        save(name,'T');
        coun = coun + 1;
    end

end
save('size_events','sizes')

duracion_eventos = sizes/200
% se requieren al menos bloques de 20 segundos, a Fs = 200, se requieren
Fs*20 %muestras
%% Organizar las etiquetas/clases Y de cada archivo
current_path = pwd;
S = dir(fullfile(current_path,'subject*'));

for i = 1:size(S,1)
    names(i) = {S(i).name};
end
Y = cell(length(names),1);
find(contains(names,'neutral'))
find(contains(names,'concentrating'))
find(contains(names,'relaxed'))
Y([find(contains(names,'neutral'))]) = {'neutral'};
Y([find(contains(names,'concentrating'))]) = {'concentrating'};
Y([find(contains(names,'relaxed'))]) = {'relaxed'};

save('Y','Y')
%% Crear eventos de 5 segundos (1k muestras) con sus etiquetas
% o de menos tiempo si se desea
clear all, clc
current_path = pwd;
load('Y')
y = Y;

Fs = 200;

w_size = Fs*5; %muestras
X = []; Y = [];
S = dir(fullfile(current_path,'subject*'));
for i = 1:size(S,1) %se recorren todos los archivos .mat
    %se carga la matriz de dicho caso:
    name = cell2mat({S(i).name}); name = name(1:end-4);
    load(name)
    blocks = size(T,1)/w_size;
    for window = 1:blocks
        Sig = T(w_size*(window-1)+1:w_size*window,:);
        %se remueven la primera y ultima columna (vec de tiempo, y ruido)
        Sig(:,1) = []; Sig(:,5) = [];
        %Se filtran todos los canales de la señal respectivamente
        %filtrado de la señal con un fir pasabajos con frecuencia de corte
        %en 90Hz
        filter = designfilt('lowpassfir', 'FilterOrder', 70, ...
            'CutoffFrequency', 70, 'SampleRate', Fs, 'Window', 'hamming');
        
        filt_Sig = filtfilt(filter, Sig);
        
        %cada uno de los 4 canales cuenta como un evento distinto (4 filas)
        %cálculo de características: 
        mean_feat = mean(filt_Sig)';
        std_feat = std(filt_Sig)';
        kurt_feat = kurtosis(filt_Sig)';
        skew_feat = skewness(filt_Sig)';
        [min_feat,i_min] = min(filt_Sig); min_feat = min_feat';
        [max_feat,i_max] = max(filt_Sig); max_feat = max_feat';
        slope_feat = abs((max_feat - min_feat)./(i_max' - i_min'));
        
        %entropía de shannon y logarítmica
        e1 = wentropy(filt_Sig(:,1),'shannon')';
        e2 = wentropy(filt_Sig(:,2),'shannon')';
        e3 = wentropy(filt_Sig(:,3),'shannon')';
        e4 = wentropy(filt_Sig(:,4),'shannon')';
        shan_e = horzcat(e1,e2,e3,e4); shan_e = shan_e';
        
        le1 = wentropy(filt_Sig(:,1),'log energy');
        le2 = wentropy(filt_Sig(:,2),'log energy');
        le3 = wentropy(filt_Sig(:,3),'log energy');
        le4 = wentropy(filt_Sig(:,4),'log energy');
        log_e = horzcat(le1,le2,le3,le4); log_e = log_e';
        
        % PSD de todas las observaciones (los 4 canales)
        N = 2^8; win = blackman(N); noverlap = 50;
        [pxx,f] = pwelch(filt_Sig,win,noverlap,N,Fs);
        pxx = pxx';
        
        %Bandas:
        % 1) delta: 1-4Hz
        % 2) theta: 4-10Hz
        % 3) alpha: 8-12Hz
        % 4) beta: 12-30Hz
        % 5) gamma:30-80Hz
        %Límites superiores e inferiores de cada banda:
        inf_f = [1,4,8,12,30]; sup_f = [4,10,12,30,80];
        for ii = 1:5 %Localización de los límites en las señales:
            [~,infloc(ii)] = min(abs(f - inf_f(ii)));
            [~,suploc(ii)] = min(abs(f - sup_f(ii)));
        end
        %Cálculo de la energía en las 5 bandas para cada observación
        for ii = 1:5 %columnas i=1:delta, 2:theta, 3:alpha, 4:beta, 5:gamma
            E_sig(:,ii) = sum(pxx(:,infloc(ii):suploc(ii)),2)/(2*pi);
        end

        window_feats = horzcat(std_feat,kurt_feat,...
            E_sig,slope_feat); 
        
        window_y = cell(1, size(Sig,2)); 
        window_y(:) = {y{i}}; window_y = window_y';
        
        X = [X;window_feats];
        Y = [Y;window_y];
    end
end
save('feats','X')
save('targets','Y')
%% PCA 

load('feats'); load('targets')
name_feats = {'std';'kurtosis';...
    'E delta';'E theta';'E alpha';'E beta';'E gamma';'Slope'};
varX = var(X);
[varSort_X,Ixvar] = sort(varX);
subplot(2,1,1), semilogy(flip(varSort_X)), grid on
xlabel('Características'); ylabel('Var Observaciones')
title('Varianza ordenada de X')
subplot(2,1,2)

% Ver los 3 eventos con menor varianza
% h = scatter3(X(:,Ixvar(end-3))',X(:,Ixvar(end-2))',X(:,Ixvar(end-1))',...
%    'filled', 'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
%Ver los 3 eventos con mayor varianza
h = scatter3(X(:,Ixvar(1))',X(:,Ixvar(2))',X(:,Ixvar(3))',...
    'filled', 'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
set(gca,'xscale','log');set(gca,'yscale','log');%set(gca,'zscale','log')
h.SizeData = 20;
title('Varianza características')
% xlabel(name_feats(Ixvar(end-3))); ylabel(name_feats(Ixvar(end-2)))
% zlabel(name_feats(Ixvar(end-1)))
xlabel(name_feats(Ixvar(1))); ylabel(name_feats(Ixvar(2)))
zlabel(name_feats(Ixvar(3)))
%Orden de características segun su varianza:
name_feats(Ixvar)

% Recorte de eventos
del_cols = [71,343,462,522,547,551,555,559,563,571,575,755,797];
for i = 1:length(del_cols)
    X(del_cols(i),:) = [];
    Y(del_cols(i),:) = [];
end
for feats = 1:size(X,2)
    prom = mean(X(:,feats));
    % se eliminan los eventos por encima del umbral promedio 
    idx = find(X(:,feats) > prom + prom/2);
    X(idx,:) = []; Y(idx) = [];
end

% Después del recorte de eventos
figure
varX = var(X);
[varSort_X,Ixvar] = sort(varX);
subplot(2,1,1), semilogy(flip(varSort_X)), grid on
xlabel('Características'); ylabel('Var Observaciones')
title('Varianza ordenada de X')
subplot(2,1,2)

%Ver los 3 eventos con menor varianza
% h = scatter3(X(:,Ixvar(end-3))',X(:,Ixvar(end-2))',X(:,Ixvar(end-1))',...
%    'filled', 'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
%Ver los 3 eventos con mayor varianza
h = scatter3(X(:,Ixvar(1))',X(:,Ixvar(2))',X(:,Ixvar(3))',...
    'filled', 'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
set(gca,'xscale','log');set(gca,'yscale','log');%set(gca,'zscale','log')
h.SizeData = 20;
title('Varianza características')
% xlabel(name_feats(Ixvar(end-3))); ylabel(name_feats(Ixvar(end-2)))
% zlabel(name_feats(Ixvar(end-1)))
xlabel(name_feats(Ixvar(1))); ylabel(name_feats(Ixvar(2)))
zlabel(name_feats(Ixvar(3)))
name_feats(Ixvar)

%% clasificación ANN
%clear all, clc
%load('feats'); load('targets')

numFolds = 3; % NÚMERO DE FOLDS
rng(2) %semilla
c = cvpartition(Y,'k',numFolds);
% table to store the results
netAry_ANN = {numFolds,1};
perfAry_ANN = zeros(numFolds,1);

for i = 1:numFolds
    
    %Se particiona los datos a Entrenamiento y Prueba para este Fold
    trIdx = c.training(i);teIdx = c.test(i);
    xTrain = X(trIdx);yTrain = Y(trIdx);
    xTest = X(teIdx);yTest = Y(teIdx);
    
    xTrain = xTrain'; xTest = xTest';
    yTrain_dumm = dummyvar(grp2idx(yTrain))';
    yTest_dumm = dummyvar(grp2idx(yTest))';
    
    %%ANN
    %Crear la red y poner en 0 los datos de Test y valida en los input data
    net = patternnet([32 16 8]);
    net.divideParam.trainRatio = 1;net.divideParam.testRatio = 0;
    net.divideParam.valRatio = 0;
    %Entrenar la red
    net = train(net,xTrain,yTrain_dumm);
    yPred_ANN = net(xTest);
    perf_ANN = perform(net,yTest_dumm,yPred_ANN);
    
    %Guardar las redes y sus desempeños
    netAry_ANN{i} = net; %red i-ésima
    perfAry_ANN(i) = perf_ANN; %desempeño
    
end
%Usar la red con menor error:
perfAry_ANN
[maxPerf,maxPerfId] = min(perfAry_ANN);
bestNet = netAry_ANN{maxPerfId};

scores_neural = sim(bestNet,xTest);
[~,yNum] = max(scores_neural);
yPred_ANN = categorical(yNum,1:length(unique(Y)),unique(Y))';
figure
plotconfusion(categorical(yTest),yPred_ANN)
title('Matriz confusión ANN')

%% visualizar eventos originales
figure
for chn = 1:size(filt_Sig,2)
    subplot(size(filt_Sig,2),1,chn), plot(filt_Sig(:,chn)), grid on
    ylabel('mV','FontSize',16)
end
xlabel('muestras','FontSize',16)
figure
for chn = 1:size(Sig,2)
    subplot(size(Sig,2),1,chn), plot(Sig(:,chn)), grid on
    ylabel('mV','FontSize',16)
end
xlabel('muestras','FontSize',16)
%% visualizar las características con todos sus eventos
figure

for feat = 1:length(name_feats)-2
    subplot(length(name_feats)-1,1,feat), plot(X(:,feat)), grid on
    ylabel(name_feats{feat},'FontSize',16)
end
subplot(616),plot(X(:,length(name_feats))), grid on
ylabel(name_feats{length(name_feats)},'FontSize',16)
xlabel('Eventos','FontSize',16)
