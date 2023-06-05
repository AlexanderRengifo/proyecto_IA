clear; clc; close all;
% Leer el archivo de datos
data = readmatrix('Encuesta.xlsx');

% Excluir la primera fila (etiquetas)
data = data(2:end, :);

% Obtener las variables independientes (X) y las variables dependientes (y1, y2)
X = data(:, 1:end-2);
y1 = data(:, end-1);
y2 = data(:, end);

% Partición del conjunto de datos
total_samples = size(X, 1);
train_samples = round(91);
train_idx = 1:train_samples;
test_idx = train_samples+1:total_samples;

% Conjunto de entrenamiento
X_train = X(train_idx, :);
X_test = X(test_idx, :);
y1_train = y1(train_idx);
y1_test = y1(test_idx);
y2_train = y2(train_idx);
y2_test = y2(test_idx);

% Cálculo de probabilidades

% Probabilidad marginal de que una persona vote por Correa
PC_SI = sum(y2_train) / length(y2_train);
%Probabilidad de que una persona izquierdista vote por Correa
PIC = sum(y2_train(y1_train == 1)) / sum(y1_train == 1);
%Probabilidad de que una persona derechista vote por Correa
PDC = sum(y2_train(y1_train == 0)) / sum(y1_train == 0);
%Probabilidad marginal de que una persona sea de izquierda
PI = sum(y1_train) / length(y1_train);
%Probabilidad marginal de que una encuesta escogida al azar pertenezca a un hombre
PH = sum(X_train(:, 5) == 0) / length(X_train);
%Probabilidad de que una mujer sea correísta:
PMC = sum(y2_train(X_train(:, 5) == 1)) / sum(X_train(:, 5) == 1);
%Probabilidad de que un hombre sea correísta:
PHC = sum(y2_train(X_train(:, 5) == 0)) / sum(X_train(:, 5) == 0);

disp(['Probabilidad marginal de que una persona vote por Correa: ', num2str(PC_SI)]);
disp(['Probabilidad de que una persona izquierdista vote por Correa: ', num2str(PIC)]);
disp(['Probabilidad de que una persona derechista vote por Correa: ', num2str(PDC)]);
disp(['Probabilidad marginal de que una persona sea de izquierda: ',num2str(PI)]);
disp(['Probabilidad marginal de que una encuesta escogida al azar pertenezca a un hombre: ', num2str(PH)]);
disp(['Probabilidad de que una mujer sea correísta: ', num2str(PMC)]);
disp(['Probabilidad de que un hombre sea correísta: ', num2str(PHC)]);

%Naive Bayes.

x_new = X_test(1, :);
disp(['No es joven: ', num2str(x_new(1))]);
disp(['Sí es pragmática: ', num2str(x_new(2))]);
disp(['Sí es extrovertida: ', num2str(x_new(3))]);
disp(['Sí es sentimental: ', num2str(x_new(4))]);
disp(['Sí es mujer: ', num2str(x_new(5))]);
disp(['No es planificada: ', num2str(x_new(6))]);

% Probabilidades condicionales de cada característica dada la clase C=1

% Calcular las probabilidades condicionales de cada característica dada la clase C=1
p_x_given_C1 = zeros(1, size(X_train, 2));
for i = 1:size(X_train, 2)
    p_x_given_C1(i) = sum(X_train(:, i) == 1 & y2_train == 1) / sum(y2_train == 1);
end

% Calcular la likelihood P(x_new|C=1)
likelihood_C1 = prod(x_new .* p_x_given_C1 + (1 - x_new) .* (1 - p_x_given_C1));

disp(['Probabilidad P(xnew|C=1): ', num2str(likelihood_C1)]);

% Calcular las probabilidades condicionales de cada característica dada la clase C=0
p_x_given_C0 = zeros(1, size(X_train, 2));
for i = 1:size(X_train, 2)
    p_x_given_C0(i) = sum(X_train(:, i) == 1 & y2_train == 0) / sum(y2_train == 0);
end

% Calcular la likelihood P(x_new|C=0)
likelihood_C0 = prod(x_new .* p_x_given_C0 + (1 - x_new) .* (1 - p_x_given_C0));

disp(['Probabilidad P(xnew|C=0): ', num2str(likelihood_C0)]);
% Calcular las probabilidades condicionales de cada característica dada la clase C=1
p_x_given_C1 = zeros(1, size(X_train, 2));
for i = 1:size(X_train, 2)
    p_x_given_C1(i) = sum(X_train(:, i) == 1 & y2_train == 1) / sum(y2_train == 1);
end

% Calcular las probabilidades condicionales de cada característica dada la clase C=0
p_x_given_C0 = zeros(1, size(X_train, 2));
for i = 1:size(X_train, 2)
    p_x_given_C0(i) = sum(X_train(:, i) == 1 & y2_train == 0) / sum(y2_train == 0);
end

% Calcular la likelihood P(x_new|C=1)
likelihood_C1 = prod(x_new .* p_x_given_C1 + (1 - x_new) .* (1 - p_x_given_C1));

% Calcular la likelihood P(x_new|C=0)
likelihood_C0 = prod(x_new .* p_x_given_C0 + (1 - x_new) .* (1 - p_x_given_C0));

% Calcular la evidence P(x_new)
evidence = likelihood_C1 * PC_SI + likelihood_C0 * (1 - PC_SI);

disp(['Evidence P(xnew): ', num2str(evidence)]);

% Calcular la posterior probability P(C=1|x_new) usando la fórmula de Bayes
posterior_C1 = likelihood_C1 * PC_SI / evidence;

disp(['Posterior probability P(C=1|xnew): ', num2str(posterior_C1)]);

% Comprobar si la primera persona del conjunto X_test es correísta
es_correista = y2_test(1);

if es_correista == 1
    disp('La primera persona del conjunto X_test es correísta.');
else
    disp('La primera persona del conjunto X_test no es correísta.');
end

% Inicializar vector de predicciones
y_hat = zeros(size(X_test, 1), 1);

% Probabilidad marginal de que una persona no vote por Correa (C=0)
PC_NO = 1 - PC_SI;


% Realizar predicciones para cada persona en X_test
for i = 1:size(X_test, 1)
    x_new = X_test(i, :);
    
    % Calcular la likelihood P(x_new|C=1)
    likelihood_C1 = prod(x_new .* p_x_given_C1 + (1 - x_new) .* (1 - p_x_given_C1));
    
    % Calcular la likelihood P(x_new|C=0)
    likelihood_C0 = prod(x_new .* p_x_given_C0 + (1 - x_new) .* (1 - p_x_given_C0));
    
    % Calcular la evidence P(x_new)
    evidence = likelihood_C1 * PC_SI + likelihood_C0 * PC_NO;
    
    % Calcular la posterior probability P(C=1|x_new) usando la fórmula de Bayes
    posterior_C1 = (likelihood_C1 * PC_SI) / evidence;
    
    % Asignar la predicción correspondiente
    if posterior_C1 >= 0.5
        y_hat(i) = 1;
    else
        y_hat(i) = 0;
    end
end

% Imprimir la tabla
fprintf('Persona\t| Correísta Real\t| Predicción Naive Bayes\n');
fprintf('------------------------------------------------------\n');
for i = 1:size(X_test, 1)
    fprintf('%d\t| %d\t\t| %d\n', i, y2_test(i), y_hat(i));
end

