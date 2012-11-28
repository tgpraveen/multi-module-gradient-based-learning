function Q_4_1_and_2()

local k_collection = {1,2,4,8,16,32,64,128,256,512,617}

local whitened_data_train_isolet_collection = {}
local whitened_data_test_isolet_collection = {}

function k_collection:size() return 11 end

local data_train_isolet, data_test_isolet = isolet:getDatasets(600,100)

for i=1, k_collection:size() do
whitened_data_train_isolet_collection[i], whitened_data_test_isolet_collection[i] = whitenDatasets(data_train_isolet, data_test_isolet, k_collection[i])
end

-- For Logistic Regression
local various_test_error_for_different_k_logistic_regression = {}
local logisticRegressionTrainer = {}
local logisticRegressionMLP = {}

for i=1, k_collection:size() do
logisticRegressionTrainer[i], logisticRegressionMLP[i], various_test_error_for_different_k_logistic_regression[i] = logisticRegression(whitened_data_train_isolet_collection[i], whitened_data_train_isolet_collection[i])
end

-- For 2 Layer Neural Network
local various_test_error_for_different_k_2_layer_neural_network = {}
local two_layer_neural_network_RegressionTrainer = {}
local two_layer_neural_network_RegressionMLP = {}

for i=1, k_collection:size() do
two_layer_neural_network_RegressionTrainer[i], two_layer_neural_network_RegressionMLP[i], various_test_error_for_different_k_2_layer_neural_network[i] = twoLayerNN(whitened_data_train_isolet_collection[i], whitened_data_train_isolet_collection[i])
end


-- For RBF Network,
local various_test_error_for_different_k_rbf_network = {}
local rbf_RegressionTrainer = {}
local rbf_RegressionMLP = {}

for i=1, k_collection:size() do
rbf_RegressionTrainer[i], rbf_RegressionMLP[i], various_test_error_for_different_k_rbf[i] = RBF(whitened_data_train_isolet_collection[i], whitened_data_train_isolet_collection[i])
end

--[[

-- For Q.4.2
-- For Beats All Models,
local various_test_error_for_different_k_beats_all_models = {}
local beats_all_models_RegressionTrainer = {}
local beats_all_models_RegressionMLP = {}

for i=1, k_collection:size() do
beats_all_models_RegressionTrainer[i], beats_all_models_RegressionMLP[i], various_test_error_for_different_k_beats_all_models[i] = beatsAllModels(whitened_data_train_isolet_collection[i], whitened_data_train_isolet_collection[i])
end

--]]

--Now let's plot it all.
    gnuplot.epsfigure('q_4_1_and_2.eps')
--	gnuplot.plot({'Deg=1', graphplotdatacollector[1],'-'},{'Deg=2', graphplotdatacollector[2],'-'},{'Deg=3', graphplotdatacollector[3],'-'},{'Deg=4', graphplotdatacollector[4],'-'})
	gnuplot.plot({'Logistic Regression', various_test_error_for_different_k_logistic_regression,'-'})
    gnuplot.plot({'2 Layer Neural Network', various_test_error_for_different_k_2_layer_neural_network,'-'})
    gnuplot.plot({'RBF', various_test_error_for_different_k_rbf,'-'})
    -- gnuplot.plot({'Beats All Models', various_test_error_for_different_k_beats_all_models,'-'})
	gnuplot.xlabel('k')
	gnuplot.ylabel('% test error')
	gnuplot.plotflush()


end

Q_4_1_and_2()
